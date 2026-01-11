# train_racetrack.py
import os
import multiprocessing
from typing import Dict, Any, List, Optional, Union, Tuple

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Modular Import
from utils import get_device, MPSCompatibleWrapper

# ============================================================================
# 1. SETUP & CONSTANTS
# ============================================================================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DEVICE = get_device()
TOTAL_CORES = multiprocessing.cpu_count()
NUM_ENVS = max(4, TOTAL_CORES - 2)

print(f"🚀 Device: {DEVICE} | Parallel Envs: {NUM_ENVS}")

M4_TRAIN_FREQ = (64, "step")
M4_GRADIENT_STEPS = 64
M4_BATCH_SIZE = 512  # TYPO FIXED (was 5122)

# ============================================================================
# 2. CUSTOM CNN
# ============================================================================
class CustomCNN(BaseFeaturesExtractor):
    """Custom CNN feature extractor for Racetrack."""
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# ============================================================================
# 3. WRAPPER & CONFIG
# ============================================================================
class CompletionWrapper(gym.Wrapper):
    """
    Racetrack completion wrapper.
    Awards survival, progress, and speed bonuses; penalizes off-road.
    """
    def __init__(self, env: gym.Env, min_speed: float = 10, max_speed: float = 25, speed_bonus_scale: float = 0.3):
        super().__init__(env)
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed_bonus_scale = speed_bonus_scale
        self.last_position: Optional[np.ndarray] = None
        
    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.last_position = None
        obs = self._sanitize_obs(obs)
        return obs, info
        
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        current_speed = 0.0
        try:
            current_speed = float(self.env.unwrapped.vehicle.speed)
        except AttributeError:
            pass
        
        on_road = True
        try:
            on_road = self.env.unwrapped.vehicle.on_road
        except AttributeError:
            pass
        
        # Bonuses / Penalties
        survival_bonus = 0.1 if on_road else 0.0
        offroad_penalty = -10.0 if not on_road else 0.0
        
        progress_bonus = 0.0
        try:
            current_pos = np.array(self.env.unwrapped.vehicle.position, dtype=np.float32)
            if self.last_position is not None and on_road:
                dist = np.linalg.norm(current_pos - self.last_position)
                progress_bonus = float(dist * 0.4)
            self.last_position = current_pos.copy()
        except AttributeError:
            pass
        
        speed_bonus = 0.0
        if on_road:
            if current_speed >= self.max_speed:
                speed_bonus = self.speed_bonus_scale
            elif current_speed >= self.min_speed:
                p = (current_speed - self.min_speed) / (self.max_speed - self.min_speed + 1e-6)
                speed_bonus = float(p * self.speed_bonus_scale)
            else:
                speed_bonus = -0.05
        
        modified_reward = float(reward) + speed_bonus + progress_bonus + survival_bonus + offroad_penalty
        
        info.update({
            'speed_bonus': speed_bonus,
            'progress_bonus': progress_bonus,
            'survival_bonus': survival_bonus,
            'offroad_penalty': offroad_penalty,
            'current_speed': current_speed,
            'on_road': on_road
        })
        
        obs = self._sanitize_obs(obs)
        return obs, modified_reward, terminated, truncated, info
    
    @staticmethod
    def _sanitize_obs(obs: np.ndarray) -> np.ndarray:
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

RACETRACK_CONFIG: Dict[str, Any] = {
    "observation": {
        "type": "OccupancyGrid",
        "features": ["presence", "on_road"],
        "grid_size": [[-18, 18], [-18, 18]],
        "grid_step": [0.5, 0.5],
        "as_image": True,
        "align_to_vehicle_axes": True
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
        "speed_range": [0, 30],
        "steering_range": [-0.5, 0.5]
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 300,
    "collision_reward": -50,
    "offroad_terminal": True,
    "lane_centering_reward": 0.4,
    "action_reward": -0.01,
}

def make_env() -> gym.Env:
    """Factory function for Racetrack."""
    env = gym.make("racetrack-v0", config=RACETRACK_CONFIG, render_mode=None)
    env = CompletionWrapper(env, min_speed=10, max_speed=25, speed_bonus_scale=0.3)
    # Apply MPS Wrapper for compatibility
    env = MPSCompatibleWrapper(env)
    return env

# ============================================================================
# 4. CALLBACKS
# ============================================================================
class ProgressMonitorCallback(BaseCallback):
    """Logs speed and progress metrics periodically."""
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            try:
                infos = self.locals.get('infos', [])
                speeds = [i.get('current_speed', 0) for i in infos if 'current_speed' in i]
                progs = [i.get('progress_bonus', 0) for i in infos if 'progress_bonus' in i]
                if speeds:
                    print(f"📊 Step {self.num_timesteps:,} | Spd: {np.mean(speeds):.1f} | Prg: {np.mean(progs):.3f}")
            except Exception: 
                pass
        return True

# ============================================================================
# 5. TRAINING FUNCTIONS
# ============================================================================
def train_from_scratch() -> None:
    """Starts training from scratch."""
    TOTAL_TIMESTEPS = 500_000
    
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    env = VecMonitor(SubprocVecEnv([make_env for _ in range(NUM_ENVS)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env]))
    
    policy_kwargs = dict(features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=256))
    
    model = SAC(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=10_000,
        batch_size=M4_BATCH_SIZE,
        train_freq=M4_TRAIN_FREQ,
        gradient_steps=M4_GRADIENT_STEPS,
        ent_coef='auto',
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=DEVICE,
        tensorboard_log="./logs"
    )
    
    callbacks = [
        EvalCallback(eval_env, best_model_save_path="./models", eval_freq=25_000, deterministic=True),
        ProgressMonitorCallback()
    ]
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
    model.save("./models/racetrack_best_model")
    env.close()

def retrain_existing(base_model_path: str) -> None:
    """Continues training from an existing model."""
    TOTAL_TIMESTEPS = 400_000
    
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    print("📥 Loading Model...")
    model = SAC.load(base_model_path, device=DEVICE)
    
    model_num_envs = model.n_envs
    print(f"⚙️ Using {model_num_envs} environments (from loaded model)...")
    
    env = VecMonitor(SubprocVecEnv([make_env for _ in range(model_num_envs)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env]))
    
    model.set_env(env)
    model.tensorboard_log = "./logs"
    
    callbacks = [
        EvalCallback(eval_env, best_model_save_path="./models", eval_freq=25_000, deterministic=True),
        ProgressMonitorCallback()
    ]
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True, reset_num_timesteps=False)
    model.save("./models/racetrack_best_model")
    env.close()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    print("\n🏁 RACETRACK MASTER")
    print("1. Train from Scratch")
    print("2. Retrain Existing Model")
    choice = input("Choice: ").strip()
    
    if choice == "1":
        train_from_scratch()
    else:
        path = input("Model Path: ").strip()
        if os.path.exists(path):
            retrain_existing(path)
        else:
            print("❌ File not found!")