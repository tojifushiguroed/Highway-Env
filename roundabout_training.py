# roundabout_training.py
import os
import multiprocessing
from typing import Any, Dict, Tuple

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from utils import MPSCompatibleWrapper, get_device

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

TOTAL_CORES = multiprocessing.cpu_count()
NUM_ENVS = max(4, TOTAL_CORES - 2)

SYSTEM_DEVICE = get_device()
TRAIN_DEVICE = "cpu"  # PPO çoğu zaman CPU'da daha stabil/hızlı

print(f"🚀 System Capability: {SYSTEM_DEVICE.upper()}")
print(f"⚙️  Training Device:   {TRAIN_DEVICE.upper()}")
print(f"🔥 Parallel Envs:      {NUM_ENVS}")


class SafeRoundaboutWrapper(gym.Wrapper):
    """Reward shaping wrapper for safer roundabout navigation."""

    def __init__(self, env: gym.Env, danger_distance: float = 8.0, caution_distance: float = 15.0):
        super().__init__(env)
        self.danger_distance = danger_distance
        self.caution_distance = caution_distance
        self.last_speed = 0.0

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.last_speed = 0.0
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        try:
            vehicle = self.env.unwrapped.vehicle
            road = self.env.unwrapped.road
            current_speed = float(vehicle.speed)
        except Exception:
            return obs, float(reward), terminated, truncated, info

        min_distance = self._get_min_vehicle_distance(vehicle, road)

        danger_penalty = 0.0
        if min_distance < self.danger_distance:
            danger_penalty = -0.8
        elif min_distance < self.caution_distance:
            ratio = (self.caution_distance - min_distance) / (self.caution_distance - self.danger_distance)
            danger_penalty = -0.3 * float(ratio)

        safe_distance_bonus = 0.1 if min_distance >= self.caution_distance else 0.0

        slowdown_bonus = 0.0
        if min_distance < self.caution_distance:
            if current_speed < self.last_speed:
                slowdown_bonus = 0.2
            elif current_speed > 15.0:
                slowdown_bonus = -0.2

        self.last_speed = current_speed

        survival_bonus = 0.05
        exit_bonus = 1.0 if truncated and not terminated else 0.0

        modified_reward = (
            float(reward)
            + danger_penalty
            + safe_distance_bonus
            + slowdown_bonus
            + survival_bonus
            + exit_bonus
        )

        info.update(
            {
                "min_distance": float(min_distance),
                "danger_penalty": float(danger_penalty),
                "slowdown_bonus": float(slowdown_bonus),
                "current_speed": float(current_speed),
            }
        )

        return obs, modified_reward, terminated, truncated, info

    @staticmethod
    def _get_min_vehicle_distance(vehicle: Any, road: Any) -> float:
        min_dist = 100.0
        vehicles = getattr(road, "vehicles", [])
        if not vehicles:
            return min_dist

        for other in vehicles:
            if other is vehicle:
                continue
            dx = float(other.position[0] - vehicle.position[0])
            dy = float(other.position[1] - vehicle.position[1])
            dist = float(np.hypot(dx, dy))
            min_dist = min(min_dist, dist)

        return float(min_dist)


ROUNDABOUT_FINETUNE_CONFIG: Dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 20,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "absolute": False,
        "normalize": True,
        "see_behind": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 60,
    "collision_reward": -60,
    "high_speed_reward": 0.0,
    "right_lane_reward": 0.05,
    "reward_speed_range": [5, 15],
}


def make_env() -> gym.Env:
    env = gym.make("roundabout-v0", config=ROUNDABOUT_FINETUNE_CONFIG, render_mode=None)
    env = SafeRoundaboutWrapper(env, danger_distance=8.0, caution_distance=15.0)
    env = MPSCompatibleWrapper(env)  # dtype standardizasyonu (M4 için iyi)
    return env


class RoundaboutMonitor(BaseCallback):
    """Logs safe driving metrics periodically."""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps % 2000 == 0:
            infos = self.locals.get("infos", [])
            distances = [i.get("min_distance", 100.0) for i in infos]
            speeds = [i.get("current_speed", 0.0) for i in infos]
            if distances and speeds:
                print(
                    f"📊 Step {self.num_timesteps:,} | "
                    f"Min Dist: {float(np.mean(distances)):.1f}m | "
                    f"Speed: {float(np.mean(speeds)):.1f}"
                )
        return True


def setup_directories() -> None:
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)


def finetune_roundabout(base_model_path: str) -> None:
    total_timesteps = 200_000
    setup_directories()

    env = VecMonitor(SubprocVecEnv([make_env for _ in range(NUM_ENVS)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env]))

    if not base_model_path.endswith(".zip") and os.path.exists(base_model_path + ".zip"):
        base_model_path = base_model_path + ".zip"

    if not os.path.exists(base_model_path):
        print(f"❌ Error: Model file not found at {base_model_path}")
        env.close()
        eval_env.close()
        return

    print(f"📥 Loading Base Model: {base_model_path}")

    model = PPO.load(
        base_model_path,
        env=env,
        device=TRAIN_DEVICE,
        custom_objects={
            "learning_rate": 5e-5,
            "clip_range": 0.1,
            "ent_coef": 0.005,
        },
    )
    model.tensorboard_log = "./logs"

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path="./models",
            eval_freq=max(1, 10_000 // NUM_ENVS),
            deterministic=True,
        ),
        RoundaboutMonitor(),
    ]

    print("\n🔄 ROUNDABOUT FINETUNING STARTED")
    print(f"   • Steps: {total_timesteps:,}")
    print("   • LR: 5e-5 | clip: 0.1 | ent: 0.005")
    print("=" * 50)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    model.save("./models/roundabout_best_model")
    env.close()
    eval_env.close()
    print("✅ Done! Model saved to: ./models/roundabout_best_model.zip")


def train_from_scratch() -> None:
    total_timesteps = 400_000
    setup_directories()

    env = VecMonitor(SubprocVecEnv([make_env for _ in range(NUM_ENVS)]))
    eval_env = VecMonitor(SubprocVecEnv([make_env]))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device=TRAIN_DEVICE,
        tensorboard_log="./logs",
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path="./models",
            eval_freq=max(1, 10_000 // NUM_ENVS),
            deterministic=True,
        ),
        RoundaboutMonitor(),
    ]

    print("\n🔄 ROUNDABOUT TRAINING STARTED (FROM SCRATCH)")
    print("=" * 50)

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    model.save("./models/roundabout_best_model")
    env.close()
    eval_env.close()
    print("✅ Done! Model saved to: ./models/roundabout_best_model.zip")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    print("\n" + "=" * 60)
    print("🔄 ROUNDABOUT SAFETY TRAINER")
    print("=" * 60)
    print("1. Finetune Existing Model (Recommended)")
    print("2. Train from Scratch")

    choice = input("\nSelect (1/2): ").strip()

    if choice == "1":
        default_path = "./training/roundabout/roundabout_ppo_model.zip"
        print(f"\nDefault Path: {default_path}")
        custom_path = input("Enter model path (Enter for default): ").strip()
        target_path = custom_path if custom_path else default_path
        finetune_roundabout(target_path)
    elif choice == "2":
        train_from_scratch()
    else:
        print("❌ Invalid selection.")