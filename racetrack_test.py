# test_racetrack.py
import os
import time
from typing import Dict, Any, Optional

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Modular Import
from utils import MPSCompatibleWrapper, get_device


# ============================================================================
# CUSTOM CNN (must match training)
# ============================================================================
class CustomCNN(BaseFeaturesExtractor):
    """Custom CNN feature extractor for Racetrack (matches training config)."""
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
# CONFIGURATION
# ============================================================================
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
    "duration": 100,
}

DEFAULT_MODEL_PATH = "./models/racetrack_best_model.zip"

def test_model(model_path: str, n_episodes: int = 5) -> Optional[Dict[str, float]]:
    """Tests the Racetrack model and prints metrics."""
    
    print("\n" + "=" * 60)
    print("🏎️ RACETRACK TEST")
    print("=" * 60)
    print(f"📁 Model: {model_path}")
    
    full_path = model_path if model_path.endswith('.zip') else model_path + ".zip"
    if not os.path.exists(full_path) and not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return None
    
    # Environment Setup
    env = gym.make("racetrack-v0", config=RACETRACK_CONFIG, render_mode="human")
    env = MPSCompatibleWrapper(env) # Standardize with wrapper
    
    print("📥 Loading Model...")
    # Must pass policy_kwargs so the correct architecture is used
    custom_objects = {
        "policy_kwargs": {
            "features_extractor_class": CustomCNN,
            "features_extractor_kwargs": {"features_dim": 256}
        }
    }
    model = SAC.load(model_path, device=get_device(), custom_objects=custom_objects)
    print("✅ Model loaded!")
    
    episode_rewards = []
    episode_lengths = []
    crashes = 0
    successes = 0 
    
    print(f"\n🚀 Starting Test ({n_episodes} Episodes)...\n")
    
    try:
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                steps += 1
                done = terminated or truncated
                
                env.render()
                time.sleep(0.02)
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Metric logic: 150+ steps usually implies completing a significant portion
            if steps >= 150:
                successes += 1
                status = "✅ SUCCESS"
            elif total_reward < 0:
                crashes += 1
                status = "💥 CRASH"
            else:
                status = "🏁 FINISHED"
            
            print(f"   Episode {episode+1}: {status} | Reward: {total_reward:.2f} | Steps: {steps}")
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    finally:
        env.close()
    
    # Statistics
    mean_rew = np.mean(episode_rewards)
    print(f"\n📊 RESULTS:")
    print(f"   • Mean Reward: {mean_rew:.2f} ± {np.std(episode_rewards):.2f}")
    print(f"   • Success Rate: {successes}/{n_episodes} ({(successes/n_episodes)*100:.0f}%)")
    print("=" * 60)
    
    return {
        "mean_reward": float(mean_rew),
        "success_rate": float(successes / n_episodes)
    }

if __name__ == "__main__":
    print("\n🏎️ RACETRACK TEST RUNNER")
    
    custom_path = input(f"Model Path (Enter for default): ").strip()
    model_path = custom_path if custom_path else DEFAULT_MODEL_PATH
    
    try:
        episodes_input = input("Episodes (Default 5): ").strip()
        n_episodes = int(episodes_input) if episodes_input else 5
    except ValueError:
        n_episodes = 5
    
    test_model(model_path, n_episodes)