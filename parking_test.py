# parking_test.py
import os
import time
from typing import Optional, Dict, Any

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch
from stable_baselines3 import SAC

# Modular import
from utils import MPSCompatibleWrapper, get_device

# ============================================================================
# CONFIGURATION
# ============================================================================
PARKING_CONFIG: Dict[str, Any] = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": True,
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
        "dynamical": True,
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 100,
    "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
    "screen_width": 600,
    "screen_height": 300,
    "show_trajectories": True,
}

DEFAULT_MODEL_PATH = "./models/parking_best_model.zip"


def find_model_path(custom_path: Optional[str] = None) -> Optional[str]:
    """Finds the model path from a custom input or the default."""
    if custom_path:
        if os.path.exists(custom_path):
            return custom_path
        if os.path.exists(f"{custom_path}.zip"):
            return f"{custom_path}.zip"
        print(f"❌ Custom path not found: {custom_path}")
        return None

    if os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH
    return None


def test_parking(model_path: Optional[str] = None, num_episodes: int = 5) -> None:
    """Runs the testing loop for the parking model."""
    
    final_path = find_model_path(model_path)
    if not final_path:
        print("❌ ERROR: No model found!")
        return

    print(f"\n🔍 Found Model: {final_path}")
    device = get_device()
    print(f"🚀 Device: {device.upper()}")

    print("🛠️  Preparing Environment...")
    env = gym.make("parking-v0", config=PARKING_CONFIG, render_mode="human")
    env = MPSCompatibleWrapper(env)

    print("📥 Loading Model...")
    try:
        model = SAC.load(
            final_path, 
            env=env, # Critical for HER
            device=device,
            print_system_info=True
        )
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        env.close()
        return

    print(f"\n🎬 Starting Test ({num_episodes} Episodes)...\n")
    total_success = 0
    
    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                env.render()
                # time.sleep(0.01) 
            
            is_success = info.get('is_success', False)
            if is_success: 
                total_success += 1
            
            status = "✅ SUCCESS" if is_success else "❌ FAIL"
            print(f"Episode {episode+1}: {status} | Reward: {episode_reward:.2f}")
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    finally:
        env.close()
        if num_episodes > 0:
            success_rate = (total_success / num_episodes) * 100
            print(f"\n📊 Success Rate: {total_success}/{num_episodes} ({success_rate:.1f}%)")

if __name__ == "__main__":
    test_parking()