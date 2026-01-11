# test_merge.py
import sys
import os
from typing import Dict, Any, Optional

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO

# Modular Import
from utils import MPSCompatibleWrapper, get_device

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================
KINEMATICS_OBS: Dict[str, Any] = {
    "type": "Kinematics",
    "vehicles_count": 10,
    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    "absolute": False,
    "normalize": True,
}

MERGE_CONFIG: Dict[str, Any] = {
    "observation": KINEMATICS_OBS,
    "action": {"type": "DiscreteMetaAction"},
    "collision_reward": -10,
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.1,
    "simulation_frequency": 15,
    "policy_frequency": 5,
}

DEFAULT_MODEL_PATH = "./models/merge_model.zip"


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


def test_merge() -> None:
    """Main testing loop for Merge environment."""
    
    model_path = find_model_path()
    if model_path is None:
        print("❌ ERROR: No trained model found for Merge!")
        sys.exit(1)

    print(f"📂 Loading Model: {model_path}")
    device = get_device()
    print(f"🚀 Device: {device.upper()}")

    env = gym.make("merge-v0", config=MERGE_CONFIG, render_mode="human")
    # Wrap for safety on Mac/MPS
    env = MPSCompatibleWrapper(env)

    model = PPO.load(model_path, device=device)

    episodes = 5
    
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step = 0
        
        print(f"\n🎬 Episode {episode} starting...")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            env.render()
            
        print(f"🏁 Episode {episode} finished. Reward: {total_reward:.2f} | Steps: {step}")
        
        if info.get("crashed", False):
            print("💥 RESULT: Crash!")
        else:
            print("✅ RESULT: Success / Timeout.")

    env.close()

if __name__ == "__main__":
    test_merge()