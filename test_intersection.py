# visualize_intersection.py
import os
from typing import Dict, Any, Optional

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO

# Modular Import
from utils import MPSCompatibleWrapper, get_device

ENV_NAME = "intersection"
ENV_ID = f"{ENV_NAME}-v0"

CONFIG: Dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "absolute": False,
        "normalize": True
    },
    "action": {"type": "DiscreteMetaAction"},
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 60 
}

DEFAULT_MODEL_PATH = "./models/intersection_model.zip"


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


def run_visualization() -> None:
    """Runs the visualization loop."""
    model_path = find_model_path()
    if not model_path:
        print(f"❌ No model found for {ENV_NAME}")
        return

    print(f"📂 Loading: {model_path}")
    device = get_device()
    env = gym.make(ENV_ID, config=CONFIG, render_mode="human")
    env = MPSCompatibleWrapper(env)
    
    model = PPO.load(model_path, device=device)

    print("🎥 Simulation starting... (Close window to exit)")

    for episode in range(5):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            
        print(f"Episode {episode+1} Finished. Score: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    run_visualization()