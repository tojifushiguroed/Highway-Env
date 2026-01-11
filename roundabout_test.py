# test_roundabout.py
import os
import sys
import time
from typing import Any, Dict, Optional, Sequence

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from stable_baselines3 import PPO

from utils import MPSCompatibleWrapper, get_device

ROUNDABOUT_CONFIG: Dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "absolute": False,
        "normalize": True,
        "see_behind": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 100,
    "collision_reward": -50,
}

DEFAULT_MODEL_PATH = "./models/roundabout_best_model.zip"


def find_model_path(custom_path: Optional[str] = None) -> Optional[str]:
    """Finds the model path from a custom input or the default."""
    if custom_path:
        if os.path.exists(custom_path):
            return custom_path
        if os.path.exists(f"{custom_path}.zip"):
            return f"{custom_path}.zip"
        print(f"❌ Custom path not found: {custom_path}")
        return None

    # Use default path
    if os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH
    return None


def test_roundabout(model_path: Optional[str] = None, n_episodes: int = 5) -> None:
    """Runs a PPO policy in roundabout-v0 and prints basic statistics."""
    print("\n" + "=" * 60)
    print("🔄 ROUNDABOUT TEST")
    print("=" * 60)

    final_path = find_model_path(model_path)
    if not final_path:
        print("❌ ERROR: No model found! Please train first or check paths.")
        sys.exit(1)

    device = get_device()
    print(f"📁 Model: {final_path}")
    print(f"🚀 Device: {device.upper()}")

    env = gym.make("roundabout-v0", config=ROUNDABOUT_CONFIG, render_mode="human")
    env = MPSCompatibleWrapper(env)

    try:
        model = PPO.load(final_path, device=device)
    except Exception as exc:
        print(f"❌ Error loading model: {exc}")
        env.close()
        return

    successes = 0
    crashes = 0
    episode_rewards = []

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                time.sleep(0.01)

            episode_rewards.append(total_reward)

            is_crashed = bool(info.get("crashed", False))
            if is_crashed:
                crashes += 1
                status = "💥 CRASH"
            else:
                successes += 1
                status = "✅ SUCCESS"

            print(f"Episode {ep + 1}: {status} | Reward: {total_reward:.2f} | Steps: {steps}")
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    finally:
        env.close()

    if n_episodes > 0:
        print("\n📊 STATISTICS:")
        print(f"   • Success Rate: {successes}/{n_episodes} ({100 * successes / n_episodes:.1f}%)")
        print(f"   • Crash Rate:   {crashes}/{n_episodes} ({100 * crashes / n_episodes:.1f}%)")
        print(f"   • Avg Reward:   {np.mean(episode_rewards):.2f}")
        print("=" * 60)


if __name__ == "__main__":
    print("\n🔄 ROUNDABOUT TEST RUNNER")
    user_path = input("Model Path (Enter for auto): ").strip() or None

    try:
        ep_input = input("Episodes (Default 5): ").strip()
        episodes = int(ep_input) if ep_input else 5
    except ValueError:
        episodes = 5

    test_roundabout(user_path, episodes)