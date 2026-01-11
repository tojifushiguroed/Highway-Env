from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
from stable_baselines3 import PPO

CONFIG: Dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 60,
    "initial_spacing": 2,
    "collision_reward": -50,
    "reward_speed_range": [30, 45],
    "high_speed_reward": 1.0,
    "right_lane_reward": 0.0,
    "lane_change_reward": 0,
    "simulation_frequency": 15,
    "policy_frequency": 5,
}

DEFAULT_MODEL_PATH = "./models/highway_aggressive.zip"


def resolve_model_path(model_path: str) -> Optional[str]:
    """Return an existing path, trying with/without .zip."""
    if os.path.exists(model_path):
        return model_path
    if os.path.exists(model_path + ".zip"):
        return model_path + ".zip"
    return None


def test_model(model_path: str, n_episodes: int = 5) -> None:
    """Run the model for N episodes with rendering."""
    final_path = resolve_model_path(model_path)
    if final_path is None:
        print(f"❌ Model not found: {model_path}")
        return

    print("\n" + "=" * 60)
    print("🚗 HIGHWAY MODEL TEST")
    print("=" * 60)
    print(f"📁 Model: {final_path}")
    print(f"🎮 Episodes: {n_episodes}")
    print("=" * 60)

    env = gym.make("highway-v0", config=CONFIG, render_mode="human")

    print("📥 Loading model...")
    try:
        model = PPO.load(final_path, device="cpu")
    except Exception as exc:
        print(f"❌ Error loading model: {exc}")
        env.close()
        return

    episode_rewards: list[float] = []
    crashes = 0
    successes = 0

    print("\n🚀 Testing...\n")
    try:
        for ep in range(n_episodes):
            obs, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                time.sleep(0.02)

            episode_rewards.append(total_reward)

            crashed = bool(info.get("crashed", False))
            if crashed:
                crashes += 1
                status = "💥 CRASH"
            else:
                successes += 1
                status = "✅ SUCCESS"

            print(
                f"Episode {ep + 1}/{n_episodes}: {status} | "
                f"Reward: {total_reward:.2f} | Steps: {steps}"
            )

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")
    finally:
        env.close()

    if episode_rewards:
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        print("\n📊 SUMMARY:")
        print(f"   • Mean reward:   {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   • Success rate:  {successes}/{n_episodes} ({100*successes/n_episodes:.0f}%)")
        print(f"   • Crash rate:    {crashes}/{n_episodes} ({100*crashes/n_episodes:.0f}%)")
    print("=" * 60)


def main() -> None:
    print("\n🚗 HIGHWAY TEST RUNNER")
    custom = input("Model path (Enter for default): ").strip()
    model_path = custom if custom else DEFAULT_MODEL_PATH

    episodes_str = input("Episodes (default 5): ").strip()
    try:
        n_episodes = int(episodes_str) if episodes_str else 5
    except ValueError:
        n_episodes = 5

    test_model(model_path, n_episodes)


if __name__ == "__main__":
    main()