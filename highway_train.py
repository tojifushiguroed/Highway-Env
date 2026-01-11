"""
Highway Model v4 - Smart Aggressive Overtaking (Fixed Logic)

Continues training from a v3 model if available.
Goal: High-speed overtaking without "wiggle" behavior.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# ============================================================================
# CONFIGURATION (Smart Aggressive)
# ============================================================================
CONFIG: Dict[str, Any] = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,
    "initial_spacing": 2,
    "collision_reward": -50,
    "reward_speed_range": [30, 45],
    "high_speed_reward": 1.0,
    "right_lane_reward": 0.0,
    "lane_change_reward": 0,
    "simulation_frequency": 15,
    "policy_frequency": 5,
}

BASE_MODEL_PATH = "./models/highway_aggressive"
FINAL_MODEL_PATH = "./models/highway_aggressive"

N_ENVS = 8
TOTAL_TIMESTEPS = 100_000


def make_env(config: Dict[str, Any]) -> Callable[[], gym.Env]:
    """Factory for multiprocessing environments."""

    def _init() -> gym.Env:
        return gym.make("highway-v0", config=config, render_mode=None)

    return _init


def resolve_model_path(path: str) -> Optional[str]:
    """Return existing model path, trying with/without .zip."""
    if os.path.exists(path):
        return path
    if os.path.exists(path + ".zip"):
        return path + ".zip"
    return None


def create_vec_env(n_envs: int) -> SubprocVecEnv:
    """Create the vectorized training environment."""
    return SubprocVecEnv([make_env(CONFIG) for _ in range(n_envs)])


def main() -> None:
    print("=" * 60)
    print("🚗 HIGHWAY v4 - SMART AGGRESSIVE TRAINING")
    print("=" * 60)

    env = create_vec_env(N_ENVS)

    base_path = resolve_model_path(BASE_MODEL_PATH)
    reset_timesteps = True

    if base_path is None:
        print(f"⚠️ Base model not found: {BASE_MODEL_PATH}(.zip)")
        print("➡️  Training from scratch...")
        model = PPO("MlpPolicy", env, verbose=1)
        reset_timesteps = True
    else:
        print(f"📥 Loading base model: {base_path}")
        model = PPO.load(base_path, env=env)
        print("✅ Model loaded!")
        reset_timesteps = False

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    print("\n🚀 Training started:")
    print(f"   • Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"   • Traffic vehicles: {CONFIG['vehicles_count']}")
    print(f"   • Target speed range: {CONFIG['reward_speed_range']}")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            progress_bar=True,
            reset_num_timesteps=reset_timesteps,
        )
    finally:
        model.save(FINAL_MODEL_PATH)
        env.close()

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED!")
    print(f"📁 Saved model: {FINAL_MODEL_PATH}.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()