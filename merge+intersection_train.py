# merge_intersection_train.py
import os
import gc
from typing import Dict, Any, List, Callable

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import multiprocessing as mp

# ===========================================================================
# 1. SHARED CONFIGURATION
# ===========================================================================
PPO_KWARGS: Dict[str, Any] = dict(
    learning_rate=3e-4,
    n_steps=512,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    device="cpu",  # Merge/Intersection are light, CPU is often faster/stable
)

KINEMATICS_OBS: Dict[str, Any] = {
    "type": "Kinematics",
    "vehicles_count": 10,
    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    "absolute": False,
    "normalize": True,
}

# ===========================================================================
# 2. ENVIRONMENT SPECS
# ===========================================================================
ENV_SPECS: List[Dict[str, Any]] = [
    {
        "name": "merge",
        "env_id": "merge-v0",
        "n_envs": 8,
        "timesteps": 200_000,
        "config": {
            "observation": KINEMATICS_OBS,
            "action": {"type": "DiscreteMetaAction"},
            "collision_reward": -10,
            "high_speed_reward": 0.4,
            "right_lane_reward": 0.1,
            "simulation_frequency": 15,
            "policy_frequency": 5,
        },
    },
    {
        "name": "intersection",
        "env_id": "intersection-v0",
        "n_envs": 8,
        "timesteps": 250_000,
        "config": {
            "observation": KINEMATICS_OBS,
            "action": {"type": "DiscreteMetaAction"},
            "collision_reward": -5,
            "high_speed_reward": 0.4,
            "simulation_frequency": 15,
            "policy_frequency": 5,
        },
    },
]

def make_env(env_id: str, config: Dict[str, Any]) -> Callable[[], gym.Env]:
    """Factory function for creating environments."""
    def _init() -> gym.Env:
        env = gym.make(env_id, config=config, render_mode=None)
        # Optional: Wrap if you face float32 issues, otherwise standard is fine here
        # env = MPSCompatibleWrapper(env) 
        return env
    return _init

def train_one(spec: Dict[str, Any]) -> None:
    """Trains a single environment specification."""
    name = spec["name"]
    env_id = spec["env_id"]
    n_envs = spec["n_envs"]
    total_timesteps = spec["timesteps"]
    config = spec["config"]

    print("\n" + "=" * 70)
    print(f"🚀 PREPARING: {name.upper()} ({env_id})")
    print("=" * 70)

    # ---------------------------------------------------------
    # 🧪 STAGE 1: PRE-TEST (Config Check)
    # ---------------------------------------------------------
    try:
        print("🔍 Checking configuration...")
        test_env = gym.make(env_id, config=config)
        test_env.reset()
        test_env.step(test_env.action_space.sample())
        test_env.close()
        print("✅ Config is valid.")
    except Exception as e:
        print(f"❌ ERROR: Invalid config for {name}! Skipping... ({e})")
        return

    # ---------------------------------------------------------
    # 🏋️ STAGE 2: TRAINING ENVIRONMENT
    # ---------------------------------------------------------
    env = SubprocVecEnv([make_env(env_id, config) for _ in range(n_envs)])

    # Directory Setup
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    model = PPO("MlpPolicy", env, tensorboard_log="./logs", **PPO_KWARGS)

    # ---------------------------------------------------------
    # 📊 STAGE 3: CALLBACKS
    # ---------------------------------------------------------
    eval_env_instance = gym.make(env_id, config=config, render_mode=None)
    eval_env = Monitor(eval_env_instance)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models",
        log_path="./logs",
        eval_freq=max(1, 10_000 // n_envs),
        deterministic=True,
        render=False,
    )

    print(f"🔥 Training started: {total_timesteps:,} steps")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback],
        progress_bar=True,
        tb_log_name=f"PPO_{name}",
    )

    # Save Final Model
    final_path = f"./models/{name}_model"
    model.save(final_path)

    print(f"✅ {name} Finished. Saved: {final_path}.zip")

    # ---------------------------------------------------------
    # 🧹 Cleanup
    # ---------------------------------------------------------
    env.close()
    eval_env.close()
    del model
    del env
    del eval_env
    gc.collect()

if __name__ == "__main__":
    import multiprocessing as mp
...
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    for spec in ENV_SPECS:
        train_one(spec)