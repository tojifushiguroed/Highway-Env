# parking_training.py
import os
from typing import Dict, Any, List

import gymnasium as gym
import highway_env  # noqa: F401
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

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
}

def make_env() -> gym.Env:
    """Creates and wraps the parking environment."""
    env = gym.make("parking-v0", config=PARKING_CONFIG)
    env = MPSCompatibleWrapper(env)
    return env

def train() -> None:
    """Main training loop for the Parking environment."""
    
    # Device setup
    torch.set_default_dtype(torch.float32)
    device = get_device()
    print(f"🚀 Training Device: {device.upper()}")

    # Paths
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Environments
    env = DummyVecEnv([make_env]) 
    eval_env = DummyVecEnv([make_env])

    # Model Setup
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_rate=1e-3,
        learning_starts=1000, 
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.05,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256, 256]), 
        device=device,
        tensorboard_log="./logs",
        verbose=1
    )

    # Callbacks
    callbacks = [
        EvalCallback(
            eval_env, 
            best_model_save_path="./models",
            log_path="./logs",
            eval_freq=2000,
            n_eval_episodes=10,
            deterministic=True,
            render=False
        )
    ]

    print("\n🅿️  PARKING TRAINING STARTED (SAC+HER)")
    print("=" * 50)

    try:
        model.learn(
            total_timesteps=300_000,
            callback=callbacks,
            progress_bar=True
        )
        print("\n✅ Training Completed!")
        model.save("./models/parking_best_model")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted! Saving current model...")
        model.save("./models/parking_best_model")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        raise
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    train()