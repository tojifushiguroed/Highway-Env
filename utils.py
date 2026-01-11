# utils.py
import gymnasium as gym
import numpy as np
import torch
import sys

def get_device() -> str:
    """Detects and returns the best available device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

class MPSCompatibleWrapper(gym.ObservationWrapper):
    """
    A wrapper to ensure observation and action spaces use float32.
    Critical for running PyTorch on Apple Silicon (MPS) or strictly typed environments.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # Fix Observation Space
        if isinstance(env.observation_space, gym.spaces.Dict):
            new_spaces = {}
            for key, space in env.observation_space.spaces.items():
                if isinstance(space, gym.spaces.Box):
                    low = space.low.astype(np.float32)
                    high = space.high.astype(np.float32)
                    new_spaces[key] = gym.spaces.Box(low=low, high=high, dtype=np.float32)
                else:
                    new_spaces[key] = space
            self.observation_space = gym.spaces.Dict(new_spaces)
        elif isinstance(env.observation_space, gym.spaces.Box):
            low = env.observation_space.low.astype(np.float32)
            high = env.observation_space.high.astype(np.float32)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Fix Action Space
        if isinstance(env.action_space, gym.spaces.Box):
            self.action_space = gym.spaces.Box(
                low=env.action_space.low.astype(np.float32),
                high=env.action_space.high.astype(np.float32),
                dtype=np.float32
            )

    def observation(self, observation: any) -> any:
        """Casts observation to float32."""
        if isinstance(observation, dict):
            return {k: v.astype(np.float32) for k, v in observation.items()}
        return observation.astype(np.float32)