from gymnasium import spaces
import gymnasium as gym
import numpy as np
from typing_extensions import TypedDict


class WidowXMessageFormat(TypedDict):
    """Message format for the WidowX client-server environment."""

    state: np.ndarray
    prompt: str  # from the eval_widowx code
    images: dict[str, np.ndarray]
    reset: bool


class WidowXEnv(gym.Env):
    """Gym compatible env for the WidowX client-server environment.

    Args:
        gym (_type_): _description_
    """

    def __init__(self, resolution: int = 256):
        self._max_episode_steps = float("inf")
        self.render_mode = "rgb_array"
        self.metadata = {"render_fps": 10}
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        "external": spaces.Box(0, 255, shape=(resolution, resolution, 3), dtype=np.uint8),
                        "over_shoulder": spaces.Box(
                            0, 255, shape=(resolution, resolution, 3), dtype=np.uint8
                        ),
                    }
                ),
                "agent_pos": spaces.Box(-np.inf, np.inf, shape=(8,)),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,))
        self.spec = {}
        self.spec["max_episode_steps"] = self._max_episode_steps

    def reset(self, seed=None, **kwargs):
        pass

    def step(self, action):
        pass
