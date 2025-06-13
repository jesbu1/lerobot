from gymnasium import spaces
import gymnasium as gym


class BRIDGEEnv(gym.Env):
    """Wrapper for the BRIDGE client-server environment. This will act as a server that receives images and actions from the client and returns the next observation.

    Args:
        gym (_type_): _description_
    """
