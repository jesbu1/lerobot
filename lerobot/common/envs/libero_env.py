import numpy as np
import os
import pathlib
import math
import h5py
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark
from gymnasium import spaces
import gymnasium as gym
from lerobot.common.envs.utils import convert_to_uint8, resize_with_pad


class LIBEROEnv(gym.Env):
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

    def __init__(
        self,
        task_suite_name: str,
        seed: int,
        task_idx: int,
        episode_idx: int,
        resolution: int = 256,
        libero_hdf5_dir: str = None,
        load_gt_initial_states: bool = False,
    ):
        super().__init__()
        self.LIBERO_ENV_RESOLUTION = resolution
        self.num_steps_wait = 10
        self._task = None
        self.task_suite_name = task_suite_name
        self.seed = seed
        if task_suite_name == "libero_spatial":
            self._max_episode_steps = 220  # longest training demo has 193 steps
        elif task_suite_name == "libero_object":
            self._max_episode_steps = 280  # longest training demo has 254 steps
        elif task_suite_name == "libero_goal":
            self._max_episode_steps = 300  # longest training demo has 270 steps
        elif task_suite_name == "libero_10":
            self._max_episode_steps = 520  # longest training demo has 505 steps
        elif task_suite_name == "libero_90":
            self._max_episode_steps = 400  # longest training demo has 373 steps
        else:
            raise ValueError(f"Unknown task suite: {task_suite_name}")
        benchmark_dict = benchmark.get_benchmark_dict()
        self._libero_task_suite = benchmark_dict[self.task_suite_name]()
        self._libero_hdf5_dir = libero_hdf5_dir
        self.load_gt_initial_states = load_gt_initial_states
        self.current_step = 0
        self.set_task_idx(task_idx)
        self.set_episode_idx(episode_idx)
        # load dummy env first
        # env, _ = self._get_libero_env()
        self.metadata = {"render_fps": 10}
        self.render_mode = "rgb_array"
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        "image": spaces.Box(
                            0,
                            255,
                            shape=(self.LIBERO_ENV_RESOLUTION, self.LIBERO_ENV_RESOLUTION, 3),
                            dtype=np.uint8,
                        ),
                        "image_wrist": spaces.Box(
                            0,
                            255,
                            shape=(self.LIBERO_ENV_RESOLUTION, self.LIBERO_ENV_RESOLUTION, 3),
                            dtype=np.uint8,
                        ),
                    }
                ),
                "agent_pos": spaces.Box(-np.inf, np.inf, shape=(8,)),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,))
        self.spec = {}
        self.spec["max_episode_steps"] = self._max_episode_steps

    @property
    def task(self):
        return str(self._task)

    @property
    def episode_idx(self):
        return self._episode_idx

    @property
    def num_tasks(self):
        return self._libero_task_suite.n_tasks

    def set_task_idx(self, task_idx: int):
        self._task_idx = task_idx

    def set_episode_idx(self, episode_idx: int):
        self._episode_idx = episode_idx

    def _construct_obs(self, obs):
        flipped_agentview = obs["agentview_image"][::-1]
        flipped_eye_in_hand = obs["robot0_eye_in_hand_image"][::-1]
        new_obs = {}
        pixels = {}
        # following stupid lerobot hardcoded pixels naming...
        pixels["image"] = convert_to_uint8(
            resize_with_pad(flipped_agentview, self.LIBERO_ENV_RESOLUTION, self.LIBERO_ENV_RESOLUTION)
        )
        pixels["image_wrist"] = convert_to_uint8(
            resize_with_pad(flipped_eye_in_hand, self.LIBERO_ENV_RESOLUTION, self.LIBERO_ENV_RESOLUTION)
        )
        new_obs["pixels"] = pixels
        # following stupid lerobot hardcoded agent_pos naming...
        new_obs["agent_pos"] = np.concatenate(
            [
                obs["robot0_eef_pos"],
                LIBEROEnv._quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ]
        )
        return new_obs

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.seed = seed
        self.env, initial_states = self._get_libero_env()
        obs = self.env.reset(**kwargs)

        if self.load_gt_initial_states:
            self.env.set_init_state(initial_states)
        current_steps_waited = 0
        while current_steps_waited < self.num_steps_wait:
            obs, _, _, info = self.env.step(self.LIBERO_DUMMY_ACTION)
            current_steps_waited += 1
        self.current_step = 0
        self.obs = self._construct_obs(obs)
        return self.obs, info

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        self.current_step += 1
        truncated = self.current_step >= self._max_episode_steps
        self.obs = self._construct_obs(obs)
        if terminated:
            info["is_success"] = True
        else:
            info["is_success"] = False
        return self.obs, reward, terminated, truncated, info

    def _load_initial_states_from_h5(self, episode_idx: int):
        """Load initial states from HDF5 file."""
        # get the hdf5 names
        hdf5_names = os.listdir(self._libero_hdf5_dir)
        task_name_underscore = self.task.replace(" ", "_")
        for hdf5_name in hdf5_names:
            if task_name_underscore not in hdf5_name:
                continue
            with h5py.File(os.path.join(self._libero_hdf5_dir, hdf5_name), "r", swmr=True) as f:
                return f["data"][f"demo_{episode_idx}"]["states"][0]

        raise ValueError(f"Could not find task name {self.task} in HDF5 files")

    def _get_libero_env(self):
        """Initializes and returns the LIBERO environment, along with the task description."""
        task_description = self._libero_task_suite.get_task(self._task_idx).language
        self._task = task_description
        task_bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / self._libero_task_suite.get_task(self._task_idx).problem_folder
            / self._libero_task_suite.get_task(self._task_idx).bddl_file
        )
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.LIBERO_ENV_RESOLUTION,
            "camera_widths": self.LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(
            self.seed
        )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
        if self.load_gt_initial_states:
            initial_states = self._load_initial_states_from_h5(self._episode_idx)
        else:
            initial_states = None

        return env, initial_states

    @staticmethod
    def _quat2axisangle(quat):
        """
        Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
        """
        # clip quaternion
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0

        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            # This is (close to) a zero degree rotation, immediately return
            return np.zeros(3)

        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    def render(self):
        return self.obs["pixels"]["image"]
