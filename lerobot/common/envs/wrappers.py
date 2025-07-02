import base64
import math
import os
import pathlib
import re
import time

import cv2
import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openai import OpenAI
from PIL import Image
from vila_utils.utils.decode import add_mask_2d_to_img, add_path_2d_to_img_alt_fast, get_path_from_answer
from vila_utils.utils.encode import scale_path
from vila_utils.utils.prompts import get_prompt

# Constants
SERVER_IP = "https://whippet-pet-singularly.ngrok.app"
DOWNSAMPLE_RESOLUTION = 256
# PATH_MODEL_NAME = "vila_3b_oxe_no_droid"
# PATH_MODEL_NAME_MASK = "vila_3b_oxe_no_droid_path_mask"
PATH_MODEL_NAME_MASK = PATH_MODEL_NAME = "vila_13b_path_mask_new"

def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def draw_onto_image(vlm_path_mask_output, prompt_type, img, verbose=False):
    # default inference code which is a bit different from the original data processing code because of legacy code reasons.
    h, w, c = img.shape
    scaled_mask = None
    if "mask" in prompt_type:
        min_in, max_in = np.zeros(2), np.array([w, h])
        min_out, max_out = np.zeros(2), np.ones(2)
        mask = vlm_path_mask_output[1] if len(vlm_path_mask_output) == 2 else vlm_path_mask_output
        scaled_mask = scale_path(mask, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)

    scaled_path = None
    if "path" in prompt_type:
        min_in, max_in = np.zeros(2), np.array([w, h])
        min_out, max_out = np.zeros(2), np.ones(2)
        path = vlm_path_mask_output[0] if len(vlm_path_mask_output) == 2 else vlm_path_mask_output
        scaled_path = scale_path(path, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)

        # check if there's any very close points in the path, get rid of duplicates
        new_path = []
        for i, point in enumerate(scaled_path):
            if i == 0:
                new_path.append(point)
            else:
                if not np.allclose(point, new_path[-1]):
                    new_path.append(point)
        scaled_path = np.array(new_path)

    if "mask" in prompt_type and scaled_mask is not None:
        if verbose:
            print("adding mask")
        img = add_mask_2d_to_img(img, scaled_mask, mask_pixels=int(h * 0.15))

    if "path" in prompt_type and scaled_path is not None:
        if verbose:
            print("adding path")
        img = add_path_2d_to_img_alt_fast(img, scaled_path, line_size=2 if h <= 128 else 3)
    return img


def preprocess_image(image, crop_type):
    """Process the image by either stretching or center cropping."""
    height, width, _ = image.shape
    if crop_type == "Center Crop":
        crop_size = min(height, width)
        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2
        image = image[start_y : start_y + crop_size, start_x : start_x + crop_size]
    # then, resize the image to DOWNSAMPLE_RESOLUTION x DOWNSAMPLE_RESOLUTION
    return cv2.resize(image, (DOWNSAMPLE_RESOLUTION, DOWNSAMPLE_RESOLUTION))


def send_request(
    image,
    quest,
    prompt_type,
    crop_type,
    server_ip,
    max_tokens=512,
    temperature=0.0,
    top_p=0.95,
    max_retries=5,
    verbose=False,
):
    """Send image and quest to HAMSTER model and get response."""
    # Ensure image is in BGR format for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    # preprocess the image
    image_bgr = preprocess_image(image_bgr, crop_type)

    if prompt_type == "path":
        model_name = PATH_MODEL_NAME
    elif prompt_type == "path_mask":
        model_name = PATH_MODEL_NAME_MASK
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    # Encode image to base64
    _, encoded_image_array = cv2.imencode(".jpg", image_bgr)
    encoded_image = base64.b64encode(encoded_image_array.tobytes()).decode("utf-8")

    if verbose:
        print(f"Sending request with quest: {quest}")

    retry_count = 0
    while retry_count < max_retries:
        try:
            start_time = time.time()  # Record start time
            client = OpenAI(base_url=server_ip, api_key="fake-key")
            prompt = get_prompt(quest, prompt_type, prompt_eval=True)
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=int(max_tokens),
                model=model_name,
                extra_body={
                    "num_beams": 1,
                    "use_cache": True,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                },
            )
            end_time = time.time()  # Record end time
            response_text = response.choices[0].message.content[0]["text"]
            duration = end_time - start_time
            if verbose:
                print(f"Server response received in {duration:.2f} seconds.")
            return response_text
        except Exception as e:
            retry_count += 1
            wait_time = 2**retry_count  # Exponential backoff
            if retry_count < max_retries:
                print(f"Error connecting to server: {e}")
                print(f"Retrying in {wait_time} seconds... (Attempt {retry_count} of {max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
    return None


def get_path_mask_from_vlm(
    image: np.ndarray,
    crop_type: str,
    task_instr: str,
    draw_path=True,
    draw_mask=True,
    verbose=False,
    vlm_server_ip: str = SERVER_IP,
    path=None,
    mask=None,
):
    # used for VLM inference during eval
    assert draw_path or draw_mask
    # try up to 5 times
    temperature = 0.0
    for _ in range(5):
        try:
            if path is None and draw_path or mask is None and draw_mask:
                prompt_type = "path_mask"
                response_text = send_request(
                    image,
                    task_instr,
                    prompt_type,
                    crop_type,
                    server_ip=vlm_server_ip,
                    verbose=verbose,
                    temperature=temperature,
                )
                path, mask = get_path_from_answer(response_text, prompt_type)
            if draw_path:
                drawn_rgb = draw_onto_image((path, mask), "path", image.copy())
                image = drawn_rgb
            if draw_mask:
                masked_rgb = draw_onto_image((path, mask), "mask", image.copy())
                image = masked_rgb

            return image, path, mask
        except Exception as e:
            print(f"Error: {e}")
            temperature += 0.1  # increase temperature for next attempt
            continue
    raise Exception("Failed to get path and mask from VLM")

class ObservationModificationWrapper(gym.Wrapper):
    def __init__(self, env, image_key: str):
        super().__init__(env)
        self.image_key = image_key

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._after_env_reset(obs, info)
        return self._modify_observation(obs), info

    def _after_env_reset(self, obs, info):
        raise NotImplementedError("Subclasses must implement this method")

    def _modify_observation(self, obs):
        raise NotImplementedError("Subclasses must implement this method")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._modify_observation(obs), reward, terminated, truncated, info


class GroundTruthPathMaskWrapper(ObservationModificationWrapper):
    """
    A gym wrapper that draws a path and mask on the observation image.
    This wrapper is designed to work with a single environment.
    """

    def __init__(
        self,
        env,
        path_and_mask_h5_file: str,
        draw_path: bool,
        draw_mask: bool,
        image_key="image",
    ):
        """
        Args:
            env: The gym environment to wrap.
            path_and_mask_h5_file: Path to the HDF5 file containing path and mask data
            draw_path: Whether to draw the path on the image
            draw_mask: Whether to draw the mask on the image
            image_key: The key in the observation dictionary that contains the image.
        """
        super().__init__(env, image_key=image_key)
        self.path_and_mask_h5_file = path_and_mask_h5_file
        self.draw_path = draw_path
        self.draw_mask = draw_mask

        self.current_path = None
        self.current_masked_images = None
        self.rng = np.random.default_rng()

    def _modify_observation(self, obs):
        """Applies path and mask drawing to a single observation."""
        if self.image_key not in obs["pixels"]:
            return obs

        img = obs["pixels"][self.image_key].copy()
        mask = self.current_mask[self.env.current_step % len(self.current_mask)]
        # mask_points = np.stack(mask.nonzero(), axis=1)
        # min_in, max_in = np.zeros(2), np.array(mask.shape)
        # min_out, max_out = np.zeros(2), np.ones(2)
        # mask_points = scale_path(mask_points, min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out)
        if self.draw_mask:
            # mask directly
            img = mask[..., None] * img

        if self.draw_path:
            img, _, _ = get_path_mask_from_vlm(
                img,
                "Center Crop",
                self.env.task,
                draw_path=self.draw_path,
                draw_mask=False,
                verbose=True,
                vlm_server_ip=None,
                path=self.current_path,
                mask=None,
            )

        obs["pixels"][self.image_key] = img
        return obs

    def _after_env_reset(self, obs, info):
        self.current_path, self.current_mask = self._load_path_and_mask_from_h5(
            self.env.task,
            self.env.episode_idx,
            obs["pixels"][self.image_key].shape,
        )

    def _load_path_and_mask_from_h5(
        self,
        task_description: str,
        episode_idx: int,
        img_shape: tuple,
    ):
        """Load path and mask data from HDF5 file.

        Args:
            path_and_mask_h5_file: Path to the HDF5 file containing path and mask data
            task_description: Description of the task
            episode_idx: Index of the episode
            img_shape: Shape of the image for scaling path and mask

        Returns:
            Tuple of (path, mask) where each can be None if loading fails
        """
        with h5py.File(self.path_and_mask_h5_file, "r", swmr=True) as f:
            # Find a key that contains the task description
            task_key = None
            task_description_clean = task_description.replace(" ", "_")
            for key in f:
                if task_description_clean in key:
                    task_key = key
                    break

            if task_key is None:
                raise KeyError(f"Could not find task key containing '{task_description_clean}' in HDF5 file")

            demo_key = f"demo_{episode_idx}"
            f_annotation = f[task_key][demo_key]["primary"]

            # Get path data
            path = f_annotation["gripper_positions"]  # Get path

            # Get mask data
            masked_images = f_annotation["masked_frames"][()]

            # Scale path and mask to 0, 1-normalized coordinates for VLM to scale back to image coords.
            w, h = img_shape[:2]
            min_in, max_in = np.zeros(2), np.array([w, h])
            min_out, max_out = np.zeros(2), np.ones(2)
            path = scale_path(path, min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out)
            # center image on path
            return path, masked_images

class VLMPathMaskWrapper(ObservationModificationWrapper):
    def __init__(
        self,
        env,
        image_key: str,
        vlm_server_ip: str = SERVER_IP,
        vlm_query_frequency: int = 50,
        draw_path: bool = True,
        draw_mask: bool = True,
        flip_image: bool = False,
        center_image_on_path: bool = False,
    ):
        super().__init__(env, image_key=image_key)
        self.vlm_server_ip = vlm_server_ip
        self.current_path = None
        self.current_mask = None
        self.current_step = 0
        self.vlm_query_frequency = vlm_query_frequency
        self.draw_path = draw_path
        self.draw_mask = draw_mask
        self.flip_image = flip_image
        self.center_image_on_path = center_image_on_path

    def _after_env_reset(self, obs, info):
        self.current_step = 0

    def _modify_observation(self, obs):
        img = obs["pixels"][self.image_key].copy()
        if self.flip_image:
            for key in obs["pixels"]:
                obs["pixels"][key] = np.fliplr(obs["pixels"][key])
        if self.current_step % self.vlm_query_frequency == 0:
            try:
                img, self.current_path, self.current_mask = get_path_mask_from_vlm(
                    image=img,
                    crop_type="Center Crop",
                    task_instr=self.env.task,
                    draw_path=self.draw_path,
                    draw_mask=self.draw_mask,
                    verbose=False,
                    vlm_server_ip=self.vlm_server_ip,
                )
            except Exception as e:
                print(f"Error: {e}")
                self.current_path = None
                self.current_mask = None
        if self.current_path is not None or self.current_mask is not None:
            # draw without querying by passing the current path and mask
            img, _, _ = get_path_mask_from_vlm(
                image=img,
                crop_type="Center Crop",
                task_instr=self.env.task,
                draw_path=self.draw_path,
                draw_mask=self.draw_mask,
                verbose=False,
                vlm_server_ip=None,
                path=self.current_path,
                mask=self.current_mask,
            )
        if self.center_image_on_path and self.current_path is not None:
            first_point = self.current_path[0]
            height, width = img.shape[:2]

            # Convert first_point to pixel coordinates
            # Assuming first_point is in normalized coordinates [0, 1]
            center_x = int(first_point[0] * width)
            center_y = int(first_point[1] * height)

            # Calculate crop boundaries
            crop_size = min(height, width) // 2  # Use half the smaller dimension
            top = center_y - crop_size
            left = center_x - crop_size

            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            cropped_tensor = F.crop(img_tensor, top, left, height, width)
            img = cropped_tensor.numpy().permute(1, 2, 0)

        obs["pixels"][self.image_key] = img
        return obs

    def step(self, action):
        self.current_step += 1
        return super().step(action)


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
        self.metadata = {"render_fps" : 10}
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
    def task_idx(self):
        return self._task_idx

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

        # Only set is_success if the underlying environment hasn't already provided it
        if "is_success" not in info:
            if terminated:
                info["is_success"] = True
            else:
                info["is_success"] = False
        # If terminated is True but the underlying env set is_success to False,
        # this likely means the task wasn't actually completed successfully
        elif terminated and not info["is_success"]:
            # Keep the underlying environment's success determination
            pass

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
