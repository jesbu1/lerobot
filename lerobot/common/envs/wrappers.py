import gymnasium as gym
import pathlib
import numpy as np
import h5py
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark

import re
import time
import base64
import cv2
import numpy as np
from openai import OpenAI
from vila_utils.utils.prompts import get_prompt
from vila_utils.utils.decode import add_path_2d_to_img_alt_fast, add_mask_2d_to_img, get_path_from_answer
from vila_utils.utils.encode import draw_path_on_img, draw_binary_mask_on_img
from vila_utils.utils.encode import scale_path


# Constants
SERVER_IP = "https://whippet-pet-singularly.ngrok.app"
DOWNSAMPLE_RESOLUTION = 256
# PATH_MODEL_NAME = "vila_3b_oxe_no_droid"
# PATH_MODEL_NAME_MASK = "vila_3b_oxe_no_droid_path_mask"
PATH_MODEL_NAME = "vila_3b_oxe_sim_path"
PATH_MODEL_NAME_MASK = "vila_3b_oxe_sim_path_mask"


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
            if path is None or mask is None:
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


class GroundTruthPathMaskWrapper(gym.Wrapper):
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
        image_key="agentview_image",
        every_n_steps: int = 50,
    ):
        """
        Args:
            env: The gym environment to wrap.
            path_and_mask_h5_file: Path to the HDF5 file containing path and mask data
            draw_path: Whether to draw the path on the image
            draw_mask: Whether to draw the mask on the image
            image_key: The key in the observation dictionary that contains the image.
            every_n_steps: The number of steps between each path and mask drawing.
        """
        super().__init__(env)
        self.path_and_mask_h5_file = path_and_mask_h5_file
        self.image_key = image_key
        self.draw_path = draw_path
        self.draw_mask = draw_mask
        self.num_steps_since_last_draw = 0
        self.every_n_steps = every_n_steps

        self.current_path = None
        self.current_mask = None
        self.rng = np.random.default_rng()

    def _modify_observation(self, obs):
        """Applies path and mask drawing to a single observation."""
        if self.image_key not in obs:
            return obs

        img = obs[self.image_key].copy()
        img, _, _ = get_path_mask_from_vlm(
            img,
            "Center Crop",
            self.env.task,
            draw_path=self.draw_path,
            draw_mask=self.draw_mask,
            verbose=True,
            vlm_server_ip=None,
            path=self.current_path,
            mask=self.current_mask,
        )

        obs[self.image_key] = img
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_path, self.current_mask = self._load_path_and_mask_from_h5(
            self.env.task,
            self.episode_idx,
            obs[self.image_key].shape,
        )
        self.num_steps_since_last_draw = 0
        return self._modify_observation(obs), info

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

            # Scale path and mask to 0, 1-normalized coordinates for VLM to scale back to image coords.
            w, h = img_shape[:2]
            min_in, max_in = np.zeros(2), np.array([w, h])
            min_out, max_out = np.zeros(2), np.ones(2)
            path = scale_path(path, min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out)
            masked_images = f_annotation["masked_frames"][()]

            return path, masked_images

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.num_steps_since_last_draw += 1
        if self.num_steps_since_last_draw >= self.every_n_steps:
            self.num_steps_since_last_draw = 0
            obs = self._modify_observation(obs)
        return obs, reward, terminated, truncated, info


class LIBEROEnv(gym.Env):
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

    def __init__(self, task_suite_name: str, seed: int, resolution: int = 256):
        super().__init__()
        self.LIBERO_ENV_RESOLUTION = resolution
        self.num_steps_wait = 10
        self._task = None
        self.task_suite_name = task_suite_name
        self.seed = seed
        if task_suite_name == "libero_spatial":
            self.max_steps = 220  # longest training demo has 193 steps
        elif task_suite_name == "libero_object":
            self.max_steps = 280  # longest training demo has 254 steps
        elif task_suite_name == "libero_goal":
            self.max_steps = 300  # longest training demo has 270 steps
        elif task_suite_name == "libero_10":
            self.max_steps = 520  # longest training demo has 505 steps
        elif task_suite_name == "libero_90":
            self.max_steps = 400  # longest training demo has 373 steps
        else:
            raise ValueError(f"Unknown task suite: {task_suite_name}")
        benchmark_dict = benchmark.get_benchmark_dict()
        self._libero_task_suite = benchmark_dict[self.task_suite_name]()

    @property
    def task(self):
        return str(self._task)

    @property
    def num_tasks(self):
        return len(self._libero_task_suite.get_tasks())

    def reset(self, episode_id=None, **kwargs):
        obs, info = self.env.reset(episode_id=episode_id, **kwargs)
        current_steps_waited = 0
        while current_steps_waited < self.num_steps_wait:
            obs, _, _, info = self.env.step(self.LIBERO_DUMMY_ACTION)
            current_steps_waited += 1
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # vertically flip the images coming from LIBERO
        flipped_agentview = obs["agentview_image"][::-1]
        flipped_eye_in_hand = obs["eye_in_hand_image"][::-1]
        obs["agentview_image"] = flipped_agentview
        obs["eye_in_hand_image"] = flipped_eye_in_hand
        return obs, reward, terminated, truncated, info

    def _get_libero_env(self, resolution, seed):
        """Initializes and returns the LIBERO environment, along with the task description."""
        task_description = self._libero_task_suite.get_task(self.task).language
        task_bddl_file = (
            pathlib.Path(get_libero_path("bddl_files"))
            / self._libero_task_suite.get_task(self.task).problem_folder
            / self._libero_task_suite.get_task(self.task).bddl_file
        )
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.LIBERO_ENV_RESOLUTION,
            "camera_widths": self.LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
        return env, task_description
