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
import torchvision.transforms.functional as F
from gymnasium import spaces
try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError:
    print("libero not found, so we can't use the LIBERO Env")
    pass

from openai import OpenAI, APIConnectionError
from PIL import Image
from vila_utils.utils.decode import add_mask_2d_to_img, add_path_2d_to_img_alt_fast, get_path_from_answer
from vila_utils.utils.encode import scale_path
from vila_utils.utils.prompts import get_prompt

# Constants
SERVER_IP = "https://whippet-pet-singularly.ngrok.app"
VLM_DOWNSAMPLE_RESOLUTION = 256
OLD_PROMPT = False
# PATH_MODEL_NAME = "vila_3b_oxe_no_droid"
# PATH_MODEL_NAME_MASK = "vila_3b_oxe_no_droid_path_mask"
PATH_MODEL_NAME_MASK = PATH_MODEL_NAME = "vila_3b_path_mask_fast"


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


def draw_onto_image(vlm_path_mask_output, prompt_type, img, mask_ratio=0.15, verbose=False):
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
        img = add_mask_2d_to_img(img, scaled_mask, mask_pixels=int(h * mask_ratio))

    if "path" in prompt_type and scaled_path is not None:
        if verbose:
            print("adding path")
        img = add_path_2d_to_img_alt_fast(img, scaled_path, line_size=2)
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
    return cv2.resize(image, (VLM_DOWNSAMPLE_RESOLUTION, VLM_DOWNSAMPLE_RESOLUTION))


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
            prompt = get_prompt(quest, prompt_type, prompt_eval=OLD_PROMPT)
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
        except APIConnectionError as e:
            print(f"Error connecting to server: {e}")
            wait_time = 2**retry_count  # Exponential backoff
            retry_count += 1 # this doesn't count as a retry
            max_retries += 1 
            print(f"Retrying in {wait_time} seconds... (Attempt {retry_count} of {max_retries})")
            time.sleep(wait_time)
            continue
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
    mask_ratio=0.15,
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
                drawn_rgb = draw_onto_image((path, mask), "path", image.copy(), mask_ratio=mask_ratio)
                image = drawn_rgb
            if draw_mask:
                masked_rgb = draw_onto_image((path, mask), "mask", image.copy(), mask_ratio=mask_ratio)
                image = masked_rgb

            return image, path, mask
        except Exception as e:
            print(f"Error: {e}")
            temperature += 0.1  # increase temperature for next attempt
            continue
    raise Exception("Failed to get path and mask from VLM")


class ObservationModificationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

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
        mask_ratio: float = 0.15,
    ):
        super().__init__(env)
        self.image_key = image_key
        self.vlm_server_ip = vlm_server_ip
        self.current_path = None
        self.current_mask = None
        self.current_step = 0
        self.vlm_query_frequency = vlm_query_frequency
        self.draw_path = draw_path
        self.draw_mask = draw_mask
        self.flip_image = flip_image
        self.center_image_on_path = center_image_on_path
        self.mask_ratio = mask_ratio

        print(
            f"VLMPathMaskWrapper initialized with mask_ratio: {self.mask_ratio}, draw_path: {self.draw_path}, draw_mask: {self.draw_mask}, center_image_on_path: {self.center_image_on_path}"
        )

    def _after_env_reset(self, obs, info):
        self.current_step = 0
        self.current_path = None
        self.current_mask = None

    def _modify_observation(self, obs):
        if self.flip_image:
            for key in obs["pixels"]:
                obs["pixels"][key] = np.fliplr(obs["pixels"][key])
        img = obs["pixels"][self.image_key].copy()
        if self.draw_path or self.draw_mask:
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
                        mask_ratio=self.mask_ratio,
                    )
                except Exception as e:
                    print(f"Error: {e}")
                    self.current_path = None
                    self.current_mask = None
            elif self.current_path is not None or self.current_mask is not None:
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
                    mask_ratio=self.mask_ratio,
                )
            if self.center_image_on_path and self.current_path is not None and len(self.current_path) > 0:
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
                cropped_tensor = F.crop(img_tensor, top, left, crop_size * 2, crop_size * 2)
                img = cropped_tensor.permute(1, 2, 0).numpy()

        obs["pixels"][self.image_key] = img

        return obs

    def step(self, action):
        self.current_step += 1
        return super().step(action)


class DownsampleObservationWrapper(ObservationModificationWrapper):
    def __init__(self, env, downsample_resolution: int = 224):
        super().__init__(env)
        self.downsample_resolution = downsample_resolution
        if self.downsample_resolution != self.env.resolution:
            for key in self.env.observation_space["pixels"]:
                self.env.observation_space["pixels"][key] = spaces.Box(
                    0,
                    255,
                    shape=(self.downsample_resolution, self.downsample_resolution, 3),
                    dtype=np.uint8,
                )

    def _modify_observation(self, obs):
        if self.downsample_resolution != self.env.resolution:
            for key in obs["pixels"]:
                obs["pixels"][key] = cv2.resize(
                    obs["pixels"][key], (self.downsample_resolution, self.downsample_resolution)
                )
        return obs

    def _after_env_reset(self, obs, info):
        pass
