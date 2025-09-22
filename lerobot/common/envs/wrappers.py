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
#from vila_utils.utils.decode import add_mask_2d_to_img, add_path_2d_to_img_alt_fast, get_path_from_answer
#from vila_utils.utils.encode import scale_path
#from vila_utils.utils.prompts import get_prompt
from peek_vlm.models.peek import add_answer_to_img, send_request

# Constants
PEEK_VLM_NAME = "peek_3b"


def get_path_mask_from_vlm(
    image: np.ndarray,
    task_instr: str,
    draw_path=True,
    draw_mask=True,
    vlm_server_ip: str = None,
    current_vlm_pred: str = None,
):
    # used for VLM inference during eval
    assert draw_path or draw_mask
    assert current_vlm_pred is not None or vlm_server_ip is not None, "Either current_vlm_pred or vlm_server_ip must be provided"
    prompt_type = "path_mask"
    if not current_vlm_pred:
        # query the VLM otherwise use the provided path and mask
        answer_pred = send_request(image, task_instr, prompt_type=prompt_type, server_ip=vlm_server_ip, model_name=PEEK_VLM_NAME)
    else:
        answer_pred = current_vlm_pred

    H, W, _ = image.shape
    line_size = int(min(H, W) * 0.01)
    mask_pixels = int(min(H, W) * 0.08)

    path_mask_image, _, _ = add_answer_to_img(
        image, answer_pred, prompt_type, line_size=line_size, add_mask=True, mask_pixels=mask_pixels
    )
    return path_mask_image, answer_pred


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
        vlm_server_ip: str = None,
        vlm_query_frequency: int = 50,
        draw_path: bool = True,
        draw_mask: bool = True,
        flip_image: bool = False,
    ):
        super().__init__(env)
        self.image_key = image_key
        self.vlm_server_ip = vlm_server_ip
        self.current_vlm_pred = None
        self.current_step = 0
        self.vlm_query_frequency = vlm_query_frequency
        self.draw_path = draw_path
        self.draw_mask = draw_mask
        self.flip_image = flip_image

        print(
            f"VLMPathMaskWrapper initialized with draw_path: {self.draw_path}, draw_mask: {self.draw_mask}"
        )

    def _after_env_reset(self, obs, info):
        self.current_step = 0
        self.current_vlm_pred = None

    def _modify_observation(self, obs):
        if self.flip_image:
            for key in obs["pixels"]:
                obs["pixels"][key] = np.fliplr(obs["pixels"][key])
        img = obs["pixels"][self.image_key].copy()
        if self.draw_path or self.draw_mask:
            if self.current_step % self.vlm_query_frequency == 0:
                try:
                    img, self.current_vlm_pred = get_path_mask_from_vlm(
                        image=img,
                        task_instr=self.env.task,
                        draw_path=self.draw_path,
                        draw_mask=self.draw_mask,
                        vlm_server_ip=self.vlm_server_ip,
                    )
                except Exception as e:
                    print(f"Error: {e}")
                    self.current_vlm_pred = None
            elif self.current_vlm_pred is not None:
                # draw without querying by passing the current path and mask
                img, _ = get_path_mask_from_vlm(
                    image=img,
                    task_instr=self.env.task,
                    draw_path=self.draw_path,
                    draw_mask=self.draw_mask,
                    vlm_server_ip=None,
                    current_vlm_pred=self.current_vlm_pred,
                )
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
