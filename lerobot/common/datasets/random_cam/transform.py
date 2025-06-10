import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RandomCamTransform:
    """Transform that randomly samples a specified number of camera observations.

    Args:
        how_many_cameras: Number of cameras to sample
        sample_cameras: Whether to randomly sample cameras or take first N
        camera_present_key: Key in the dataset that indicates which cameras are present
    """

    how_many_cameras: int = 2
    sample_cameras: bool = True
    camera_present_key: str = "camera_present"

    def __call__(self, sample: Dict[str, Any], original_image_keys: list[str]) -> Dict[str, Any]:
        # Get all camera keys
        camera_keys = [key for key in sample if key.startswith("observation.images")]

        # Get available cameras based on camera_present
        if self.camera_present_key in sample:
            camera_present = sample[self.camera_present_key]
            available_cameras = np.count_nonzero(camera_present)
            camera_idx_to_include = np.arange(available_cameras)[: self.how_many_cameras]

            if self.sample_cameras and available_cameras >= self.how_many_cameras:
                camera_idx_to_include = np.random.choice(
                    available_cameras, self.how_many_cameras, replace=False
                )
        else:
            # If no camera_present key, assume all cameras are available
            available_cameras = len(camera_keys)
            camera_idx_to_include = np.arange(available_cameras)[: self.how_many_cameras]
            if self.sample_cameras and available_cameras >= self.how_many_cameras:
                camera_idx_to_include = np.random.choice(
                    available_cameras, self.how_many_cameras, replace=False
                )

        camera_keys_to_include = [camera_keys[i] for i in camera_idx_to_include]

        # Create output dictionary with sampled cameras
        output = {}
        for key, value in sample.items():
            if not key.startswith("observation.images"):
                output[key] = value
                continue

            # Get camera index from key name
            if key in camera_keys_to_include:
                idx = camera_keys_to_include.index(key)
                # map to some original key so the collate fn doesn't break with different number of keys in the batch
                output[original_image_keys[idx]] = value
        if available_cameras < self.how_many_cameras:
            # pad it with black images
            for i in range(self.how_many_cameras - available_cameras):
                output[original_image_keys[i + available_cameras]] = torch.zeros_like(
                    output[original_image_keys[0]]
                )

        return output
