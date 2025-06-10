import torch
from torch.utils.data import Dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.random_cam.transform import RandomCamTransform


class RandomCamDataset(Dataset):
    """Wrapper around LeRobotDataset that applies random camera sampling."""

    def __init__(
        self,
        dataset: LeRobotDataset,
        how_many_cameras: int = 2,
        sample_cameras: bool = True,
        camera_present_key: str = "camera_present",
    ):
        self.dataset = dataset
        # find the image keys that are in the remap keys and replace them with the new keys in any order
        if dataset.remap_keys:
            self.image_keys = [
                dataset.remap_keys.get(key, key)
                for key in dataset.remap_keys.values()
                if key.startswith("observation.images")
            ]
        else:
            self.image_keys = [key for key in dataset.meta.features if key.startswith("observation.images")]
        self.transform = RandomCamTransform(
            how_many_cameras=how_many_cameras,
            sample_cameras=sample_cameras,
            camera_present_key=camera_present_key,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.transform(sample, self.image_keys)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped dataset."""
        return getattr(self.dataset, name)
