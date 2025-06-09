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
        self.transform = RandomCamTransform(
            how_many_cameras=how_many_cameras,
            sample_cameras=sample_cameras,
            camera_present_key=camera_present_key,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self.transform(sample)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped dataset."""
        return getattr(self.dataset, name)
