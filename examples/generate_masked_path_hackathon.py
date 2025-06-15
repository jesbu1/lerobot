import argparse
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm  # type: ignore

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from vila_utils.utils.decode import add_path_2d_to_img_alt_fast, add_mask_2d_to_img
from vila_utils.utils.encode import scale_path, smooth_path_rdp

MASK_CAM_NAME = "stationary"


def process_path_obs(sample_img, path, path_line_size=3, apply_rdp=False):
    """Process path observation by drawing it onto the image."""
    height, width = sample_img.shape[:2]

    # Scale path to image size
    min_in, max_in = np.zeros(2), np.array([width, height])
    min_out, max_out = np.zeros(2), np.ones(2)
    path_scaled = scale_path(path, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)

    if apply_rdp:
        length_before = len(path_scaled)
        path_scaled = smooth_path_rdp(path_scaled, tolerance=0.05)
        length_after = len(path_scaled)
        if length_before > length_after:
            print(f"RDP reduced path length from {length_before} to {length_after}")
    # Draw path
    return add_path_2d_to_img_alt_fast(sample_img, path_scaled, line_size=path_line_size)


def process_mask_obs(sample_img, mask_points, mask_pixels=25, scale_mask=False, apply_rdp=False):
    """Process mask observation by applying it to the image."""
    if scale_mask:
        height, width = sample_img.shape[:2]

        # Scale mask points to image size
        min_in, max_in = np.zeros(2), np.array([width, height])
        min_out, max_out = np.zeros(2), np.ones(2)
        mask_points_scaled = scale_path(
            mask_points, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in
        )
    else:
        mask_points_scaled = mask_points

    if apply_rdp:
        length_before = len(mask_points_scaled)
        mask_points_scaled = smooth_path_rdp(mask_points_scaled, tolerance=0.05)
        length_after = len(mask_points_scaled)
        if length_before > length_after:
            print(f"RDP reduced mask length from {length_before} to {length_after}")

    return add_mask_2d_to_img(sample_img, mask_points_scaled, mask_pixels=mask_pixels)


def convert_lerobot_dataset_to_masked_path_dataset(
    original_dataset: LeRobotDataset,
    hdf5_path: str,
    new_repo_id: str,
    new_dataset_root: str,
    path_line_size: int = 2,
    path_mask_ratio: float = 0.15,
    push_to_hub: bool = False,
) -> LeRobotDataset:
    """
    Converts an existing LeRobotDataset by iterating over its episodes and frames,
    adding a `masked_path` image observation from an HDF5 file, and saving a new dataset.

    Args:
        original_dataset (LeRobotDataset): The source dataset.
        hdf5_path (str): Path to the HDF5 file containing the masked_path data.
        new_repo_id (str): Repository id for the new dataset.
        new_dataset_root (str): The root directory where the new dataset will be written.
        push_to_hub (bool, optional): Whether to push the new dataset to the hub. Defaults to False.


    Returns:
        LeRobotDataset: A new LeRobotDataset with the added `masked_path` observation.
    """
    # 1. Define the features for the new dataset, including the new `masked_path` observation.
    new_features = deepcopy(original_dataset.meta.info["features"])

    # Determine image shape from an existing camera key
    camera_key = original_dataset.meta.camera_keys[0]
    # In LeRobot datasets, info has (h, w, c) but data is (c, h, w)
    if len(new_features[camera_key]["shape"]) == 3:
        h, w, c = new_features[camera_key]["shape"]
    else:
        # Fallback for different conventions
        c, h, w = original_dataset[0][camera_key].shape

    new_features["observation.images.image_path"] = {
        "shape": [3, h, w],
        "dtype": "video",
        "names": ["height", "width", "channels"],
    }

    new_features["observation.images.image_masked_path"] = {
        "shape": [3, h, w],
        "dtype": "video",
        "names": ["height", "width", "channels"],
    }

    # 2. Create a new (empty) LeRobotDataset for writing.
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=original_dataset.fps,
        root=new_dataset_root,
        robot_type=original_dataset.meta.robot_type,
        features=new_features,
        use_videos=len(original_dataset.meta.video_keys) > 0,
    )

    # 3. Iterate through episodes, load data, add mask, and save to new dataset.
    with h5py.File(hdf5_path, "r") as hdf5_file:
        for episode_idx in tqdm(range(original_dataset.num_episodes), desc="Processing episodes"):
            # --- HDF5 Loading Logic ---
            # This is a placeholder for loading your HDF5 data.
            # You will need to adapt this based on the structure of your HDF5 file.
            # Assumption: The HDF5 file has groups named 'episode_0', 'episode_1', etc.
            # and each group contains a 'masked_path' dataset.
            episode_group_name = f"episode_{episode_idx}"
            if episode_group_name not in hdf5_file:
                raise ValueError(
                    f"Episode group '{episode_group_name}' not found in HDF5 file '{hdf5_path}'. "
                    "Please adapt the script to your HDF5 file structure."
                )
            episode_group = hdf5_file[episode_group_name]
            path_data = episode_group[f"{MASK_CAM_NAME}_paths"]
            mask_data = episode_group[f"{MASK_CAM_NAME}_masks"]
            path_lengths = (
                episode_group[f"{MASK_CAM_NAME}_path_lengths"][:]
                if f"{MASK_CAM_NAME}_path_lengths" in episode_group
                else None
            )
            mask_lengths = (
                episode_group[f"{MASK_CAM_NAME}_mask_lengths"][:]
                if f"{MASK_CAM_NAME}_mask_lengths" in episode_group
                else None
            )
            # Timesteps correspond to which steps have the path and masks
            path_timesteps = (
                episode_group[f"{MASK_CAM_NAME}_path_timesteps"][:]
                if f"{MASK_CAM_NAME}_path_timesteps" in episode_group
                else None
            )
            mask_timesteps = (
                episode_group[f"{MASK_CAM_NAME}_mask_timesteps"][:]
                if f"{MASK_CAM_NAME}_mask_timesteps" in episode_group
                else None
            )
            # --- End HDF5 Loading Logic ---

            from_idx = original_dataset.episode_data_index["from"][episode_idx].item()
            to_idx = original_dataset.episode_data_index["to"][episode_idx].item()

            next_path_timestep_idx = 0
            next_mask_timestep_idx = 0

            for step_idx, frame_idx in enumerate(range(from_idx, to_idx)):
                frame = original_dataset[frame_idx]
                new_frame = {}
                for key, value in frame.items():
                    new_frame[key] = value

                original_img = frame[f"observation.images.{MASK_CAM_NAME}"]
                # Skip if any of the path or mask data is not available or if the camera is not present
                if any(data is None for data in [path_data, path_lengths, mask_data, mask_lengths]):
                    new_frame["observation.images.image_path"] = np.zeros_like(original_img)
                    new_frame["observation.images.image_masked_path"] = np.zeros_like(original_img)
                    continue

                # Process path and mask if available and enabled
                if path_data is not None:
                    # Get the current step's path
                    # because we query the path and mask data for each step but it's only generated every N steps in the HDF5, we check if the step_idx has reached the next timestep
                    if step_idx == path_timesteps[
                        next_path_timestep_idx % len(path_timesteps)
                    ] and next_path_timestep_idx < len(path_timesteps):
                        next_path_timestep_idx += 1
                    if step_idx == mask_timesteps[
                        next_mask_timestep_idx % len(mask_timesteps)
                    ] and next_mask_timestep_idx < len(mask_timesteps):
                        next_mask_timestep_idx += 1

                    current_path_idx = next_path_timestep_idx - 1
                    current_mask_idx = next_mask_timestep_idx - 1

                    current_path = path_data[current_path_idx, : path_lengths[current_path_idx]]
                    # Add path to image
                    path_img = process_path_obs(
                        original_img.copy(),
                        current_path,
                        path_line_size=path_line_size,
                        apply_rdp=False,
                    )
                    new_frame[f"observation.images.image_path"] = path_img

                    # Add mask if available
                    if mask_data is not None:
                        current_mask = mask_data[current_mask_idx, : mask_lengths[current_mask_idx]]
                        # Apply mask
                        height, width = original_img.shape[:2]
                        masked_img = process_mask_obs(
                            original_img.copy(),
                            current_mask,
                            mask_pixels=int(height * path_mask_ratio),
                            scale_mask=np.all(current_mask <= 1),
                            apply_rdp=False,
                        )
                        # frame[f"observation.mask.{cam}"] = masked_img

                        # Combine path and mask
                        masked_path_img = process_path_obs(
                            masked_img.copy(),
                            current_path,
                            path_line_size=path_line_size,
                        )
                        new_frame[f"observation.images.image_masked_path"] = masked_path_img
                assert "task" in new_frame, "Task is required"
                new_dataset.add_frame(new_frame)

            new_dataset.save_episode()

    if push_to_hub:
        new_dataset.push_to_hub()

    return new_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add a masked_path observation to a LeRobot dataset from an HDF5 file."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="The repository id of the LeRobot dataset to process.",
    )
    parser.add_argument(
        "--hdf5-path",
        type=str,
        required=True,
        help="Path to the HDF5 file containing the masked_path data.",
    )
    parser.add_argument(
        "--new-repo-id",
        type=str,
        required=True,
        help="The repository id for the new dataset.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="The root directory of the LeRobot dataset.",
    )
    parser.add_argument(
        "--new-dataset-root",
        type=str,
        default=None,
        help="The root directory where the new dataset will be written.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Whether to push the new dataset to the hub.",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    new_dataset_root_path = args.new_dataset_root
    if new_dataset_root_path is None:
        new_dataset_root_path = Path(str(dataset.root or args.repo_id.split("/")[-1]) + "_masked_path")

    new_dataset = convert_lerobot_dataset_to_masked_path_dataset(
        original_dataset=dataset,
        hdf5_path=args.hdf5_path,
        new_repo_id=args.new_repo_id,
        new_dataset_root=new_dataset_root_path,
        push_to_hub=args.push_to_hub,
    )

    print(f"Successfully created dataset at {new_dataset_root_path}")
