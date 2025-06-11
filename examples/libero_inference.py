"""
Script to run LIBERO task evaluation using VLM server for path and mask inference.
"""

import os
import logging
import collections
import datetime
import wandb
import dataclasses
import math
import pathlib
from pathlib import Path
from typing import Union

import imageio
import numpy as np
import tqdm
import tyro
import sys
import h5py
import subprocess
import time
import requests

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from vila_utils.utils.encode import scale_path
from src.openpi.policies.eval_maskpath_utils import get_path_mask_from_vlm
from lerobot.inference import ACTInference
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    vlm_server_ip: str = "http://0.0.0.0:8000"
    vlm_query_frequency: int = 20  # call VLM once every how many action chunks
    vlm_model_path: str = "memmelma/vila_3b_path_mask"  # VLM model to use
    vlm_checkpoint: str = "checkpoint-6600"  # VLM checkpoint to use

    #################################################################################################################
    # Dataset and model parameters
    #################################################################################################################
    dataset_id: str = "jesbu1/libero_90_lerobot_pathmask_rdp_full_path_mask"  # Dataset to use
    model_id: str = "jesbu1/act-bridge-v2"  # Pretrained model to use

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    draw_path: bool = False
    draw_mask: bool = False
    flip_image_horizontally: bool = False

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)
    use_wandb: bool = True  # Whether to also log results in Weights & Biases
    wandb_project: str = "p-masked-vla"  # Name of W&B project to log to
    wandb_entity: str = "clvr"  # Name of entity to log under
    wandb_name_suffix: str = ""

def _quat2axisangle(quat):
    """
    Convert quaternion to axis-angle representation.
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

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def start_vlm_server(args: Args) -> subprocess.Popen:
    """Start the VLM server in a subprocess."""
    # Download VLM model if not already downloaded
    model_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", 
                            f"models--{args.vlm_model_path.replace('/', '--')}")
    if not os.path.exists(model_path):
        logging.info(f"Downloading VLM model: {args.vlm_model_path}")
        subprocess.run(["huggingface-cli", "download", args.vlm_model_path], check=True)

    # Find the checkpoint directory
    checkpoint_dir = None
    for root, dirs, files in os.walk(model_path):
        if args.vlm_checkpoint in dirs:
            checkpoint_dir = os.path.join(root, args.vlm_checkpoint)
            break
    
    if checkpoint_dir is None:
        raise ValueError(f"Could not find checkpoint {args.vlm_checkpoint} in {model_path}")

    # Start VLM server
    logging.info(f"Starting VLM server with model at {checkpoint_dir}")
    server_process = subprocess.Popen(
        ["python", "-W", "ignore", "vila_3b_server.py", 
         "--model-paths", checkpoint_dir],
        env={"CUDA_VISIBLE_DEVICES": "0"}
    )

    # Wait for server to start
    max_retries = 30
    retry_interval = 2
    for i in range(max_retries):
        try:
            response = requests.get(f"{args.vlm_server_ip}/health")
            if response.status_code == 200:
                logging.info("VLM server started successfully")
                return server_process
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                logging.info(f"Waiting for VLM server to start... ({i+1}/{max_retries})")
                time.sleep(retry_interval)
            else:
                raise RuntimeError("Failed to start VLM server")

def load_pretrained_model(args: Args) -> ACTInference:
    """Load the pretrained model for inference."""
    logging.info(f"Loading pretrained model: {args.model_id}")
    config_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", 
                             "models--" + args.model_id.replace("/", "--"), "train_config.json")
    
    if not os.path.exists(config_path):
        logging.info("Downloading config file...")
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=args.model_id, filename="train_config.json")
    
    return ACTInference(config_path, args.model_id)

def eval_libero(args: Args) -> None:
    # Start VLM server
    server_process = start_vlm_server(args)
    try:
        # Load dataset
        logging.info(f"Loading dataset: {args.dataset_id}")
        dataset = LeRobotDataset(args.dataset_id)

        # Load pretrained model
        inference = load_pretrained_model(args)

        # Set random seed
        np.random.seed(args.seed)

        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        logging.info(f"Task suite: {args.task_suite_name}")

        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

        # Set max steps based on task suite
        if args.task_suite_name == "libero_spatial":
            max_steps = 220
        elif args.task_suite_name == "libero_object":
            max_steps = 280
        elif args.task_suite_name == "libero_goal":
            max_steps = 300
        elif args.task_suite_name == "libero_10":
            max_steps = 520
        elif args.task_suite_name == "libero_90":
            max_steps = 400
        else:
            raise ValueError(f"Unknown task suite: {args.task_suite_name}")

        if args.use_wandb:
            run_name = f"pi0-{args.task_suite_name}_date-{datetime.datetime.now().strftime('%Y-%m-%d')}_seed-{args.seed}_replan-{args.replan_steps}-draw{args.draw_path}-mask{args.draw_mask}-{args.wandb_name_suffix}"
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=args)

        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

        # Start evaluation
        total_episodes, total_successes = 0, 0
        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            # Get task
            task = task_suite.get_task(task_id)
            task_description = task.language

            # Initialize video tracking for this task
            success_videos_saved = 0
            failure_videos_saved = 0

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                logging.info(f"\nTask: {task_description}")

                # Reset environment
                env.reset()
                action_plan = collections.deque()

                # Set initial states
                obs = env.set_init_state(initial_states[episode_idx])

                # Setup
                t = 0
                replay_images = []

                # initialize vlm path and and query counter
                path = None
                mask = None
                vlm_query_counter = 0

                logging.info(f"Starting episode {task_episodes+1}...")
                while t < max_steps + args.num_steps_wait:
                    try:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        # Get preprocessed image
                        # IMPORTANT: rotate 180 degrees to match train preprocessing
                        img = np.ascontiguousarray(obs["agentview_image"][::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_robot0_eye_in_hand_image"][::-1])
                        if args.flip_image_horizontally:
                            img = img[:, ::-1]
                            wrist_img = wrist_img[:, ::-1]
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )

                        if not action_plan:
                            # Finished executing previous action chunk -- compute new chunk

                            # get path and mask from VLM
                            if args.draw_path or args.draw_mask:
                                # Use VLM to get path and mask
                                if vlm_query_counter % args.vlm_query_frequency == 0:
                                    vlm_query_counter = 0
                                    # setting path and mask to None so that the VLM is called
                                    path, mask = None, None
                                img, path, mask = get_path_mask_from_vlm(
                                    img,
                                    "Center Crop",
                                    str(task_description),
                                    draw_path=args.draw_path,
                                    draw_mask=args.draw_mask,
                                    verbose=True,
                                    vlm_server_ip=args.vlm_server_ip,
                                    path=path,
                                    mask=mask,
                                )
                                vlm_query_counter += 1
                            img = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                            )

                            # Prepare observations dict
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                                "prompt": str(task_description),
                            }

                            # Query model to get action
                            action_chunk = client.infer(element)["actions"]
                            assert (
                                len(action_chunk) >= args.replan_steps
                            ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[: args.replan_steps])
                        elif args.draw_path or args.draw_mask:
                            # draw path and mask on image just for visualization when action chunk is still being used
                            # Use VLM to get path and mask
                            img, path, mask = get_path_mask_from_vlm(
                                img,
                                "Center Crop",
                                str(task_description),
                                draw_path=args.draw_path,
                                draw_mask=args.draw_mask,
                                verbose=True,
                                vlm_server_ip=args.vlm_server_ip,
                                path=path,
                                mask=mask,
                            )
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        )

                        action = action_plan.popleft()

                        replay_images.append(img)

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        logging.error(f"Caught exception: {e}")
                        break

                task_episodes += 1
                total_episodes += 1

                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4"
                imageio.mimwrite(
                    video_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                # wandb stuff
                if args.use_wandb:
                    if done and success_videos_saved <= 2:
                        success_videos_saved += 1
                        wandb.log({
                            f"videos/{task_description}/success_{success_videos_saved}": wandb.Video(str(video_path), fps=10)
                        })
                    elif not done and failure_videos_saved <= 2:
                        failure_videos_saved += 1
                        wandb.log({
                            f"videos/{task_description}/failure_{failure_videos_saved}": wandb.Video(str(video_path), fps=10)
                        })
                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
        logging.info(f"Total episodes: {total_episodes}")
        if args.use_wandb:
            wandb.log({f"{args.task_suite_name}/success_rate": float(total_successes) / float(total_episodes)})
            wandb.finish()

    finally:
        # Clean up VLM server process
        if server_process:
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero) 