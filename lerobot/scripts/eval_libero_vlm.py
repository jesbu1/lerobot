"""
python lerobot/scripts/eval_libero_vlm.py \
    --env.type=libero \
    --policy.path=outputs/train_act_libero_path_mask_vlm/checkpoints/last/pretrained_model/ \
    --policy.use_amp=false \
    --policy.device=cuda \
    --env.task_suite_name libero_object \
    --eval.n_episodes=50 \
    --eval.batch_size 1 \
    --wandb_name_suffix=fml_test --draw_path=true --draw_mask=true
"""

import json
import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import gymnasium as gym
import numpy as np
import torch
from datasets import load_dataset
from termcolor import colored
import wandb

from lerobot.common.envs.factory import make_env
from lerobot.common.policies.factory import make_policy
from lerobot.scripts.eval import eval_policy
import gymnasium as gym
from lerobot.common.envs import LIBEROEnv as LIBEROEnvConfig
from lerobot.common.envs.wrappers import LIBEROEnv, VLMPathMaskWrapper, DownsampleObservationWrapper
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig as BaseEvalPipelineConfig

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class EvalPipelineConfig(BaseEvalPipelineConfig):
    vlm_server_ip: str = "https://whippet-pet-singularly.ngrok.app"
    vlm_query_frequency: int = 50
    draw_path: bool = True
    draw_mask: bool = True
    image_key: str = "image"
    flip_image: bool = (
        True  # flips the image horizontally to match with how we labeled the LIBERO dataset with the VLM
    )
    center_image_on_path: bool = False
    mask_ratio: float = 0.15
    # Wandb configuration
    wandb_enable: bool = True
    wandb_project: str = "lerobot-eval"
    wandb_entity: str | None = None
    wandb_name_suffix: str = ""
    wandb_group: str = ""
    wandb_notes: str | None = None
    wandb_mode: str = "online"  # Allowed values: 'online', 'offline', 'disabled'
    downsample_resolution: int = 224


CURRENT_EPISODE_IDX = 0


def finished_task():
    global CURRENT_EPISODE_IDX
    CURRENT_EPISODE_IDX = 0


def reset_callback(envs: gym.vector.VectorEnv):
    global CURRENT_EPISODE_IDX
    for env in envs.envs:
        env.set_episode_idx(CURRENT_EPISODE_IDX)
        print(f"setting episode idx to {CURRENT_EPISODE_IDX}")
        CURRENT_EPISODE_IDX += 1


def make_libero_env(
    env_cfg: LIBEROEnvConfig,
    vlm_server_ip: str,
    vlm_query_frequency: int,
    draw_path: bool,
    draw_mask: bool,
    image_key: str,
    task_idx: int,
    start_episode_idx: int,
    n_envs: int,
    flip_image: bool,
    center_image_on_path: bool,
    downsample_resolution: int,
    mask_ratio: float = 0.15,
) -> gym.vector.VectorEnv | None:
    """Makes a gym vector environment according to the config.

    Args:
        cfg (EnvConfig): the config of the environment to instantiate.
        n_envs (int, optional): The number of parallelized env to return. Defaults to 1.
        use_async_envs (bool, optional): Whether to return an AsyncVectorEnv or a SyncVectorEnv. Defaults to
            False.

    Raises:
        ValueError: if n_envs < 1
        ModuleNotFoundError: If the requested env package is not installed

    Returns:
        gym.vector.VectorEnv: The parallelized gym.env instance.
    """
    # TODO: fix this later for lerobot[libero]
    # package_name = f"gym_{cfg.type}"

    # try:
    #    importlib.import_module(package_name)
    # except ModuleNotFoundError as e:
    #    print(f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.type}]'`")
    #    raise e
    env = gym.vector.SyncVectorEnv(
        [
            lambda i=i: DownsampleObservationWrapper(
                VLMPathMaskWrapper(
                    LIBEROEnv(
                        task_suite_name=env_cfg.task_suite_name,
                        seed=env_cfg.seed,
                        resolution=env_cfg.resolution,
                        libero_hdf5_dir=env_cfg.libero_hdf5_dir,
                        load_gt_initial_states=env_cfg.load_gt_initial_states,
                        task_idx=task_idx,
                        episode_idx=start_episode_idx,
                        include_wrist_image=env_cfg.include_wrist_image,
                    ),
                    vlm_server_ip=vlm_server_ip,
                    vlm_query_frequency=vlm_query_frequency,
                    draw_path=draw_path,
                    draw_mask=draw_mask,
                    image_key=image_key,
                    flip_image=flip_image,
                    center_image_on_path=center_image_on_path,
                    mask_ratio=mask_ratio,
                ),
                downsample_resolution=downsample_resolution,
            )
            for i in range(n_envs)
        ]
    )

    return env


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))
    # assert cfg.eval.n_episodes == 50, "n_episodes must be 50 for libero"

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    # Use the passed wandb name directly
    wandb_name = cfg.wandb_name_suffix

    if cfg.wandb_enable:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=wandb_name,
            group=cfg.wandb_group if cfg.wandb_group else None,
            notes=cfg.wandb_notes,
            config=asdict(cfg),
            mode=cfg.wandb_mode if cfg.wandb_mode in ["online", "offline", "disabled"] else "online",
        )
        logging.info(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")

    # eval metrics from loggeing utils.py
    eval_metrics = {
        "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
        "pc_success": AverageMeter("success", ":.1f"),
        "eval_s": AverageMeter("eval_s", ":.3f"),
    }
    eval_tracker = MetricsTracker(
        cfg.eval.batch_size,
        1,  # num_frames (not used in eval)
        1,  # num_episodes (not used in eval)
        eval_metrics,
        initial_step=0,
    )

    # Initialize overall metrics aggregators
    overall_metrics = {
        "total_successes": 0,
        "total_episodes": 0,
        "total_reward": 0.0,
        "total_eval_time": 0.0,
    }

    env = make_libero_env(
        env_cfg=cfg.env,
        vlm_server_ip=cfg.vlm_server_ip,
        vlm_query_frequency=cfg.vlm_query_frequency,
        draw_path=cfg.draw_path,
        draw_mask=cfg.draw_mask,
        image_key=cfg.image_key,
        task_idx=0,
        start_episode_idx=0,
        n_envs=1,
        flip_image=cfg.flip_image,
        center_image_on_path=cfg.center_image_on_path,
        downsample_resolution=cfg.downsample_resolution,
        mask_ratio=cfg.mask_ratio,
    )
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        for task_idx in range(env.envs[0].num_tasks):
            task_successes = 0
            task_episodes = 0
            task_reward = 0.0
            task_eval_time = 0.0

            # Clear videos directory to avoid logging videos from previous tasks
            videos_dir = Path(cfg.output_dir) / "videos"
            if videos_dir.exists():
                import shutil

                shutil.rmtree(videos_dir)

            eval_tracker.reset_averages()

            finished_task() # reset the episode idx to 0 for the next task

            logging.info(f"Making environment for task {task_idx}.")
            env = make_libero_env(
                env_cfg=cfg.env,
                vlm_server_ip=cfg.vlm_server_ip,
                vlm_query_frequency=cfg.vlm_query_frequency,
                draw_path=cfg.draw_path,
                draw_mask=cfg.draw_mask,
                image_key=cfg.image_key,
                task_idx=task_idx,
                start_episode_idx=0,
                n_envs=cfg.eval.batch_size,
                flip_image=cfg.flip_image,
                center_image_on_path=cfg.center_image_on_path,
                downsample_resolution=cfg.downsample_resolution,
                mask_ratio=cfg.mask_ratio,
            )

            policy = make_policy(
                cfg=cfg.policy,
                env_cfg=cfg.env,
            )
            policy.eval()
            info = eval_policy(
                env,
                policy,
                cfg.eval.n_episodes,
                max_episodes_rendered=cfg.eval.n_episodes,
                videos_dir=videos_dir,
                start_seed=cfg.seed,
                reset_callback=reset_callback,
            )

            eval_tracker.eval_s = info["aggregated"].pop("eval_s", 0)
            eval_tracker.avg_sum_reward = info["aggregated"].pop("avg_sum_reward", 0)
            eval_tracker.pc_success = info["aggregated"].pop("pc_success", 0)

            task_successes = sum(e["success"] for e in info["per_episode"])

            assert cfg.eval.n_episodes == len(
                info["per_episode"]
            ), "number of episodes in VALID_EPISODE_LIST and info['per_episode'] should be the same"
            task_episodes += cfg.eval.n_episodes

            total_reward_for_task = sum(e["sum_reward"] for e in info["per_episode"])
            task_reward += total_reward_for_task

            task_eval_time += eval_tracker.eval_s.sum

            overall_metrics["total_successes"] += task_successes
            overall_metrics["total_episodes"] += cfg.eval.n_episodes
            overall_metrics["total_reward"] += total_reward_for_task
            overall_metrics["total_eval_time"] += eval_tracker.eval_s.sum

            # Log episode-level metrics to wandb
            if cfg.wandb_enable:
                # Log videos if available
                if "video_paths" in info and len(info["video_paths"]) > 0:
                    for i, ep_info in enumerate(info["per_episode"]):
                        if i >= len(info["video_paths"]):
                            break
                        if ep_info["success"]:
                            wandb_prefix = "success"
                        else:
                            wandb_prefix = "failure"
                        wandb.log(
                            {
                                f"task_{task_idx}/video_episode_{ep_info['episode_ix']}_{wandb_prefix}": wandb.Video(
                                    info["video_paths"][i], fps=30, format="mp4"
                                )
                            }
                        )

                # Print current metrics
                print(info["aggregated"])
                logging.info(f"Task {task_idx} success rate: {task_successes / task_episodes:.2f}")

            # Log task-level aggregated metrics
            if cfg.wandb_enable:
                wandb.log(
                    {
                        f"task_{task_idx}/success_rate": task_successes / task_episodes,
                        f"task_{task_idx}/total_episodes": task_episodes,
                        f"task_{task_idx}/avg_reward": task_reward / task_episodes,
                        f"task_{task_idx}/avg_eval_time": task_eval_time / task_episodes,
                    }
                )

    # Log overall aggregated metrics
    if cfg.wandb_enable:
        wandb.log(
            {
                "overall/success_rate": overall_metrics["total_successes"]
                / overall_metrics["total_episodes"],
                "overall/total_episodes": overall_metrics["total_episodes"],
                "overall/avg_reward": overall_metrics["total_reward"] / overall_metrics["total_episodes"],
                "overall/avg_eval_time": overall_metrics["total_eval_time"]
                / overall_metrics["total_episodes"],
            }
        )

    # Save info with aggregated metrics
    info["aggregated"] = {
        "overall_success_rate": overall_metrics["total_successes"] / overall_metrics["total_episodes"],
        "total_episodes": overall_metrics["total_episodes"],
        "avg_reward": overall_metrics["total_reward"] / overall_metrics["total_episodes"],
        "avg_eval_time": overall_metrics["total_eval_time"] / overall_metrics["total_episodes"],
    }

    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    env.close()

    if cfg.wandb_enable:
        wandb.finish()

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
