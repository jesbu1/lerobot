t"""Evaluate a policy on an environment with ground truth path/mask visualizations.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
python lerobot/scripts/eval.py \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
python lerobot/scripts/eval.py \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --use_amp=false \
    --device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
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
import importlib
import gymnasium as gym
from lerobot.common.envs import LIBEROEnv as LIBEROEnvConfig
from lerobot.common.envs.wrappers import LIBEROEnv, GroundTruthPathMaskWrapper
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig as BaseEvalPipelineConfig


@dataclass
class EvalPipelineConfig(BaseEvalPipelineConfig):
    path_and_mask_h5_file: str | None = None
    draw_path: bool = True
    draw_mask: bool = True
<<<<<<< HEAD
    image_key: str = "agentview_image"
    every_n_steps: int = 50  # how many steps to wait before redrawing path/mask on the image
    # Wandb configuration
    wandb_enable: bool = True
    wandb_project: str = "lerobot-eval"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_notes: str | None = None
    wandb_mode: str = "online"  # Allowed values: 'online', 'offline', 'disabled'
=======
    image_key: str = "image"
>>>>>>> d3af5beea044e59e923035f6b8228cdcd770f434


def make_libero_env(
    env_cfg: LIBEROEnvConfig,
    path_and_mask_h5_file: str,
    draw_path: bool,
    draw_mask: bool,
    image_key: str,
    task_idx: int,
    episode_idx: int,
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

    n_envs = 1

    env = gym.vector.SyncVectorEnv(
        [
            lambda: GroundTruthPathMaskWrapper(
                LIBEROEnv(
                    task_suite_name=env_cfg.task_suite_name,
                    seed=env_cfg.seed,
                    resolution=env_cfg.resolution,
                    libero_hdf5_dir=env_cfg.libero_hdf5_dir,
                    load_gt_initial_states=env_cfg.load_gt_initial_states,
                    task_idx=task_idx,
                    episode_idx=episode_idx,
                ),
                path_and_mask_h5_file=path_and_mask_h5_file,
                draw_path=draw_path,
                draw_mask=draw_mask,
                image_key=image_key,
            )
            for _ in range(n_envs)
        ]
    )

    return env

@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    assert cfg.path_and_mask_h5_file is not None, "path_and_mask_h5_file is required"

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    if cfg.wandb_enable:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            notes=cfg.wandb_notes,
            config=asdict(cfg),
            mode=cfg.wandb_mode if cfg.wandb_mode in ["online", "offline", "disabled"] else "online",
        )
        logging.info(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")

    #eval metrics from loggeing utils.py
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
        path_and_mask_h5_file=cfg.path_and_mask_h5_file,
        draw_path=cfg.draw_path,
        draw_mask=cfg.draw_mask,
        image_key=cfg.image_key,
        task_idx=0,
        episode_idx=0,
    )
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        for task_idx in range(env.envs[0].num_tasks):
            task_successes = 0
            task_episodes = 0
            task_reward = 0.0
            task_eval_time = 0.0
            
            for episode_idx in range(50):
                logging.info(f"Making environment for task {task_idx} and episode {episode_idx}.")
                env = make_libero_env(
                    env_cfg=cfg.env,
                    path_and_mask_h5_file=cfg.path_and_mask_h5_file,
                    draw_path=cfg.draw_path,
                    draw_mask=cfg.draw_mask,
                    image_key=cfg.image_key,
                    task_idx=task_idx,
                    episode_idx=episode_idx,
                )

                logging.info("Wrapping environment with GroundTruthPathMaskWrapper.")
                logging.info("Making policy.")

                policy = make_policy(
                    cfg=cfg.policy,
                    env_cfg=cfg.env,
                )
                policy.eval()
                info = eval_policy(
                    env,
                    policy,
                    cfg.eval.n_episodes,
                    max_episodes_rendered=10,
                    videos_dir=Path(cfg.output_dir) / "videos",
                    start_seed=cfg.seed,
                )
                
                eval_tracker.eval_s = info["aggregated"].pop("eval_s", 0)
                eval_tracker.avg_sum_reward = info["aggregated"].pop("avg_sum_reward", 0)
                eval_tracker.pc_success = info["aggregated"].pop("pc_success", 0)
                task_successes += int(eval_tracker.pc_success > 0)
                task_episodes += 1
                task_reward += eval_tracker.avg_sum_reward
                task_eval_time += eval_tracker.eval_s
                overall_metrics["total_successes"] += int(eval_tracker.pc_success > 0)
                overall_metrics["total_episodes"] += 1
                overall_metrics["total_reward"] += eval_tracker.avg_sum_reward
                overall_metrics["total_eval_time"] += eval_tracker.eval_s
                
                # Log episode-level metrics to wandb
                if cfg.wandb_enable:
                    wandb.log({
                        f"task_{task_idx}/episode_{episode_idx}/success_rate": eval_tracker.pc_success,
                        f"task_{task_idx}/episode_{episode_idx}/avg_reward": eval_tracker.avg_sum_reward,
                        f"task_{task_idx}/episode_{episode_idx}/eval_time": eval_tracker.eval_s,
                    })
                    
                    # Log videos if available
                    if "video_paths" in info and len(info["video_paths"]) > 0:
                        wandb.log({
                            f"task_{task_idx}/episode_{episode_idx}/video": wandb.Video(info["video_paths"][0], fps=10)
                        })

                # Print current metrics
                print(info["aggregated"])
                logging.info(f"Task {task_idx} success rate: {task_successes / task_episodes:.2f}")
            
            # Log task-level aggregated metrics
            if cfg.wandb_enable:
                wandb.log({
                    f"task_{task_idx}/success_rate": task_successes / task_episodes,
                    f"task_{task_idx}/total_episodes": task_episodes,
                    f"task_{task_idx}/avg_reward": task_reward / task_episodes,
                    f"task_{task_idx}/avg_eval_time": task_eval_time / task_episodes,
                })

    # Log overall aggregated metrics
    if cfg.wandb_enable:
        wandb.log({
            "overall/success_rate": overall_metrics["total_successes"] / overall_metrics["total_episodes"],
            "overall/total_episodes": overall_metrics["total_episodes"],
            "overall/avg_reward": overall_metrics["total_reward"] / overall_metrics["total_episodes"],
            "overall/avg_eval_time": overall_metrics["total_eval_time"] / overall_metrics["total_episodes"],
        })

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
