#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err


source ~/.bashrc
module load cuda
module load glew
module load patchelf
module load git-lfs
export PATH="/apps/conda/.local/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
policy_path="outputs/train_smolvla_libero_path_mask_vlm_centered_0.1mask_nowrist/checkpoints/last/pretrained_model"
libero_envs="libero_goal libero_spatial libero_10 libero_object"

for env in $libero_envs; do
    name="eval_vlm_smolvla_centered_5ep_0.1mask_nowrist_$env"
    CMD="conda run -n lerobot --no-capture-output python lerobot/scripts/eval_libero_vlm.py \
        --env.type=libero \
        --env.include_wrist_image=false \
        --policy.path=$policy_path \
        --policy.use_amp=false \
        --policy.device=cuda \
        --env.task_suite_name $env \
        --eval.n_episodes=10 \
        --eval.batch_size 1 \
        --wandb_name_suffix=$name \
        --draw_path=true \
        --draw_mask=true \
        --mask_ratio=0.1 \
        --center_image_on_path=true"
    echo "Executing command: $CMD"
    $CMD
done
