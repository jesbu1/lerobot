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
RESUME=true
policy_path="outputs/train_act_libero_path_mask_vlm_centered/checkpoints/last/pretrained_model"
libero_envs="libero_goal libero_spatial libero_10 libero_object"

for env in $libero_envs; do
    name="eval_vlm_act_centered_$env"
    CMD="conda run -n lerobot --no-capture-output python lerobot/scripts/eval_libero_vlm.py \
        --env.type=libero \
        --policy.path=$policy_path \
        --policy.use_amp=false \
        --policy.device=cuda \
        --env.task_suite_name $env \
        --eval.n_episodes=50 \
        --eval.batch_size 2 \
        --wandb_name_suffix=$name \
        --draw_path=true \
        --draw_mask=true"
    echo "Executing command: $CMD"
    $CMD
done
