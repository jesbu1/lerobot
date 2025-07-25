#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
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

EXP_NAME=train_act_libero_path_mask_vlm_0.01-0.12mask
PROJ_NAME=lerobot
RESUME=false

BASE_TRAIN_CMD="--config_path=train_configs/train_act_libero_path_mask_vlm.yaml --output_dir=outputs/$EXP_NAME --job_name=$EXP_NAME --wandb.run_id=$EXP_NAME"
RESUME_TRAIN_CMD="--config_path=outputs/$EXP_NAME/checkpoints/last/pretrained_model/train_config.json --resume=true"

TRAIN_CMD="conda run -n lerobot --no-capture-output python lerobot/scripts/train.py"
if [ "$RESUME" = true ]; then
    TRAIN_CMD="$TRAIN_CMD $RESUME_TRAIN_CMD"
else
    TRAIN_CMD="$TRAIN_CMD $BASE_TRAIN_CMD"
fi
echo "Executing command: $TRAIN_CMD"
$TRAIN_CMD
