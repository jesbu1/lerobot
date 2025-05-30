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

if [ $RELAUNCH_COUNT -ge $MAX_RELAUNCHES ]; then
  echo "Maximum relaunch limit ($MAX_RELAUNCHES) reached. Exiting."
  exit 1
fi

echo "Starting job... Relaunch attempt: $((RELAUNCH_COUNT + 1))/$MAX_RELAUNCHES"
source ~/.bashrc
module load cuda
module load glew
module load patchelf
module load git-lfs
export PATH="/apps/conda/.local/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl

EXP_NAME=train_act_libero_path
PROJ_NAME=lerobot

BASE_TRAIN_CMD="python lerobot/scripts/train.py --config_path train_configs/train_act_libero_path.yaml --wandb.run_id=$EXP_NAME"

#TRAIN_CMD="$BASE_TRAIN_CMD --overwrite"
TRAIN_CMD="$BASE_TRAIN_CMD"
#TRAIN_CMD="$BASE_TRAIN_CMD --resume"
echo "Executing command: $TRAIN_CMD"
$TRAIN_CMD