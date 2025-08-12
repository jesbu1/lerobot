#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

source ~/.bashrc

RESUME=false

EXP_NAME=train_smolvla_libero_path_vlm_test_5ep
PROJ_NAME=lerobot

BASE_TRAIN_CMD="--config_path=train_configs/train_smolvla_libero_path_vlm.yaml --wandb.run_id=$EXP_NAME --output_dir=outputs/$EXP_NAME --job_name=$EXP_NAME --dataset.repo_id=jesbu1/libero_test_lerobot_pathmask_vlm_preds_max_ep_per_task_5"
RESUME_TRAIN_CMD="--config_path=outputs/$EXP_NAME/checkpoints/last/pretrained_model/train_config.json --resume=true"

TRAIN_CMD="conda run -n lerobot --no-capture-output python lerobot/scripts/train.py"
if [ "$RESUME" = true ]; then
    TRAIN_CMD="$TRAIN_CMD $RESUME_TRAIN_CMD"
else
    TRAIN_CMD="$TRAIN_CMD $BASE_TRAIN_CMD"
fi
echo "Executing command: $TRAIN_CMD"
$TRAIN_CMD 
