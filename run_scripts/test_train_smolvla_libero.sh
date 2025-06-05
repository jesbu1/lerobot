#!/bin/bash

# Test configuration
export MUJOCO_GL=egl
RESUME=false

EXP_NAME=test_smolvla_libero
PROJ_NAME=lerobot

# Test with a smaller number of steps
BASE_TRAIN_CMD="--config_path=train_configs/train_smolvla_libero.yaml --wandb.run_id=$EXP_NAME --steps=100 --save_freq=50 --log_freq=10"
RESUME_TRAIN_CMD="--config_path=outputs/train_smolvla_libero/checkpoints/last/pretrained_model/train_config.json --resume=true"

TRAIN_CMD="python lerobot/scripts/train.py"
if [ "$RESUME" = true ]; then
    TRAIN_CMD="$TRAIN_CMD $RESUME_TRAIN_CMD"
else
    TRAIN_CMD="$TRAIN_CMD $BASE_TRAIN_CMD"
fi
echo "Executing command: $TRAIN_CMD"
$TRAIN_CMD 