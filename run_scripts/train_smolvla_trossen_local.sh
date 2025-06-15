#!/bin/bash

# Set up environment variables
export MUJOCO_GL=egl

# Add Hugging Face timeout and retry settings
export HF_HUB_DOWNLOAD_TIMEOUT=300  # Increase timeout to 5 minutes
export HF_HUB_ENABLE_HF_TRANSFER=1  # Enable HF transfer for better download speeds
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1
export HF_HUB_DISABLE_PROGRESS_BARS=0

RESUME=false

EXP_NAME=train_smolvla_trossen
PROJ_NAME=lerobot

BASE_TRAIN_CMD="--config_path=train_configs/train_smolvla_trossen.yaml --wandb.run_id=$EXP_NAME"
RESUME_TRAIN_CMD="--config_path=outputs/train_smolvla_trossen/checkpoints/last/pretrained_model/train_config.json --resume=true"

# Check if conda environment exists
if conda env list | grep -q "lerobot"; then
    TRAIN_CMD="conda run -n lerobot --no-capture-output python lerobot/scripts/train.py"
else
    echo "Warning: 'lerobot' conda environment not found. Using system Python."
    TRAIN_CMD="python lerobot/scripts/train.py"
fi

if [ "$RESUME" = true ]; then
    TRAIN_CMD="$TRAIN_CMD $RESUME_TRAIN_CMD"
else
    TRAIN_CMD="$TRAIN_CMD $BASE_TRAIN_CMD"
fi

echo "Executing command: $TRAIN_CMD"
$TRAIN_CMD 