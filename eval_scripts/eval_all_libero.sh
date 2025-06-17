#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=15
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

# Common settings
path_mask_h5_loc=/home1/jessez/scratch_data/libero_processed_256_05_12/masked_vla_data/dataset_movement_and_masks.h5
libero_env_suite=libero_10
libero_hdf5_dir=/home1/jessez/scratch_data/libero_processed_256_05_12/

# Define all model variants and their settings
declare -A model_configs
# ACT variants
model_configs["train_act_libero_lang"]="false false"
model_configs["train_act_libero_path_lang"]="true false"
model_configs["train_act_libero_path_mask_lang"]="true true"
# SmolVLA variants
model_configs["train_smolvla_libero"]="false false"
model_configs["train_smolvla_libero_path"]="true false"
model_configs["train_smolvla_libero_path_mask"]="true true"

# Function to run evaluation for a single model
run_eval() {
    local model_name=$1
    local draw_path=$2
    local draw_mask=$3
    
    local policy_path="outputs/${model_name}/checkpoints/last/pretrained_model"
    local wandb_name_suffix="${model_name}"
    
    echo "Running evaluation for ${model_name}"
    echo "Path: ${draw_path}, Mask: ${draw_mask}"
    
    RUN_SCRIPT="conda run -n lerobot --no-capture-output python lerobot/scripts/eval_libero_gt_pathmask.py \
        --env.type=libero \
        --policy.path=$policy_path \
        --policy.use_amp=false \
        --policy.device=cuda \
        --path_and_mask_h5_file $path_mask_h5_loc \
        --env.task_suite_name $libero_env_suite \
        --env.libero_hdf5_dir $libero_hdf5_dir \
        --eval.n_episodes=50 \
        --eval.batch_size 10 \
        --wandb_name_suffix=$wandb_name_suffix \
        --draw_path=$draw_path \
        --draw_mask=$draw_mask"
    
    echo "Executing command: $RUN_SCRIPT"
    $RUN_SCRIPT
    
    # Wait a bit between runs to ensure clean GPU memory
    sleep 10
}

# Run evaluations for all models
for model_name in "${!model_configs[@]}"; do
    read -r draw_path draw_mask <<< "${model_configs[$model_name]}"
    run_eval "$model_name" "$draw_path" "$draw_mask"
done 