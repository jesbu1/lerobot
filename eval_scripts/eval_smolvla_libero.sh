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
path_mask_h5_loc=/scratch1/jessez/libero_processed_256_05_12/masked_vla_data/dataset_movement_and_masks.h5
libero_hdf5_dir=/scratch1/jessez/libero_processed_256_05_12/

# Define task suites to cycle through
declare -a task_suites=("libero_10" "libero_spatial" "libero_object" "libero_goal")

# Define SmolVLA model variants and their settings
declare -A model_configs
model_configs["train_smolvla_libero"]="false false lang"
model_configs["train_smolvla_libero_path"]="true false path"
model_configs["train_smolvla_libero_path_mask"]="true true path+mask"

# Function to run evaluation for a single model
run_eval() {
    local model_name=$1
    local draw_path=$2
    local draw_mask=$3
    local suffix=$4
    local task_suite=$5
    
    local policy_path="outputs/${model_name}/checkpoints/last/pretrained_model"
    local wandb_name="SmolVLA_${suffix}_${task_suite}"
    
    echo "Running evaluation for ${model_name} on ${task_suite}"
    echo "Path: ${draw_path}, Mask: ${draw_mask}"
    echo "Wandb name: ${wandb_name}"
    
    RUN_SCRIPT="conda run -n lerobot --no-capture-output python lerobot/scripts/eval_libero_gt_pathmask.py \
        --env.type=libero \
        --policy.path=$policy_path \
        --policy.use_amp=false \
        --policy.device=cuda \
        --path_and_mask_h5_file $path_mask_h5_loc \
        --env.task_suite_name $task_suite \
        --env.libero_hdf5_dir $libero_hdf5_dir \
        --eval.n_episodes=50 \
        --eval.batch_size 10 \
        --wandb_name_suffix=$wandb_name \
        --draw_path=$draw_path \
        --draw_mask=$draw_mask"
    
    echo "Executing command: $RUN_SCRIPT"
    $RUN_SCRIPT
    
    # Wait a bit between runs to ensure clean GPU memory
    sleep 10
}

# Run evaluations for all models and task suites
for task_suite in "${task_suites[@]}"; do
    for model_name in "${!model_configs[@]}"; do
        read -r draw_path draw_mask suffix <<< "${model_configs[$model_name]}"
        run_eval "$model_name" "$draw_path" "$draw_mask" "$suffix" "$task_suite"
    done
done 