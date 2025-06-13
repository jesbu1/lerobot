#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
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
policy_path=outputs/train_act_libero/checkpoints/last/pretrained_model
path_mask_h5_loc=/home1/jessez/scratch_data/libero_processed_256_05_12/masked_vla_data/dataset_movement_and_masks.h5
libero_env_suite=libero_10
libero_hdf5_dir=/home1/jessez/scratch_data/libero_processed_256_05_12/
wandb_name_suffix=test
draw_path=false
draw_mask=false

RUN_SCRIPT="python lerobot/scripts/eval_libero_gt_pathmask.py \
    --env.type=libero \
    --policy.path=$policy_path \
    --policy.use_amp=false \
    --policy.device=cuda \
    --path_and_mask_h5_file $path_mask_h5_loc \
    --env.task_suite_name $libero_env_suite \
    --env.libero_hdf5_dir $libero_hdf5_dir \
    --eval.n_episodes=50 \
    --eval.batch_size 10 \
    --wandb.name_suffix=$wandb_name_suffix \
    --draw_path=$draw_path \
    --draw_mask=$draw_mask"

EXP_NAME=train_act_libero
PROJ_NAME=lerobot

echo "Executing command: $RUN_SCRIPT"

$RUN_SCRIPT