#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

VILA_GPU_ID=1
POLICY_GPU_ID=0

source ~/.bashrc


cd /home1/jessez/nvidia/VILA
echo "Running VILA server"
conda run -n vila --no-capture-output /bin/bash -c "CUDA_VISIBLE_DEVICES=$VILA_GPU_ID python -W ignore vila_3b_server.py --model-paths ~/.cache/huggingface/hub/models--memmelma--vila_3b_path_mask_fast/snapshots/12df7a04221a50e88733cd2f1132eb01257aba0d/checkpoint-11700/" &

# Wait for the model to load
sleep 30

policy_path="outputs/train_act_libero_path_mask_0.01_0.12/checkpoints/last/pretrained_model"
libero_envs="libero_goal libero_spatial libero_10 libero_object"

for env in $libero_envs; do
    for mask_ratio in 0.05 0.08 0.12; do
        #name="eval_fixres_vlm_act_5ep_0.1mask_$env"
        #name="eval_vlm_act_0.1mask_nowrist_5ep_$env"
        name="eval_vlm_gttrain_0.01-0.12mask_act_${mask_ratio}mask_25ep_$env"
        CMD="conda run -n lerobot --no-capture-output python lerobot/scripts/eval_libero_vlm.py \
            --env.type=libero \
            --env.include_wrist_image=false \
            --policy.path=$policy_path \
            --policy.use_amp=false \
            --policy.device=cuda \
            --env.task_suite_name $env \
            --eval.n_episodes=25 \
            --eval.batch_size 5 \
            --wandb_name_suffix=$name \
            --draw_path=true \
            --mask_ratio=$mask_ratio \
            --draw_mask=true"
            echo "Executing command: $CMD"
            $CMD
    done
done
