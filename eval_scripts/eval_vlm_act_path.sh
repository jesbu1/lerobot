#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=185G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

VILA_GPU_ID=0
POLICY_GPU_ID=0

source ~/.bashrc


cd /home1/jessez/nvidia/VILA
echo "Running VILA server"
conda run -n vila --no-capture-output /bin/bash -c "CUDA_VISIBLE_DEVICES=$VILA_GPU_ID python -W ignore vila_3b_server.py --model-paths ~/.cache/huggingface/hub/models--memmelma--vila_3b_path_mask_fast/snapshots/12df7a04221a50e88733cd2f1132eb01257aba0d/checkpoint-11700/" &

# Wait for the model to load
sleep 90

cd /home1/jessez/nvidia/my_lerobot

policy_path="outputs/train_act_libero_path_test_10ep/checkpoints/last/pretrained_model"
libero_envs="libero_goal libero_spatial libero_10 libero_object"

for env in $libero_envs; do
    name="eval_vlm_act_path_fixinit_test_10ep_$env"
    group="eval_vlm_act_path_fixinit_test_10ep"
    CMD="CUDA_VISIBLE_DEVICES=$POLICY_GPU_ID python lerobot/scripts/eval_libero_vlm.py \
        --vlm_server_ip=http://0.0.0.0:8000 \
        --env.type=libero \
        --env.include_wrist_image=false \
        --policy.path=$policy_path \
        --policy.use_amp=false \
        --policy.device=cuda \
        --env.task_suite_name $env \
        --eval.n_episodes=50 \
        --eval.batch_size 5 \
        --wandb_name_suffix=$name \
        --wandb_group=$group \
        --draw_path=true \
        --draw_mask=false"
    FINAL_CMD="conda run -n lerobot --no-capture-output /bin/bash -c \"$CMD\""
    echo "Executing command: $FINAL_CMD"
    eval $FINAL_CMD
done
