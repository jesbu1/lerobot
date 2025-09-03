#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G 
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
##SBATCH --gres=gpu:1
#SBATCH --gres=shard:20

policy_port=8001
#checkpoint=outputs/train_smolvla_bridge_pathmask_imgtransforms/checkpoints/last/pretrained_model
#checkpoint=outputs/train_diffusion_bridge_pathmask/checkpoints/last/pretrained_model
checkpoint=outputs/train_diffusion_bridge/checkpoints/last/pretrained_model/
#checkpoint=outputs/train_smolvla_bridge_imgtransforms/checkpoints/last/pretrained_model/
serve_policy_vlm_freq=10

if [[ "$checkpoint" == *"path"* ]]; then
    cd ~/VILA
    echo "Running VILA server"
    conda run -n vila --no-capture-output /bin/bash -c "python -W ignore vila_3b_server.py --model-paths ~/.cache/huggingface/hub/models--memmelma--vila_3b_path_mask_fast/snapshots/12df7a04221a50e88733cd2f1132eb01257aba0d/checkpoint-11700/" &
    sleep 10
fi

cd ~/lerobot
if [[ "$checkpoint" == *"path"* ]]; then                
    conda run -n lerobot --no-capture-output /bin/bash -c "python lerobot/scripts/serve_widowx.py --policy.path=$checkpoint --policy.use_amp=false --policy.device=cuda --use_vlm true --port $policy_port --vlm_query_frequency=$serve_policy_vlm_freq"
else
    conda run -n lerobot --no-capture-output /bin/bash -c "python lerobot/scripts/serve_widowx.py --policy.path=$checkpoint --policy.use_amp=false --policy.device=cuda --use_vlm false --port $((policy_port+1))"
fi