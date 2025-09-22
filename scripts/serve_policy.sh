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
#checkpoint=outputs/act-bridge-v2/checkpoints/050000/pretrained_model # standard ACT
checkpoint=outputs/peek-act-bridge-v2/checkpoints/050000/pretrained_model # ACT+PEEK
serve_policy_vlm_freq=10 # how many action chunks between VLM queries
PEEK_VLM_PORT=8000
PEEK_VLM_IP=http://localhost

vlm_server_ip=$PEEK_VLM_IP:$PEEK_VLM_PORT

cd ~/lerobot
if [[ "$checkpoint" == *"peek"* ]]; then                
    conda run -n lerobot --no-capture-output /bin/bash -c "python lerobot/scripts/serve_widowx.py --policy.path=$checkpoint --policy.use_amp=false --policy.device=cuda --use_vlm true --port $policy_port --vlm_server_ip=$vlm_server_ip --vlm_query_frequency=$serve_policy_vlm_freq"
else
    conda run -n lerobot --no-capture-output /bin/bash -c "python lerobot/scripts/serve_widowx.py --policy.path=$checkpoint --policy.use_amp=false --policy.device=cuda --use_vlm false --port $((policy_port+1))"
fi
