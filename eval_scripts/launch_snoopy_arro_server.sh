#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G 
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#SBATCH --gres=shard:12

policy_port=8001
#checkpoint=outputs/train_smolvla_bridge_pathmask_imgtransforms/checkpoints/last/pretrained_model
checkpoint=outputs/train_act_bridge_arro/checkpoints/last/pretrained_model
#checkpoint=outputs/train_act_bridge_pathmask_imgtransforms/checkpoints/last/pretrained_model
#checkpoint=outputs/train_diffusion_bridge_pathmask/checkpoints/last/pretrained_model
#checkpoint=outputs/train_diffusion_bridge_imgtransforms/checkpoints/last/pretrained_model/
#checkpoint=outputs/train_diffusion_bridge_pathmask/checkpoints/last/pretrained_model/
#checkpoint=outputs/train_smolvla_bridge_imgtransforms/checkpoints/last/pretrained_model/

cd ~/lerobot
conda run -n lerobot --no-capture-output /bin/bash -c "python lerobot/scripts/serve_arro_widowx.py --policy.path=$checkpoint --policy.use_amp=false --policy.device=cuda --port $policy_port"
