"""
Example script demonstrating how to use the ACT policy inference with various model sources.
Supports:
- Default model (jesbu1/act-bridge-v2)
- Any other Hugging Face model
- Local checkpoint loading
"""

import os
import argparse
import numpy as np
from PIL import Image
from lerobot.inference import ACTInference
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

def parse_args():
    parser = argparse.ArgumentParser(description='ACT Policy Inference')
    
    # Model source group (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--hf_model', type=str, 
                           help='Hugging Face model ID (e.g., "username/model-name")')
    model_group.add_argument('--local_checkpoint', type=str,
                           help='Path to local checkpoint directory containing model files')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default="minjunkevink/trossen_objects_pick_place",
                       help='Hugging Face dataset ID to use for inference')
    parser.add_argument('--split', type=str, default="train",
                       help='Dataset split to use')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of the sample to use for inference')
    
    # Config arguments
    parser.add_argument('--config_path', type=str,
                       help='Path to config file (only needed for local checkpoints)')
    
    args = parser.parse_args()
    
    # Set default model if none specified
    if not args.hf_model and not args.local_checkpoint:
        args.local_checkpoint = "outputs/train_act_trossen_pathmask/checkpoints/last/pretrained_model"
    
    return args

def load_model(args):
    """Load model based on the provided arguments."""
    if args.hf_model:
        print(f"[INFO] Loading model from Hugging Face Hub: {args.hf_model}")
        # For HF models, we need to download the config file
        config_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--" + args.hf_model.replace("/", "--"), "train_config.json")
        if not os.path.exists(config_path):
            print(f"[INFO] Downloading config file for {args.hf_model}...")
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_id=args.hf_model, filename="train_config.json")
        return ACTInference(config_path, args.hf_model)
    else:
        print(f"[INFO] Loading model from local checkpoint: {args.local_checkpoint}")
        if not args.config_path:
            # Try to find train_config.json in the checkpoint directory
            config_path = os.path.join(args.local_checkpoint, "train_config.json")
            if not os.path.exists(config_path):
                raise ValueError(f"No train_config.json found at {config_path}. Please specify --config_path")
        else:
            config_path = args.config_path
        # Use the pretrained_model directory for both config and model
        model_path = args.local_checkpoint
        return ACTInference(config_path, model_path)

def main():
    args = parse_args()
    
    # Load dataset metadata first to analyze structure
    print(f"[INFO] Loading dataset metadata for {args.dataset}...")
    ds_meta = LeRobotDatasetMetadata(args.dataset)
    print("\nDataset Metadata:")
    print(f"Total number of episodes: {ds_meta.total_episodes}")
    print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
    print(f"Frames per second: {ds_meta.fps}")
    print(f"Robot type: {ds_meta.robot_type}")
    print(f"Camera keys: {ds_meta.camera_keys}")
    print("\nFeatures:")
    print(ds_meta.features)
    
    # Load dataset
    print(f"\n[INFO] Loading dataset {args.dataset}...")
    dataset = LeRobotDataset(args.dataset)
    print(f"[INFO] Dataset loaded. Number of frames: {dataset.num_frames}")
    
    # Get sample
    sample = dataset[args.sample_idx]
    print("\n[INFO] Sample keys:", sample.keys())
    
    # Extract state and images
    print("\n[INFO] Extracting state and images from sample...")
    state = sample["observation.state"]
    images = [sample[key] for key in ds_meta.camera_keys] if ds_meta.camera_keys else []
    
    print("[INFO] State shape:", state.shape)
    if images:
        print("[INFO] Number of images:", len(images))
        print("[INFO] Image shape:", images[0].shape)
    
    # Load model
    inference = load_model(args)
    print("[INFO] Model loaded successfully.")
    
    # Run inference
    print("[INFO] Running inference...")
    action = inference.get_action(state, images)
    print("[INFO] Inference complete.")
    print("State shape:", state.shape)
    print("Predicted action shape:", action.shape)
    print("Predicted action:", action)

if __name__ == "__main__":
    main() 