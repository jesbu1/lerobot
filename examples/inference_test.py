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
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='ACT Policy Inference')
    
    # Model source group (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--hf_model', type=str, 
                           help='Hugging Face model ID (e.g., "username/model-name")')
    model_group.add_argument('--local_checkpoint', type=str,
                           help='Path to local checkpoint directory containing model files')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default="jesbu1/bridge_v2_lerobot",
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
        args.hf_model = "jesbu1/act-bridge-v2"
    
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
        return ACTInference(config_path, args.local_checkpoint)

def main():
    args = parse_args()
    
    # Load dataset
    print(f"[INFO] Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, split=args.split)
    print(f"[INFO] Dataset loaded. Number of samples: {len(dataset)}")
    
    # Get sample
    sample = dataset[args.sample_idx]
    print("[INFO] Sample keys:", sample.keys())
    
    # Extract state
    print("[INFO] Extracting state from sample...")
    state = np.array(sample["observation.state"])
    print("[INFO] State extracted. Shape:", state.shape)
    
    # Load model
    inference = load_model(args)
    print("[INFO] Model loaded successfully.")
    
    # Run inference
    print("[INFO] Running inference...")
    action = inference.get_action(state, [])  # For now, we'll just use the state without images
    print("[INFO] Inference complete.")
    print("State shape:", state.shape)
    print("Predicted action shape:", action.shape)
    print("Predicted action:", action)

if __name__ == "__main__":
    main() 