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
import torch

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
    
    # Always use the local model
    args.local_checkpoint = "outputs/train_act_trossen_pathmask/checkpoints/last/pretrained_model"
    
    return args

def load_model(args):
    """Load model based on the provided arguments."""
    # Hardcode the absolute path to the model directory
    model_dir = os.path.expanduser("~/lerobot/outputs/train_act_trossen_pathmask/checkpoints/last")
    print(f"[INFO] Loading model from local checkpoint: {model_dir}")
    
    # Use both config files from the pretrained_model directory
    train_config_path = os.path.join(model_dir, "train_config.json")
    config_path = os.path.join(model_dir, "config.json")
    # if not os.path.exists(train_config_path):
    #     raise ValueError(f"No train_config.json found at {train_config_path}")
    # if not os.path.exists(config_path):
    #     raise ValueError(f"No config.json found at {config_path}")
    # return ACTInference(train_config_path, model_dir)

def generate_test_data():
    """Generate synthetic test data for inference."""

    state = torch.randn(7) 
    
    # Generate a random RGB image
    image = torch.randn(3, 224, 224)  # Random image data
    
    print("[INFO] Generated test data:")
    print(f"State shape: {state.shape}")
    print(f"Image shape: {image.shape}")
    
    return state, [image]

def main():
    args = parse_args()
    
    state, images = generate_test_data()
    
    inference = load_model(args)
    print("[INFO] Model loaded successfully.")
    
    print("[INFO] Running inference...")
    action = inference.get_action(state, images)
    print("[INFO] Inference complete.")
    print("State shape:", state.shape)
    print("Predicted action shape:", action.shape)
    print("Predicted action:", action)

if __name__ == "__main__":
    main() 