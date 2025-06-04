"""
Example script demonstrating how to use the ACT policy inference.
"""

import os
import numpy as np
from PIL import Image
from lerobot.inference import ACTInference
from datasets import load_dataset

def main():
    config_path = "train_configs/train_act_bridge.yaml"
    checkpoint_path = "jesbu1/act-bridge-v2"  # Using the pretrained model from Hugging Face
    
    # Load a sample from the dataset
    dataset = load_dataset("jesbu1/bridge_v2_lerobot", split="train")
    sample = dataset[0]  # Get first sample
    
    # Extract state and images
    state = np.array(sample["observation"]["state"])
    images = [
        np.array(sample["observation"]["images"]["image_0"]),
        np.array(sample["observation"]["images"]["image_1"])
    ]
    
    # Initialize inference
    inference = ACTInference(config_path, checkpoint_path)
    
    # Get action prediction
    action = inference.get_action(state, images)
    print("State shape:", state.shape)
    print("Image shapes:", [img.shape for img in images])
    print("Predicted action shape:", action.shape)
    print("Predicted action:", action)

if __name__ == "__main__":
    main() 