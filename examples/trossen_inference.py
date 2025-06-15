"""
Simple script to run ACT policy inference using the default model.
"""

import os
from lerobot.inference import ACTInference
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main():
    print(f"Loading dataset: {dataset_id}")
    dataset = LeRobotDataset(dataset_id)
    
    # Get first sample
    sample_idx = 0
    sample = dataset[sample_idx]
    
    # Extract state and images
    state = sample["observation.state"]
    images = [sample[key] for key in ["image", "wrist_image"]] #masked_path_image , 'maksed_image'
    # Load default model
    model_id = "jesbu1/act-bridge-v2"
    print(f"Loading model: {model_id}")
    config_path = "outputs/train_act_trossen_pathmask/train_config.json"
    checkpoint_path = "outputs/train_act_trossen_pathmask/checkpoints"
    
    if not os.path.exists(config_path):
        print("Downloading config file...")
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=model_id, filename="train_config.json")
    
    # Initialize inference
    inference = ACTInference(config_path, checkpoint_path)
    
    # Run inference
    print("Running inference...")
    action = inference.get_action(state, images)
    
    print("\nResults:")
    print(f"State shape: {state.shape}")
    print(f"Number of images: {len(images)}")
    print(f"Predicted action shape: {action.shape}")
    print(f"Predicted action: {action}")

if __name__ == "__main__":
    main() 