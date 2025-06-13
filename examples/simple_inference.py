"""
Simple script to run ACT policy inference using the default model.
"""

import os
from lerobot.inference import ACTInference
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main():
    # Load dataset
    dataset_id = "jesbu1/libero_90_lerobot_pathmask_rdp_full_path_mask"
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
    config_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", 
                             "models--" + model_id.replace("/", "--"), "train_config.json")
    
    if not os.path.exists(config_path):
        print("Downloading config file...")
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=model_id, filename="train_config.json")
    
    # Initialize inference
    inference = ACTInference(config_path, model_id)
    
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