"""
Example script demonstrating how to use the ACT policy inference.
"""

import os
import numpy as np
from PIL import Image
from lerobot.inference import ACTInference

def main():
    config_path = "train_configs/train_act_bridge.yaml"
    checkpoint_path = "" 
    
    inference = ACTInference(config_path, checkpoint_path)
    
    state = np.zeros(7)  #
    images = [np.zeros((224, 224, 3))] * 2 
    action = inference.get_action(state, images)
    print("Example 1 - Action from numpy arrays:", action)
    

if __name__ == "__main__":
    main() 