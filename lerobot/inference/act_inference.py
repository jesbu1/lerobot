"""
ACT (Action-Constrained Transformer) inference implementation.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from PIL import Image
import yaml
import os
from typing import List, Union, Dict, Any

# Add import for ACTPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy

class ACTInference:
    """Inference class for ACT policies."""
    
    def __init__(self, config_path: str, checkpoint_path: str = None):
        """
        Initialize ACT inference.
        
        Args:
            config_path: Path to the training config YAML file
            checkpoint_path: Path to the model checkpoint directory (should contain config.json and model.safetensors)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['policy']['device'])
        
        # Use ACTPolicy.from_pretrained to load model from safetensors
        if checkpoint_path is not None:
            # Use the directory containing config.json and model.safetensors
            model_dir = os.path.dirname(os.path.abspath(checkpoint_path))
            self.model = ACTPolicy.from_pretrained(model_dir)
        else:
            raise ValueError("checkpoint_path must be provided and point to a directory with config.json and model.safetensors")
        self.model.to(self.device)
        self.model.eval()
        
        # Tokenizer is not needed for ACTPolicy, so we skip it
        
        self.state_shape = next(f['shape'] for f in self.config['policy']['input_features'] 
                              if f['type'] == 'state')
        self.image_shape = next(f['shape'] for f in self.config['policy']['input_features'] 
                              if f['type'] == 'visual')
        self.action_shape = next(f['shape'] for f in self.config['policy']['output_features'] 
                               if f['type'] == 'action')
        
        self.chunk_size = self.config['policy']['chunk_size']
        self.n_action_steps = self.config['policy']['n_action_steps']

    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
 
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize to match config shape
        image = image.resize((self.image_shape[2], self.image_shape[1]))
        # Convert to tensor and normalize
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC to CHW
        return image

    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32)

    def get_action(self, state: np.ndarray, images: List[Union[str, np.ndarray, Image.Image]]) -> np.ndarray:

        with torch.no_grad():
            state_tensor = self.preprocess_state(state).to(self.device)
            image_tensors = [self.preprocess_image(img).to(self.device) for img in images]
            
            inputs = {
                'state': state_tensor.unsqueeze(0),  # Add batch dimension
                'images': [img.unsqueeze(0) for img in image_tensors]  # Add batch dimension
            }
            
            # Predict
            outputs = self.model(**inputs)
            action = outputs.logits[0].cpu().numpy()
            
            return action 