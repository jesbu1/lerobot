"""
ACT (Action-Constrained Transformer) inference implementation.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import yaml
import os
from typing import List, Union, Dict, Any

class ACTInference:
    """Inference class for ACT policies."""
    
    def __init__(self, config_path: str, checkpoint_path: str = None):
        """
        Initialize ACT inference.
        
        Args:
            config_path: Path to the training config YAML file
            checkpoint_path: Path to the model checkpoint (optional)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(self.config['policy']['device'])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['policy']['pretrained_policy_name_or_path'],
            torch_dtype=torch.float16 if self.config['policy']['use_amp'] else torch.float32
        ).to(self.device)
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['policy']['pretrained_policy_name_or_path']
        )
        
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