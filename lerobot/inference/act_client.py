"""
Client for interfacing with ACT policy server and VLM server.
"""

import asyncio
import json
import logging
import numpy as np
import websockets
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ACTPolicyClient:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize the ACT policy client.
        
        Args:
            host: Host of the policy server
            port: Port of the policy server
        """
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        
    async def infer(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send observations to the policy server and get actions.
        
        Args:
            element: Dictionary containing:
                - observation/image: Main camera image
                - observation/wrist_image: Wrist camera image
                - observation/state: Robot state
                - prompt: Task description (optional)
        
        Returns:
            Dictionary containing:
                - actions: List of actions
        """
        try:
            async with websockets.connect(self.uri) as websocket:
                # Send observations
                await websocket.send(json.dumps(element))
                
                # Get response
                response = await websocket.recv()
                return json.loads(response)
                
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

class VLMClient:
    def __init__(self, server_ip: str = "http://0.0.0.0:8000"):
        """
        Initialize the VLM client.
        
        Args:
            server_ip: IP address of the VLM server
        """
        self.server_ip = server_ip
        
    async def get_path_mask(
        self,
        image: np.ndarray,
        task_description: str,
        draw_path: bool = False,
        draw_mask: bool = False,
        path: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get path and mask from VLM server.
        
        Args:
            image: Input image
            task_description: Description of the task
            draw_path: Whether to draw the path on the image
            draw_mask: Whether to draw the mask on the image
            path: Existing path (optional)
            mask: Existing mask (optional)
            
        Returns:
            Tuple of (processed_image, path, mask)
        """
        # TODO: Implement VLM client
        # For now, return the input image and None for path/mask
        return image, None, None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='ACT Policy Client')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                       help='Host of the policy server')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port of the policy server')
    parser.add_argument('--vlm_server_ip', type=str, default="http://0.0.0.0:8000",
                       help='IP address of the VLM server')
    
    args = parser.parse_args()
    
    # Create clients
    policy_client = ACTPolicyClient(args.host, args.port)
    vlm_client = VLMClient(args.vlm_server_ip)
    
    # Example usage
    async def run():
        # Example observations
        element = {
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(7, dtype=np.float32),
            "prompt": "Example task"
        }
        
        # Get path and mask from VLM
        image, path, mask = await vlm_client.get_path_mask(
            element["observation/image"],
            element["prompt"]
        )
        element["observation/image"] = image
        
        # Get action from policy
        response = await policy_client.infer(element)
        print("Actions:", response["actions"])
    
    # Run example
    asyncio.run(run())

if __name__ == "__main__":
    main() 