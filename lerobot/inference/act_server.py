"""
Websocket server for ACT policy inference.
"""

import asyncio
import json
import logging
import numpy as np
from PIL import Image
import websockets
from typing import Dict, Any

from lerobot.inference import ACTInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ACTPolicyServer:
    def __init__(self, config_path: str, checkpoint_path: str, host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize the ACT policy server.
        
        Args:
            config_path: Path to the training config YAML file
            checkpoint_path: Path to the model checkpoint or Hugging Face model ID
            host: Host to bind the server to
            port: Port to bind the server to
        """
        self.host = host
        self.port = port
        self.inference = ACTInference(config_path, checkpoint_path)
        
    async def handle_client(self, websocket):
        """Handle a client connection."""
        try:
            async for message in websocket:
                # Parse the message
                data = json.loads(message)
                
                # Extract observations
                state = np.array(data["observation/state"])
                images = [
                    np.array(data["observation/image"]),
                    np.array(data["observation/wrist_image"])
                ]
                
                # Get action from model
                action = self.inference.get_action(state, images)
                
                # Send response
                response = {
                    "actions": [action.tolist()]  # Wrap in list to match Pi0 interface
                }
                await websocket.send(json.dumps(response))
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            await websocket.send(json.dumps({"error": str(e)}))
    
    async def start(self):
        """Start the websocket server."""
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        logger.info(f"Server started on ws://{self.host}:{self.port}")
        await server.wait_closed()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='ACT Policy Server')
    parser.add_argument('--config_path', type=str, default="train_configs/train_act_bridge.yaml",
                       help='Path to training config YAML file')
    parser.add_argument('--checkpoint_path', type=str, default="jesbu1/act-bridge-v2",
                       help='Path to checkpoint or Hugging Face model ID')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                       help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind the server to')
    
    args = parser.parse_args()
    
    # Create and start server
    server = ACTPolicyServer(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        host=args.host,
        port=args.port
    )
    
    # Run the server
    asyncio.run(server.start())

if __name__ == "__main__":
    main() 