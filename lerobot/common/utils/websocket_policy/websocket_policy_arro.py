import asyncio
from datetime import datetime
import logging
import traceback
import os
import numpy as np
import torch
import websockets.asyncio.server
import websockets.frames

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.websocket_policy import msgpack_numpy
from lerobot.common.envs.widowx_env import WidowXMessageFormat
from lerobot.common.envs.utils import preprocess_observation
from PIL import Image
from lerobot.common.utils.websocket_policy.arro_client import GroundedSam2TrackerClient

import spacy
import cv2
nlp = spacy.load("en_core_web_sm")
def _apply_masks_to_frames(frames, masks_list):
    """
    frames: list of PIL.Image or np.ndarray [H,W,3]
    masks_list: list of torch.BoolTensor or np.ndarray [N,H,W]
    returns: list of np.ndarray frames with masks applied
    """
    out_frames = []
    for frame, masks in zip(frames, masks_list):
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        if hasattr(masks, "numpy"):  # torch.Tensor
            masks = masks.cpu().numpy()
        masks = masks.squeeze(1)
        if masks.ndim == 3:  # multiple objects -> combine
            mask = np.any(masks, axis=0)
        else:
            mask = masks

        mask3 = np.repeat(mask[..., None], 3, axis=-1)
        out = frame * mask3

        out_frames.append(out.astype(np.uint8))
    return out_frames
def instruction_to_dino_instr(instruction):
    # find all the nouns in the image and use gripper via spacy
    doc = nlp(instruction)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    nouns = nouns + ["robot gripper"]

    # add "a " prefix to each object
    objects = ["a " + o for o in nouns]
    # separate objects with ". "
    dino_instr = ". ".join(objects)
    return dino_instr

IMAGE_SIZE = 224
class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `infer` method.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        device: torch.device,
        host: str = "0.0.0.0",
        port: int = 8000,
        arro_server_address: str = "127.0.0.1",
        arro_img_key: str = "image_0",
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._device = device
        # Set more verbose logging for websockets
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        logging.getLogger("websockets.protocol").setLevel(logging.INFO)
        self._arro_server_address = arro_server_address
        self._arro_client = GroundedSam2TrackerClient(self._arro_server_address)
        self._arro_img_key = arro_img_key
        if self._arro_server_address is not None:
            self._arro_save_dir = os.path.join(os.getcwd(), "arro_tmp_visualizations")
            os.makedirs(self._arro_save_dir, exist_ok=True)
            logging.info(f"Arro images will be saved to: {self._arro_save_dir}")

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        logging.info(f"Starting WebSocket server on {self._host}:{self._port}")
        print(f"🔌 WebSocket server starting on {self._host}:{self._port}")
        
        try:
            async with websockets.asyncio.server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                # Add additional server options for better connection handling
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            ) as server:
                print(f"✅ WebSocket server is running and listening for connections")
                print(f"📡 Server address: ws://{self._host}:{self._port}")
                await server.serve_forever()
        except Exception as e:
            logging.error(f"Failed to start WebSocket server: {e}")
            print(f"❌ Failed to start WebSocket server: {e}")
            raise

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        client_addr = websocket.remote_address
        logging.info(f"🔗 New connection from {client_addr}")
        print(f"🔗 New connection from {client_addr}")
        
        packer = msgpack_numpy.Packer()

        try:
            # Send metadata to client
            await websocket.send(packer.pack(self._metadata))
            logging.info(f"📤 Sent metadata to {client_addr}")
            
            connection_count = 0
            while True:
                try:
                    # Receive observation from client
                    obs_data = await websocket.recv()
                    connection_count += 1
                    
                    if connection_count % 10 == 0:  # Log every 10th request
                        logging.info(f"📥 Received request #{connection_count} from {client_addr}")
                    
                    obs: WidowXMessageFormat = msgpack_numpy.unpackb(obs_data)

                    if obs["reset"]:
                        # resetting the policy
                        logging.info(f"🔄 Resetting policy and VLM step")
                        self._policy.reset()
                        self._arro_client.reset(init_frame=obs["images"][self._arro_img_key], text=instruction_to_dino_instr(obs["prompt"]))

                    # generate ARRO masks 
                    idx, masks = self._arro_client.step(Image.fromarray(obs["images"][self._arro_img_key]))
                    masked_frame = _apply_masks_to_frames([obs["images"][self._arro_img_key]], [masks])[0]
                    obs["images"][self._arro_img_key] = masked_frame

                    # save the masked frame to the _arro_save_dir
                    Image.fromarray(masked_frame).save(os.path.join(self._arro_save_dir, obs["prompt"], f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.png"))


                    # Process observation for policy
                    policy_obs = {
                        "agent_pos": obs["state"].copy(),
                        "pixels": {},
                    }
                    for cam_name, img in obs["images"].items():
                        # resize the image to 224x224
                        img = Image.fromarray(img)
                        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                        img = np.array(img)
                        policy_obs["pixels"][cam_name] = img
                    policy_obs = preprocess_observation(policy_obs)
                    policy_obs = {
                        key: policy_obs[key].to(self._device, non_blocking=self._device.type == "cuda")
                        for key in policy_obs
                    }
                    policy_obs["task"] = obs["prompt"]
                    
                    # Run policy inference
                    with torch.inference_mode():
                        self._policy.reset()  # clears the action chunk
                        action = self._policy.select_action_chunk(policy_obs) # get full action chunk
                        action = torch.stack(list(action), dim=0).to("cpu")
                        assert action.ndim == 3, "Action dimensions should be (chunk_size, batch, action_dim)"
                        assert action.shape[1] == 1, "Batch size should be 1"
                    action = action[:, 0]  # get first batch item from the chunk
                    action = {"actions": action.numpy().tolist()}  # convert to list for JSON serialization
                    
                    # Send action back to client
                    await websocket.send(packer.pack(action))
                    
                except websockets.ConnectionClosed:
                    logging.info(f"🔌 Connection from {client_addr} closed normally")
                    print(f"🔌 Connection from {client_addr} closed")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    logging.warning(f"🔌 Connection from {client_addr} closed with error: {e}")
                    print(f"🔌 Connection from {client_addr} closed with error: {e}")
                    break
                except Exception as e:
                    error_msg = f"❌ Error processing request from {client_addr}: {e}\n{traceback.format_exc()}"
                    logging.error(error_msg)
                    print(error_msg)
                    
                    try:
                        # Send error message to client
                        await websocket.send(traceback.format_exc())
                        await websocket.close(
                            code=websockets.frames.CloseCode.INTERNAL_ERROR,
                            reason="Internal server error. Traceback included in previous frame.",
                        )
                    except Exception as close_error:
                        logging.error(f"Failed to send error to client {client_addr}: {close_error}")
                    raise
                    
        except websockets.ConnectionClosed:
            logging.info(f"🔌 Connection from {client_addr} closed during setup")
            print(f"🔌 Connection from {client_addr} closed during setup")
        except Exception as e:
            logging.error(f"❌ Unexpected error with client {client_addr}: {e}")
            print(f"❌ Unexpected error with client {client_addr}: {e}")
            try:
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason=f"Server error: {str(e)}"
                )
            except Exception:
                pass  # Ignore errors during close
