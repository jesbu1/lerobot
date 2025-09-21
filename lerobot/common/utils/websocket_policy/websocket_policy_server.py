"""Taken from OpenPI: https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi/serving/websocket_policy_server.py"""

import asyncio
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
from lerobot.common.envs.wrappers import get_path_mask_from_vlm
from PIL import Image
from torch import Tensor
import einops

def preprocess_observation_custom_hacky(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    # TODO(aliberts, rcadene): refactor this to use features from the environment (no hardcoding)
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            # remove the "image" prefix from the keys
            imgs = {f"observation.{key}": img for key, img in observations["pixels"].items()}
        else:
            imgs = {"observation": observations["pixels"]}

        for imgkey, img in imgs.items():
            # TODO(aliberts, rcadene): use transforms.ToTensor()?
            img = torch.from_numpy(img)

            # When preprocessing observations in a non-vectorized environment, we need to add a batch dimension.
            # This is the case for human-in-the-loop RL where there is only one environment.
            if img.ndim == 3:
                img = img.unsqueeze(0)
            # sanity check that images are channel last
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

            # sanity check that images are uint8
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

            # convert to channel first of type float32 in range [0,1]
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255

            return_observations[imgkey] = img

    if "environment_state" in observations:
        env_state = torch.from_numpy(observations["environment_state"]).float()
        if env_state.dim() == 1:
            env_state = env_state.unsqueeze(0)

        return_observations["observation.environment_state"] = env_state

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    agent_pos = torch.from_numpy(observations["agent_pos"]).float()
    if agent_pos.dim() == 1:
        agent_pos = agent_pos.unsqueeze(0)
    return_observations["observation.state"] = agent_pos

    return return_observations

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
        metadata: dict | None = None,
        vlm_img_key: str | None = None,
        vlm_server_ip: str | None = None,
        vlm_query_frequency: int = 10,
        vlm_draw_path: bool = True,
        vlm_draw_mask: bool = True,
        vlm_updated_img_key_name: str | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._device = device
        # Set more verbose logging for websockets
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        logging.getLogger("websockets.protocol").setLevel(logging.INFO)

        # Optional VLM path / mask overlay settings
        self._vlm_img_key = vlm_img_key
        self._vlm_server_ip = vlm_server_ip
        self._vlm_query_frequency = int(vlm_query_frequency)
        self._vlm_draw_path = bool(vlm_draw_path)
        self._vlm_draw_mask = bool(vlm_draw_mask)
        self._vlm_current_vlm_pred = None
        self._vlm_step = 0
        self._vlm_updated_img_key_name = vlm_updated_img_key_name
        # Save directory in current working directory for VLM images
        self._vlm_save_dir = None
        if self._vlm_img_key is not None:
            self._vlm_save_dir = os.path.join(os.getcwd(), "vlm_tmp")
            os.makedirs(self._vlm_save_dir, exist_ok=True)
            logging.info(f"VLM images will be saved to: {self._vlm_save_dir}")

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
                        self._vlm_step = 0
                    
                    # Optionally augment selected image with VLM path/mask
                    if self._vlm_img_key is not None and self._vlm_img_key in obs.get("images", {}):
                        try:
                            img = obs["images"][self._vlm_img_key]
                            if self._vlm_draw_path or self._vlm_draw_mask:
                                if self._vlm_step % self._vlm_query_frequency == 0:
                                    try:
                                        img, self._vlm_current_vlm_pred = get_path_mask_from_vlm(
                                            image=img,
                                            task_instr=obs.get("prompt", ""),
                                            draw_path=self._vlm_draw_path,
                                            draw_mask=self._vlm_draw_mask,
                                            vlm_server_ip=self._vlm_server_ip,
                                        )
                                    except Exception as e:
                                        logging.warning(f"VLM overlay error on query: {e}")
                                        self._vlm_current_vlm_pred = None
                                    else:
                                        # Save the overlaid image for this fresh query
                                        if self._vlm_save_dir is not None:
                                            try:
                                                save_name = f"vlm_{self._vlm_img_key}_{self._vlm_step:06d}.png"
                                                save_path = os.path.join(self._vlm_save_dir, save_name)
                                                Image.fromarray(img).save(save_path)
                                                print(f"🖼️ Saved VLM image to {save_path}")
                                            except Exception as save_err:
                                                logging.warning(f"Failed to save VLM image to {self._vlm_save_dir}: {save_err}")
                                elif self._vlm_current_vlm_pred is not None:
                                    try:
                                        img, _ = get_path_mask_from_vlm(
                                            image=img,
                                            task_instr=obs.get("prompt", ""),
                                            draw_path=self._vlm_draw_path,
                                            draw_mask=self._vlm_draw_mask,
                                            vlm_server_ip=None,
                                            current_vlm_pred=self._vlm_current_vlm_pred,
                                        )
                                    except Exception as e:
                                        logging.warning(f"VLM overlay error on reuse: {e}")
                            # write back possibly modified image
                            obs["images"][self._vlm_img_key] = img
                        finally:
                            self._vlm_step += 1

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
                        if cam_name == self._vlm_img_key:
                            cam_name = self._vlm_updated_img_key_name
                        policy_obs["pixels"][cam_name] = img
                    if self._vlm_draw_path or self._vlm_draw_mask:
                        policy_obs = preprocess_observation_custom_hacky(policy_obs)
                    else:
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
