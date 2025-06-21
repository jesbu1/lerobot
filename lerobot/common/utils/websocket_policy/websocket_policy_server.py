"""Taken from OpenPI: https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi/serving/websocket_policy_server.py"""

import asyncio
import logging
import traceback

import torch
import websockets.asyncio.server
import websockets.frames

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.websocket_policy import msgpack_numpy
from lerobot.common.envs.widowx_env import WidowXMessageFormat
from lerobot.common.envs.utils import preprocess_observation


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
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._device = device
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                obs: WidowXMessageFormat = msgpack_numpy.unpackb(await websocket.recv())
                policy_obs = {
                    "agent_pos": obs["state"].copy(),
                    "task": obs["prompt"],
                }
                for cam_name, img in obs["images"].items():
                    policy_obs[f"observation.images.{cam_name}"] = img
                policy_obs = preprocess_observation(policy_obs)

                policy_obs = {
                    key: policy_obs[key].to(self._device, non_blocking=self._device.type == "cuda")
                    for key in policy_obs
                }
                with torch.inference_mode():
                    self._policy.reset()  # clears the action chunk
                    action = self._policy.select_action_chunk(policy_obs).to("cpu")  # get full action chunk
                    assert action.ndim == 3, "Action dimensions should be (chunk_size, batch, action_dim)"
                    assert action.shape[1] == 1, "Batch size should be 1"
                action = action[1]  # get first batch item from the chunk
                action = {"actions": action.tolist()}
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
