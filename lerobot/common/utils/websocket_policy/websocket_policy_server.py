"""Taken from OpenPI: https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi/serving/websocket_policy_server.py"""

import asyncio
import torch
import logging
import traceback

from lerobot.common.utils.websocket_policy import msgpack_numpy
from lerobot.common.utils.websocket_policy.base_policy import BasePolicy
import websockets.asyncio.server
import websockets.frames


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: BasePolicy,
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
                obs = msgpack_numpy.unpackb(await websocket.recv())
                obs["task"] = obs.pop("prompt")
                with torch.inference_mode():
                    action = self._policy.select_action(obs)
                assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"
                action = {
                    key: action[key].to(self._device, non_blocking=self._device.type == "cuda")
                    for key in action
                }
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
