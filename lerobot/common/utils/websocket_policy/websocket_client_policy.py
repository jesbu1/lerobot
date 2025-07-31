"""Taken from OpenPI: https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/websocket_client_policy.py"""

import logging
import time
from typing import Dict, Tuple, Union
from urllib.parse import urlparse

import websockets.sync.client
from typing_extensions import override

from lerobot.common.utils.websocket_policy.base_policy import BasePolicy
import lerobot.common.utils.websocket_policy.msgpack_numpy as msgpack_numpy


class WebsocketClientPolicy(BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "localhost", port: int = 8000, address: Union[str, None] = None) -> None:
        if address is None:
            address = f"ws://{host}:{port}"
        else:
            if not address.startswith(("ws://", "wss://", "http://", "https://")):
                address = f"ws://{address}"

        parsed_url = urlparse(address)

        scheme = parsed_url.scheme
        hostname = parsed_url.hostname
        port = parsed_url.port

        if hostname is None:
            raise ValueError(f"Could not extract hostname from address: {address}")

        ws_scheme = "ws"
        if scheme in ["https", "wss"]:
            ws_scheme = "wss"
            if port is None:
                port = 443
        elif scheme in ["http", "ws"]:
            ws_scheme = "ws"
            if port is None:
                port = 80
        else:
            if port is None:
                print(f"Warning: Unknown scheme '{scheme}' or no scheme, defaulting to port 8000 for ws://")
                port = 8000

        self._uri = f"{ws_scheme}://{hostname}:{port}{parsed_url.path or ''}"
        logging.info(f"🔗 Client connecting to: {self._uri}")
        print(f"🔗 Client connecting to: {self._uri}")

        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"⏳ Waiting for server at {self._uri}...")
        print(f"⏳ Waiting for server at {self._uri}...")
        
        attempt = 0
        while True:
            attempt += 1
            try:
                print(f"🔄 Attempt {attempt}: Connecting to {self._uri}")
                conn = websockets.sync.client.connect(
                    self._uri, 
                    compression=None, 
                    max_size=None,
                    # Add connection options for better reliability
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                )
                print(f"✅ Successfully connected to server!")
                
                # Receive metadata from server
                metadata = msgpack_numpy.unpackb(conn.recv())
                print(f"📥 Received server metadata")
                logging.info(f"✅ Connected to server and received metadata")
                return conn, metadata
                
            except ConnectionRefusedError:
                print(f"❌ Connection refused (attempt {attempt}) - server may not be running")
                logging.info(f"Connection refused (attempt {attempt}) - server may not be running")
                if attempt == 1:
                    print(f"💡 Make sure the server is running with: python lerobot/scripts/serve_widowx.py")
                    print(f"💡 Check that port is not in use and firewall allows connections")
                time.sleep(5)
            except websockets.exceptions.InvalidURI as e:
                print(f"❌ Invalid URI: {e}")
                logging.error(f"Invalid URI: {e}")
                raise
            except websockets.exceptions.InvalidHandshake as e:
                print(f"❌ Handshake failed (attempt {attempt}): {e}")
                logging.warning(f"Handshake failed (attempt {attempt}): {e}")
                time.sleep(2)
            except Exception as e:
                print(f"❌ Connection error (attempt {attempt}): {e}")
                logging.warning(f"Connection error (attempt {attempt}): {e}")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        try:
            data = self._packer.pack(obs)
            self._ws.send(data)
            response = self._ws.recv()
            if isinstance(response, str):
                # we're expecting bytes; if the server sends a string, it's an error.
                error_msg = f"Error in inference server:\n{response}"
                print(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            return msgpack_numpy.unpackb(response)
        except websockets.exceptions.ConnectionClosed as e:
            print(f"❌ Connection to server was closed: {e}")
            logging.error(f"Connection to server was closed: {e}")
            raise
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            logging.error(f"Error during inference: {e}")
            raise

    @override
    def reset(self) -> None:
        pass
