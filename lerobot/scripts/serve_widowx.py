# partially from openpi: https://github.com/Physical-Intelligence/openpi/scripts/serve_policy.py
from dataclasses import dataclass
import enum
import logging
import socket

import tyro

from lerobot.common.envs.factory import make_env
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.websocket_policy import websocket_policy_server


from lerobot.configs.eval import EvalPipelineConfig as BaseEvalPipelineConfig


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    WIDOWX = "widowx"


@dataclass
class EvalPipelineConfig(BaseEvalPipelineConfig):
    path_and_mask_h5_file: str | None = None
    draw_path: bool = True
    draw_mask: bool = True
    image_keys: list[str] = ["image_0", "image_1"]


class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.WIDOWX

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Checkpoint)


def create_policy(args: Args) -> Policy:
    """Create a policy from the given arguments."""
    env_config = make_env(args.env.value, return_config_only=True)
    match args.policy:
        case Checkpoint():
            policy_config = PolicyConfig(path=args.policy.path)
            return make_policy(cfg=policy_config, env_cfg=env_config)
        case Default():
            raise NotImplementedError("Default policies are not yet supported for WidowX.")


def main(args: Args) -> None:
    policy = create_policy(args)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        # TODO(rcadene): metadata not used for now, but could be useful for some checks on client side
        # metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
