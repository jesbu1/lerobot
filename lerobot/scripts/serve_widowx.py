# partially from openpi: https://github.com/Physical-Intelligence/openpi/scripts/serve_policy.py
import logging
import socket
from dataclasses import asdict, dataclass, field
from pprint import pformat

from lerobot.common import envs
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_logging
from lerobot.common.utils.websocket_policy import websocket_policy_server
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class WidowXEvalConfig:
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch
    # (useful for debugging). This argument is mutually exclusive with `--config`.
    draw_path: bool = True
    draw_mask: bool = True
    image_keys: list[str] = ["image_0", "image_1"]
    env: envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    port: int = 8000  # Port to serve the policy on.

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]

@parser.wrap()
def main(cfg: WidowXEvalConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    logging.info("Making environment.")

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy.eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=cfg.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    init_logging()
    main()
