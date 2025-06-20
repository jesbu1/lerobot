# partially from openpi: https://github.com/Physical-Intelligence/openpi/scripts/serve_policy.py
import logging
import socket
import sys
from dataclasses import asdict, dataclass, field
from pprint import pformat

from lerobot.common import envs
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_logging
from lerobot.common.utils.websocket_policy import websocket_policy_server
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.policies import PreTrainedConfig


def custom_wrap():
    """Custom wrapper that allows both --policy.path and --policy.type arguments"""
    from functools import wraps
    import draccus
    from lerobot.configs.parser import parse_plugin_args, load_plugin, filter_arg, parse_arg, get_cli_overrides, get_path_arg, get_type_arg
    from lerobot.common.utils.utils import has_method
    
    def wrapper_outer(fn):
        @wraps(fn)
        def wrapper_inner(*args, **kwargs):
            import inspect
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]
            if len(args) > 0 and type(args[0]) is argtype:
                cfg = args[0]
                args = args[1:]
            else:
                cli_args = sys.argv[1:]
                plugin_args = parse_plugin_args("discover_packages_path", cli_args)
                for plugin_cli_arg, plugin_path in plugin_args.items():
                    try:
                        load_plugin(plugin_path)
                    except Exception as e:
                        raise Exception(f"{e}\nFailed plugin CLI Arg: {plugin_cli_arg}") from e
                    cli_args = filter_arg(plugin_cli_arg, cli_args)
                
                # Custom handling for policy arguments - allow both path and type
                policy_path = get_path_arg("policy", cli_args)
                policy_type = get_type_arg("policy", cli_args)
                
                # Filter out ONLY policy arguments from CLI args for draccus parsing
                # Keep other arguments like --env
                cli_args = [arg for arg in cli_args if not arg.startswith("--policy.")]
                
                cfg = draccus.parse(config_class=argtype, args=cli_args)
                
                # Handle policy loading in __post_init__
                
            response = fn(cfg, *args, **kwargs)
            return response

        return wrapper_inner

    return wrapper_outer


@dataclass
class WidowXEvalConfig:
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch
    # (useful for debugging). This argument is mutually exclusive with `--config`.

    env: envs.EnvConfig = field(default_factory=lambda: envs.WidowXEnv())
    draw_path: bool = True
    draw_mask: bool = True
    # image_keys: list[str] = ["external_img", "over_shoulder"]
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    port: int = 8001  # Port to serve the policy on.

    def __post_init__(self):
        # Check if policy path is provided via CLI
        policy_path = parser.get_path_arg("policy")
        policy_type = parser.get_type_arg("policy")
        
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            # Filter out both --type=... and --policy.type=... from CLI overrides
            cli_overrides = [arg for arg in cli_overrides if not (arg.startswith("--type=") or arg.startswith("--policy.type="))]
            
            # If both path and type are specified, we need to handle this specially
            if policy_type:
                # Create a temporary config with the specified type to get the right class
                from lerobot.common.policies.factory import make_policy_config
                temp_config = make_policy_config(policy_type)
                # Load the config from the path but use the CLI overrides
                self.policy = temp_config.__class__.from_pretrained(policy_path, cli_overrides=cli_overrides)
            else:
                # Standard case: just load from path
                self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        elif self.policy is None:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@custom_wrap()
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
        host="1.0.0.0",
        port=cfg.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    init_logging()
    main()
