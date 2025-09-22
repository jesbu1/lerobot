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

import torch


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
    # VLM overlay optiona
    use_vlm: bool = False
    vlm_img_key: str = "image_0"  # e.g., "image" or "image_wrist"; None disables overlay
    vlm_server_ip: str = "http://localhost:8000"  # defaults to wrapper's SERVER_IP when None
    vlm_query_frequency: int = 5  # how many action chunks between VLM queries
    # image_keys: list[str] = ["external_img", "over_shoulder"]
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    port: int = 8001  # Port to serve the policy on.

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


def filter_type_overrides(overrides):
    filtered = []
    skip_next = False
    for i, arg in enumerate(overrides):
        if skip_next:
            skip_next = False
            continue
        # Remove --type or --policy.type and their value if split
        if arg in ("--type", "--policy.type"):
            skip_next = True
            continue
        if arg.startswith("--type=") or arg.startswith("--policy.type="):
            continue
        filtered.append(arg)
    return filtered


@custom_wrap()
def main(cfg: WidowXEvalConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    logging.info("Making environment.")

    logging.info("Making policy.")

    updated_vlm_img_key_name = (
        "path_image_0"
        if cfg.draw_path and not cfg.draw_mask
        else "masked_path_image_0"
        if cfg.draw_path and cfg.draw_mask
        else "image_0"
    )
    if not cfg.use_vlm:
        # if we are not using vlm, we don't need to draw path or mask
        cfg.draw_path = cfg.draw_mask = False
    else:
        # rename env cfg so policy sees the right key
        cfg.env.features[f"pixels/{updated_vlm_img_key_name}"] = cfg.env.features[f"pixels/{cfg.vlm_img_key}"]
        cfg.env.features_map[f"pixels/{updated_vlm_img_key_name}"] = cfg.env.features_map[
            f"pixels/{cfg.vlm_img_key}"
        ].replace(cfg.vlm_img_key, updated_vlm_img_key_name)

        # must do this because i forgot to add "images" in the prefix for the keys in the bridge dataset lol
        cfg.env.features_map[f"pixels/{updated_vlm_img_key_name}"] = cfg.env.features_map[f"pixels/{updated_vlm_img_key_name}"].replace("images.", "")

        cfg.env.features.pop(f"pixels/{cfg.vlm_img_key}")
        cfg.env.features_map.pop(f"pixels/{cfg.vlm_img_key}")
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy.eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    print("hostname:", hostname)
    print("local_ip:", local_ip)
    
    # Add more detailed logging for debugging
    logging.info(f"Starting WebSocket server on 0.0.0.0:{cfg.port}")
    print(f"WebSocket server will be available at:")
    print(f"  - ws://localhost:{cfg.port}")
    print(f"  - ws://{local_ip}:{cfg.port}")
    print(f"  - ws://{hostname}:{cfg.port}")
    
    # Test if port is available
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(1)
        result = test_socket.connect_ex(('localhost', cfg.port))
        test_socket.close()
        if result == 0:
            logging.warning(f"Port {cfg.port} appears to be in use. This might cause connection issues.")
            print(f"⚠️  Warning: Port {cfg.port} appears to be in use!")
        else:
            logging.info(f"Port {cfg.port} is available")
    except Exception as e:
        logging.warning(f"Could not test port availability: {e}")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=cfg.port,
        device=torch.device(policy.config.device),
        # VLM overlay wiring
        vlm_img_key=cfg.vlm_img_key if cfg.use_vlm else None,
        vlm_server_ip=cfg.vlm_server_ip,
        vlm_query_frequency=cfg.vlm_query_frequency,
        vlm_draw_path=cfg.draw_path,
        vlm_draw_mask=cfg.draw_mask,
        vlm_updated_img_key_name=updated_vlm_img_key_name,
    )
    
    print(f"🚀 Starting WebSocket policy server...")
    print(f"📡 Server will accept connections on port {cfg.port}")
    print(f"🔗 Clients should connect to: ws://localhost:{cfg.port}")
    print(f"⏹️  Press Ctrl+C to stop the server")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {e}")
        print(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    init_logging()
    main()
