import json
from typing import Any, Dict, Tuple

import yaml
from torch import nn

from model import FFTNet
from fftnet_block import FFTNetBlock

# Registry mapping string names to block classes
BLOCK_REGISTRY = {
    "FFTNetBlock": FFTNetBlock,
}


def load_config(config_path: str, modules_path: str) -> Dict[str, Any]:
    """Load base configuration and block definitions.

    Parameters
    ----------
    config_path: str
        Path to a JSON file containing base hyperparameters.
    modules_path: str
        Path to a YAML file defining the model's blocks.
    """
    with open(config_path, "r") as f:
        cfg = json.load(f)
    with open(modules_path, "r") as f:
        modules = yaml.safe_load(f) or {}
    blocks = modules.get("blocks", [])
    cfg["blocks"] = blocks
    cfg["num_blocks"] = len(blocks)
    return cfg


def build_model_from_config(cfg: Dict[str, Any]) -> FFTNet:
    """Instantiate an ``FFTNet`` from a configuration dictionary."""
    blocks = []
    if "blocks" in cfg:
        for block_cfg in cfg["blocks"]:
            block_type = block_cfg["type"]
            block_cls = BLOCK_REGISTRY.get(block_type)
            if block_cls is None:
                raise ValueError(f"Unknown block type {block_type}")
            params = {k: v for k, v in block_cfg.items() if k not in {"type", "name"}}
            blocks.append(block_cls(cfg["dim"], **params))
        num_blocks = len(blocks)
    else:
        num_blocks = int(cfg.get("num_blocks", 0))
        blocks = [FFTNetBlock(cfg["dim"]) for _ in range(num_blocks)]

    model = FFTNet(cfg["vocab_size"], cfg["dim"], num_blocks)
    model.blocks = nn.ModuleList(blocks)
    return model


def build_model(config_path: str, modules_path: str) -> Tuple[FFTNet, Dict[str, Any]]:
    """Load configuration files and build the corresponding model."""
    cfg = load_config(config_path, modules_path)
    model = build_model_from_config(cfg)
    return model, cfg
