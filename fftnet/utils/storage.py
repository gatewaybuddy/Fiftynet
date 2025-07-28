import json
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from safetensors.torch import save_file, safe_open

from model import FFTNet


def _prepare_state_for_save(state: dict) -> tuple[dict, list[str]]:
    processed = {}
    complex_keys = []
    for k, v in state.items():
        if torch.is_complex(v):
            processed[k] = torch.view_as_real(v)
            complex_keys.append(k)
        else:
            processed[k] = v
    return processed, complex_keys


def _restore_complex(state: dict, complex_keys: list[str]) -> dict:
    for k in complex_keys:
        state[k] = torch.view_as_complex(state[k])
    return state


def save_model(model: nn.Module, path: str, config: dict) -> None:
    """Save model weights and config using safetensors."""
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    weights_path = base.with_suffix('.safetensors')
    state, complex_keys = _prepare_state_for_save(model.state_dict())
    metadata = {"complex": ",".join(complex_keys)} if complex_keys else None
    save_file(state, str(weights_path), metadata=metadata)
    config_path = base.as_posix() + '_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f)


def load_model(path: str) -> Tuple[nn.Module, dict]:
    """Load model weights and configuration."""
    base = Path(path)
    config_path = base.as_posix() + '_config.json'
    weights_path = base.with_suffix('.safetensors')
    if not Path(config_path).exists() or not weights_path.exists():
        raise FileNotFoundError(f'Model files for {path} not found')
    with open(config_path, 'r') as f:
        config = json.load(f)
    tensors = {}
    complex_keys = []
    with safe_open(str(weights_path), framework="pt") as f:
        meta = f.metadata()
        if meta and "complex" in meta:
            complex_keys = [k for k in meta["complex"].split(',') if k]
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    tensors = _restore_complex(tensors, complex_keys)
    model = FFTNet(**{k: config[k] for k in ('vocab_size', 'dim', 'num_blocks')})
    model.load_state_dict(tensors)
    return model, config
