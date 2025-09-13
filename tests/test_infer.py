import os
import sys
import subprocess
from pathlib import Path

import torch
import pytest

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tokenizer import SimpleTokenizer
from fftnet.utils.config import build_model_from_config
from fftnet_infer import generate


@pytest.fixture
def tiny_tokenizer(tmp_path):
    """Create and persist a minimal tokenizer for testing."""
    tokenizer = SimpleTokenizer.train_from_iterator([
        "hello world"
    ], vocab_size=32, min_frequency=1)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return tokenizer, path


def _build_model(tokenizer: SimpleTokenizer):
    cfg = {
        "vocab_size": len(tokenizer),
        "dim": 2,
        "blocks": [
            {"type": "FFTNetBlock", "name": "block_0"},
        ],
    }
    return build_model_from_config(cfg)


def test_generate_shapes(tiny_tokenizer):
    tokenizer, _ = tiny_tokenizer
    model = _build_model(tokenizer)
    input_ids = torch.tensor([tokenizer.encode("hi")], dtype=torch.long)
    generated, logits = generate(model, input_ids, max_new_tokens=2)
    assert generated.shape == (1, input_ids.shape[1] + 2)
    assert logits.shape == (1, generated.shape[1], len(tokenizer))


def test_cli_modes(tiny_tokenizer):
    tokenizer, path = tiny_tokenizer
    repo_root = Path(__file__).resolve().parents[1]
    base_cmd = [
        sys.executable,
        str(repo_root / "fftnet_infer.py"),
        "--prompt",
        "hi",
        "--max-new-tokens",
        "1",
        "--tokenizer-path",
        str(path),
    ]
    env = dict(os.environ, MPLBACKEND="Agg")
    subprocess.run(base_cmd + ["--mode", "text"], check=True, cwd=repo_root, env=env)
    subprocess.run(base_cmd + ["--mode", "spectrum"], check=True, cwd=repo_root, env=env)
