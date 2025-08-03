import argparse
import os
import sys
from pathlib import Path

import torch

# Allow importing utility modules from the scripts directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from tokenizer import SimpleTokenizer
try:
    from main import plot_embedding_spectrum as fft_visualizer  # type: ignore
except Exception:  # pragma: no cover
    # Fallback visualizer if scripts/main.py is not available
    import matplotlib.pyplot as plt

    def fft_visualizer(embeddings: torch.Tensor) -> None:
        """Plot average frequency magnitude across tokens in a sequence."""
        if embeddings.ndim != 3:
            raise ValueError("embeddings must be 3D [batch, seq_len, dim]")
        with torch.no_grad():
            freq = torch.fft.fft(embeddings, dim=1)
            magnitude = freq.abs().mean(dim=(0, 2))
        plt.figure()
        plt.plot(torch.arange(magnitude.size(0)), magnitude.cpu())
        plt.xlabel("Frequency index")
        plt.ylabel("Magnitude")
        plt.title("Average Frequency Spectrum")
        plt.tight_layout()
        plt.show()
        plt.close()


from model import FFTNet
from fftnet.utils import storage
from fftnet.utils.config import load_config, build_model_from_config


def _tokenize(tokenizer: SimpleTokenizer, prompt: str) -> list[int]:
    """Convert prompt text to token IDs."""
    return tokenizer.encode(prompt)


def _decode(tokenizer: SimpleTokenizer, tokens: torch.Tensor) -> str:
    return tokenizer.decode(tokens.tolist())


@torch.no_grad()
def generate(model: FFTNet, input_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy generation returning final tokens and logits."""
    device = next(model.parameters()).device
    generated = input_ids.to(device)
    for _ in range(max_new_tokens):
        logits = model(generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    logits = model(generated)
    return generated, logits


def main() -> None:
    parser = argparse.ArgumentParser(description="FFTNet inference")
    parser.add_argument("--model", help="Model version name to load", metavar="VERSION", nargs="?")
    parser.add_argument("--prompt", default="the quick", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=5, help="Number of tokens to generate")
    parser.add_argument("--mode", choices=["text", "logits", "spectrum"], default="text")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k predictions for logits mode")
    parser.add_argument("--tokenizer-path", default="tokenizer.json", help="Tokenizer file")
    args = parser.parse_args()

    tokenizer = SimpleTokenizer.load(args.tokenizer_path)

    if args.model:
        model, cfg = storage.load_model(Path("weights") / args.model)
    else:
        cfg = load_config("config/fiftynet_config.json", "config/fiftynet_modules.yaml")
        cfg["vocab_size"] = len(tokenizer)
        model = build_model_from_config(cfg)

    tokens = _tokenize(tokenizer, args.prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated, logits = generate(model, input_ids, args.max_new_tokens)

    if args.mode == "text":
        print(_decode(tokenizer, generated[0]))
    elif args.mode == "logits":
        last_logits = logits[0, -1]
        k = min(args.top_k, last_logits.size(0))
        values, indices = torch.topk(last_logits, k)
        for idx, val in zip(indices.tolist(), values.tolist()):
            word = tokenizer.decode([idx])
            print(f"{word}: {val:.4f}")
    else:  # spectrum
        embeddings = model.embedding(generated)
        fft_visualizer(embeddings)


if __name__ == "__main__":
    main()
