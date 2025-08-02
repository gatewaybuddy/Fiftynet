import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Allow importing utility modules from the scripts directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from train_fresh import DummyWikiDataset  # type: ignore
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


def _tokenize(prompt: str) -> list[int]:
    """Convert prompt words to token IDs."""
    mapping = DummyWikiDataset.WORD_TO_ID
    return [mapping.get(w, 0) for w in prompt.split()]


def _decode(tokens: torch.Tensor) -> str:
    vocab = DummyWikiDataset.VOCAB
    return " ".join(vocab[t.item()] for t in tokens)


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
    args = parser.parse_args()

    if args.model:
        model, cfg = storage.load_model(Path("weights") / args.model)
    else:
        with open("config/fiftynet_config.json") as f:
            cfg = json.load(f)
        model = FFTNet(**{k: cfg[k] for k in ("vocab_size", "dim", "num_blocks")})

    tokens = _tokenize(args.prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated, logits = generate(model, input_ids, args.max_new_tokens)

    if args.mode == "text":
        print(_decode(generated[0]))
    elif args.mode == "logits":
        last_logits = logits[0, -1]
        k = min(args.top_k, last_logits.size(0))
        values, indices = torch.topk(last_logits, k)
        vocab = DummyWikiDataset.VOCAB
        for idx, val in zip(indices.tolist(), values.tolist()):
            word = vocab[idx] if idx < len(vocab) else str(idx)
            print(f"{word}: {val:.4f}")
    else:  # spectrum
        embeddings = model.embedding(generated)
        fft_visualizer(embeddings)


if __name__ == "__main__":
    main()
