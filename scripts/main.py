import argparse
import json
import sys
import torch
import matplotlib.pyplot as plt

from model import FFTNet
from fftnet.utils import storage


def plot_embedding_spectrum(embeddings: torch.Tensor) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="FFTNet demo")
    parser.add_argument("--model", metavar="VERSION", help="Load model version", nargs="?")
    args = parser.parse_args()

    if args.model:
        try:
            model, cfg = storage.load_model(f"weights/{args.model}")
            print(f"Loaded model {args.model}")
        except FileNotFoundError:
            print(f"Model {args.model} not found", file=sys.stderr)
            return
    else:
        with open("config/fiftynet_config.json") as f:
            cfg = json.load(f)
        model = FFTNet(**{k: cfg[k] for k in ("vocab_size", "dim", "num_blocks")})

    input_ids = torch.randint(0, cfg["vocab_size"], (1, 8))
    embeddings = model.embedding(input_ids)
    plot_embedding_spectrum(embeddings)


if __name__ == "__main__":
    main()
