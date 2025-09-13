import argparse
import os
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from fftnet.utils import storage
from fftnet.utils.config import build_model



def plot_embedding_spectrum(embeddings: torch.Tensor, save_path: Optional[str] = None) -> None:
    """Plot average frequency magnitude across tokens in a sequence.

    Args:
        embeddings: Input embedding tensor of shape [batch, seq_len, dim].
        save_path: If provided, write the plot to this path instead of showing.
    """
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
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FFTNet demo. Saves spectrum when no display is available."
    )
    parser.add_argument("--model", metavar="VERSION", help="Load model version", nargs="?")
    parser.add_argument(
        "--save-plot",
        metavar="PATH",
        help=(
            "Save spectrum image to PATH instead of displaying it. "
            "When no display is available the plot is saved to PATH or 'spectrum.png' by default."
        ),
    )
    args = parser.parse_args()

    if args.model:
        try:
            model, cfg = storage.load_model(f"weights/{args.model}")
            print(f"Loaded model {args.model}")
        except FileNotFoundError:
            print(f"Model {args.model} not found", file=sys.stderr)
            return
    else:
        model, cfg = build_model(
            "config/fiftynet_config.json", "config/fiftynet_modules.yaml"
        )

    input_ids = torch.randint(0, cfg["vocab_size"], (1, 8))
    embeddings = model.embedding(input_ids)

    save_path = args.save_plot
    if save_path is None and not os.environ.get("DISPLAY"):
        save_path = "spectrum.png"

    plot_embedding_spectrum(embeddings, save_path)


if __name__ == "__main__":
    main()
