import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


import torch

from fftnet.utils import storage
from fftnet.utils.config import build_model
from fftnet.utils.visualization import plot_embedding_spectrum


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
