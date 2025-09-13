import argparse
import sys
import torch

from fftnet.utils import storage
from fftnet.utils.config import build_model
from fftnet.utils.visualization import plot_embedding_spectrum


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
        model, cfg = build_model(
            "config/fiftynet_config.json", "config/fiftynet_modules.yaml"
        )

    input_ids = torch.randint(0, cfg["vocab_size"], (1, 8))
    embeddings = model.embedding(input_ids)
    plot_embedding_spectrum(embeddings)


if __name__ == "__main__":
    main()
