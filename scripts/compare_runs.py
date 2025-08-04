import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from model import FFTNet
from fftnet.data import TextFileDataset
from fftnet.utils.storage import load_model
from tokenizer import SimpleTokenizer




def load_losses(path: Path) -> Tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    if not path.exists():
        return steps, losses
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            steps.append(rec.get("step", len(steps) + 1))
            losses.append(rec.get("loss", 0.0))
    return steps, losses
def average_logit_spectrum(
    model: FFTNet, dataset: TextFileDataset, num_batches: int = 5
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    mags = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            logits = model(batch)
            freq = torch.fft.fft(logits, dim=1)
            mag = freq.abs().mean(dim=(0, 2))
            mags.append(mag.cpu())
    return torch.stack(mags).mean(dim=0)


def plot_comparison(args: argparse.Namespace) -> None:
    fresh_steps, fresh_loss = load_losses(args.fresh_log)
    distill_steps, distill_loss = load_losses(args.distill_log)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(fresh_steps, fresh_loss, label="fresh")
    axes[0].plot(distill_steps, distill_loss, label="distill")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    if args.fresh_model and args.distill_model and args.data_path:
        tokenizer = SimpleTokenizer.load(args.tokenizer_path)
        dataset = TextFileDataset(args.data_path, tokenizer, seq_len=args.seq_len)
        fresh_model, _ = load_model(args.fresh_model)
        distill_model, _ = load_model(args.distill_model)
        fresh_fft = average_logit_spectrum(fresh_model, dataset)
        distill_fft = average_logit_spectrum(distill_model, dataset)
        axes[1].plot(fresh_fft.numpy(), label="fresh")
        axes[1].plot(distill_fft.numpy(), label="distill")
        axes[1].set_title("Average Logit Spectrum")
        axes[1].set_xlabel("Frequency Index")
        axes[1].set_ylabel("Magnitude")
        axes[1].legend()
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("training_comparison.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare training logs")
    parser.add_argument("--fresh-log", type=Path, default=Path("logs/fresh_run.jsonl"))
    parser.add_argument("--distill-log", type=Path, default=Path("logs/distill_run.jsonl"))
    parser.add_argument("--fresh-model", type=Path, help="Path to fresh model for FFT analysis")
    parser.add_argument("--distill-model", type=Path, help="Path to distilled model for FFT analysis")
    parser.add_argument("--data-path", type=Path, help="Path to text file for FFT analysis")
    parser.add_argument(
        "--tokenizer-path", type=Path, default=Path("tokenizer.json"), help="Tokenizer path"
    )
    parser.add_argument("--seq-len", type=int, default=8)
    args = parser.parse_args()
    plot_comparison(args)
    print("Saved training_comparison.png")


if __name__ == "__main__":
    main()
