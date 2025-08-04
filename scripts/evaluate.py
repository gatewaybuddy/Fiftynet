import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM

from model import FFTNet
from fftnet.data import TextFileDataset
from fftnet.utils.storage import load_model
from tokenizer import SimpleTokenizer


@torch.no_grad()
def compute_spectrum(model: FFTNet, dataset: torch.utils.data.Dataset, cfg: dict) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model.to(device)
    mags = []
    for i, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device)
        logits = model(batch)
        freq = torch.fft.fft(logits, dim=1)
        mag = freq.abs().mean(dim=(0, 2))
        mags.append(mag.cpu())
    return torch.stack(mags).mean(dim=0)


@torch.no_grad()
def evaluate(model: FFTNet, dataset: torch.utils.data.Dataset, cfg: dict, args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    count = 0

    teacher_loss = 0.0
    teacher = None
    if args.teacher_model:
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model).to(device)
        teacher.eval()
        dist_loss_fn = nn.MSELoss()

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            x, y = batch, batch
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        logits = logits.view(-1, cfg["vocab_size"])
        targets = y.view(-1)
        loss = loss_fn(logits, targets)
        total_loss += loss.item() * targets.numel()
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        count += targets.numel()
        if teacher is not None:
            teach_logits = teacher(x).logits[..., : cfg["vocab_size"]]
            tloss = dist_loss_fn(logits, teach_logits.view(-1, cfg["vocab_size"]))
            teacher_loss += tloss.item() * targets.numel()

    results = {
        "loss": total_loss / count,
        "accuracy": correct / count,
    }
    if teacher is not None:
        results["teacher_similarity"] = teacher_loss / count
    return results


def plot_and_save_spectrum(spectrum: torch.Tensor, output_path: Path) -> None:
    plt.figure()
    plt.plot(spectrum.numpy())
    plt.xlabel("Frequency index")
    plt.ylabel("Magnitude")
    plt.title("Average Logit Spectrum")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved FFTNet model")
    parser.add_argument("--model", required=True, help="Model version name")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--teacher-model", help="Optional teacher model for similarity")
    parser.add_argument("--data-path", required=True, help="Path to evaluation text file")
    parser.add_argument(
        "--tokenizer-path", default="tokenizer.json", help="Path to tokenizer"
    )
    args = parser.parse_args()

    model, cfg = load_model(Path("weights") / args.model)
    tokenizer = SimpleTokenizer.load(args.tokenizer_path)
    full_dataset = TextFileDataset(args.data_path, tokenizer, seq_len=args.seq_len)
    test_indices = range(max(0, len(full_dataset) - 20), len(full_dataset))
    test_dataset = Subset(full_dataset, list(test_indices))

    results = evaluate(model, test_dataset, cfg, args)
    spectrum = compute_spectrum(model, test_dataset, cfg)
    plot_and_save_spectrum(spectrum, Path("evaluate_spectrum.png"))

    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print("Saved evaluate_spectrum.png")


if __name__ == "__main__":
    main()
