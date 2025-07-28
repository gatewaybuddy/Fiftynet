import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import FFTNet
from fftnet.utils.storage import save_model, load_model


class DummyWikiDataset(Dataset):
    """Simple token dataset built from a tiny Wikipedia-like text."""

    VOCAB = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "lorem",
        "ipsum",
    ]

    WORD_TO_ID = {w: i for i, w in enumerate(VOCAB)}

    def __init__(self, seq_len: int) -> None:
        sample_text = (
            "the quick brown fox jumps over lazy dog lorem ipsum " * 100
        ).strip()
        tokens = [self.WORD_TO_ID[w] for w in sample_text.split()]
        self.seq_len = seq_len
        self.seqs = [tokens[i : i + seq_len + 1] for i in range(len(tokens) - seq_len)]

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        seq = self.seqs[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


def train(model: FFTNet, dataset: Dataset, cfg: dict, args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    log_path = Path("logs/fresh_run.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0

    with log_path.open("w") as log_file:
        for epoch in range(1, args.epochs + 1):
            total_loss = 0.0
            correct = 0
            count = 0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                logits = model(x)
                logits = logits.view(-1, cfg["vocab_size"])
                targets = y.view(-1)
                loss = loss_fn(logits, targets)
                loss.backward()
                opt.step()
                total_loss += loss.item() * targets.numel()
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                count += targets.numel()

                step += 1
                entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "accuracy": (preds == targets).float().mean().item(),
                    "timestamp": time.time(),
                }
                log_file.write(json.dumps(entry) + "\n")

            avg_loss = total_loss / count
            acc = correct / count
            print(f"Epoch {epoch}: loss={avg_loss:.4f} acc={acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FFTNet from scratch")
    parser.add_argument("--resume", metavar="VERSION", help="Resume from saved model", nargs="?")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-name", default="trained", help="Version name to save")
    args = parser.parse_args()

    if args.resume:
        model, cfg = load_model(Path("weights") / args.resume)
    else:
        with open("config/fiftynet_config.json") as f:
            cfg = json.load(f)
        model = FFTNet(**{k: cfg[k] for k in ("vocab_size", "dim", "num_blocks")})

    dataset = DummyWikiDataset(seq_len=args.seq_len)
    train(model, dataset, cfg, args)

    save_path = Path("weights") / args.save_name
    save_model(model, str(save_path), cfg)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
