import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import FFTNet
from fftnet.data import TextFileDataset
from fftnet.utils.config import load_config, build_model_from_config
from fftnet.utils.storage import save_model, load_model
from tokenizer import SimpleTokenizer


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
    parser.add_argument("--data-path", required=True, help="Path to training text file")
    parser.add_argument(
        "--tokenizer-path",
        default="tokenizer.json",
        help="Path to load/save the tokenizer",
    )
    parser.add_argument("--vocab-size", type=int, default=5000, help="Tokenizer vocab size")
    args = parser.parse_args()

    if args.resume:
        model, cfg = load_model(Path("weights") / args.resume)
        tokenizer = SimpleTokenizer.load(args.tokenizer_path)
    else:
        tokenizer_path = Path(args.tokenizer_path)
        if tokenizer_path.exists():
            tokenizer = SimpleTokenizer.load(str(tokenizer_path))
        else:
            corpus = Path(args.data_path).read_text(encoding="utf-8")
            tokenizer = SimpleTokenizer.train_from_iterator([corpus], vocab_size=args.vocab_size)
            tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(tokenizer_path))

        cfg = load_config("config/fiftynet_config.json", "config/fiftynet_modules.yaml")
        cfg["vocab_size"] = len(tokenizer)
        model = build_model_from_config(cfg)

    dataset = TextFileDataset(args.data_path, tokenizer, seq_len=args.seq_len)
    train(model, dataset, cfg, args)

    save_path = Path("weights") / args.save_name
    save_model(model, str(save_path), cfg)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
