import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split

from model import FFTNet
from fftnet.data import TextFileDataset
from fftnet.utils.config import load_config, build_model_from_config
from fftnet.utils.storage import save_model, load_model
from tokenizer import SimpleTokenizer


def train(
    model: FFTNet,
    train_ds: Dataset,
    val_ds: Dataset | None,
    cfg: dict,
    args: argparse.Namespace,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size) if val_ds is not None else None
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=args.mixed_precision)

    log_path = Path("logs/fresh_run.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    best_val = float("inf")
    patience_cntr = 0

    with log_path.open("w") as log_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            count = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                with autocast(enabled=args.mixed_precision):
                    logits = model(x)
                    logits = logits.view(-1, cfg["vocab_size"])
                    targets = y.view(-1)
                    loss = loss_fn(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
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

            if val_loader is not None:
                model.eval()
                v_total = 0.0
                v_correct = 0
                v_count = 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(device)
                        y = y.to(device)
                        with autocast(enabled=args.mixed_precision):
                            logits = model(x)
                            logits = logits.view(-1, cfg["vocab_size"])
                            targets = y.view(-1)
                            v_loss = loss_fn(logits, targets)
                        v_total += v_loss.item() * targets.numel()
                        preds = logits.argmax(dim=-1)
                        v_correct += (preds == targets).sum().item()
                        v_count += targets.numel()

                val_loss = v_total / v_count
                val_acc = v_correct / v_count
                print(
                    f"Validation: loss={val_loss:.4f} acc={val_acc:.4f}"
                )
                val_entry = {
                    "step": step,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "timestamp": time.time(),
                }
                log_file.write(json.dumps(val_entry) + "\n")

                if val_loss < best_val:
                    best_val = val_loss
                    patience_cntr = 0
                    if args.checkpoint_path is not None:
                        save_model(model, str(args.checkpoint_path), cfg)
                else:
                    patience_cntr += 1
                    if args.patience and patience_cntr >= args.patience:
                        print("Early stopping triggered")
                        break


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
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--val-data-path",
        help="Optional path to validation text file",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation if no val-data-path provided",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (0 to disable)",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Where to save best model checkpoint",
    )
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

    full_dataset = TextFileDataset(args.data_path, tokenizer, seq_len=args.seq_len)
    if args.val_data_path:
        val_dataset = TextFileDataset(args.val_data_path, tokenizer, seq_len=args.seq_len)
        train_dataset = full_dataset
    else:
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    if args.checkpoint_path is None:
        args.checkpoint_path = Path("weights") / f"{args.save_name}_best"
    else:
        args.checkpoint_path = Path(args.checkpoint_path)

    train(model, train_dataset, val_dataset, cfg, args)

    save_path = Path("weights") / args.save_name
    save_model(model, str(save_path), cfg)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
