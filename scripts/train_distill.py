import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM

from model import FFTNet
from fftnet.data import TextFileDataset
from fftnet.utils.config import build_model_from_config, load_config
from fftnet.utils.storage import load_model, save_model
from tokenizer import SimpleTokenizer


def distill(
    model: FFTNet,
    teacher: AutoModelForCausalLM,
    train_ds: Dataset,
    val_ds: Dataset | None,
    cfg: dict,
    args: argparse.Namespace,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size) if val_ds is not None else None
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    scaler = GradScaler(enabled=args.mixed_precision)

    log_path = Path("logs/distill_run.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    best_val = float("inf")
    patience_cntr = 0

    with log_path.open("w") as log_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            correct = 0
            count = 0
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                opt.zero_grad()
                with autocast(enabled=args.mixed_precision):
                    with torch.no_grad():
                        teach_logits = teacher(inputs).logits[..., : cfg["vocab_size"]]
                    stud_logits = model(inputs)
                    loss = loss_fn(stud_logits, teach_logits)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item() * inputs.size(0)

                preds = stud_logits.argmax(dim=-1)
                teach_preds = teach_logits.argmax(dim=-1)
                correct += (preds == teach_preds).sum().item()
                count += inputs.numel()

                step += 1
                entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "accuracy": (preds == teach_preds).float().mean().item(),
                    "timestamp": time.time(),
                }
                log_file.write(json.dumps(entry) + "\n")

            epoch_loss = total / (len(train_loader.dataset))
            epoch_acc = correct / count if count else 0.0
            print(f"Epoch {epoch}: distill_loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

            if val_loader is not None:
                model.eval()
                v_total = 0.0
                v_correct = 0
                v_count = 0
                with torch.no_grad():
                    for inputs, _ in val_loader:
                        inputs = inputs.to(device)
                        with autocast(enabled=args.mixed_precision):
                            teach_logits = teacher(inputs).logits[..., : cfg["vocab_size"]]
                            stud_logits = model(inputs)
                            v_loss = loss_fn(stud_logits, teach_logits)
                        v_total += v_loss.item() * inputs.size(0)
                        preds = stud_logits.argmax(dim=-1)
                        teach_preds = teach_logits.argmax(dim=-1)
                        v_correct += (preds == teach_preds).sum().item()
                        v_count += inputs.numel()

                val_loss = v_total / len(val_loader.dataset)
                val_acc = v_correct / v_count if v_count else 0.0
                print(f"Validation: distill_loss={val_loss:.4f} acc={val_acc:.4f}")
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
    parser = argparse.ArgumentParser(description="Distill GPT2 into FFTNet")
    parser.add_argument("--resume", metavar="VERSION", help="Resume FFTNet", nargs="?")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher-model", default="gpt2")
    parser.add_argument("--save-name", default="distilled")
    parser.add_argument("--data-path", required=True, help="Path to training text file")
    parser.add_argument(
        "--tokenizer-path", default="tokenizer.json", help="Path to load/save the tokenizer"
    )
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data reserved for validation",
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
            tokenizer = SimpleTokenizer.train_from_iterator(
                [corpus], vocab_size=args.vocab_size
            )
            tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
            tokenizer.save(str(tokenizer_path))

        cfg = load_config("config/fiftynet_config.json", "config/fiftynet_modules.yaml")
        cfg["vocab_size"] = len(tokenizer)
        model = build_model_from_config(cfg)

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model)
    dataset = TextFileDataset(args.data_path, tokenizer, seq_len=args.seq_len)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    if args.checkpoint_path is None:
        args.checkpoint_path = Path("weights") / f"{args.save_name}_best"
    else:
        args.checkpoint_path = Path(args.checkpoint_path)

    distill(model, teacher, train_ds, val_ds, cfg, args)

    save_path = Path("weights") / args.save_name
    save_model(model, str(save_path), cfg)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
