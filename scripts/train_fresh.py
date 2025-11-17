import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split

from model import FFTNet
from fftnet.data import TextFileDataset
from fftnet.utils.config import load_config, build_model_from_config
from fftnet.utils.storage import save_model, load_model
from tokenizer import SimpleTokenizer


def save_checkpoint(
    checkpoint_dir: Path,
    model: FFTNet,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
    best_val: float,
    patience_cntr: int,
    cfg: dict,
) -> None:
    """Save a complete training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val": best_val,
        "patience_cntr": patience_cntr,
        "config": cfg,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)

    # Save a symlink to the latest checkpoint
    latest_path = checkpoint_dir / "latest.pt"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(checkpoint_path.name)

    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: FFTNet,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> tuple[int, int, float, int, dict]:
    """Load a training checkpoint and return training state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    step = checkpoint["step"]
    best_val = checkpoint["best_val"]
    patience_cntr = checkpoint["patience_cntr"]
    cfg = checkpoint["config"]

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}, step {step}")

    return epoch, step, best_val, patience_cntr, cfg


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int = 3) -> None:
    """Keep only the last N checkpoints to save disk space."""
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda p: p.stat().st_mtime
    )

    # Remove all but the last N checkpoints
    for checkpoint in checkpoints[:-keep_last_n]:
        checkpoint.unlink()
        print(f"Removed old checkpoint: {checkpoint.name}")


def train(
    model: FFTNet,
    train_ds: Dataset,
    val_ds: Dataset | None,
    cfg: dict,
    args: argparse.Namespace,
    start_epoch: int = 1,
    start_step: int = 0,
    start_best_val: float = float("inf"),
    start_patience_cntr: int = 0,
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

    # Learning rate scheduler
    scheduler = None
    if args.scheduler != "none":
        total_steps = len(train_loader) * args.epochs
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_steps, eta_min=args.lr * 0.1
            )
        elif args.scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
            )
        elif args.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=2, verbose=True
            )

    log_path = Path("logs/fresh_run.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step = start_step
    best_val = start_best_val
    patience_cntr = start_patience_cntr

    # Append to log file if resuming, otherwise create new
    log_mode = "a" if start_epoch > 1 else "w"
    with log_path.open(log_mode) as log_file:
        for epoch in range(start_epoch, args.epochs + 1):
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

                # Gradient clipping
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                    )
                else:
                    grad_norm = None

                scaler.step(opt)
                scaler.update()

                # Update learning rate scheduler (step-based schedulers)
                if scheduler is not None and args.scheduler != "plateau":
                    scheduler.step()

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
                    "lr": opt.param_groups[0]["lr"],
                    "timestamp": time.time(),
                }
                if grad_norm is not None:
                    entry["grad_norm"] = grad_norm.item()
                log_file.write(json.dumps(entry) + "\n")

                # Periodic checkpoint saving
                if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                    checkpoint_dir = Path(args.checkpoint_dir)
                    save_checkpoint(
                        checkpoint_dir, model, opt, scaler, scheduler,
                        epoch, step, best_val, patience_cntr, cfg
                    )
                    if args.keep_last_n > 0:
                        cleanup_old_checkpoints(checkpoint_dir, args.keep_last_n)

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

                # Update plateau scheduler
                if scheduler is not None and args.scheduler == "plateau":
                    scheduler.step(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    patience_cntr = 0
                    if args.checkpoint_path is not None:
                        save_model(model, str(args.checkpoint_path), cfg)
                        print(f"Best model saved to {args.checkpoint_path}")
                else:
                    patience_cntr += 1
                    if args.patience and patience_cntr >= args.patience:
                        print("Early stopping triggered")
                        break

            # Save checkpoint at end of epoch
            if args.checkpoint_dir:
                checkpoint_dir = Path(args.checkpoint_dir)
                save_checkpoint(
                    checkpoint_dir, model, opt, scaler, scheduler,
                    epoch, step, best_val, patience_cntr, cfg
                )
                if args.keep_last_n > 0:
                    cleanup_old_checkpoints(checkpoint_dir, args.keep_last_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FFTNet from scratch")
    parser.add_argument("--resume", metavar="VERSION", help="Resume from saved model", nargs="?")
    parser.add_argument("--resume-from", type=str, help="Resume from checkpoint file")
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
        "--split-seed",
        type=int,
        default=42,
        help="Random seed to reproduce train/val splits",
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
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory to save periodic checkpoints",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save checkpoint every N steps (0 to disable)",
    )
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=3,
        help="Keep only last N checkpoints",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine", "onecycle", "plateau"],
        default="none",
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm (0 to disable)",
    )
    args = parser.parse_args()

    # Initialize training state
    start_epoch = 1
    start_step = 0
    start_best_val = float("inf")
    start_patience_cntr = 0

    # Handle resuming from checkpoint
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load tokenizer first
        tokenizer = SimpleTokenizer.load(args.tokenizer_path)

        # Load config from checkpoint
        temp_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = temp_checkpoint["config"]

        # Build model from config
        model = build_model_from_config(cfg)

        # Create optimizer and scaler (will be loaded from checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler = GradScaler(enabled=args.mixed_precision)

        # Create scheduler if needed
        scheduler = None
        if args.scheduler != "none":
            full_dataset = TextFileDataset(args.data_path, tokenizer, seq_len=args.seq_len)
            train_loader_temp = DataLoader(full_dataset, batch_size=args.batch_size)
            total_steps = len(train_loader_temp) * args.epochs
            if args.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=total_steps, eta_min=args.lr * 0.1
                )
            elif args.scheduler == "onecycle":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1
                )
            elif args.scheduler == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode="min", factor=0.5, patience=2, verbose=True
                )

        # Load checkpoint state
        start_epoch, start_step, start_best_val, start_patience_cntr, cfg = load_checkpoint(
            checkpoint_path, model, opt, scaler, scheduler
        )
        start_epoch += 1  # Start from next epoch

        print(f"Resuming training from epoch {start_epoch}, step {start_step}")

    elif args.resume:
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
        generator = torch.Generator().manual_seed(args.split_seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

    if args.checkpoint_path is None:
        args.checkpoint_path = Path("weights") / f"{args.save_name}_best"
    else:
        args.checkpoint_path = Path(args.checkpoint_path)

    train(
        model, train_dataset, val_dataset, cfg, args,
        start_epoch, start_step, start_best_val, start_patience_cntr
    )

    save_path = Path("weights") / args.save_name
    save_model(model, str(save_path), cfg)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
