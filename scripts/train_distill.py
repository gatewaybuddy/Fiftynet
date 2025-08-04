import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

from model import FFTNet
from fftnet.data import TextFileDataset
from fftnet.utils.config import build_model_from_config, load_config
from fftnet.utils.storage import load_model, save_model
from tokenizer import SimpleTokenizer


def distill(model: FFTNet, teacher: AutoModelForCausalLM, dataset: Dataset, cfg: dict, args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    log_path = Path("logs/distill_run.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0

    with log_path.open("w") as log_file:
        for epoch in range(1, args.epochs + 1):
            total = 0.0
            correct = 0
            count = 0
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                opt.zero_grad()
                with torch.no_grad():
                    teach_logits = teacher(batch).logits[..., : cfg["vocab_size"]]
                stud_logits = model(batch)
                loss = loss_fn(stud_logits, teach_logits)
                loss.backward()
                opt.step()
                total += loss.item() * batch.size(0)

                preds = stud_logits.argmax(dim=-1)
                teach_preds = teach_logits.argmax(dim=-1)
                correct += (preds == teach_preds).sum().item()
                count += batch.numel()

                step += 1
                entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "accuracy": (preds == teach_preds).float().mean().item(),
                    "timestamp": time.time(),
                }
                log_file.write(json.dumps(entry) + "\n")

            epoch_loss = total / (len(loader.dataset))
            epoch_acc = correct / count if count else 0.0
            print(f"Epoch {epoch}: distill_loss={epoch_loss:.4f} acc={epoch_acc:.4f}")


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

    distill(model, teacher, dataset, cfg, args)

    save_path = Path("weights") / args.save_name
    save_model(model, str(save_path), cfg)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
