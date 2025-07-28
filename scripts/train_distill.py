import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM

from model import FFTNet
from fftnet.utils.storage import save_model, load_model


class DummyWikiDataset(Dataset):
    """Very small dataset of repeated words."""

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
        text = ("the quick brown fox jumps over lazy dog lorem ipsum " * 100).strip()
        ids = [self.WORD_TO_ID[w] for w in text.split()]
        self.seq_len = seq_len
        self.seqs = [ids[i : i + seq_len] for i in range(len(ids) - seq_len)]

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.seqs[idx], dtype=torch.long)
        return x


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
    args = parser.parse_args()

    if args.resume:
        model, cfg = load_model(Path("weights") / args.resume)
    else:
        with open("config/fiftynet_config.json") as f:
            cfg = json.load(f)
        model = FFTNet(**{k: cfg[k] for k in ("vocab_size", "dim", "num_blocks")})

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model)
    dataset = DummyWikiDataset(seq_len=args.seq_len)

    distill(model, teacher, dataset, cfg, args)

    save_path = Path("weights") / args.save_name
    save_model(model, str(save_path), cfg)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
