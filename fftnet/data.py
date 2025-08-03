from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from tokenizer import SimpleTokenizer


class TextFileDataset(Dataset):
    """Dataset that tokenizes a text file into fixed-length sequences."""

    def __init__(self, path: str | Path, tokenizer: SimpleTokenizer, seq_len: int) -> None:
        text = Path(path).read_text(encoding="utf-8")
        self.tokens: Sequence[int] = tokenizer.encode(text)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.tokens[idx : idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1 : idx + self.seq_len + 1], dtype=torch.long)
        return x, y


