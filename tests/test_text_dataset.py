from pathlib import Path

import torch

from fftnet.data import TextFileDataset
from tokenizer import SimpleTokenizer


def test_text_file_dataset(tmp_path: Path) -> None:
    text = "hello world hello"
    file_path = tmp_path / "corpus.txt"
    file_path.write_text(text)

    tokenizer = SimpleTokenizer.train_from_iterator([text], vocab_size=10)

    ds = TextFileDataset(file_path, tokenizer, seq_len=2)
    assert len(ds) == len(ds.tokens) - 2
    x, y = ds[0]
    assert x.shape == (2,)
    assert y.shape == (2,)
    # successive tokens shifted by one
    assert torch.equal(x[1:], y[:-1])
