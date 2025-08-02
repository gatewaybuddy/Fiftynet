import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tokenizer import SimpleTokenizer


def test_train_encode_decode(tmp_path):
    corpus = ["hello world", "hello there"]
    tokenizer = SimpleTokenizer.train_from_iterator(corpus, vocab_size=100)

    ids = tokenizer.encode("hello world")
    assert tokenizer.decode(ids) == "hello world"

    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))

    loaded = SimpleTokenizer.load(str(path))
    assert loaded.decode(ids) == "hello world"
    assert len(loaded) == len(tokenizer)
