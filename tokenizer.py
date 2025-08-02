from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


@dataclass
class SimpleTokenizer:
    """Byte-level BPE tokenizer with minimal persistence helpers.

    This wrapper is intentionally lightweight.  It exposes the methods needed
    by the training scripts while keeping the underlying HuggingFace tokenizer
    configurable for future extensions.
    """

    tokenizer: Tokenizer

    @classmethod
    def train_from_iterator(
        cls,
        iterator: Iterable[str],
        vocab_size: int = 5000,
        min_frequency: int = 2,
        special_tokens: Sequence[str] = ("<pad>", "<unk>", "<bos>", "<eos>"),
    ) -> "SimpleTokenizer":
        """Train a byte-level BPE tokenizer from an iterable of texts.

        Args:
            iterator: Iterable yielding strings for training.
            vocab_size: Target vocabulary size.
            min_frequency: Minimum frequency for merges.
            special_tokens: Tokens always kept in the vocabulary.
        """

        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(special_tokens),
        )
        tok.train_from_iterator(iterator, trainer=trainer)
        tok.post_processor = processors.ByteLevel(trim_offsets=False)
        tok.decoder = decoders.ByteLevel()
        return cls(tok)

    def encode(self, text: str) -> List[int]:
        """Encode a string into token ids."""

        return self.tokenizer.encode(text).ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token ids into a string.

        The underlying byte-level tokenizer includes a leading space token.
        Stripping the result provides a more conventional text round-trip.
        """

        return self.tokenizer.decode(ids).strip()

    def save(self, path: str) -> None:
        """Serialize the tokenizer to ``path``."""

        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """Load a previously saved tokenizer from ``path``."""

        return cls(Tokenizer.from_file(path))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.tokenizer.get_vocab_size()
