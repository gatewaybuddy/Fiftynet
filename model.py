import torch
from torch import nn

from fftnet_block import FFTNetBlock


class FFTNet(nn.Module):
    """Stack of FFTNetBlocks with token embeddings and output projection."""

    def __init__(self, vocab_size: int, dim: int, num_blocks: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([FFTNetBlock(dim, base=base) for _ in range(num_blocks)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
