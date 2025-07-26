import torch
from torch import nn

class NeuralFourierOperator(nn.Module):
    """Apply a learnable complex-valued filter in the frequency domain."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        param = torch.randn(dim, dtype=torch.cfloat)
        self.filter = nn.Parameter(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("input must be complex-valued")
        if x.size(-1) != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {x.size(-1)}")
        filt = self.filter
        if filt.device != x.device:
            filt = filt.to(x.device)
        return x * filt.view(1, 1, -1)
