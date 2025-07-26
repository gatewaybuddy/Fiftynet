import torch
from torch import nn

class ComplexRoPE(nn.Module):
    """Complex rotary position embedding with log-scaled frequencies."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        self.dim = dim
        self.base = base
        inv_freq = base ** (-2 * torch.arange(dim // 2, dtype=torch.float32) / dim)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {x.size(-1)}")
        b, seq_len, _ = x.shape
        x_complex = torch.view_as_complex(x.contiguous().view(b, seq_len, self.dim // 2, 2))
        device = x.device
        inv_freq = self.inv_freq.to(device)
        positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        theta = torch.outer(positions, inv_freq)
        rot = torch.polar(torch.ones_like(theta), theta)
        rot = rot.unsqueeze(0)
        return x_complex * rot
