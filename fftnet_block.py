import torch
from torch import nn
from complex_rope import ComplexRoPE
from neural_fourier_operator import NeuralFourierOperator


class FFTNetBlock(nn.Module):
    """FFT-based block with complex RoPE, Fourier filtering and MLP."""

    def __init__(self, dim: int, mlp_hidden_dim: int | None = None, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        self.dim = dim
        self.rope = ComplexRoPE(dim, base)
        complex_dim = dim // 2
        self.filter = NeuralFourierOperator(complex_dim)
        hidden = mlp_hidden_dim or dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {x.size(-1)}")
        residual = x
        x_complex = self.rope(x)
        x_freq = torch.fft.fft(x_complex, dim=1)
        x_filtered = self.filter(x_freq)
        x_time = torch.fft.ifft(x_filtered, dim=1)
        b, seq_len, _ = x_time.shape
        x_real = torch.view_as_real(x_time).reshape(b, seq_len, self.dim)
        out = self.mlp(x_real)
        return out + residual
