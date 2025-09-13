"""Visualization utilities for FFTNet."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch

__all__ = ["plot_embedding_spectrum"]


def plot_embedding_spectrum(
    embeddings: torch.Tensor, save_path: Optional[str | Path] = None
) -> None:
    """Plot average frequency magnitude across tokens in a sequence.

    Parameters
    ----------
    embeddings:
        Tensor of shape ``(batch, seq_len, dim)`` representing token embeddings.
    save_path:
        If provided, save the plot to this path instead of displaying it.
    """
    if embeddings.ndim != 3:
        raise ValueError("embeddings must be 3D [batch, seq_len, dim]")

    with torch.no_grad():
        freq = torch.fft.fft(embeddings, dim=1)
        magnitude = freq.abs().mean(dim=(0, 2))

    plt.figure()
    plt.plot(torch.arange(magnitude.size(0)), magnitude.cpu())
    plt.xlabel("Frequency index")
    plt.ylabel("Magnitude")
    plt.title("Average Frequency Spectrum")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
