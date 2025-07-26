import torch
import matplotlib.pyplot as plt


def plot_embedding_spectrum(embeddings: torch.Tensor) -> None:
    """Plot average frequency magnitude across tokens in a sequence."""
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
    plt.show()
    plt.close()
