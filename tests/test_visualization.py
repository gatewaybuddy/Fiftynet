import os
import sys
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from main import plot_embedding_spectrum


def test_plot_embedding_spectrum():
    embeddings = torch.randn(2, 8, 4)
    plot_embedding_spectrum(embeddings)
    plt.close("all")
