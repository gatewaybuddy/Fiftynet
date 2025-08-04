"""Benchmark throughput and memory usage of the full FFTNet model."""

import os
import sys
import torch

# Ensure repository root is on the path for direct execution
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import FFTNet
from benchmarks.benchmark_utils import benchmark_module


def main():
    vocab_size = 5000
    dim = 64
    num_blocks = 2
    seq_len = 128
    batch_size = 4
    model = FFTNet(vocab_size, dim, num_blocks)
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    throughput, mem_mb = benchmark_module(model, dummy_input)
    print(f"FFTNet: {throughput:.2f} it/s, peak memory {mem_mb:.2f} MB")


if __name__ == "__main__":
    main()
