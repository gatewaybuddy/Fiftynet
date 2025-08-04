"""Benchmark throughput and memory usage of FFTNetBlock."""

import os
import sys
import torch

# Ensure repository root is on the path for direct execution
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fftnet_block import FFTNetBlock
from benchmarks.benchmark_utils import benchmark_module


def main():
    dim = 64
    seq_len = 128
    batch_size = 4
    block = FFTNetBlock(dim)
    dummy_input = torch.randn(batch_size, seq_len, dim)
    throughput, mem_mb = benchmark_module(block, dummy_input)
    print(f"FFTNetBlock: {throughput:.2f} it/s, peak memory {mem_mb:.2f} MB")


if __name__ == "__main__":
    main()
