"""Tests for sampling utilities."""

import torch
import pytest
from fftnet.utils.sampling import (
    sample_with_temperature,
    sample_top_k,
    sample_nucleus,
    sample_top_k_top_p,
)


def test_sample_with_temperature_greedy():
    """Test that temperature=0 gives greedy sampling."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    sampled = sample_with_temperature(logits, temperature=0.0)
    assert sampled.item() == 4  # Index of max logit (5.0)


def test_sample_with_temperature_deterministic():
    """Test that same seed gives same results with temperature."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

    torch.manual_seed(42)
    sampled1 = sample_with_temperature(logits, temperature=1.0)

    torch.manual_seed(42)
    sampled2 = sample_with_temperature(logits, temperature=1.0)

    assert sampled1.item() == sampled2.item()


def test_sample_with_temperature_high_temp_more_random():
    """Test that higher temperature increases entropy."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]).repeat(100, 1)

    torch.manual_seed(42)
    samples_low = sample_with_temperature(logits, temperature=0.1)

    torch.manual_seed(42)
    samples_high = sample_with_temperature(logits, temperature=2.0)

    # Lower temperature should have less variety
    unique_low = len(torch.unique(samples_low))
    unique_high = len(torch.unique(samples_high))

    assert unique_high >= unique_low


def test_sample_top_k_basic():
    """Test top-k sampling keeps only top k tokens."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

    # With k=1, should always pick the max
    torch.manual_seed(42)
    sampled = sample_top_k(logits, k=1, temperature=1.0)
    assert sampled.item() == 4  # Index of max logit


def test_sample_top_k_filters_tokens():
    """Test that top-k only samples from top k tokens."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]).repeat(100, 1)

    # Sample 100 times with k=2
    torch.manual_seed(42)
    samples = sample_top_k(logits, k=2, temperature=1.0)

    # Should only see indices 3 and 4 (top 2 logits)
    unique_samples = torch.unique(samples)
    assert len(unique_samples) <= 2
    assert all(idx in [3, 4] for idx in unique_samples.tolist())


def test_sample_top_k_invalid_k():
    """Test that invalid k raises error."""
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        sample_top_k(logits, k=0)

    with pytest.raises(ValueError):
        sample_top_k(logits, k=-1)


def test_sample_nucleus_basic():
    """Test nucleus sampling with p threshold."""
    # Create uniform logits
    logits = torch.ones(1, 10)

    torch.manual_seed(42)
    sampled = sample_nucleus(logits, p=0.5, temperature=1.0)

    # Should return a valid index
    assert 0 <= sampled.item() < 10


def test_sample_nucleus_high_p_includes_more():
    """Test that higher p includes more tokens."""
    # Create logits where one token dominates
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])

    # With very low p, should mostly pick the dominant token
    torch.manual_seed(42)
    samples_low = torch.stack([sample_nucleus(logits, p=0.1, temperature=1.0) for _ in range(100)])

    # With high p, should have more variety
    torch.manual_seed(42)
    samples_high = torch.stack([sample_nucleus(logits, p=0.99, temperature=1.0) for _ in range(100)])

    unique_low = len(torch.unique(samples_low))
    unique_high = len(torch.unique(samples_high))

    assert unique_high >= unique_low


def test_sample_nucleus_invalid_p():
    """Test that invalid p raises error."""
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError):
        sample_nucleus(logits, p=0.0)

    with pytest.raises(ValueError):
        sample_nucleus(logits, p=1.5)


def test_sample_top_k_top_p_combined():
    """Test combined top-k and top-p sampling."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

    torch.manual_seed(42)
    sampled = sample_top_k_top_p(logits, k=3, p=0.9, temperature=1.0)

    # Should return a valid index from top 3
    assert 0 <= sampled.item() < 5


def test_sample_top_k_top_p_only_topk():
    """Test that k=0 disables top-k filtering."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]).repeat(100, 1)

    torch.manual_seed(42)
    samples = sample_top_k_top_p(logits, k=0, p=0.5, temperature=1.0)

    # Should still work with only top-p
    assert len(samples) == 100


def test_sample_top_k_top_p_only_topp():
    """Test that p=1.0 disables top-p filtering."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]).repeat(100, 1)

    torch.manual_seed(42)
    samples = sample_top_k_top_p(logits, k=2, p=1.0, temperature=1.0)

    # Should only sample from top 2
    unique_samples = torch.unique(samples)
    assert len(unique_samples) <= 2


def test_sample_shapes():
    """Test that all sampling functions handle batch dimensions correctly."""
    batch_size = 4
    vocab_size = 100
    logits = torch.randn(batch_size, vocab_size)

    torch.manual_seed(42)
    samples_temp = sample_with_temperature(logits, temperature=1.0)
    assert samples_temp.shape == (batch_size,)

    torch.manual_seed(42)
    samples_topk = sample_top_k(logits, k=10, temperature=1.0)
    assert samples_topk.shape == (batch_size,)

    torch.manual_seed(42)
    samples_topp = sample_nucleus(logits, p=0.9, temperature=1.0)
    assert samples_topp.shape == (batch_size,)

    torch.manual_seed(42)
    samples_combined = sample_top_k_top_p(logits, k=10, p=0.9, temperature=1.0)
    assert samples_combined.shape == (batch_size,)


def test_sample_output_dtype():
    """Test that sampled tokens are long integers."""
    logits = torch.randn(2, 50)

    sampled = sample_with_temperature(logits, temperature=1.0)
    assert sampled.dtype == torch.int64

    sampled = sample_top_k(logits, k=10, temperature=1.0)
    assert sampled.dtype == torch.int64

    sampled = sample_nucleus(logits, p=0.9, temperature=1.0)
    assert sampled.dtype == torch.int64
