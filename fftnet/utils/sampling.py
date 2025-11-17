"""
Sampling strategies for text generation.

Provides various decoding methods beyond greedy sampling for better quality
and diversity in generated text.
"""

import torch
import torch.nn.functional as F


def sample_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample from logits with temperature scaling.

    Args:
        logits: Tensor of shape (batch_size, vocab_size) containing unnormalized log probabilities
        temperature: Sampling temperature. Higher values (>1) increase randomness,
                    lower values (<1) make sampling more deterministic.
                    temperature=1.0 is standard sampling, temperature→0 is greedy.

    Returns:
        Sampled token indices of shape (batch_size,)
    """
    if temperature <= 0:
        # Greedy sampling
        return logits.argmax(dim=-1)

    # Scale logits by temperature
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_top_k(
    logits: torch.Tensor, k: int, temperature: float = 1.0
) -> torch.Tensor:
    """
    Sample from top-k tokens only.

    Keeps only the k most likely tokens, redistributes probability mass among them,
    and samples from this restricted distribution.

    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        k: Number of top tokens to keep (k > 0)
        temperature: Temperature for scaling logits

    Returns:
        Sampled token indices of shape (batch_size,)
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    # Scale by temperature
    logits = logits / temperature

    # Get top-k values and indices
    top_k_values, top_k_indices = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1)

    # Set all non-top-k logits to -inf
    logits_filtered = torch.full_like(logits, float("-inf"))
    logits_filtered.scatter_(-1, top_k_indices, top_k_values)

    # Sample from filtered distribution
    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_nucleus(
    logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling.

    Samples from the smallest set of tokens whose cumulative probability
    exceeds the threshold p. This dynamically adjusts the number of tokens
    considered based on the probability distribution.

    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        p: Cumulative probability threshold (0 < p <= 1.0)
        temperature: Temperature for scaling logits

    Returns:
        Sampled token indices of shape (batch_size,)
    """
    if not (0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1], got {p}")

    # Scale by temperature
    logits = logits / temperature

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    # Keep at least one token
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Set filtered logits to -inf
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    # Map back to original indices
    logits_filtered = torch.full_like(logits, float("-inf"))
    logits_filtered.scatter_(-1, sorted_indices, sorted_logits)

    # Sample from filtered distribution
    probs = F.softmax(logits_filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_top_k_top_p(
    logits: torch.Tensor,
    k: int = 0,
    p: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Combined top-k and nucleus sampling.

    First applies top-k filtering, then applies nucleus (top-p) filtering
    to the remaining tokens. This provides fine-grained control over
    the sampling distribution.

    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        k: Number of top tokens to keep (0 to disable top-k)
        p: Cumulative probability threshold (1.0 to disable top-p)
        temperature: Temperature for scaling logits

    Returns:
        Sampled token indices of shape (batch_size,)
    """
    # Apply temperature
    logits = logits / temperature

    # Apply top-k filtering
    if k > 0:
        top_k_values, top_k_indices = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1)
        logits_filtered = torch.full_like(logits, float("-inf"))
        logits_filtered.scatter_(-1, top_k_indices, top_k_values)
        logits = logits_filtered

    # Apply nucleus filtering
    if p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_indices, sorted_logits)

    # Sample from filtered distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
