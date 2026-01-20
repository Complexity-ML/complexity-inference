"""
Mu Sampling
===========

Token sampling strategies for text generation.

Supports:
- Temperature scaling
- Top-k sampling
- Top-p (nucleus) sampling
- Repetition penalty
"""

from typing import Optional, List

import torch
import torch.nn.functional as F


def prepare_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    past_tokens: Optional[List[int]] = None,
    repetition_window: int = 0,
) -> torch.Tensor:
    """
    Prepare logits for sampling.

    Applies temperature, top-k, top-p, and repetition penalty.

    Args:
        logits: Raw logits [batch, vocab_size]
        temperature: Temperature for scaling (>1 = more random)
        top_k: Keep only top k tokens (0 = disabled)
        top_p: Keep tokens with cumulative prob <= top_p
        repetition_penalty: Penalize repeated tokens (>1 = less repetition)
        past_tokens: List of previously generated tokens
        repetition_window: Only consider last N tokens (0 = all)

    Returns:
        Processed logits
    """
    logits = logits.clone()

    # Apply repetition penalty
    if repetition_penalty != 1.0 and past_tokens:
        # Apply window if specified
        tokens_to_check = past_tokens
        if repetition_window > 0:
            tokens_to_check = past_tokens[-repetition_window:]

        for token_id in set(tokens_to_check):
            if token_id < logits.shape[-1]:
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

    # Apply temperature
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-8)

    # Apply top-k
    if top_k > 0 and top_k < logits.shape[-1]:
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits = torch.where(
            logits < min_top_k,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    # Apply top-p (nucleus sampling)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift to keep first token above threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Scatter back to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


def sample_token(
    logits: torch.Tensor,
    do_sample: bool = True,
) -> int:
    """
    Sample a token from logits.

    Args:
        logits: Processed logits [batch, vocab_size]
        do_sample: If True, sample; if False, use argmax

    Returns:
        Sampled token ID
    """
    if do_sample:
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Handle edge cases
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure at least one valid probability
        if probs.sum() == 0:
            probs = torch.ones_like(probs) / probs.shape[-1]

        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()
    else:
        # Greedy decoding
        return logits.argmax(dim=-1).item()


def sample_tokens_batch(
    logits: torch.Tensor,
    do_sample: bool = True,
) -> torch.Tensor:
    """
    Sample tokens for a batch.

    Args:
        logits: Processed logits [batch, vocab_size]
        do_sample: If True, sample; if False, use argmax

    Returns:
        Token IDs [batch]
    """
    if do_sample:
        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        return logits.argmax(dim=-1)


def apply_mu_to_logits(
    logits: torch.Tensor,
    mu: float = 0.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Apply Mu centering to logits.

    Shifts logits toward equilibrium before sampling.

    Args:
        logits: Raw logits
        mu: Equilibrium point
        scale: Scaling factor

    Returns:
        Mu-adjusted logits
    """
    # Center around mu
    centered = logits - mu

    # Scale
    scaled = centered * scale

    return scaled + mu
