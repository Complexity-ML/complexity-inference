"""
Mu Layers
=========

Common neural network layers with Mu dynamics integration.

All layers apply Mu-clamping for numerical stability.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mu_inference.core.mu import mu_clamp, soft_clamp
from mu_inference.core.config import MuConfig


class MuRMSNorm(nn.Module):
    """
    RMSNorm with Mu dynamics.

    Simpler than LayerNorm, no mean centering.
    Mu-clamping applied to output for stability.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        mu_config: Optional[MuConfig] = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.mu_config = mu_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        output = self.weight * x

        # Apply Mu clamping
        if self.mu_config is not None and self.mu_config.enabled:
            output = mu_clamp(
                output,
                self.mu_config.mu,
                self.mu_config.clamp_min,
                self.mu_config.clamp_max,
            )

        return output


class MuLayerNorm(nn.Module):
    """
    LayerNorm with Mu dynamics.

    Standard LayerNorm with Mu-clamping for stability.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        mu_config: Optional[MuConfig] = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        self.mu_config = mu_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)

        if self.mu_config is not None and self.mu_config.enabled:
            output = mu_clamp(
                output,
                self.mu_config.mu,
                self.mu_config.clamp_min,
                self.mu_config.clamp_max,
            )

        return output


class MuRotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) with Mu integration.

    Encodes position information through rotation matrices.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings, device or "cpu")

    def _set_cos_sin_cache(self, seq_len: int, device: str):
        """Pre-compute cos/sin embeddings."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos/sin for given positions.

        Args:
            x: Input tensor (for dtype)
            position_ids: Position indices [batch_size, seq_len]

        Returns:
            (cos, sin) each with shape [batch_size, seq_len, head_dim]
        """
        seq_len = position_ids.max() + 1

        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)

        cos = self.cos_cached[position_ids].to(x.dtype)
        sin = self.sin_cached[position_ids].to(x.dtype)

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to Q and K.

    Args:
        q: Query tensor [batch, seq, heads, head_dim]
        k: Key tensor [batch, seq, heads, head_dim]
        cos: Cosine embeddings [batch, seq, head_dim]
        sin: Sine embeddings [batch, seq, head_dim]

    Returns:
        (rotated_q, rotated_k)
    """
    # Add head dimension to cos/sin
    cos = cos.unsqueeze(2)  # [batch, seq, 1, head_dim]
    sin = sin.unsqueeze(2)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class MuMLP(nn.Module):
    """
    Standard MLP with Mu dynamics.

    Used by Llama, Mistral, etc.
    SiLU activation (SwiGLU variant).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mu_config: Optional[MuConfig] = None,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.mu_config = mu_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(silu(gate(x)) * up(x))
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        output = self.down_proj(gate * up)

        if self.mu_config is not None and self.mu_config.enabled:
            output = mu_clamp(
                output,
                self.mu_config.mu,
                self.mu_config.clamp_min,
                self.mu_config.clamp_max,
            )

        return output


class MuTokenRoutedMLP(nn.Module):
    """
    Token-Routed MLP with Mu dynamics.

    Deterministic routing: expert_id = token_id % num_experts
    Used by Pacific Prime / DeepForCausalLM.

    This is NOT MoE - it's deterministic and doesn't need load balancing.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 4,
        mu_config: Optional[MuConfig] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.mu_config = mu_config

        # Per-expert weights
        self.gate_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for param in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward with token-based routing.

        Args:
            hidden_states: [batch, seq, hidden]
            token_ids: [batch, seq] - token IDs for routing

        Returns:
            output: [batch, seq, hidden]
        """
        batch_size, seq_len, _ = hidden_states.shape

        if token_ids is None:
            # Fallback: use expert 0
            token_ids = torch.zeros(
                batch_size, seq_len,
                dtype=torch.long, device=hidden_states.device
            )

        # Route tokens to experts
        expert_ids = token_ids % self.num_experts

        # Process each token
        output = torch.zeros_like(hidden_states)

        for expert_idx in range(self.num_experts):
            # Find tokens for this expert
            mask = expert_ids == expert_idx

            if not mask.any():
                continue

            # Get hidden states for this expert
            expert_hidden = hidden_states[mask]  # [num_tokens, hidden]

            # Expert computation (SwiGLU)
            gate = F.silu(expert_hidden @ self.gate_proj[expert_idx])
            up = expert_hidden @ self.up_proj[expert_idx]
            expert_output = (gate * up) @ self.down_proj[expert_idx]

            output[mask] = expert_output

        # Apply Mu clamping
        if self.mu_config is not None and self.mu_config.enabled:
            output = mu_clamp(
                output,
                self.mu_config.mu,
                self.mu_config.clamp_min,
                self.mu_config.clamp_max,
            )

        return output


class MuAttention(nn.Module):
    """
    Multi-Head Attention with Mu dynamics.

    Features:
    - Grouped Query Attention (GQA) support
    - Optional QK normalization
    - RoPE integration
    - Mu-clamping on attention output
    - Uses PyTorch's scaled_dot_product_attention (FlashAttention when available)
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 2048,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
        mu_config: Optional[MuConfig] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.qk_norm = qk_norm
        self.mu_config = mu_config

        # Number of Q heads per KV head (for GQA)
        self.num_groups = self.num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        # QK normalization (optional)
        if qk_norm:
            self.q_norm = MuRMSNorm(self.head_dim, mu_config=mu_config)
            self.k_norm = MuRMSNorm(self.head_dim, mu_config=mu_config)

        # RoPE
        self.rotary_emb = MuRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_base,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq, hidden]
            position_ids: [batch, seq]
            attention_mask: [batch, 1, seq, seq] or None
            past_key_value: Cached (K, V) or None
            use_cache: Whether to return updated cache

        Returns:
            (output, updated_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: [batch, seq, heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK normalization
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE
        cos, sin = self.rotary_emb(q, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        new_cache = (k, v) if use_cache else None

        # Expand KV for GQA
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=2)
            v = v.repeat_interleave(self.num_groups, dim=2)

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention (uses FlashAttention when available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None and seq_len > 1,
        )

        # Reshape back: [batch, seq, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.num_heads * self.head_dim)

        # Output projection
        output = self.o_proj(attn_output)

        # Apply Mu clamping
        if self.mu_config is not None and self.mu_config.enabled:
            output = mu_clamp(
                output,
                self.mu_config.mu,
                self.mu_config.clamp_min,
                self.mu_config.clamp_max,
            )

        return output, new_cache
