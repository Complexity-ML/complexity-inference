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


class INLDynamics(nn.Module):
    """
    INL Dynamics - Robotics-grade control with velocity tracking.

    This is the key component that makes Pacific Prime unique.
    Creates smooth, stable trajectories like a robot controller.

    Equations:
        error = h - mu(h)                   # deviation from contextual equilibrium
        v_next = alpha * v - beta * error   # velocity update
        h_next = h + dt * gate * v_next     # position update

    Parameters:
        - mu: learnable equilibrium target
        - controller_in/out: MLP that computes alpha, beta, gate from context
        - mu_proj: context-dependent equilibrium adjustment
    """

    def __init__(
        self,
        hidden_size: int,
        controller_hidden: int = 64,
        dt: float = 0.1,
        init_alpha: float = 0.9,
        init_beta: float = 0.1,
        init_gate: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.controller_hidden = controller_hidden

        # Learnable equilibrium (target position)
        self.mu = nn.Parameter(torch.zeros(hidden_size))

        # Contextual mu projection - adapts equilibrium per token
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)

        # Controller MLP - computes alpha, beta, gate from context
        # Input: [h, v] concatenated (2H -> controller_hidden -> 3H)
        self.controller_in = nn.Linear(hidden_size * 2, controller_hidden)
        self.controller_out = nn.Linear(controller_hidden, hidden_size * 3)

        # Initialize controller biases from config values
        with torch.no_grad():
            bias = self.controller_out.bias
            # alpha in [0,1] via sigmoid - inverse sigmoid to get bias
            alpha_bias = math.log(init_alpha / (1.0 - init_alpha + 1e-8))
            bias[:hidden_size].fill_(alpha_bias)
            # beta in [0,inf) via softplus - inverse softplus to get bias
            beta_bias = math.log(math.exp(init_beta) - 1.0 + 1e-8)
            bias[hidden_size:hidden_size*2].fill_(beta_bias)
            # gate in [0,1] via sigmoid - inverse sigmoid to get bias
            gate_bias = math.log(init_gate / (1.0 - init_gate + 1e-8))
            bias[hidden_size*2:].fill_(gate_bias)

            # Small weights for stable start
            self.controller_out.weight.normal_(0, 0.01)

    def forward(
        self,
        h: torch.Tensor,
        v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply dynamics update.

        Args:
            h: Hidden states [batch, seq_len, hidden_size]
            v: Velocity states [batch, seq_len, hidden_size] (None = init to zero)

        Returns:
            h_next: Updated hidden states
            v_next: Updated velocity states
            mu_contextual: Contextual equilibrium
        """
        # Initialize velocity if not provided
        if v is None:
            v = torch.zeros_like(h)

        # Controller: [h, v] -> alpha, beta, gate
        hv = torch.cat([h, v], dim=-1)  # [B, S, 2H]
        ctrl_hidden = F.silu(self.controller_in(hv))  # [B, S, ctrl_hidden]
        controller_out = self.controller_out(ctrl_hidden)  # [B, S, 3H]

        # Split and apply activations
        alpha_raw, beta_raw, gate_raw = torch.split(
            controller_out, self.hidden_size, dim=-1
        )
        alpha = torch.sigmoid(alpha_raw)      # [0, 1] - inertia
        beta = torch.clamp(F.softplus(beta_raw), max=2.0)  # [0, 2] - correction
        gate = torch.sigmoid(gate_raw)        # [0, 1] - amplitude

        # Contextual mu: base equilibrium + context adjustment
        mu_contextual = self.mu + self.mu_proj(h)

        # Dynamics equations
        error = h - mu_contextual
        v_next = alpha * v - beta * error

        # Clamp velocity for stability
        v_next = torch.clamp(v_next, min=-10.0, max=10.0)

        h_next = h + self.dt * gate * v_next

        return h_next, v_next, mu_contextual


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

    Uses fused gate_up_proj for efficiency (matches HuggingFace format).
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
        # intermediate_size is total, divide by num_experts for per-expert size
        self.expert_intermediate_size = intermediate_size // num_experts
        self.mu_config = mu_config

        # Fused gate and up projection: [num_experts, hidden_size, 2 * expert_intermediate]
        # First half is gate, second half is up
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, 2 * self.expert_intermediate_size)
        )
        # Down projection: [num_experts, expert_intermediate, hidden_size]
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, self.expert_intermediate_size, hidden_size)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_uniform_(self.gate_up_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))

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

            # Fused gate_up projection
            gate_up = expert_hidden @ self.gate_up_proj[expert_idx]  # [num_tokens, 2 * intermediate]

            # Split into gate and up
            gate, up = gate_up.chunk(2, dim=-1)

            # SwiGLU: silu(gate) * up
            expert_output = (F.silu(gate) * up) @ self.down_proj[expert_idx]

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
    Multi-Head Attention with Mu dynamics and Mu-guided attention.

    Features:
    - Grouped Query Attention (GQA) support
    - QK normalization (using nn.RMSNorm for checkpoint compatibility)
    - RoPE integration
    - Mu-guided attention (mu_to_q, mu_to_k, mu_to_v projections)
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
        self.use_qk_norm = qk_norm
        self.mu_config = mu_config

        # Number of Q heads per KV head (for GQA)
        self.num_groups = self.num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

        # Mu-guided attention projections (INL 2025)
        # mu from previous layer biases Q, K, and V
        self.mu_to_q = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.mu_to_k = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.mu_to_v = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)

        # QK normalization (using nn.RMSNorm for checkpoint compatibility)
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

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
        mu_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with Mu-guided attention.

        Args:
            hidden_states: [batch, seq, hidden]
            position_ids: [batch, seq]
            attention_mask: [batch, 1, seq, seq] or None
            past_key_value: Cached (K, V) or None
            use_cache: Whether to return updated cache
            mu_prev: Mu from previous layer (guides Q, K, V)

        Returns:
            (output, updated_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Mu-guided attention: mu from previous layer biases Q, K, V
        if mu_prev is not None:
            q = q + self.mu_to_q(mu_prev)
            k = k + self.mu_to_k(mu_prev)
            v = v + self.mu_to_v(mu_prev)

        # Reshape: [batch, seq, heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK normalization
        if self.use_qk_norm:
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
