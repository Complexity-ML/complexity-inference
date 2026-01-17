"""
DeepForCausalLM Model
=====================

Mu-compatible implementation of DeepForCausalLM (Pacific Prime).

Key features:
- Token-Routed MLP (deterministic routing: expert_id = token_id % num_experts)
- Grouped Query Attention (GQA)
- QK Normalization
- RoPE embeddings
- Mu dynamics throughout
"""

from typing import Optional, List, Tuple, Dict, Any
import logging

import torch
import torch.nn as nn

from mu_inference.core.config import MuConfig, ModelConfig
from mu_inference.models.base import MuModelBase, MuModelOutput
from mu_inference.models.registry import register_model
from mu_inference.models.layers import (
    MuRMSNorm,
    MuAttention,
    MuTokenRoutedMLP,
    MuMLP,
    INLDynamics,
)
from mu_inference.core.mu import mu_clamp

logger = logging.getLogger(__name__)


class DeepDecoderLayer(nn.Module):
    """
    Single decoder layer for DeepForCausalLM.

    Architecture:
    - Pre-norm with RMSNorm
    - Attention with GQA and QK norm
    - Token-Routed MLP (or standard MLP for some layers)
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        mu_config: Optional[MuConfig] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.mu_config = mu_config

        # Input layernorm
        self.input_layernorm = MuRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            mu_config=mu_config,
        )

        # Attention
        self.self_attn = MuAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_base=config.rope_theta,
            qk_norm=getattr(config, 'use_qk_norm', False),
            mu_config=mu_config,
        )

        # Post-attention layernorm
        self.post_attention_layernorm = MuRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            mu_config=mu_config,
        )

        # INL Dynamics (robotics-grade control)
        dynamics_controller_hidden = getattr(config, 'dynamics_controller_hidden', 64)
        self.dynamics = INLDynamics(
            hidden_size=config.hidden_size,
            controller_hidden=dynamics_controller_hidden,
        )

        # MLP (Token-Routed or standard based on layer)
        # Use Token-Routed MLP if num_experts > 1
        if config.num_experts and config.num_experts > 1:
            self.mlp = MuTokenRoutedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                mu_config=mu_config,
            )
            self.use_token_routing = True
        else:
            self.mlp = MuMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mu_config=mu_config,
            )
            self.use_token_routing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        token_ids: Optional[torch.Tensor] = None,
        velocity_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Forward pass through the layer.

        Args:
            hidden_states: [batch, seq, hidden]
            position_ids: [batch, seq]
            attention_mask: Optional attention mask
            past_key_value: Cached KV for this layer
            use_cache: Whether to return updated cache
            token_ids: Token IDs for routing (if using Token-Routed MLP)
            velocity_states: Velocity for INL dynamics

        Returns:
            (output_hidden_states, updated_cache, velocity_states)
        """
        residual = hidden_states

        # Pre-norm + Attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, new_cache = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # INL Dynamics (after attention, before residual)
        hidden_states, velocity_states, _ = self.dynamics(attn_output, velocity_states)

        hidden_states = residual + hidden_states

        # Pre-norm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.use_token_routing:
            hidden_states = self.mlp(hidden_states, token_ids=token_ids)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        # Apply Mu clamping to residual output
        if self.mu_config is not None and self.mu_config.enabled:
            hidden_states = mu_clamp(
                hidden_states,
                self.mu_config.mu,
                self.mu_config.clamp_min,
                self.mu_config.clamp_max,
            )

        return hidden_states, new_cache, velocity_states


@register_model("deep")
class DeepForCausalLM(MuModelBase):
    """
    DeepForCausalLM with Mu dynamics.

    Pacific Prime / Complexity Deep model architecture.
    """

    def __init__(
        self,
        config: ModelConfig,
        mu_config: Optional[MuConfig] = None,
    ):
        super().__init__(config, mu_config)

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            DeepDecoderLayer(config, layer_idx, mu_config)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = MuRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            mu_config=mu_config,
        )

        # LM head (tied to embeddings by default)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._is_initialized = True

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
    ) -> Tuple[MuModelOutput, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq]
            position_ids: [batch, seq] or None
            attention_mask: [batch, seq] or None
            past_key_values: List of (K, V) per layer
            use_cache: Whether to return updated cache

        Returns:
            (MuModelOutput, updated_past_key_values)
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Apply Mu clamping to embeddings
        if self.mu_config.enabled:
            hidden_states = mu_clamp(
                hidden_states,
                self.mu_config.mu,
                self.mu_config.clamp_min,
                self.mu_config.clamp_max,
            )

        # Create position IDs if not provided
        if position_ids is None:
            past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Create causal mask if needed
        if attention_mask is not None:
            # Convert from [batch, seq] to [batch, 1, seq, full_seq]
            full_seq_len = seq_len
            if past_key_values is not None:
                full_seq_len += past_key_values[0][0].shape[2]

            causal_mask = self._make_causal_mask(
                seq_len, full_seq_len, hidden_states.dtype, hidden_states.device
            )
            # Apply padding mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = causal_mask + (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        else:
            attention_mask = None  # Let SDPA handle causal

        # Process through layers
        new_key_values = [] if use_cache else None
        velocity_states = None  # Track velocity across layers

        for layer_idx, layer in enumerate(self.layers):
            past_kv = past_key_values[layer_idx] if past_key_values else None

            hidden_states, new_kv, velocity_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
                token_ids=input_ids,  # For Token-Routed MLP
                velocity_states=velocity_states,
            )

            if use_cache:
                new_key_values.append(new_kv)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM head (only last token for generation)
        logits = self.lm_head(hidden_states[:, -1, :])

        # Collect Mu stats
        mu_stats = self.get_mu_stats() if self.mu_dynamics else None

        output = MuModelOutput(
            logits=logits,
            hidden_states=hidden_states if getattr(self.config, 'output_hidden_states', False) else None,
            mu_stats=mu_stats,
        )

        return output, new_key_values

    def _make_causal_mask(
        self,
        query_len: int,
        key_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.full((query_len, key_len), float("-inf"), dtype=dtype, device=device),
            diagonal=key_len - query_len + 1,
        )
        return mask.unsqueeze(0).unsqueeze(0)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[ModelConfig] = None,
        mu_config: Optional[MuConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> "DeepForCausalLM":
        """Load from pretrained weights."""
        from mu_inference.models.loader import load_model
        return load_model(
            model_path,
            model_type="deep",
            config=config,
            mu_config=mu_config,
            device=device,
            dtype=dtype,
            **kwargs,
        )
