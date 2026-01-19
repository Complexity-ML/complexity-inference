"""
LlamaForCausalLM Model
======================

Mu-compatible implementation of LlamaForCausalLM.

Compatible with:
- Llama 1, 2, 3
- CodeLlama
- Any Llama-architecture model

Key features:
- Grouped Query Attention (GQA)
- RoPE embeddings
- SwiGLU MLP
- Mu dynamics throughout
"""

from typing import Optional, List, Tuple
import logging

import torch
import torch.nn as nn

from mu_inference.core.config import MuConfig, ModelConfig
from mu_inference.models.base import MuModelBase, MuModelOutput
from mu_inference.models.registry import register_model
from mu_inference.models.layers import (
    MuRMSNorm,
    MuAttention,
    MuMLP,
)
from mu_inference.core.mu import mu_clamp

logger = logging.getLogger(__name__)


class LlamaDecoderLayer(nn.Module):
    """
    Single decoder layer for Llama.

    Standard Llama architecture:
    - Pre-norm with RMSNorm
    - Attention with GQA
    - SwiGLU MLP
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        mu_config: Optional[MuConfig] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
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
            qk_norm=False,  # Llama doesn't use QK norm
            mu_config=mu_config,
        )

        # Post-attention layernorm
        self.post_attention_layernorm = MuRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            mu_config=mu_config,
        )

        # MLP (SwiGLU)
        self.mlp = MuMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            mu_config=mu_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the layer."""
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
        hidden_states = residual + attn_output

        # Pre-norm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Mu clamping
        if self.mu_config is not None and self.mu_config.enabled:
            hidden_states = mu_clamp(
                hidden_states,
                self.mu_config.mu,
                self.mu_config.clamp_min,
                self.mu_config.clamp_max,
            )

        return hidden_states, new_cache


@register_model("llama")
class LlamaForCausalLM(MuModelBase):
    """
    LlamaForCausalLM with Mu dynamics.

    Standard Llama architecture compatible with all Llama variants.
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
            LlamaDecoderLayer(config, layer_idx, mu_config)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = MuRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            mu_config=mu_config,
        )

        # LM head
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
        """Forward pass."""
        batch_size, seq_len = input_ids.shape

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Mu clamping on embeddings
        if self.mu_config.enabled:
            hidden_states = mu_clamp(
                hidden_states,
                self.mu_config.mu,
                self.mu_config.clamp_min,
                self.mu_config.clamp_max,
            )

        # Position IDs
        if position_ids is None:
            # KV cache shape is [batch, seq, heads, head_dim], so seq_len is dim=1
            past_len = 0 if past_key_values is None else past_key_values[0][0].shape[1]
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        # Process layers
        new_key_values = [] if use_cache else None

        for layer_idx, layer in enumerate(self.layers):
            past_kv = past_key_values[layer_idx] if past_key_values else None

            hidden_states, new_kv = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=None,  # Let SDPA handle causal
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            if use_cache:
                new_key_values.append(new_kv)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states[:, -1, :])

        output = MuModelOutput(
            logits=logits,
            hidden_states=hidden_states if getattr(self.config, 'output_hidden_states', False) else None,
            mu_stats=self.get_mu_stats() if self.mu_dynamics else None,
        )

        return output, new_key_values

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[ModelConfig] = None,
        mu_config: Optional[MuConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> "LlamaForCausalLM":
        """Load from pretrained weights."""
        from mu_inference.models.loader import load_model
        return load_model(
            model_path,
            model_type="llama",
            config=config,
            mu_config=mu_config,
            device=device,
            dtype=dtype,
            **kwargs,
        )


# Register Mistral as Llama variant (same architecture)
@register_model("mistral")
class MistralForCausalLM(LlamaForCausalLM):
    """
    MistralForCausalLM - same as Llama architecture.

    Mistral uses sliding window attention in the original,
    but we use full attention with Mu dynamics.
    """
    pass


@register_model("qwen")
class Qwen2ForCausalLM(LlamaForCausalLM):
    """
    Qwen2ForCausalLM - based on Llama architecture.

    Minor differences handled by config.
    """
    pass


@register_model("gemma")
class GemmaForCausalLM(LlamaForCausalLM):
    """
    GemmaForCausalLM - based on Llama architecture.

    Minor differences handled by config.
    """
    pass
