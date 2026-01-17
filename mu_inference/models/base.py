"""
Mu Model Base
=============

Abstract base class for all Mu-compatible models.

All model implementations must inherit from MuModelBase
and implement the required methods for engine integration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn

from mu_inference.core.mu import mu_clamp, MuDynamics
from mu_inference.core.config import MuConfig, ModelConfig


@dataclass
class MuModelOutput:
    """
    Standard output from Mu models.

    Consistent interface for the engine regardless of model architecture.
    """

    # Logits for next token prediction
    # Shape: [batch_size, vocab_size]
    logits: torch.Tensor

    # Hidden states (optional, for debugging/analysis)
    # Shape: [batch_size, seq_len, hidden_size]
    hidden_states: Optional[torch.Tensor] = None

    # Attention weights (optional, for visualization)
    # Dict[layer_idx, Tensor[batch, heads, seq, seq]]
    attentions: Optional[Dict[int, torch.Tensor]] = None

    # Router logits for MoE/Token-Routed MLP (optional)
    # Dict[layer_idx, Tensor[batch, seq, num_experts]]
    router_logits: Optional[Dict[int, torch.Tensor]] = None

    # Mu statistics (optional, for monitoring)
    mu_stats: Optional[Dict[str, float]] = None


class MuModelBase(ABC, nn.Module):
    """
    Abstract base class for Mu-compatible models.

    Provides common functionality and enforces interface
    required by the Mu inference engine.
    """

    def __init__(
        self,
        config: ModelConfig,
        mu_config: Optional[MuConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.mu_config = mu_config or MuConfig()

        # Mu dynamics tracker
        self.mu_dynamics = MuDynamics(self.mu_config) if self.mu_config.enabled else None

        # Model metadata
        self._is_initialized = False
        self._device = None
        self._dtype = None

    @property
    def device(self) -> torch.device:
        """Get model device."""
        if self._device is None:
            # Get from first parameter
            try:
                self._device = next(self.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        if self._dtype is None:
            try:
                self._dtype = next(self.parameters()).dtype
            except StopIteration:
                self._dtype = torch.float16
        return self._dtype

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
    ) -> Tuple[MuModelOutput, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            past_key_values: Cached KV pairs from previous forward
            use_cache: Whether to return updated KV cache

        Returns:
            Tuple of (MuModelOutput, updated_past_key_values)
        """
        pass

    @abstractmethod
    def get_input_embeddings(self) -> nn.Module:
        """Get the input embedding layer."""
        pass

    @abstractmethod
    def get_output_embeddings(self) -> nn.Module:
        """Get the output embedding/LM head layer."""
        pass

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation step.

        Handles KV cache logic - only pass last token if cache exists.
        """
        if past_key_values is not None:
            # Only need last token when using cache
            input_ids = input_ids[:, -1:]

        # Create position IDs
        batch_size, seq_len = input_ids.shape

        if past_key_values is not None:
            # Position = cache length
            past_len = past_key_values[0][0].shape[2]
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def apply_mu_clamping(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply Mu clamping to a tensor."""
        if self.mu_config.enabled:
            return mu_clamp(
                tensor,
                self.mu_config.mu,
                self.mu_config.clamp_min,
                self.mu_config.clamp_max,
            )
        return tensor

    def update_mu_stats(self, name: str, tensor: torch.Tensor):
        """Update Mu dynamics statistics."""
        if self.mu_dynamics is not None:
            self.mu_dynamics.update(name, tensor)

    def get_mu_stats(self) -> Dict[str, float]:
        """Get current Mu statistics."""
        if self.mu_dynamics is not None:
            return self.mu_dynamics.get_stats()
        return {}

    def reset_mu_stats(self):
        """Reset Mu statistics."""
        if self.mu_dynamics is not None:
            self.mu_dynamics.reset()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[ModelConfig] = None,
        mu_config: Optional[MuConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> "MuModelBase":
        """
        Load a pretrained model.

        This is implemented by each model class.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement from_pretrained. "
            "Use the load_model() function from mu_inference.models instead."
        )

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        params = self.parameters()
        if trainable_only:
            params = filter(lambda p: p.requires_grad, params)
        return sum(p.numel() for p in params)

    def get_memory_footprint(self) -> Dict[str, float]:
        """
        Get memory footprint in MB.

        Returns:
            Dict with 'parameters_mb' and 'buffers_mb'
        """
        param_bytes = sum(
            p.numel() * p.element_size() for p in self.parameters()
        )
        buffer_bytes = sum(
            b.numel() * b.element_size() for b in self.buffers()
        )
        return {
            "parameters_mb": param_bytes / (1024 * 1024),
            "buffers_mb": buffer_bytes / (1024 * 1024),
            "total_mb": (param_bytes + buffer_bytes) / (1024 * 1024),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  hidden_size={self.config.hidden_size},\n"
            f"  num_layers={self.config.num_hidden_layers},\n"
            f"  num_heads={self.config.num_attention_heads},\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  mu_enabled={self.mu_config.enabled},\n"
            f"  parameters={self.num_parameters() / 1e6:.1f}M\n"
            f")"
        )
