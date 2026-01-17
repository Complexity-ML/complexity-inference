"""
Mu Model Loader
===============

Load pretrained models into Mu-compatible format.

Supports loading from:
- Local paths
- HuggingFace Hub
- Safetensors and PyTorch formats
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

import torch
import torch.nn as nn

from mu_inference.core.config import MuConfig, ModelConfig, ModelType
from mu_inference.models.base import MuModelBase
from mu_inference.models.registry import (
    MODEL_REGISTRY,
    detect_architecture,
    get_model_class,
)

logger = logging.getLogger(__name__)


def load_config(model_path: str) -> Dict[str, Any]:
    """
    Load model config from path.

    Args:
        model_path: Local path or HuggingFace model ID

    Returns:
        Config dictionary
    """
    # Check if local path
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        raise FileNotFoundError(f"No config.json found in {model_path}")

    # Try HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=model_path, filename="config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load config from {model_path}: {e}")


def config_from_dict(config_dict: Dict[str, Any]) -> ModelConfig:
    """
    Create ModelConfig from config dictionary.

    Handles various HuggingFace config formats.
    """
    # Detect model type
    arch = detect_architecture(config_dict)
    model_type = ModelType.AUTO
    if arch:
        model_type = ModelType[arch.upper()] if arch.upper() in ModelType.__members__ else ModelType.AUTO

    return ModelConfig(
        model_type=model_type,
        vocab_size=config_dict.get("vocab_size", 32000),
        hidden_size=config_dict.get("hidden_size", 2048),
        intermediate_size=config_dict.get("intermediate_size", 5632),
        num_hidden_layers=config_dict.get("num_hidden_layers", 24),
        num_attention_heads=config_dict.get("num_attention_heads", 16),
        num_key_value_heads=config_dict.get("num_key_value_heads", config_dict.get("num_attention_heads", 16)),
        head_dim=config_dict.get("head_dim", config_dict.get("hidden_size", 2048) // config_dict.get("num_attention_heads", 16)),
        max_position_embeddings=config_dict.get("max_position_embeddings", 2048),
        rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
        rope_theta=config_dict.get("rope_theta", 10000.0),
        num_experts=config_dict.get("num_experts", config_dict.get("num_local_experts")),
        qk_norm=config_dict.get("qk_norm", False),
        tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
    )


def load_weights(
    model: nn.Module,
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> None:
    """
    Load weights into model.

    Supports safetensors and PyTorch formats.
    """
    weight_files = []

    # Check if local path
    if os.path.isdir(model_path):
        # Look for safetensors first
        safetensor_files = list(Path(model_path).glob("*.safetensors"))
        if safetensor_files:
            weight_files = [str(f) for f in safetensor_files]
        else:
            # Fall back to PyTorch
            pt_files = list(Path(model_path).glob("*.bin"))
            pt_files.extend(Path(model_path).glob("*.pt"))
            weight_files = [str(f) for f in pt_files]
    else:
        # HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            files = list_repo_files(model_path)

            # Prefer safetensors
            safetensor_files = [f for f in files if f.endswith(".safetensors")]
            if safetensor_files:
                weight_files = [
                    hf_hub_download(repo_id=model_path, filename=f)
                    for f in safetensor_files
                ]
            else:
                pt_files = [f for f in files if f.endswith(".bin") or f.endswith(".pt")]
                weight_files = [
                    hf_hub_download(repo_id=model_path, filename=f)
                    for f in pt_files
                ]
        except Exception as e:
            raise ValueError(f"Failed to download weights from {model_path}: {e}")

    if not weight_files:
        raise FileNotFoundError(f"No weight files found for {model_path}")

    # Load weights
    state_dict = {}

    for weight_file in weight_files:
        logger.info(f"Loading weights from {weight_file}")

        if weight_file.endswith(".safetensors"):
            from safetensors.torch import load_file
            weights = load_file(weight_file, device="cpu")
        else:
            weights = torch.load(weight_file, map_location="cpu", weights_only=True)

        state_dict.update(weights)

    # Map weight names to our format
    mapped_state_dict = map_weight_names(state_dict, model)

    # Load into model
    missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)

    if missing:
        logger.warning(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # Move to device and dtype
    model.to(device=device, dtype=dtype)


def map_weight_names(
    state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Map HuggingFace weight names to our format.

    Handles various naming conventions.
    """
    mapped = {}
    model_keys = set(n for n, _ in model.named_parameters())

    # Common prefix mappings
    prefix_maps = [
        ("model.", ""),  # Remove 'model.' prefix
        ("transformer.", ""),
        ("decoder.", ""),
    ]

    # Layer name mappings
    name_maps = {
        # Embeddings
        "embed_tokens.weight": "embed_tokens.weight",
        "wte.weight": "embed_tokens.weight",

        # LM head
        "lm_head.weight": "lm_head.weight",

        # Final norm
        "model.norm.weight": "norm.weight",
        "norm.weight": "norm.weight",
        "ln_f.weight": "norm.weight",

        # Attention
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",

        # Layer norms
        "input_layernorm": "input_layernorm",
        "post_attention_layernorm": "post_attention_layernorm",
        "ln_1": "input_layernorm",
        "ln_2": "post_attention_layernorm",

        # MLP
        "gate_proj": "mlp.gate_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
    }

    for key, tensor in state_dict.items():
        new_key = key

        # Apply prefix mappings
        for old_prefix, new_prefix in prefix_maps:
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]

        # Handle layer patterns
        # e.g., layers.0.self_attn.q_proj.weight
        parts = new_key.split(".")
        for i, part in enumerate(parts):
            if part in name_maps:
                parts[i] = name_maps[part]

        new_key = ".".join(parts)

        # Check if key exists in model
        if new_key in model_keys:
            mapped[new_key] = tensor
        elif key in model_keys:
            mapped[key] = tensor
        else:
            # Try original key
            mapped[key] = tensor

    return mapped


def load_model(
    model_path: str,
    model_type: Optional[str] = None,
    config: Optional[ModelConfig] = None,
    mu_config: Optional[MuConfig] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    **kwargs,
) -> MuModelBase:
    """
    Load a pretrained model.

    Args:
        model_path: Local path or HuggingFace model ID
        model_type: Model type (auto-detected if None)
        config: ModelConfig (loaded from model if None)
        mu_config: Mu dynamics config
        device: Device to load on
        dtype: Data type

    Returns:
        Loaded MuModelBase instance
    """
    logger.info(f"Loading model from {model_path}")

    # Load config
    config_dict = load_config(model_path)

    # Create model config
    if config is None:
        config = config_from_dict(config_dict)

    # Auto-detect model type
    if model_type is None:
        model_type = detect_architecture(config_dict)
        if model_type is None:
            raise ValueError(
                f"Could not detect model architecture. "
                f"Please specify model_type explicitly."
            )

    logger.info(f"Detected architecture: {model_type}")

    # Get model class
    model_class = get_model_class(model_type)
    if model_class is None:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported: {list(MODEL_REGISTRY.keys())}"
        )

    # Create Mu config
    if mu_config is None:
        mu_config = MuConfig()

    # Instantiate model
    logger.info(f"Creating {model_class.__name__}")
    model = model_class(config=config, mu_config=mu_config)

    # Load weights
    load_weights(model, model_path, device=device, dtype=dtype)

    logger.info(f"Model loaded: {model.num_parameters() / 1e6:.1f}M parameters")

    return model


def get_model_class(model_type: str):
    """Get model class from registry."""
    # Import models to register them
    from mu_inference.models import deep, llama

    model_type = model_type.lower()
    return MODEL_REGISTRY.get(model_type)
