"""
Mu Model Registry
=================

Central registry for all supported model architectures.

Allows dynamic model loading based on config or auto-detection.
"""

from typing import Dict, Type, Optional, Callable
import logging

logger = logging.getLogger(__name__)


# Global model registry
MODEL_REGISTRY: Dict[str, Type] = {}

# Architecture aliases (HuggingFace names -> our names)
ARCHITECTURE_ALIASES = {
    # Deep / Pacific Prime
    "DeepForCausalLM": "deep",
    "DeepseekForCausalLM": "deep",
    "PacificPrimeForCausalLM": "deep",

    # Llama family
    "LlamaForCausalLM": "llama",
    "Llama2ForCausalLM": "llama",
    "Llama3ForCausalLM": "llama",
    "CodeLlamaForCausalLM": "llama",

    # Mistral family
    "MistralForCausalLM": "mistral",
    "Mixtral": "mistral",

    # Qwen family
    "Qwen2ForCausalLM": "qwen",
    "QwenForCausalLM": "qwen",

    # Phi family
    "PhiForCausalLM": "phi",
    "Phi3ForCausalLM": "phi",

    # Gemma family
    "GemmaForCausalLM": "gemma",
    "Gemma2ForCausalLM": "gemma",
}


def register_model(name: str) -> Callable:
    """
    Decorator to register a model class.

    Usage:
        @register_model("llama")
        class LlamaForCausalLM(MuModelBase):
            ...
    """
    def decorator(cls: Type) -> Type:
        MODEL_REGISTRY[name.lower()] = cls
        logger.debug(f"Registered model: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_model_class(model_type: str) -> Optional[Type]:
    """
    Get model class by type name.

    Args:
        model_type: Model type (e.g., "llama", "deep", "mistral")

    Returns:
        Model class or None if not found
    """
    model_type = model_type.lower()

    # Direct lookup
    if model_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_type]

    # Check aliases
    for alias, canonical in ARCHITECTURE_ALIASES.items():
        if alias.lower() == model_type:
            return MODEL_REGISTRY.get(canonical)

    return None


def detect_architecture(config_dict: dict) -> Optional[str]:
    """
    Auto-detect model architecture from config.

    Args:
        config_dict: Model config dictionary (from config.json)

    Returns:
        Architecture name or None if unknown
    """
    # Check architectures field
    architectures = config_dict.get("architectures", [])
    for arch in architectures:
        if arch in ARCHITECTURE_ALIASES:
            return ARCHITECTURE_ALIASES[arch]

    # Check model_type field
    model_type = config_dict.get("model_type", "")

    type_mapping = {
        "llama": "llama",
        "mistral": "mistral",
        "qwen": "qwen",
        "qwen2": "qwen",
        "phi": "phi",
        "phi3": "phi",
        "gemma": "gemma",
        "gemma2": "gemma",
        "deep": "deep",
        "deepseek": "deep",
    }

    model_type_lower = model_type.lower()
    for key, value in type_mapping.items():
        if key in model_type_lower:
            return value

    # Check for Token-Routed MLP (Pacific Prime indicator)
    if config_dict.get("num_experts") is not None:
        return "deep"

    return None


def list_supported_models() -> list:
    """List all registered model types."""
    return list(MODEL_REGISTRY.keys())


def is_supported(model_type: str) -> bool:
    """Check if a model type is supported."""
    return get_model_class(model_type) is not None
