"""
Mu Inference Models
===================

Model implementations compatible with the Mu inference engine.

Supported architectures:
- DeepForCausalLM (Token-Routed MLP)
- LlamaForCausalLM
- MistralForCausalLM
- Qwen2ForCausalLM
- PhiForCausalLM
- GemmaForCausalLM

All models implement the MuModelBase interface for
consistent engine integration.
"""

from mu_inference.models.base import MuModelBase, MuModelOutput
from mu_inference.models.loader import load_model, get_model_class
from mu_inference.models.registry import MODEL_REGISTRY, register_model

__all__ = [
    "MuModelBase",
    "MuModelOutput",
    "load_model",
    "get_model_class",
    "MODEL_REGISTRY",
    "register_model",
]
