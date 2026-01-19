"""
Mu Inference Engine
===================

High-performance LLM inference with Mu-guided dynamics.

Features:
- Mu dynamics for numerical stability (soft-clamping)
- Multi-user scheduling with priority support
- Paged KV cache (no custom CUDA kernels)
- OpenAI-compatible REST API
- Support for DeepForCausalLM, Llama, Mistral, Qwen, Phi, Gemma

Quick Start:
    >>> from mu_inference import MuEngine, SamplingParams
    >>> import asyncio
    >>>
    >>> async def main():
    ...     engine = MuEngine()
    ...     await engine.initialize("path/to/model")
    ...     output = await engine.generate("Hello!")
    ...     print(output.text)
    ...
    >>> asyncio.run(main())

CLI Server:
    $ mu-serve --model path/to/model --port 8000

CLI Generation:
    $ mu-generate --model path/to/model --prompt "Hello!"
"""

__version__ = "0.2.4"
__author__ = "Complexity ML"

# Core
from mu_inference.core.mu import mu_clamp, soft_clamp, MuDynamics
from mu_inference.core.config import (
    MuConfig,
    EngineConfig,
    SamplingParams,
    ModelConfig,
    CacheConfig,
    SchedulerConfig,
)

# Models
from mu_inference.models.base import MuModelBase, MuModelOutput
from mu_inference.models.loader import load_model
from mu_inference.models.registry import MODEL_REGISTRY, register_model

# Serving
from mu_inference.serving.engine import MuEngine, MuBatchEngine

__all__ = [
    # Version
    "__version__",
    # Core - Mu dynamics
    "mu_clamp",
    "soft_clamp",
    "MuDynamics",
    # Core - Config
    "MuConfig",
    "EngineConfig",
    "SamplingParams",
    "ModelConfig",
    "CacheConfig",
    "SchedulerConfig",
    # Models
    "MuModelBase",
    "MuModelOutput",
    "load_model",
    "MODEL_REGISTRY",
    "register_model",
    # Engine
    "MuEngine",
    "MuBatchEngine",
]
