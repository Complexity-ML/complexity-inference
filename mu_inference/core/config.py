"""
Mu Inference Configuration
==========================

All configuration classes for the inference engine.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal, Dict, Any
from enum import Enum


class ModelType(str, Enum):
    """Supported model architectures."""
    DEEP = "deep"           # DeepForCausalLM (Token-Routed MLP)
    LLAMA = "llama"         # LlamaForCausalLM
    MISTRAL = "mistral"     # MistralForCausalLM
    QWEN = "qwen"           # Qwen2ForCausalLM
    PHI = "phi"             # PhiForCausalLM
    GEMMA = "gemma"         # GemmaForCausalLM
    AUTO = "auto"           # Auto-detect from config


@dataclass
class MuConfig:
    """
    Configuration for Mu dynamics.

    Mu (μ) is the equilibrium point that all values seek.
    This provides numerical stability through soft-clamping.

    NOTE: mu_clamp should only be enabled for models trained WITH mu_clamp.
    complexity-deep (Pacific Prime) was NOT trained with mu_clamp, so
    it should be disabled by default for correct inference.
    """

    # Master switch - disabled by default for compatibility with
    # models not trained with mu_clamp (like complexity-deep)
    enabled: bool = False

    # Equilibrium point
    mu: float = 0.0

    # Value bounds (deviation from mu)
    clamp_min: float = -10.0
    clamp_max: float = 10.0

    # Velocity bounds
    velocity_min: float = -10.0
    velocity_max: float = 10.0

    # Adaptation rate
    beta: float = 1.0
    beta_min: float = 0.0
    beta_max: float = 2.0

    # Smoothing
    ema_alpha: float = 0.3

    # Apply to specific components
    apply_to_attention: bool = True
    apply_to_mlp: bool = True
    apply_to_layernorm: bool = True
    apply_to_kv_cache: bool = True


@dataclass
class SamplingParams:
    """Parameters for token sampling."""

    # Temperature (0 = greedy, 1 = default, >1 = more random)
    temperature: float = 1.0

    # Top-k sampling (0 = disabled)
    top_k: int = 50

    # Nucleus sampling threshold
    top_p: float = 1.0

    # Repetition penalty
    repetition_penalty: float = 1.0

    # Presence/frequency penalties (OpenAI style)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # Generation limits
    max_tokens: int = 512
    min_tokens: int = 0

    # Stop conditions
    stop_sequences: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)

    # Reproducibility
    seed: Optional[int] = None

    # Logprobs
    logprobs: bool = False
    top_logprobs: int = 0

    # Skip special tokens in output
    skip_special_tokens: bool = True

    def __post_init__(self):
        """Validate parameters."""
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be >= 1")


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Architecture
    model_type: ModelType = ModelType.AUTO
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: Optional[int] = None  # Computed if None

    # Position embeddings
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None

    # Normalization
    rms_norm_eps: float = 1e-6

    # Activation
    hidden_act: str = "silu"

    # Token-Routed MLP (DeepForCausalLM specific)
    use_token_routed_mlp: bool = False
    num_experts: int = 4

    # QK Normalization
    use_qk_norm: bool = False

    # Embedding
    tie_word_embeddings: bool = True

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # INL Dynamics (Pacific-Prime specific)
    dynamics_alpha: float = 0.9
    dynamics_beta: float = 0.1
    dynamics_gate: float = 0.5
    dynamics_dt: float = 0.1
    dynamics_controller_hidden: int = 64

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create from HuggingFace config dict."""
        # Map architecture
        arch = config_dict.get("architectures", [""])[0]
        model_type = ModelType.AUTO
        if "Deep" in arch:
            model_type = ModelType.DEEP
        elif "Llama" in arch:
            model_type = ModelType.LLAMA
        elif "Mistral" in arch:
            model_type = ModelType.MISTRAL
        elif "Qwen" in arch:
            model_type = ModelType.QWEN
        elif "Phi" in arch:
            model_type = ModelType.PHI
        elif "Gemma" in arch:
            model_type = ModelType.GEMMA

        return cls(
            model_type=model_type,
            vocab_size=config_dict.get("vocab_size", 32000),
            hidden_size=config_dict.get("hidden_size", 2048),
            intermediate_size=config_dict.get("intermediate_size", 5632),
            num_hidden_layers=config_dict.get("num_hidden_layers", 24),
            num_attention_heads=config_dict.get("num_attention_heads", 16),
            num_key_value_heads=config_dict.get(
                "num_key_value_heads",
                config_dict.get("num_attention_heads", 16)
            ),
            max_position_embeddings=config_dict.get("max_position_embeddings", 2048),
            rope_theta=config_dict.get("rope_theta", 10000.0),
            rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
            hidden_act=config_dict.get("hidden_act", "silu"),
            use_token_routed_mlp=config_dict.get("use_token_routed_mlp", False),
            num_experts=config_dict.get("num_experts", 4),
            use_qk_norm=config_dict.get("use_qk_norm", False),
            tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
            pad_token_id=config_dict.get("pad_token_id", 0),
            bos_token_id=config_dict.get("bos_token_id", 1),
            eos_token_id=config_dict.get("eos_token_id", 2),
            # INL Dynamics
            dynamics_alpha=config_dict.get("dynamics_alpha", 0.9),
            dynamics_beta=config_dict.get("dynamics_beta", 0.1),
            dynamics_gate=config_dict.get("dynamics_gate", 0.5),
            dynamics_dt=config_dict.get("dynamics_dt", 0.1),
            dynamics_controller_hidden=config_dict.get("dynamics_controller_hidden", 64),
        )


@dataclass
class CacheConfig:
    """KV Cache configuration."""

    # Maximum sequence length
    max_seq_len: int = 2048

    # Block size for paged attention
    block_size: int = 16

    # Maximum number of blocks
    max_num_blocks: int = 2048

    # Maximum concurrent requests
    max_num_seqs: int = 256

    # Data type
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"

    # Paged attention (more memory efficient)
    use_paged: bool = True

    # Apply Mu dynamics to cached values
    mu_enabled: bool = True


@dataclass
class SchedulerConfig:
    """Request scheduler configuration."""

    # Maximum batch size
    max_batch_size: int = 64

    # Maximum tokens per batch (prefill + decode)
    max_num_batched_tokens: int = 8192

    # Maximum waiting requests
    max_waiting_requests: int = 1000

    # Scheduling policy
    policy: Literal["fcfs", "priority", "fair"] = "fcfs"

    # Preemption
    enable_preemption: bool = True
    preemption_mode: Literal["recompute", "swap"] = "recompute"

    # Chunked prefill
    enable_chunked_prefill: bool = True
    max_prefill_tokens: int = 4096

    # Continuous batching
    enable_continuous_batching: bool = True
    batch_wait_time_ms: float = 5.0  # Max wait time to form a batch
    min_batch_size: int = 1  # Minimum requests before executing


@dataclass
class EngineConfig:
    """
    Main engine configuration.

    Combines all sub-configurations.
    """

    # Model
    model_path: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)

    # Mu dynamics
    mu: MuConfig = field(default_factory=MuConfig)

    # Sampling defaults
    sampling: SamplingParams = field(default_factory=SamplingParams)

    # Cache
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Scheduler
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Hardware
    device: str = "cuda"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    tensor_parallel_size: int = 1  # Number of GPUs

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Torch compile
    use_torch_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "EngineConfig":
        """Create config from pretrained model path."""
        import json
        from pathlib import Path
        from huggingface_hub import snapshot_download

        # Download if HuggingFace model
        if not Path(model_path).exists():
            model_path = snapshot_download(model_path)

        # Load model config
        config_path = Path(model_path) / "config.json"
        with open(config_path) as f:
            model_dict = json.load(f)

        model_config = ModelConfig.from_dict(model_dict)

        # Update cache config based on model
        cache_config = CacheConfig(
            max_seq_len=model_config.max_position_embeddings,
        )

        return cls(
            model_path=str(model_path),
            model=model_config,
            cache=cache_config,
            **kwargs,
        )
