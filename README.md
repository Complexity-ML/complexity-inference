# Mu Inference Engine

**High-performance inference engine with Mu dynamics for numerical stability.**

A clean, modular inference engine built from scratch - like vLLM but simpler, without CUDA kernel dependencies.

## Features

- **Mu Dynamics**: Equilibrium-seeking soft-clamping for numerical stability
- **Multi-user Support**: Concurrent request handling with intelligent scheduling
- **Paged KV Cache**: Memory-efficient caching without custom CUDA kernels
- **OpenAI-compatible API**: Drop-in replacement for OpenAI API
- **Multiple Architectures**: DeepForCausalLM, Llama, Mistral, Qwen, Gemma, Phi
- **No Custom Kernels**: Pure PyTorch - runs anywhere PyTorch runs

## Supported Models

| Architecture | Description |
|--------------|-------------|
| **DeepForCausalLM** | Pacific Prime / Complexity Deep with Token-Routed MLP |
| **LlamaForCausalLM** | Llama 1/2/3, CodeLlama |
| **MistralForCausalLM** | Mistral, Mixtral |
| **Qwen2ForCausalLM** | Qwen, Qwen2 |
| **PhiForCausalLM** | Phi, Phi-3 |
| **GemmaForCausalLM** | Gemma, Gemma 2 |

## Installation

```bash
# Clone the repository
git clone https://github.com/Complexity-ML/mu-inference
cd mu-inference

# Install
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 2.0+ (for FlashAttention via SDPA)
- transformers
- safetensors
- fastapi
- uvicorn

## Quick Start

### Start the API Server

```bash
# Start server with your model
mu-serve --model /path/to/model --port 8000

# Or with HuggingFace model
mu-serve --model INL/pacific-prime-1.5B --port 8000
```

### Generate from CLI

```bash
# Simple generation
mu-generate --model /path/to/model --prompt "Hello, how are you?"

# With parameters
mu-generate --model /path/to/model \
    --prompt "Write a poem about AI" \
    --max-tokens 256 \
    --temperature 0.8 \
    --stream
```

### Python API

```python
import asyncio
from mu_inference import MuEngine, SamplingParams

async def main():
    # Initialize engine
    engine = MuEngine()
    await engine.initialize("/path/to/model")

    # Generate
    output = await engine.generate(
        prompt="Hello, world!",
        sampling_params=SamplingParams(
            max_tokens=100,
            temperature=0.7,
        ),
    )

    print(output.text)
    await engine.shutdown()

asyncio.run(main())
```

### OpenAI-compatible Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

# Chat completion
response = client.chat.completions.create(
    model="mu-model",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
)

print(response.choices[0].message.content)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Text completion |
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |
| `/stats` | GET | Engine statistics |

## Mu Dynamics

Mu dynamics provide numerical stability through soft-clamping:

```python
# Soft clamp function
def soft_clamp(value, min_val, max_val):
    range_val = (max_val - min_val) * 0.5
    center = (max_val + min_val) * 0.5
    normalized = (value - center) / range_val
    return torch.tanh(normalized) * range_val + center

# Mu clamp - equilibrium-seeking
def mu_clamp(value, mu=0.0, clamp_min=-10.0, clamp_max=10.0):
    deviation = value - mu
    clamped_deviation = soft_clamp(deviation, clamp_min, clamp_max)
    return mu + clamped_deviation
```

Enable/disable via CLI:
```bash
mu-serve --model /path/to/model --mu-enabled --mu 0.0
```

## Configuration

### Engine Configuration

```python
from mu_inference.core.config import (
    EngineConfig,
    MuConfig,
    ModelConfig,
    CacheConfig,
    SchedulerConfig,
)

config = EngineConfig(
    mu=MuConfig(
        enabled=True,
        mu=0.0,
        clamp_min=-10.0,
        clamp_max=10.0,
    ),
    model=ModelConfig(
        dtype="float16",
    ),
    cache=CacheConfig(
        use_paged=True,
        max_seq_len=2048,
    ),
    scheduler=SchedulerConfig(
        max_batch_size=32,
        policy="fcfs",
    ),
)
```

### Sampling Parameters

```python
from mu_inference import SamplingParams

params = SamplingParams(
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
)
```

## Benchmarks

Run benchmarks:

```bash
mu-bench --model /path/to/model \
    --num-prompts 100 \
    --prompt-len 128 \
    --output-len 128
```

## Architecture

```
mu_inference/
├── core/           # Core components
│   ├── mu.py       # Mu dynamics (soft-clamp, equilibrium)
│   ├── config.py   # Configuration dataclasses
│   ├── scheduler.py # Multi-user scheduling
│   └── cache.py    # KV cache (simple + paged)
│
├── models/         # Model implementations
│   ├── base.py     # MuModelBase interface
│   ├── layers.py   # Common layers (attention, MLP, norms)
│   ├── deep.py     # DeepForCausalLM (Token-Routed MLP)
│   ├── llama.py    # Llama/Mistral/Qwen/Gemma
│   ├── loader.py   # Model loading
│   └── registry.py # Model registry
│
├── serving/        # Inference serving
│   ├── engine.py   # Main MuEngine
│   ├── worker.py   # GPU worker
│   └── server.py   # OpenAI API server
│
├── utils/          # Utilities
│   ├── sampling.py # Token sampling
│   └── tokenizer.py # Tokenizer wrapper
│
└── cli.py          # CLI commands
```

## Token-Routed MLP

Pacific Prime uses deterministic expert routing:

```python
# Expert selection based on token ID
expert_id = token_id % num_experts
```

Unlike MoE (Mixture of Experts):
- **No router network** - routing is deterministic
- **No load balancing loss** - no training overhead
- **Consistent routing** - same token always goes to same expert
- **Cache-friendly** - predictable memory access patterns

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black mu_inference/
isort mu_inference/
```

## License

MIT License

## Citation

```bibtex
@software{mu_inference,
  title = {Mu Inference Engine},
  author = {Complexity ML},
  year = {2024},
  url = {https://github.com/Complexity-ML/mu-inference}
}
```

---

**Built with Mu dynamics for stable, efficient inference.**
