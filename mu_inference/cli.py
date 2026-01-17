"""
Mu CLI
======

Command-line interface for the Mu inference engine.

Commands:
- mu-serve: Start OpenAI-compatible API server
- mu-generate: Generate text from command line
- mu-bench: Run benchmarks
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def serve_main():
    """
    Start the Mu inference server.

    Usage:
        mu-serve --model <model_path> [--port 8000] [--host 0.0.0.0]
    """
    parser = argparse.ArgumentParser(
        description="Mu Inference Server - OpenAI-compatible API"
    )

    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to model (local or HuggingFace)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Data type (default: float16)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--mu-enabled",
        action="store_true",
        default=True,
        help="Enable Mu dynamics (default: True)",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.0,
        help="Mu equilibrium value (default: 0.0)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    import uvicorn
    from mu_inference.core.config import EngineConfig, MuConfig, ModelConfig, CacheConfig
    from mu_inference.serving.server import create_app

    # Build config
    mu_config = MuConfig(
        enabled=args.mu_enabled,
        mu=args.mu,
    )

    model_config = ModelConfig(
        max_position_embeddings=args.max_model_len,
        dtype=args.dtype,
    )

    cache_config = CacheConfig(
        max_seq_len=args.max_model_len,
    )

    engine_config = EngineConfig(
        model=model_config,
        mu=mu_config,
        cache=cache_config,
        device=args.device,
    )

    # Create app
    logger.info(f"Starting Mu Inference Server")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}, dtype: {args.dtype}")
    logger.info(f"Mu dynamics: {'enabled' if args.mu_enabled else 'disabled'}")

    app = create_app(
        model_path=args.model,
        config=engine_config,
    )

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


def generate_main():
    """
    Generate text from command line.

    Usage:
        mu-generate --model <model_path> --prompt "Hello, world!"
    """
    parser = argparse.ArgumentParser(
        description="Mu Text Generation"
    )

    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to model",
    )
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="Input prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Data type (default: float16)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output token by token",
    )
    parser.add_argument(
        "--mu-enabled",
        action="store_true",
        default=True,
        help="Enable Mu dynamics",
    )

    args = parser.parse_args()

    # Import
    from mu_inference.core.config import EngineConfig, MuConfig, SamplingParams, ModelConfig
    from mu_inference.serving.engine import MuEngine

    # Build config
    mu_config = MuConfig(enabled=args.mu_enabled)
    model_config = ModelConfig(dtype=args.dtype)
    engine_config = EngineConfig(
        model=model_config,
        mu=mu_config,
        device=args.device,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # Create engine
    engine = MuEngine(config=engine_config)

    async def run():
        await engine.initialize(args.model)

        print(f"\nPrompt: {args.prompt}\n")
        print("=" * 50)
        print("Generated:")

        if args.stream:
            async for chunk in engine.generate_stream(
                prompt=args.prompt,
                sampling_params=sampling_params,
            ):
                print(chunk.text, end="", flush=True)
            print()
        else:
            output = await engine.generate(
                prompt=args.prompt,
                sampling_params=sampling_params,
            )
            print(output.text)
            print("=" * 50)
            print(f"Tokens: {output.usage['completion_tokens']}")
            print(f"Finish reason: {output.finish_reason}")

        await engine.shutdown()

    asyncio.run(run())


def bench_main():
    """
    Run benchmarks.

    Usage:
        mu-bench --model <model_path> [--num-prompts 10]
    """
    parser = argparse.ArgumentParser(
        description="Mu Inference Benchmarks"
    )

    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to model",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts to run (default: 10)",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=128,
        help="Prompt length in tokens (default: 128)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Output length in tokens (default: 128)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Data type (default: float16)",
    )

    args = parser.parse_args()

    import time
    from mu_inference.core.config import EngineConfig, MuConfig, SamplingParams, ModelConfig
    from mu_inference.serving.engine import MuEngine

    # Config
    engine_config = EngineConfig(
        model=ModelConfig(dtype=args.dtype),
        mu=MuConfig(enabled=True),
        device=args.device,
    )

    sampling_params = SamplingParams(
        max_tokens=args.output_len,
        temperature=0.0,  # Greedy for reproducibility
    )

    engine = MuEngine(config=engine_config)

    async def run_bench():
        await engine.initialize(args.model)

        # Create prompts
        prompts = [
            "The quick brown fox " * (args.prompt_len // 4)
            for _ in range(args.num_prompts)
        ]

        print(f"\nMu Inference Benchmark")
        print(f"=" * 50)
        print(f"Model: {args.model}")
        print(f"Device: {args.device}, dtype: {args.dtype}")
        print(f"Prompts: {args.num_prompts}")
        print(f"Prompt length: ~{args.prompt_len} tokens")
        print(f"Output length: {args.output_len} tokens")
        print(f"=" * 50)

        total_prompt_tokens = 0
        total_output_tokens = 0
        start_time = time.time()

        for i, prompt in enumerate(prompts):
            output = await engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
            )
            total_prompt_tokens += output.usage["prompt_tokens"]
            total_output_tokens += output.usage["completion_tokens"]

            if (i + 1) % 5 == 0:
                print(f"Completed {i + 1}/{args.num_prompts} prompts...")

        elapsed = time.time() - start_time
        total_tokens = total_prompt_tokens + total_output_tokens

        print(f"\nResults:")
        print(f"-" * 50)
        print(f"Total time: {elapsed:.2f}s")
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(f"Throughput: {total_tokens / elapsed:.2f} tokens/s")
        print(f"Output throughput: {total_output_tokens / elapsed:.2f} tokens/s")
        print(f"Latency per request: {elapsed / args.num_prompts * 1000:.2f}ms")

        await engine.shutdown()

    asyncio.run(run_bench())


if __name__ == "__main__":
    # For direct execution
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        sys.argv = sys.argv[1:]  # Remove script name

        if cmd == "serve":
            serve_main()
        elif cmd == "generate":
            generate_main()
        elif cmd == "bench":
            bench_main()
        else:
            print(f"Unknown command: {cmd}")
            print("Available: serve, generate, bench")
    else:
        print("Mu Inference Engine")
        print("Commands: serve, generate, bench")
