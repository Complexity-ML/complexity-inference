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
        default=False,
        help="Enable Mu clamping (only for models trained with mu_clamp)",
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
    from mu_inference.models.loader import load_config, config_from_dict

    # Load model config from model directory (IMPORTANT: gets use_qk_norm etc)
    config_dict = load_config(args.model)
    model_config = config_from_dict(config_dict)

    # Override max_position_embeddings if specified
    if args.max_model_len:
        model_config.max_position_embeddings = args.max_model_len

    logger.info(f"Model config: use_qk_norm={model_config.use_qk_norm}, num_experts={model_config.num_experts}")

    # Build config
    mu_config = MuConfig(
        enabled=args.mu_enabled,
        mu=args.mu,
    )

    cache_config = CacheConfig(
        max_seq_len=model_config.max_position_embeddings,
    )

    engine_config = EngineConfig(
        model=model_config,
        mu=mu_config,
        cache=cache_config,
        device=args.device,
        dtype=args.dtype,
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
        default=0.8,
        help="Sampling temperature (default: 0.8)",
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
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1)",
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
        default=False,
        help="Enable Mu clamping (only for models trained with mu_clamp)",
    )
    parser.add_argument(
        "--reflection",
        action="store_true",
        help="Enable reflection mode (reasoning -> answer loop)",
    )
    parser.add_argument(
        "--thinking-tokens",
        type=int,
        default=200,
        help="Max tokens for reasoning phase (default: 200)",
    )
    parser.add_argument(
        "--multi-clone",
        type=int,
        default=0,
        help="Number of clone passes for collective reasoning (0=disabled)",
    )
    parser.add_argument(
        "--clone-tokens",
        type=int,
        default=100,
        help="Max tokens per clone pass (default: 100)",
    )

    args = parser.parse_args()

    # Import
    from mu_inference.core.config import EngineConfig, MuConfig, SamplingParams, ModelConfig
    from mu_inference.serving.engine import MuEngine
    from mu_inference.models.loader import load_config, config_from_dict

    # Load model config from model directory (IMPORTANT: gets use_qk_norm etc)
    config_dict = load_config(args.model)
    model_config = config_from_dict(config_dict)

    logger.info(f"Model config: use_qk_norm={model_config.use_qk_norm}, num_experts={model_config.num_experts}")

    # Build config
    mu_config = MuConfig(enabled=args.mu_enabled)
    engine_config = EngineConfig(
        model=model_config,
        mu=mu_config,
        device=args.device,
        dtype=args.dtype,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    # Create engine
    engine = MuEngine(config=engine_config)

    async def run():
        await engine.initialize(args.model)

        print(f"\nPrompt: {args.prompt}\n")
        print("=" * 50)

        if args.multi_clone > 0:
            # === MULTI-CLONE MODE ===
            # Multiple "clones" of the same model discuss, last one synthesizes
            print(f"Multi-Clone Mode: {args.multi_clone} passes")
            print("-" * 50)

            clone_params = SamplingParams(
                max_tokens=args.clone_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )

            # Build accumulated context
            context = args.prompt
            total_clone_tokens = 0

            # Clone warmup passes (N-1 clones discuss)
            for i in range(args.multi_clone - 1):
                clone_prompt = f"{context}\n\n[Clone {i + 1}]:"

                print(f"\n[Clone {i + 1}]", end="")

                if args.stream:
                    clone_text = ""
                    async for chunk in engine.generate_stream(
                        prompt=clone_prompt,
                        sampling_params=clone_params,
                    ):
                        print(chunk.text, end="", flush=True)
                        clone_text += chunk.text
                    print()
                    clone_tokens = len(clone_text.split())  # Approximate
                else:
                    clone_output = await engine.generate(
                        prompt=clone_prompt,
                        sampling_params=clone_params,
                    )
                    clone_text = clone_output.text
                    clone_tokens = clone_output.usage["completion_tokens"]
                    print(f" {clone_text}")

                # Accumulate context for next clone
                context = f"{context}\n\n[Clone {i + 1}]: {clone_text}"
                total_clone_tokens += clone_tokens

            # Final clone: synthesis
            print(f"\n[Final - Synthesis]", end="")
            synthesis_prompt = f"{context}\n\n[Summary]: Based on the above discussion,"

            if args.stream:
                synthesis_text = ""
                async for chunk in engine.generate_stream(
                    prompt=synthesis_prompt,
                    sampling_params=sampling_params,
                ):
                    print(chunk.text, end="", flush=True)
                    synthesis_text += chunk.text
                print()
                synthesis_tokens = len(synthesis_text.split())
            else:
                synthesis = await engine.generate(
                    prompt=synthesis_prompt,
                    sampling_params=sampling_params,
                )
                synthesis_text = synthesis.text
                synthesis_tokens = synthesis.usage["completion_tokens"]
                print(f" {synthesis_text}")

            print("=" * 50)
            print(f"Clone passes: {args.multi_clone - 1}")
            print(f"Clone tokens: {total_clone_tokens}")
            print(f"Synthesis tokens: {synthesis_tokens}")
            print(f"Total tokens: {total_clone_tokens + synthesis_tokens}")

        elif args.reflection:
            # === REFLECTION MODE ===
            # Phase 1: Reasoning
            print("Phase 1: Thinking...")
            reasoning_params = SamplingParams(
                max_tokens=args.thinking_tokens,
                temperature=0.5,  # Focused reasoning
                top_k=40,
                repetition_penalty=args.repetition_penalty,
            )

            reasoning_prompt = f"{args.prompt}\n\nLet me think step by step:"

            if args.stream:
                reasoning_text = ""
                async for chunk in engine.generate_stream(
                    prompt=reasoning_prompt,
                    sampling_params=reasoning_params,
                ):
                    print(chunk.text, end="", flush=True)
                    reasoning_text += chunk.text
                print("\n")
                reasoning_tokens = len(reasoning_text.split())  # Approximate
            else:
                reasoning = await engine.generate(
                    prompt=reasoning_prompt,
                    sampling_params=reasoning_params,
                )
                reasoning_text = reasoning.text
                reasoning_tokens = reasoning.usage["completion_tokens"]
                print(f"{reasoning_text}\n")

            # Phase 2: Answer
            print("Phase 2: Answering...")
            answer_prompt = f"{args.prompt}\n\nLet me think step by step:{reasoning_text}\n\nTherefore, my answer is:"

            if args.stream:
                answer_text = ""
                async for chunk in engine.generate_stream(
                    prompt=answer_prompt,
                    sampling_params=sampling_params,
                ):
                    print(chunk.text, end="", flush=True)
                    answer_text += chunk.text
                print()
                answer_tokens = len(answer_text.split())  # Approximate
            else:
                answer = await engine.generate(
                    prompt=answer_prompt,
                    sampling_params=sampling_params,
                )
                print(answer.text)
                answer_tokens = answer.usage["completion_tokens"]

            print("=" * 50)
            print(f"Reasoning tokens: {reasoning_tokens}")
            print(f"Answer tokens: {answer_tokens}")
            print(f"Total tokens: {reasoning_tokens + answer_tokens}")

        else:
            # === NORMAL MODE ===
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
    Run comprehensive benchmark tests.

    Usage:
        mu-bench --model <model_path>
        mu-bench --model <model_path> --output-len 256 --num-prompts 20
    """
    parser = argparse.ArgumentParser(
        description="Mu Inference Benchmark"
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
        "--warmup",
        type=int,
        default=2,
        help="Warmup runs (default: 2)",
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
    import statistics
    from mu_inference.core.config import EngineConfig, MuConfig, SamplingParams
    from mu_inference.serving.engine import MuEngine
    from mu_inference.models.loader import load_config, config_from_dict

    # Load model config
    config_dict = load_config(args.model)
    model_config = config_from_dict(config_dict)

    # Config
    engine_config = EngineConfig(
        model=model_config,
        mu=MuConfig(enabled=False),
        device=args.device,
        dtype=args.dtype,
    )

    sampling_params = SamplingParams(
        max_tokens=args.output_len,
        temperature=0.0,  # Greedy for reproducibility
    )

    engine = MuEngine(config=engine_config)

    async def run_bench():
        await engine.initialize(args.model)

        # Get VRAM usage
        vram_allocated = 0
        vram_reserved = 0
        if args.device == "cuda":
            import torch
            vram_allocated = torch.cuda.memory_allocated() / 1024**3
            vram_reserved = torch.cuda.memory_reserved() / 1024**3

        # Create prompts
        prompts = [
            "The quick brown fox jumps over the lazy dog. " * (args.prompt_len // 10)
            for _ in range(args.num_prompts + args.warmup)
        ]

        print(f"\n{'='*60}")
        print(f"  MU-INFERENCE BENCHMARK")
        print(f"{'='*60}")
        print(f"  Model: {args.model}")
        print(f"  Device: {args.device}, dtype: {args.dtype}")
        print(f"  VRAM: {vram_allocated:.2f} GB allocated, {vram_reserved:.2f} GB reserved")
        print(f"  Prompts: {args.num_prompts} (+ {args.warmup} warmup)")
        print(f"  Prompt length: ~{args.prompt_len} tokens")
        print(f"  Output length: {args.output_len} tokens")
        print(f"{'='*60}\n")

        # Warmup
        print(f"Warming up ({args.warmup} runs)...")
        for i in range(args.warmup):
            await engine.generate(prompt=prompts[i], sampling_params=sampling_params)
        print("Warmup complete.\n")

        # Benchmark
        print(f"Running benchmark...")
        latencies = []
        total_prompt_tokens = 0
        total_output_tokens = 0

        for i in range(args.num_prompts):
            prompt = prompts[args.warmup + i]

            # Measure total latency
            start = time.perf_counter()
            output = await engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
            )
            elapsed = time.perf_counter() - start

            latencies.append(elapsed * 1000)  # ms
            prompt_tokens = output.usage["prompt_tokens"]
            output_tokens = output.usage["completion_tokens"]
            total_prompt_tokens += prompt_tokens
            total_output_tokens += output_tokens

            if (i + 1) % 5 == 0 or i == args.num_prompts - 1:
                print(f"  [{i + 1}/{args.num_prompts}] Latency: {elapsed*1000:.1f}ms, Tokens: {output_tokens}")

        # Calculate statistics
        total_time = sum(latencies) / 1000  # seconds
        total_tokens = total_prompt_tokens + total_output_tokens

        # TPOT (Time Per Output Token)
        tpot_list = [lat / (total_output_tokens / args.num_prompts) for lat in latencies]

        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"\n  Throughput:")
        print(f"    Total tokens/s:  {total_tokens / total_time:.1f}")
        print(f"    Output tokens/s: {total_output_tokens / total_time:.1f}")
        print(f"    Requests/s:      {args.num_prompts / total_time:.2f}")

        print(f"\n  Latency (end-to-end):")
        print(f"    Mean:   {statistics.mean(latencies):.1f} ms")
        print(f"    Median: {statistics.median(latencies):.1f} ms")
        if len(latencies) >= 20:
            print(f"    P95:    {sorted(latencies)[int(len(latencies) * 0.95)]:.1f} ms")
            print(f"    P99:    {sorted(latencies)[int(len(latencies) * 0.99)]:.1f} ms")
        print(f"    Min:    {min(latencies):.1f} ms")
        print(f"    Max:    {max(latencies):.1f} ms")

        print(f"\n  Time per Output Token (TPOT):")
        avg_output_tokens = total_output_tokens / args.num_prompts
        print(f"    Mean:   {statistics.mean(latencies) / avg_output_tokens:.2f} ms/token")

        print(f"\n  Tokens:")
        print(f"    Total prompt:  {total_prompt_tokens}")
        print(f"    Total output:  {total_output_tokens}")
        print(f"    Avg per request: {total_output_tokens / args.num_prompts:.1f}")

        print(f"\n  Memory:")
        if args.device == "cuda":
            import torch
            print(f"    VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"    VRAM reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            print(f"    Peak VRAM:      {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

        print(f"\n{'='*60}\n")

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
