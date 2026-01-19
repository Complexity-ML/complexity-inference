#!/usr/bin/env python3
"""
Debug script to compare mu-inference vs complexity-deep generation.

Tests:
1. Single forward pass (no cache) - should match complexity-deep
2. Generation with KV cache - might have issues
"""

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from mu_inference.models.loader import load_config, config_from_dict, load_weights
from mu_inference.models.deep import DeepForCausalLM
from mu_inference.core.config import MuConfig

def test_single_forward():
    """Test single forward pass without KV cache."""
    model_path = "C:/INL/pacific-prime/hf-pacific-prime"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_path}")
    print(f"Device: {device}")

    # Load config
    config_dict = load_config(model_path)
    config = config_from_dict(config_dict)

    print(f"Config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")
    print(f"use_qk_norm={config.use_qk_norm}, num_experts={config.num_experts}")

    # Create model with mu_clamp disabled
    mu_config = MuConfig(enabled=False)
    model = DeepForCausalLM(config=config, mu_config=mu_config)

    # Load weights
    load_weights(model, model_path, device=device, dtype=torch.float16)
    model.eval()

    # Load tokenizer
    tokenizer = Tokenizer.from_file(f"{model_path}/tokenizer.json")

    # Test prompt
    prompt = "The meaning of life is"
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long).to(device)

    print(f"\nPrompt: {prompt}")
    print(f"Input IDs: {input_ids}")
    print(f"Input shape: {input_ids.shape}")

    # === Test 1: Single forward WITHOUT cache (like complexity-deep generate.py) ===
    print("\n" + "="*60)
    print("TEST 1: Generation WITHOUT KV cache (like complexity-deep)")
    print("="*60)

    generated_ids = input_ids.clone()

    with torch.no_grad():
        for step in range(20):
            # Forward pass with ALL tokens each time (no cache)
            output, _ = model(
                input_ids=generated_ids,
                use_cache=False,
            )

            # Get logits for last token
            logits = output.logits
            if logits.dim() == 2:
                # Model already returns only last token logits
                last_logits = logits
            else:
                # [batch, seq, vocab]
                last_logits = logits[:, -1, :]

            # Greedy sampling
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Print token
            token_text = tokenizer.decode([next_token.item()])
            print(token_text, end="", flush=True)

            # Stop at EOS
            if next_token.item() == config.eos_token_id:
                break

    print("\n")
    full_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"Full output (no cache): {full_text}")

    # === Test 2: Generation WITH cache ===
    print("\n" + "="*60)
    print("TEST 2: Generation WITH KV cache")
    print("="*60)

    generated_ids = input_ids.clone()
    past_key_values = None

    with torch.no_grad():
        for step in range(20):
            if past_key_values is not None:
                # Use only last token for decode
                curr_input = generated_ids[:, -1:]
            else:
                curr_input = generated_ids

            output, past_key_values = model(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Check KV cache shape
            if step == 0:
                k_shape = past_key_values[0][0].shape
                print(f"KV cache shape: {k_shape}")
                print(f"  dim 0 (batch): {k_shape[0]}")
                print(f"  dim 1 (heads): {k_shape[1]}")
                print(f"  dim 2 (seq): {k_shape[2]}")
                print(f"  dim 3 (head_dim): {k_shape[3]}")

            # Get logits
            logits = output.logits
            if logits.dim() == 3:
                logits = logits[:, -1, :]

            # Greedy sampling
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Print token
            token_text = tokenizer.decode([next_token.item()])
            print(token_text, end="", flush=True)

            # Stop at EOS
            if next_token.item() == config.eos_token_id:
                break

    print("\n")
    full_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"Full output (with cache): {full_text}")

    print("\n" + "="*60)
    print("If Test 1 works but Test 2 doesn't, the issue is in KV cache handling")
    print("If both fail, the issue is in the model forward pass itself")
    print("="*60)


if __name__ == "__main__":
    test_single_forward()
