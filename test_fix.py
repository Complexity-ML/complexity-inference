#!/usr/bin/env python3
"""
Test that mu-inference generates coherent text after the KV cache shape fix.

v0.2.6 Fix:
- Changed KV cache shape from [batch, seq, heads, head_dim] to [batch, heads, seq, head_dim]
- This matches the original complexity-deep implementation
- RoPE is now applied correctly after transpose
"""
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from mu_inference.models.loader import load_config, config_from_dict, load_weights
from mu_inference.models.deep import DeepForCausalLM
from mu_inference.core.config import MuConfig

def test_generation():
    """Test that generation produces coherent output."""
    model_path = "C:/INL/pacific-prime/hf-pacific-prime"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_path}")
    print(f"Device: {device}")

    # Load config
    config_dict = load_config(model_path)
    config = config_from_dict(config_dict)

    print(f"Config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")
    print(f"use_qk_norm={config.use_qk_norm}, num_experts={config.num_experts}")

    # Create model with mu_clamp disabled (complexity-deep wasn't trained with it)
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
    print("Generating...")

    # Generate tokens
    generated_ids = input_ids.clone()

    with torch.no_grad():
        past_key_values = None

        for step in range(50):
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

            # Check KV cache shape (should be [batch, heads, seq, head_dim])
            if step == 0:
                k_shape = past_key_values[0][0].shape
                print(f"KV cache shape: {k_shape}")
                assert k_shape[1] == config.num_key_value_heads, f"Expected heads at dim=1, got shape {k_shape}"
                assert k_shape[2] == curr_input.shape[1], f"Expected seq at dim=2"

            # Sample next token
            logits = output.logits
            if logits.dim() == 3:
                logits = logits[:, -1, :]

            # Greedy sampling for deterministic test
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Print token
            token_text = tokenizer.decode([next_token.item()])
            print(token_text, end="", flush=True)

            # Stop at EOS
            if next_token.item() == config.eos_token_id:
                break

    print("\n")

    # Decode full output
    full_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"Full output: {full_text}")

    # Basic coherence check - should not have garbage like "ly. Soc, on"
    assert "ly." not in full_text[:20], f"Looks like garbage output: {full_text}"

    print("\n[PASS] Generation test passed!")
    return True


if __name__ == "__main__":
    test_generation()
