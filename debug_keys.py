#!/usr/bin/env python3
"""Debug weight key mapping."""
import torch
from safetensors.torch import load_file
from mu_inference.models.loader import load_config, config_from_dict
from mu_inference.models.deep import DeepForCausalLM
from mu_inference.core.config import MuConfig

# Load checkpoint keys
state_dict = load_file("../pacific-prime/model.safetensors")
checkpoint_keys = set(state_dict.keys())

# Create model and get its keys
config_dict = load_config('../pacific-prime')
config = config_from_dict(config_dict)
model = DeepForCausalLM(config=config, mu_config=MuConfig(enabled=False))
model_keys = set(n for n, _ in model.named_parameters())

# Find q_norm related keys
print("=== Checkpoint q_norm/k_norm keys ===")
for k in sorted(checkpoint_keys):
    if 'q_norm' in k or 'k_norm' in k:
        print(f"  {k}")

print("\n=== Model q_norm/k_norm keys ===")
for k in sorted(model_keys):
    if 'q_norm' in k or 'k_norm' in k:
        print(f"  {k}")

# Check first layer attention
print("\n=== Layer 0 self_attn attributes ===")
attn = model.layers[0].self_attn
print(f"use_qk_norm: {attn.use_qk_norm}")
print(f"has q_norm: {hasattr(attn, 'q_norm')}")
if hasattr(attn, 'q_norm'):
    print(f"q_norm type: {type(attn.q_norm)}")
    print(f"q_norm state_dict: {attn.q_norm.state_dict().keys()}")
