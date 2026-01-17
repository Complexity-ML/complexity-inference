"""
Mu KV Cache
===========

High-performance KV Cache with Mu dynamics.

Two implementations:
1. MuKVCache - Simple pre-allocated cache (fast, fixed memory)
2. MuPagedCache - Paged cache like vLLM (memory efficient, dynamic)

Both apply Mu-clamping to cached values for numerical stability.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import time

import torch

from mu_inference.core.mu import mu_clamp, soft_clamp
from mu_inference.core.config import MuConfig, CacheConfig


@dataclass
class CacheBlock:
    """
    Single cache block for one request.

    Holds KV tensors for all layers.
    """

    request_id: str

    # Cache tensors: [num_layers, seq_len, num_kv_heads, head_dim]
    key_cache: torch.Tensor
    value_cache: torch.Tensor

    # Current sequence length
    seq_len: int = 0

    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self):
        """Update last accessed time."""
        self.last_accessed = time.time()


class MuKVCache:
    """
    Simple pre-allocated KV Cache with Mu dynamics.

    Fast and predictable, but uses fixed memory per request.
    Best for: Small number of concurrent requests, predictable workloads.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        max_requests: int = 64,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        mu_config: Optional[MuConfig] = None,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.max_requests = max_requests
        self.dtype = dtype
        self.device = device
        self.mu_config = mu_config or MuConfig()

        # Active caches
        self.caches: Dict[str, CacheBlock] = {}

        # Lock for thread safety
        self._lock = threading.RLock()

    def allocate(self, request_id: str) -> bool:
        """
        Allocate cache for a new request.

        Returns:
            True if allocated, False if at capacity
        """
        with self._lock:
            if len(self.caches) >= self.max_requests:
                # Try to evict LRU
                if not self._evict_lru():
                    return False

            # Allocate tensors
            key_cache = torch.zeros(
                self.num_layers,
                self.max_seq_len,
                self.num_kv_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            value_cache = torch.zeros(
                self.num_layers,
                self.max_seq_len,
                self.num_kv_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )

            self.caches[request_id] = CacheBlock(
                request_id=request_id,
                key_cache=key_cache,
                value_cache=value_cache,
            )
            return True

    def append(
        self,
        request_id: str,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> int:
        """
        Append new KV pairs to cache.

        Args:
            request_id: Request ID
            layer_idx: Layer index
            keys: [batch, seq_len, num_kv_heads, head_dim]
            values: [batch, seq_len, num_kv_heads, head_dim]

        Returns:
            New sequence length, or -1 if failed
        """
        with self._lock:
            block = self.caches.get(request_id)
            if block is None:
                return -1

            new_len = keys.shape[1]
            start = block.seq_len
            end = start + new_len

            if end > self.max_seq_len:
                return -1

            # Apply Mu-clamping
            if self.mu_config.enabled and self.mu_config.apply_to_kv_cache:
                keys = mu_clamp(
                    keys,
                    self.mu_config.mu,
                    self.mu_config.clamp_min,
                    self.mu_config.clamp_max,
                )
                values = mu_clamp(
                    values,
                    self.mu_config.mu,
                    self.mu_config.clamp_min,
                    self.mu_config.clamp_max,
                )

            # Store (squeeze batch dim if present)
            block.key_cache[layer_idx, start:end] = keys.squeeze(0)
            block.value_cache[layer_idx, start:end] = values.squeeze(0)
            block.seq_len = end
            block.touch()

            return end

    def get(
        self,
        request_id: str,
        layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get cached KV for a layer.

        Returns:
            (keys, values) or (None, None) if not found
            Shape: [1, seq_len, num_kv_heads, head_dim]
        """
        with self._lock:
            block = self.caches.get(request_id)
            if block is None:
                return None, None

            seq_len = block.seq_len
            if seq_len == 0:
                return None, None

            keys = block.key_cache[layer_idx, :seq_len].unsqueeze(0)
            values = block.value_cache[layer_idx, :seq_len].unsqueeze(0)
            block.touch()

            return keys, values

    def get_seq_len(self, request_id: str) -> int:
        """Get current sequence length for a request."""
        with self._lock:
            block = self.caches.get(request_id)
            return block.seq_len if block else 0

    def free(self, request_id: str):
        """Free cache for a request."""
        with self._lock:
            self.caches.pop(request_id, None)

    def _evict_lru(self) -> bool:
        """Evict least recently used cache."""
        if not self.caches:
            return False

        lru_id = min(
            self.caches.keys(),
            key=lambda k: self.caches[k].last_accessed
        )
        del self.caches[lru_id]
        return True

    def clear(self):
        """Clear all caches."""
        with self._lock:
            self.caches.clear()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total_tokens = sum(b.seq_len for b in self.caches.values())
            max_tokens = self.max_requests * self.max_seq_len
            return {
                "type": "simple",
                "active_requests": len(self.caches),
                "max_requests": self.max_requests,
                "total_tokens": total_tokens,
                "max_tokens": max_tokens,
                "utilization": total_tokens / max_tokens if max_tokens > 0 else 0,
                "memory_mb": self._estimate_memory_mb(),
            }

    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        dtype_size = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        bytes_per_block = (
            2 * self.num_layers * self.max_seq_len *
            self.num_kv_heads * self.head_dim * dtype_size
        )
        return (len(self.caches) * bytes_per_block) / (1024 * 1024)


class MuPagedCache:
    """
    Paged KV Cache with Mu dynamics.

    Memory-efficient like vLLM's PagedAttention, but without custom CUDA kernels.
    Uses fixed-size blocks allocated on demand.

    Best for: Many concurrent requests, variable-length sequences.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        max_num_blocks: int = 2048,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        mu_config: Optional[MuConfig] = None,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.dtype = dtype
        self.device = device
        self.mu_config = mu_config or MuConfig()

        # Pre-allocate block pool
        # Shape: [max_blocks, num_layers, 2, block_size, num_kv_heads, head_dim]
        self.block_pool = torch.zeros(
            max_num_blocks,
            num_layers,
            2,  # 0=key, 1=value
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )

        # Free block indices
        self.free_blocks: List[int] = list(range(max_num_blocks))

        # Request -> block indices
        self.request_blocks: Dict[str, List[int]] = {}

        # Request -> position in last block
        self.request_pos: Dict[str, int] = {}

        self._lock = threading.RLock()

    def allocate(self, request_id: str) -> bool:
        """Allocate initial block for a request."""
        with self._lock:
            if not self.free_blocks:
                return False

            block_idx = self.free_blocks.pop()
            self.request_blocks[request_id] = [block_idx]
            self.request_pos[request_id] = 0
            return True

    def append(
        self,
        request_id: str,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> int:
        """
        Append new KV pairs.

        Returns:
            Total sequence length, or -1 if failed
        """
        with self._lock:
            if request_id not in self.request_blocks:
                return -1

            blocks = self.request_blocks[request_id]
            pos = self.request_pos[request_id]

            # Apply Mu-clamping
            if self.mu_config.enabled and self.mu_config.apply_to_kv_cache:
                keys = mu_clamp(
                    keys,
                    self.mu_config.mu,
                    self.mu_config.clamp_min,
                    self.mu_config.clamp_max,
                )
                values = mu_clamp(
                    values,
                    self.mu_config.mu,
                    self.mu_config.clamp_min,
                    self.mu_config.clamp_max,
                )

            new_tokens = keys.shape[1]
            total_len = (len(blocks) - 1) * self.block_size + pos

            for i in range(new_tokens):
                # Need new block?
                if pos >= self.block_size:
                    if not self.free_blocks:
                        return -1
                    new_block = self.free_blocks.pop()
                    blocks.append(new_block)
                    pos = 0

                block_idx = blocks[-1]
                self.block_pool[block_idx, layer_idx, 0, pos] = keys[0, i]
                self.block_pool[block_idx, layer_idx, 1, pos] = values[0, i]
                pos += 1

            self.request_pos[request_id] = pos
            return total_len + new_tokens

    def get(
        self,
        request_id: str,
        layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get all cached KV for a request/layer.

        Returns:
            (keys, values) with shape [1, total_tokens, num_kv_heads, head_dim]
        """
        with self._lock:
            blocks = self.request_blocks.get(request_id)
            if not blocks:
                return None, None

            pos = self.request_pos[request_id]

            all_keys = []
            all_values = []

            for i, block_idx in enumerate(blocks):
                if i == len(blocks) - 1:
                    # Last block: only up to current position
                    all_keys.append(self.block_pool[block_idx, layer_idx, 0, :pos])
                    all_values.append(self.block_pool[block_idx, layer_idx, 1, :pos])
                else:
                    all_keys.append(self.block_pool[block_idx, layer_idx, 0])
                    all_values.append(self.block_pool[block_idx, layer_idx, 1])

            if not all_keys:
                return None, None

            keys = torch.cat(all_keys, dim=0).unsqueeze(0)
            values = torch.cat(all_values, dim=0).unsqueeze(0)

            return keys, values

    def get_seq_len(self, request_id: str) -> int:
        """Get sequence length for a request."""
        with self._lock:
            blocks = self.request_blocks.get(request_id)
            if not blocks:
                return 0
            pos = self.request_pos[request_id]
            return (len(blocks) - 1) * self.block_size + pos

    def free(self, request_id: str):
        """Free all blocks for a request."""
        with self._lock:
            blocks = self.request_blocks.pop(request_id, [])
            self.request_pos.pop(request_id, None)
            self.free_blocks.extend(blocks)

    def clear(self):
        """Clear all caches."""
        with self._lock:
            self.request_blocks.clear()
            self.request_pos.clear()
            self.free_blocks = list(range(self.max_num_blocks))

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            used = self.max_num_blocks - len(self.free_blocks)
            total_tokens = sum(self.get_seq_len(r) for r in self.request_blocks)
            return {
                "type": "paged",
                "active_requests": len(self.request_blocks),
                "used_blocks": used,
                "free_blocks": len(self.free_blocks),
                "max_blocks": self.max_num_blocks,
                "block_size": self.block_size,
                "total_tokens": total_tokens,
                "utilization": used / self.max_num_blocks if self.max_num_blocks > 0 else 0,
                "memory_mb": self._estimate_memory_mb(),
            }

    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage."""
        dtype_size = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        bytes_total = (
            self.max_num_blocks * self.num_layers * 2 *
            self.block_size * self.num_kv_heads * self.head_dim * dtype_size
        )
        return bytes_total / (1024 * 1024)


def create_cache(
    config: CacheConfig,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    device: str = "cuda",
    mu_config: Optional[MuConfig] = None,
):
    """
    Factory function to create appropriate cache.

    Args:
        config: Cache configuration
        num_layers: Number of transformer layers
        num_kv_heads: Number of KV attention heads
        head_dim: Head dimension
        device: Device to allocate on
        mu_config: Mu dynamics configuration

    Returns:
        MuKVCache or MuPagedCache
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.dtype, torch.float16)

    if config.use_paged:
        return MuPagedCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=config.block_size,
            max_num_blocks=config.max_num_blocks,
            dtype=dtype,
            device=device,
            mu_config=mu_config,
        )
    else:
        return MuKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=config.max_seq_len,
            max_requests=config.max_num_seqs,
            dtype=dtype,
            device=device,
            mu_config=mu_config,
        )
