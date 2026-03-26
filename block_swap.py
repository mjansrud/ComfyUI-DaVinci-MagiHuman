"""
Block swapping for VRAM-constrained GPUs (RTX 5090 32GB).
The 15B DiT model is ~30GB in bf16 - doesn't fit fully in 32GB VRAM.
This module handles layer-by-layer CPU<->GPU offloading during forward pass.
"""

import torch
import torch.nn as nn
from typing import Optional
import gc


class BlockSwapManager:
    """Manages CPU<->GPU block swapping for the DiT transformer layers.

    Strategy: Only 1 block on GPU at a time + async prefetch of the next.
    Between denoising steps, all blocks are evicted to CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        blocks_on_gpu: int = 8,
        device: torch.device = None,
        offload_device: torch.device = None,
    ):
        self.model = model
        self.blocks_on_gpu = blocks_on_gpu
        self.device = device or torch.device("cuda")
        self.offload_device = offload_device or torch.device("cpu")
        self.num_layers = len(model.layers)

        # Prefetch stream for async transfers
        if self.device.type == "cuda":
            self.prefetch_stream = torch.cuda.Stream(device=self.device)
        else:
            self.prefetch_stream = None

        self._gpu_blocks = set()

    def setup(self):
        """Move adapter/final layers to GPU, transformer blocks to CPU."""
        # Keep embedders and output heads on GPU (small)
        self.model.video_embedder.to(self.device)
        self.model.audio_embedder.to(self.device)
        self.model.text_embedder.to(self.device)
        self.model.rope.to(self.device)
        self.model.final_norm_video.data = self.model.final_norm_video.data.to(self.device)
        self.model.final_norm_audio.data = self.model.final_norm_audio.data.to(self.device)
        self.model.final_linear_video.to(self.device)
        self.model.final_linear_audio.to(self.device)

        # Move ALL blocks to CPU
        for i in range(self.num_layers):
            self.model.layers[i].to(self.offload_device)
        self._gpu_blocks.clear()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _move_to_gpu(self, block_idx: int):
        """Move a block to GPU."""
        if block_idx not in self._gpu_blocks:
            self.model.layers[block_idx].to(self.device)
            self._gpu_blocks.add(block_idx)

    def _move_to_cpu(self, block_idx: int):
        """Move a block to CPU (synchronous to ensure memory is freed)."""
        if block_idx in self._gpu_blocks:
            self.model.layers[block_idx].to(self.offload_device)
            self._gpu_blocks.discard(block_idx)

    def _evict_all(self):
        """Move all blocks back to CPU and free VRAM."""
        for idx in list(self._gpu_blocks):
            self.model.layers[idx].to(self.offload_device)
        self._gpu_blocks.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def forward_with_swap(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        modality_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        callback=None,
    ) -> torch.Tensor:
        """Run all transformer layers with block swapping.

        At start: evict any stale blocks from previous call.
        During: load current block, prefetch next, evict old.
        At end: last few blocks remain on GPU until next call.
        """
        # Evict any stale blocks from previous denoising step
        self._evict_all()

        for i in range(self.num_layers):
            # Load current block
            self._move_to_gpu(i)

            # Sync prefetch from previous iteration
            if self.prefetch_stream is not None:
                self.prefetch_stream.synchronize()

            # Prefetch next block asynchronously
            if i + 1 < self.num_layers:
                if self.prefetch_stream is not None:
                    with torch.cuda.stream(self.prefetch_stream):
                        self._move_to_gpu(i + 1)
                else:
                    self._move_to_gpu(i + 1)

            # Execute
            x = self.model.layers[i](x, rope_cos, rope_sin, modality_ids, attention_mask)

            # Evict current block (keep at most blocks_on_gpu)
            evict_idx = i - self.blocks_on_gpu + 1
            if evict_idx >= 0:
                self._move_to_cpu(evict_idx)

            if callback:
                callback(i, self.num_layers)

        return x

    def cleanup(self):
        """Move everything back to CPU and free VRAM."""
        self._evict_all()
        gc.collect()
