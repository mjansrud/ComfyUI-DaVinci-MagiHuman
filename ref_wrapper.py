"""
Wrapper around the reference daVinci-MagiHuman inference code.
Uses the original model, data proxy, and scheduler for correctness.
Adds block swapping for consumer GPUs (32GB VRAM).
"""

import sys
import os
import torch
import gc
from dataclasses import dataclass
from typing import Optional, Callable

# Add reference code to Python path
REF_PATH = os.path.join(os.path.dirname(__file__), "davinci_ref")
if REF_PATH not in sys.path:
    sys.path.insert(0, REF_PATH)

from inference.model.dit.dit_module import DiTModel
from inference.pipeline.scheduler_unipc import FlowUniPCMultistepScheduler
from inference.pipeline.data_proxy import MagiDataProxy
from inference.common.config import ModelConfig, EvaluationConfig, DataProxyConfig
from inference.common.sequence_schema import Modality


@dataclass
class EvalInput:
    """Matches reference EvalInput exactly."""
    x_t: torch.Tensor              # [B, 48, T, H, W] video latent
    audio_x_t: torch.Tensor        # [B, num_frames, 64] audio latent
    audio_feat_len: list            # [int] actual audio length
    txt_feat: torch.Tensor          # [B, seq_len, 3584] text embeddings
    txt_feat_len: list              # [int] actual text length


def load_ref_model(model_dir: str, dtype: torch.dtype = torch.bfloat16) -> DiTModel:
    """Load the reference DiTModel from sharded safetensors."""
    import json
    from safetensors.torch import load_file

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    model_config = ModelConfig()
    # Computed fields that are normally set by MagiPipelineConfig.post_override_config
    model_config.num_heads_q = model_config.hidden_size // model_config.head_dim
    model_config.num_heads_kv = model_config.num_query_groups
    model = DiTModel(model_config)

    shard_to_keys = {}
    for key, shard_file in index["weight_map"].items():
        shard_to_keys.setdefault(shard_file, []).append(key)

    state_dict = {}
    for shard_file in sorted(shard_to_keys.keys()):
        shard_path = os.path.join(model_dir, shard_file)
        if not os.path.exists(shard_path):
            continue
        print(f"  Loading {shard_file}...")
        shard_data = load_file(shard_path, device="cpu")
        for key in shard_to_keys[shard_file]:
            if key in shard_data:
                state_dict[key] = shard_data[key].to(dtype)
        del shard_data

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded: {len(state_dict)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        for k in missing[:5]:
            print(f"    missing: {k}")
    if unexpected:
        for k in unexpected[:5]:
            print(f"    unexpected: {k}")

    model = model.to(dtype).eval()
    return model


class RefBlockSwapManager:
    """Block swapping for the reference DiTModel.
    Reference model structure: model.block.layers[i] = TransformerLayer
    """

    def __init__(self, model: DiTModel, blocks_on_gpu: int = 8,
                 device=None, offload_device=None):
        self.model = model
        self.blocks_on_gpu = blocks_on_gpu
        self.device = device or torch.device("cuda")
        self.offload_device = offload_device or torch.device("cpu")
        self.num_layers = len(model.block.layers)
        self._gpu_blocks = set()

    def setup(self):
        """Move small components to GPU, blocks to CPU."""
        self.model.adapter.to(self.device)
        self.model.final_norm_video.to(self.device)
        self.model.final_norm_audio.to(self.device)
        self.model.final_linear_video.to(self.device)
        self.model.final_linear_audio.to(self.device)

        for i in range(self.num_layers):
            self.model.block.layers[i].to(self.offload_device)
        self._gpu_blocks.clear()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _evict_all(self):
        for idx in list(self._gpu_blocks):
            self.model.block.layers[idx].to(self.offload_device)
        self._gpu_blocks.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _move_to_gpu(self, idx):
        if idx not in self._gpu_blocks:
            self.model.block.layers[idx].to(self.device)
            self._gpu_blocks.add(idx)

    def _move_to_cpu(self, idx):
        if idx in self._gpu_blocks:
            self.model.block.layers[idx].to(self.offload_device)
            self._gpu_blocks.discard(idx)

    def cleanup(self):
        self._evict_all()
        gc.collect()


def create_data_proxy() -> MagiDataProxy:
    """Create a MagiDataProxy with default config."""
    dp_config = DataProxyConfig()
    return MagiDataProxy(config=dp_config)


def run_distill_sampling(
    model: DiTModel,
    swap_manager: RefBlockSwapManager,
    data_proxy: MagiDataProxy,
    text_embeds: torch.Tensor,
    text_len: int,
    width: int = 448,
    height: int = 256,
    num_frames: int = 126,
    steps: int = 8,
    shift: float = 5.0,
    seed: int = 0,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
    callback: Optional[Callable] = None,
) -> dict:
    """Run distill sampling using reference components.

    The key loop:
    1. Pack latent volumes into tokens (data_proxy.process_input)
    2. Run model forward (adapter -> blocks -> output heads)
    3. Unpack velocity predictions back to volumes (data_proxy.process_output)
    4. Scheduler step on volumes (step_ddim)
    5. Repeat
    """
    if device is None:
        device = torch.device("cuda")

    torch.manual_seed(seed)

    eval_config = EvaluationConfig()
    vae_stride = eval_config.vae_stride
    patch_size = eval_config.patch_size
    z_dim = eval_config.z_dim

    # Compute latent dims (matching reference exactly)
    latent_h = height // vae_stride[1] // patch_size[1] * patch_size[1]
    latent_w = width // vae_stride[2] // patch_size[2] * patch_size[2]
    latent_t = (num_frames - 1) // vae_stride[0] + 1

    print(f"[Ref] Generating: {width}x{height}, {num_frames}f, latent={latent_t}x{latent_h}x{latent_w}, steps={steps}")

    # Initialize random latents (matching reference)
    latent_video = torch.randn(1, z_dim, latent_t, latent_h, latent_w,
                                dtype=torch.float32, device=device)
    latent_audio = torch.randn(1, num_frames, 64,
                                dtype=torch.float32, device=device)

    # Schedulers (separate for video and audio, matching reference)
    video_scheduler = FlowUniPCMultistepScheduler()
    audio_scheduler = FlowUniPCMultistepScheduler()
    video_scheduler.set_timesteps(steps, device="cpu", shift=shift)
    audio_scheduler.set_timesteps(steps, device="cpu", shift=shift)
    timesteps = video_scheduler.timesteps

    # Denoising loop (matching reference evaluate_with_latent for cfg_number=1)
    for idx, t in enumerate(timesteps):
        # Build EvalInput (full latent volumes)
        eval_input = EvalInput(
            x_t=latent_video,
            audio_x_t=latent_audio,
            audio_feat_len=[latent_audio.shape[1]],
            txt_feat=text_embeds.to(device=device, dtype=dtype),
            txt_feat_len=[text_len],
        )

        with torch.no_grad():
            # Pack tokens via data proxy
            packed = data_proxy.process_input(eval_input)
            x, coords_mapping, modality_mapping, varlen_handler, local_attn_handler = packed

            # Everything in bf16 to match model weights
            x = x.to(device=device, dtype=dtype)
            coords_mapping = coords_mapping.to(device=device)

            # --- Model forward with block swapping ---
            swap_manager._evict_all()

            # Adapter: input embedding + RoPE
            from inference.model.dit.dit_module import ModalityDispatcher
            modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
            permute_mapping = modality_dispatcher.permute_mapping
            inv_permute_mapping = modality_dispatcher.inv_permute_mapping
            video_mask = modality_mapping == Modality.VIDEO
            audio_mask = modality_mapping == Modality.AUDIO
            text_mask = modality_mapping == Modality.TEXT

            x, rope = model.adapter(x, coords_mapping, video_mask, audio_mask, text_mask)
            x = x.to(dtype)
            x = ModalityDispatcher.permute(x, permute_mapping)

            # cp_split_sizes: for single GPU, just [total_seq_len]
            cp_split_sizes = [x.shape[0]]

            # Run transformer blocks with swapping
            for layer_idx in range(swap_manager.num_layers):
                swap_manager._move_to_gpu(layer_idx)
                if layer_idx + 1 < swap_manager.num_layers:
                    swap_manager._move_to_gpu(layer_idx + 1)

                x = model.block.layers[layer_idx](
                    x, rope,
                    permute_mapping=permute_mapping,
                    inv_permute_mapping=inv_permute_mapping,
                    varlen_handler=varlen_handler,
                    local_attn_handler=local_attn_handler,
                    modality_dispatcher=modality_dispatcher,
                    cp_split_sizes=cp_split_sizes,
                )

                evict = layer_idx - swap_manager.blocks_on_gpu + 1
                if evict >= 0:
                    swap_manager._move_to_cpu(evict)

            # Unpermute and output heads (cast to float32 matching reference post_process_dtype)
            x = ModalityDispatcher.inv_permute(x, inv_permute_mapping)

            x_video = x[video_mask].float()
            x_video = model.final_norm_video(x_video)
            x_video = model.final_linear_video.float()(x_video)

            x_audio = x[audio_mask].float()
            x_audio = model.final_norm_audio(x_audio)
            x_audio = model.final_linear_audio.float()(x_audio)

            # Combine into output tensor for data_proxy.process_output
            x_out = torch.zeros(
                x.shape[0],
                max(model.config.video_in_channels, model.config.audio_in_channels),
                device=x.device, dtype=torch.float32,
            )
            x_out[video_mask, :model.config.video_in_channels] = x_video
            x_out[audio_mask, :model.config.audio_in_channels] = x_audio

            # Restore linear layers back to original dtype
            model.final_linear_video.to(dtype)
            model.final_linear_audio.to(dtype)

            # Unpack velocity back to latent volumes
            v_video, v_audio = data_proxy.process_output(x_out)

        # Scheduler step on VOLUMES (not tokens!) — cfg_number=1 -> step_ddim
        latent_video = video_scheduler.step_ddim(v_video, idx, latent_video)
        latent_audio = audio_scheduler.step_ddim(v_audio, idx, latent_audio)

        if callback:
            callback(idx, steps, latent_video, latent_audio)

    print(f"[Ref] Done. latent_video={latent_video.shape}, latent_audio={latent_audio.shape}")

    return {
        "video_latent": latent_video.cpu(),
        "audio_tokens": latent_audio.cpu(),
        "width": width,
        "height": height,
        "num_frames": num_frames,
    }
