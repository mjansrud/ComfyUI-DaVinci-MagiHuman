"""
Wrapper around the reference daVinci-MagiHuman inference code.
Uses the original model, data proxy, and scheduler for correctness.
Adds block swapping for consumer GPUs (32GB VRAM).
"""

import sys
import os
import torch
import torch.nn.functional as F
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


def _patch_fp8_linear():
    """Monkey-patch nn.Linear to handle FP8 weights with dequantization."""
    import torch.nn as nn

    if getattr(nn.Linear, '_fp8_patched', False):
        return  # Already patched

    _orig_forward = nn.Linear.forward
    fp8_dtype = torch.float8_e4m3fn

    def _fp8_linear_forward(self, input):
        if self.weight.dtype == fp8_dtype and hasattr(self, 'weight_scale'):
            w = self.weight.to(input.dtype) * self.weight_scale
            return F.linear(input, w, self.bias)
        return _orig_forward(self, input)

    nn.Linear.forward = _fp8_linear_forward
    nn.Linear._fp8_patched = True


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

    # Check if this is a pre-quantized FP8 model
    scales_path = os.path.join(model_dir, "fp8_scales.json")
    is_fp8 = os.path.exists(scales_path)
    fp8_scales = {}
    if is_fp8:
        with open(scales_path) as f:
            fp8_scales = json.load(f)
        print(f"  FP8 model detected ({len(fp8_scales)} quantized layers)")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded: {len(state_dict)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        for k in missing[:5]:
            print(f"    missing: {k}")
    if unexpected:
        for k in unexpected[:5]:
            print(f"    unexpected: {k}")

    # For FP8: apply per-tensor scales to linear layers
    if is_fp8:
        _patch_fp8_linear()
        import torch.nn as nn
        named_modules = dict(model.named_modules())
        for key, scale_val in fp8_scales.items():
            # key is like "block.layers.0.attention.linear_qkv.weight"
            # module path is everything before ".weight"
            module_path = key.rsplit(".weight", 1)[0]
            if module_path in named_modules:
                mod = named_modules[module_path]
                if isinstance(mod, nn.Linear):
                    mod.weight_scale = scale_val
        param_bytes = sum(p.numel() * p.element_size() for p in model.block.parameters())
        print(f"  FP8 block VRAM: {param_bytes / 1e9:.1f} GB")
        # Move non-FP8 parts to proper dtypes
        model.eval()
    else:
        model = model.to(dtype).eval()

    # Reference keeps adapter and output heads in float32
    model.adapter.to(torch.float32)
    model.final_norm_video.to(torch.float32)
    model.final_norm_audio.to(torch.float32)
    model.final_linear_video.to(torch.float32)
    model.final_linear_audio.to(torch.float32)

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
    latent_image: Optional[torch.Tensor] = None,
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
    video_scheduler.set_timesteps(steps, device=device, shift=shift)
    audio_scheduler.set_timesteps(steps, device=device, shift=shift)
    timesteps = video_scheduler.timesteps

    # Move latent_image to device if provided
    if latent_image is not None:
        latent_image = latent_image.to(device=device, dtype=torch.float32)
        print(f"[Ref] I2V mode: latent_image={latent_image.shape}")

    # Denoising loop (matching reference evaluate_with_latent for cfg_number=1)
    for idx, t in enumerate(timesteps):
        # I2V: overwrite first frame with reference image latent (matching reference)
        if latent_image is not None:
            latent_video[:, :, :1] = latent_image[:, :, :1]

        # Build EvalInput (full latent volumes) — text stays float32 (reference keeps it float32)
        eval_input = EvalInput(
            x_t=latent_video,
            audio_x_t=latent_audio,
            audio_feat_len=[latent_audio.shape[1]],
            txt_feat=text_embeds.to(device=device, dtype=torch.float32),
            txt_feat_len=[text_len],
        )

        with torch.no_grad():
            # Pack tokens via data proxy
            packed = data_proxy.process_input(eval_input)
            x, coords_mapping, modality_mapping, varlen_handler, local_attn_handler = packed

            x = x.to(device=device)
            coords_mapping = coords_mapping.to(device=device)

            # --- Model forward with block swapping ---
            swap_manager._evict_all()

            # Adapter (runs in float32 — matching reference)
            from inference.model.dit.dit_module import ModalityDispatcher
            modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
            permute_mapping = modality_dispatcher.permute_mapping
            inv_permute_mapping = modality_dispatcher.inv_permute_mapping
            video_mask = modality_mapping == Modality.VIDEO
            audio_mask = modality_mapping == Modality.AUDIO
            text_mask = modality_mapping == Modality.TEXT

            x, rope = model.adapter(x, coords_mapping, video_mask, audio_mask, text_mask)
            # Cast to bf16 for transformer blocks (blocks use _BF16ComputeLinear internally)
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

            # Unpermute and output heads (float32 — adapter/output heads kept in float32)
            x = ModalityDispatcher.inv_permute(x, inv_permute_mapping)

            x_video = x[video_mask].float()
            x_video = model.final_norm_video(x_video)
            x_video = model.final_linear_video(x_video)

            x_audio = x[audio_mask].float()
            x_audio = model.final_norm_audio(x_audio)
            x_audio = model.final_linear_audio(x_audio)

            # Combine into output tensor (bf16 matching reference x.dtype)
            x_out = torch.zeros(
                x.shape[0],
                max(model.config.video_in_channels, model.config.audio_in_channels),
                device=x.device, dtype=x.dtype,
            )
            x_out[video_mask, :model.config.video_in_channels] = x_video.to(x.dtype)
            x_out[audio_mask, :model.config.audio_in_channels] = x_audio.to(x.dtype)

            # Unpack velocity back to latent volumes
            v_video, v_audio = data_proxy.process_output(x_out)

        # Scheduler step on VOLUMES (not tokens!) — cfg_number=1 -> step_ddim
        latent_video = video_scheduler.step_ddim(v_video, idx, latent_video)
        latent_audio = audio_scheduler.step_ddim(v_audio, idx, latent_audio)

        if callback:
            callback(idx, steps, latent_video, latent_audio)

    # Final I2V overwrite (matching reference)
    if latent_image is not None:
        latent_video[:, :, :1] = latent_image[:, :, :1]

    print(f"[Ref] Done. latent_video={latent_video.shape}, latent_audio={latent_audio.shape}")

    return {
        "video_latent": latent_video.cpu(),
        "audio_tokens": latent_audio.cpu(),
        "width": width,
        "height": height,
        "num_frames": num_frames,
    }


def create_sr_data_proxy() -> MagiDataProxy:
    """Create a MagiDataProxy with SR-specific config (coords_style=v1)."""
    import copy
    dp_config = DataProxyConfig()
    dp_config.coords_style = "v1"
    return MagiDataProxy(config=dp_config)


def _build_renoise_sigmas():
    """Build the ZeroSNRDDPM sigma schedule used for SR re-noising (matching reference)."""
    import numpy as np
    from functools import partial
    linear_start = 0.00085
    linear_end = 0.0120
    num_timesteps = 1000
    betas = torch.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sigmas = (1 - alphas_cumprod) / alphas_cumprod
    sigmas = sigmas.sqrt().float()
    # Reference uses flip=True: reversed so index 0 = most noise, 999 = least
    sigmas = sigmas.flip(0)
    return sigmas


def run_sr_sampling(
    sr_model: DiTModel,
    swap_manager: RefBlockSwapManager,
    sr_data_proxy: MagiDataProxy,
    text_embeds: torch.Tensor,
    text_len: int,
    br_latent_video: torch.Tensor,
    br_latent_audio: torch.Tensor,
    sr_width: int = 1920,
    sr_height: int = 1088,
    num_frames: int = 126,
    sr_steps: int = 5,
    noise_value: int = 220,
    shift: float = 5.0,
    seed: int = 0,
    sr_audio_noise_scale: float = 0.7,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
    callback: Optional[Callable] = None,
    latent_image: Optional[torch.Tensor] = None,
) -> dict:
    """Run SR sampling using reference components.

    Matches reference: interpolate base latent to SR size, re-noise,
    run SR model for sr_steps using UniPC step (not step_ddim).
    Audio is NOT updated during SR.
    """
    if device is None:
        device = torch.device("cuda")

    torch.manual_seed(seed)

    eval_config = EvaluationConfig()
    vae_stride = eval_config.vae_stride
    patch_size = eval_config.patch_size

    # Compute SR latent dims
    sr_latent_h = sr_height // vae_stride[1] // patch_size[1] * patch_size[1]
    sr_latent_w = sr_width // vae_stride[2] // patch_size[2] * patch_size[2]
    sr_latent_t = br_latent_video.shape[2]  # same temporal length

    print(f"[SR] Upscaling to {sr_width}x{sr_height}, latent={sr_latent_t}x{sr_latent_h}x{sr_latent_w}, steps={sr_steps}")

    # Trilinear interpolation of base latent to SR size (matching reference)
    latent_video = torch.nn.functional.interpolate(
        br_latent_video.to(device=device, dtype=torch.float32),
        size=(sr_latent_t, sr_latent_h, sr_latent_w),
        mode="trilinear",
        align_corners=True,
    )

    # Re-noise using ZeroSNRDDPM sigmas (matching reference exactly)
    if noise_value != 0:
        renoise_sigmas = _build_renoise_sigmas().to(device)
        sigma = renoise_sigmas[noise_value]
        noise = torch.randn_like(latent_video)
        latent_video = latent_video * sigma + noise * (1 - sigma**2) ** 0.5
        print(f"[SR] Re-noised at noise_value={noise_value}, sigma={sigma:.4f}")

    # Audio: partially re-noise (matching reference)
    latent_audio = br_latent_audio.to(device=device, dtype=torch.float32)
    audio_noise = torch.randn_like(latent_audio)
    latent_audio_sr = audio_noise * sr_audio_noise_scale + latent_audio * (1 - sr_audio_noise_scale)

    # I2V image for SR (if provided)
    if latent_image is not None:
        latent_image = latent_image.to(device=device, dtype=torch.float32)

    # Schedulers (on device, matching reference)
    video_scheduler = FlowUniPCMultistepScheduler()
    audio_scheduler = FlowUniPCMultistepScheduler()
    video_scheduler.set_timesteps(sr_steps, device=device, shift=shift)
    audio_scheduler.set_timesteps(sr_steps, device=device, shift=shift)
    timesteps = video_scheduler.timesteps

    # SR denoising loop
    for idx, t in enumerate(timesteps):
        # I2V: overwrite first frame
        if latent_image is not None:
            latent_video[:, :, :1] = latent_image[:, :, :1]

        eval_input = EvalInput(
            x_t=latent_video,
            audio_x_t=latent_audio_sr,
            audio_feat_len=[latent_audio_sr.shape[1]],
            txt_feat=text_embeds.to(device=device, dtype=torch.float32),
            txt_feat_len=[text_len],
        )

        with torch.no_grad():
            packed = sr_data_proxy.process_input(eval_input)
            x, coords_mapping, modality_mapping, varlen_handler, local_attn_handler = packed

            x = x.to(device=device)
            coords_mapping = coords_mapping.to(device=device)

            swap_manager._evict_all()

            from inference.model.dit.dit_module import ModalityDispatcher
            modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
            permute_mapping = modality_dispatcher.permute_mapping
            inv_permute_mapping = modality_dispatcher.inv_permute_mapping
            video_mask = modality_mapping == Modality.VIDEO
            audio_mask = modality_mapping == Modality.AUDIO
            text_mask = modality_mapping == Modality.TEXT

            # Adapter runs in float32
            x, rope = sr_model.adapter(x, coords_mapping, video_mask, audio_mask, text_mask)
            x = x.to(dtype)  # cast to bf16 for transformer blocks
            x = ModalityDispatcher.permute(x, permute_mapping)

            cp_split_sizes = [x.shape[0]]

            for layer_idx in range(swap_manager.num_layers):
                swap_manager._move_to_gpu(layer_idx)
                if layer_idx + 1 < swap_manager.num_layers:
                    swap_manager._move_to_gpu(layer_idx + 1)

                x = sr_model.block.layers[layer_idx](
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

            x = ModalityDispatcher.inv_permute(x, inv_permute_mapping)

            # Output heads in float32 (kept in float32 by load_ref_model)
            x_video = x[video_mask].float()
            x_video = sr_model.final_norm_video(x_video)
            x_video = sr_model.final_linear_video(x_video)

            x_audio = x[audio_mask].float()
            x_audio = sr_model.final_norm_audio(x_audio)
            x_audio = sr_model.final_linear_audio(x_audio)

            x_out = torch.zeros(
                x.shape[0],
                max(sr_model.config.video_in_channels, sr_model.config.audio_in_channels),
                device=x.device, dtype=x.dtype,
            )
            x_out[video_mask, :sr_model.config.video_in_channels] = x_video.to(x.dtype)
            x_out[audio_mask, :sr_model.config.audio_in_channels] = x_audio.to(x.dtype)

            v_video, v_audio = sr_data_proxy.process_output(x_out)

        # SR uses UniPC step (not step_ddim), audio NOT updated
        latent_video = video_scheduler.step(v_video, t, latent_video, return_dict=False)[0]

        if callback:
            callback(idx, sr_steps, latent_video, latent_audio)

    # Final I2V overwrite
    if latent_image is not None:
        latent_video[:, :, :1] = latent_image[:, :, :1]

    print(f"[SR] Done. latent_video={latent_video.shape}")

    return {
        "video_latent": latent_video.cpu(),
        "audio_tokens": latent_audio.cpu(),  # original audio, not SR'd
        "width": sr_width,
        "height": sr_height,
        "num_frames": num_frames,
    }
