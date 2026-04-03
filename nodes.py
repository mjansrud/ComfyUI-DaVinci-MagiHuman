"""
ComfyUI nodes for daVinci-MagiHuman.
Optimized for RTX 5090 (32GB VRAM) with block-level CPU offloading.
All models loaded from local paths — no downloads.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import gc
import json
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar

from .ref_wrapper import (
    load_ref_model, RefBlockSwapManager, create_data_proxy, create_sr_data_proxy,
    run_distill_sampling, run_sr_sampling, EvalInput,
)
import sys as _sys
_ref_path = os.path.join(os.path.dirname(__file__), "davinci_ref")
if _ref_path not in _sys.path:
    _sys.path.insert(0, _ref_path)
from inference.model.turbo_vaed.turbo_vaed_model import get_turbo_vaed

# Register model paths
DAVINCI_MODELS_DIR = os.path.join(folder_paths.models_dir, "daVinci-MagiHuman")

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()


def _encode_ref_image(img, height, width, device_target, dtype, vae):
    """Encode a reference image using provided Wan2.2 VAE (z_dim=48)."""
    vae.to(device_target)

    x = img.permute(0, 3, 1, 2)  # [1, 3, H, W]
    x = x * 2.0 - 1.0
    x = x.unsqueeze(2)  # [1, 3, 1, H, W]
    x = x.to(device=device_target, dtype=dtype)

    with torch.no_grad():
        latent = vae.encode(x)

    vae.to("cpu")
    torch.cuda.empty_cache()

    return latent.to(torch.float32)


class DaVinciModelLoader:
    """Load daVinci-MagiHuman DiT model with VRAM-optimized block swapping."""

    @classmethod
    def INPUT_TYPES(s):
        model_types = []
        if os.path.isdir(DAVINCI_MODELS_DIR):
            for d in os.listdir(DAVINCI_MODELS_DIR):
                full = os.path.join(DAVINCI_MODELS_DIR, d)
                if os.path.isdir(full) and os.path.exists(os.path.join(full, "model.safetensors.index.json")):
                    model_types.append(d)
        if not model_types:
            model_types = ["distill", "base", "1080p_sr", "540p_sr"]

        return {
            "required": {
                "model_variant": (model_types, {"default": model_types[0] if model_types else "distill"}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "blocks_on_gpu": ("INT", {
                    "default": 8, "min": 1, "max": 40, "step": 1,
                    "tooltip": "Number of transformer blocks kept on GPU. Lower = less VRAM but slower. 8 works for 32GB."
                }),
            },
        }

    RETURN_TYPES = ("DAVINCI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "DaVinci-MagiHuman"

    def load(self, model_variant, dtype="bf16", blocks_on_gpu=8):
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map[dtype]

        model_dir = os.path.join(DAVINCI_MODELS_DIR, model_variant)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        print(f"[DaVinci] Loading {model_variant} model from {model_dir}...")

        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
        for sf in shard_files:
            sp = os.path.join(model_dir, sf)
            if not os.path.exists(sp):
                raise FileNotFoundError(f"Model shard not found: {sp}")
            if os.path.getsize(sp) < 1024:
                raise FileNotFoundError(f"Model shard is an LFS pointer: {sp}. Run git lfs pull.")

        pbar = ProgressBar(len(shard_files) + 1)

        model = load_ref_model(model_dir, dtype=torch_dtype)
        pbar.update(len(shard_files))

        swap_manager = RefBlockSwapManager(
            model=model,
            blocks_on_gpu=blocks_on_gpu,
            device=device,
            offload_device=offload_device,
        )
        swap_manager.setup()
        pbar.update(1)

        print(f"[DaVinci] Model loaded. {blocks_on_gpu}/{swap_manager.num_layers} blocks on GPU.")

        return ({
            "model": model,
            "swap_manager": swap_manager,
            "dtype": torch_dtype,
            "variant": model_variant,
            "is_distill": model_variant == "distill",
        },)


class DaVinciTurboVAELoader:
    """Load TurboVAE decoder for fast video decoding.

    Expects turbo_vae directory inside models/daVinci-MagiHuman/ with
    a .json config and .ckpt checkpoint file.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("DAVINCI_VAE",)
    RETURN_NAMES = ("turbo_vae",)
    FUNCTION = "load"
    CATEGORY = "DaVinci-MagiHuman"

    def load(self, dtype="bf16"):
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map[dtype]

        vae_dir = os.path.join(DAVINCI_MODELS_DIR, "turbo_vae")
        if not os.path.isdir(vae_dir):
            raise FileNotFoundError(f"TurboVAE directory not found: {vae_dir}")

        config_path = None
        ckpt_path = None
        for f in os.listdir(vae_dir):
            if f.endswith('.json') and 'index' not in f:
                config_path = os.path.join(vae_dir, f)
            if f.endswith('.ckpt'):
                ckpt_path = os.path.join(vae_dir, f)

        if config_path is None or ckpt_path is None:
            raise FileNotFoundError(f"TurboVAE config (.json) or checkpoint (.ckpt) not found in {vae_dir}")

        print(f"[DaVinci] Loading TurboVAE from {vae_dir}...")
        vae = get_turbo_vaed(config_path, ckpt_path, device="cpu", weight_dtype=torch_dtype)
        print(f"[DaVinci] TurboVAE loaded.")

        return ({"vae": vae, "dtype": torch_dtype},)


class DaVinciWan22VAELoader:
    """Load Wan2.2 VAE (z_dim=48) for encoding reference images.

    Point to a local Wan2.2_VAE.pth file. Required for I2V mode.
    """

    @classmethod
    def INPUT_TYPES(s):
        # Scan for .pth files in common locations
        vae_files = []
        search_dirs = [
            os.path.join(folder_paths.models_dir, "wan_vae"),
            os.path.join(folder_paths.models_dir, "vae"),
            folder_paths.models_dir,
        ]
        for d in search_dirs:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith('.pth') and 'wan' in f.lower():
                        vae_files.append(os.path.join(d, f))
        # Also check HF cache structure
        hf_cache = os.path.join(folder_paths.models_dir, "wan_vae")
        if os.path.isdir(hf_cache):
            for root, dirs, files in os.walk(hf_cache):
                for f in files:
                    if f.endswith('.pth'):
                        vae_files.append(os.path.join(root, f))

        if not vae_files:
            vae_files = ["models/wan_vae/Wan2.2_VAE.pth"]

        return {
            "required": {
                "vae_path": (vae_files, {"tooltip": "Path to Wan2.2_VAE.pth file (z_dim=48)"}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("DAVINCI_WAN_VAE",)
    RETURN_NAMES = ("wan_vae",)
    FUNCTION = "load"
    CATEGORY = "DaVinci-MagiHuman"

    def load(self, vae_path, dtype="bf16"):
        from inference.model.vae2_2.vae2_2_model import get_vae2_2

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map[dtype]

        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"Wan2.2 VAE not found: {vae_path}")

        print(f"[DaVinci] Loading Wan2.2 VAE from {vae_path}...")
        vae = get_vae2_2(vae_path, device="cpu", weight_dtype=torch_dtype)
        print(f"[DaVinci] Wan2.2 VAE loaded (z_dim=48).")

        return ({"vae": vae, "dtype": torch_dtype},)


class DaVinciT5GemmaLoader:
    """Load T5Gemma-9B text encoder from a local directory.

    Point to the directory containing the T5Gemma model files
    (config.json, tokenizer files, model shards).
    """

    @classmethod
    def INPUT_TYPES(s):
        # Scan for T5Gemma directories
        t5_dirs = []
        search_dirs = [
            os.path.join(folder_paths.models_dir, "t5gemma"),
            os.path.join(folder_paths.models_dir, "text_encoders"),
            folder_paths.models_dir,
        ]
        for d in search_dirs:
            if os.path.isdir(d):
                for sub in os.listdir(d):
                    full = os.path.join(d, sub)
                    if os.path.isdir(full) and os.path.exists(os.path.join(full, "config.json")):
                        t5_dirs.append(full)
                # Also check if the dir itself has config.json
                if os.path.exists(os.path.join(d, "config.json")):
                    t5_dirs.append(d)

        if not t5_dirs:
            t5_dirs = ["models/t5gemma"]

        return {
            "required": {
                "model_path": (t5_dirs, {"tooltip": "Path to T5Gemma model directory"}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("DAVINCI_T5GEMMA",)
    RETURN_NAMES = ("t5gemma",)
    FUNCTION = "load"
    CATEGORY = "DaVinci-MagiHuman"

    def load(self, model_path, dtype="bf16"):
        from transformers import AutoTokenizer
        from transformers.models.t5gemma import T5GemmaEncoderModel

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map[dtype]

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"T5Gemma model directory not found: {model_path}")

        print(f"[DaVinci] Loading T5Gemma from {model_path} ({dtype})...")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5GemmaEncoderModel.from_pretrained(
            model_path,
            is_encoder_decoder=False,
            dtype=torch_dtype,
        )
        model = model.to("cpu").eval()

        print(f"[DaVinci] T5Gemma loaded.")

        return ({"model": model, "tokenizer": tokenizer, "dtype": torch_dtype},)


class DaVinciTextEncode:
    """Encode text prompt for daVinci-MagiHuman.

    Connect the DaVinci T5Gemma Loader for proper text conditioning.
    Without it, uses placeholder embeddings (won't produce meaningful video).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A person speaking naturally, looking at the camera.",
                    "tooltip": "Text description of the video to generate."
                }),
                "max_tokens": ("INT", {"default": 640, "min": 64, "max": 1024, "step": 64}),
            },
            "optional": {
                "t5gemma": ("DAVINCI_T5GEMMA", {
                    "tooltip": "T5Gemma encoder from DaVinci T5Gemma Loader node."
                }),
            }
        }

    RETURN_TYPES = ("DAVINCI_TEXT_EMBEDS",)
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "encode"
    CATEGORY = "DaVinci-MagiHuman"

    def encode(self, prompt, max_tokens=640, t5gemma=None):
        embed_dim = 3584

        if t5gemma is not None:
            model = t5gemma["model"]
            tokenizer = t5gemma["tokenizer"]

            print(f"[DaVinci] Encoding prompt with T5Gemma: {prompt[:80]}...")

            model = model.to(device)

            with torch.no_grad():
                inputs = tokenizer([prompt], return_tensors="pt").to(device)
                outputs = model(**inputs)
                embeds = outputs["last_hidden_state"].float()

            model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if embeds.shape[1] < max_tokens:
                pad = torch.zeros(1, max_tokens - embeds.shape[1], embed_dim,
                                   device=embeds.device, dtype=embeds.dtype)
                embeds = torch.cat([embeds, pad], dim=1)
            elif embeds.shape[1] > max_tokens:
                embeds = embeds[:, :max_tokens]

            print(f"[DaVinci] T5Gemma embeddings: {embeds.shape}")
            return ({"embeds": embeds.cpu(), "prompt": prompt},)

        # Fallback: deterministic prompt-seeded embeddings
        print(f"[DaVinci] WARNING: No T5Gemma connected. Using placeholder embeddings.")

        prompt_hash = hash(prompt) & 0xFFFFFFFF
        words = prompt.lower().split()
        num_words = min(len(words), max_tokens)

        embeds = torch.zeros(1, max_tokens, embed_dim, dtype=torch.float32)
        for i, word in enumerate(words[:num_words]):
            word_hash = hash(word) & 0xFFFFFFFF
            word_gen = torch.Generator().manual_seed(word_hash)
            embeds[0, i] = torch.randn(embed_dim, generator=word_gen) * 0.5

        positions = torch.arange(max_tokens).float().unsqueeze(1) / max_tokens
        pos_signal = torch.sin(positions * torch.arange(embed_dim).float().unsqueeze(0) * 0.01) * 0.1
        embeds[0] = embeds[0] + pos_signal

        return ({"embeds": embeds, "prompt": prompt},)


class DaVinciSampler:
    """Run denoising loop for daVinci-MagiHuman video generation."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("DAVINCI_MODEL",),
                "text_embeds": ("DAVINCI_TEXT_EMBEDS",),
                "width": ("INT", {"default": 448, "min": 64, "max": 1920, "step": 16,
                                   "tooltip": "256p base: 448."}),
                "height": ("INT", {"default": 256, "min": 64, "max": 1088, "step": 16,
                                    "tooltip": "256p base: 256."}),
                "num_frames": ("INT", {"default": 126, "min": 5, "max": 250, "step": 1,
                                        "tooltip": "126 = 5 seconds at 25fps (seconds*fps+1)."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50, "step": 1,
                                   "tooltip": "Distill: 8 steps. Base: 32 steps."}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True,
                                               "tooltip": "Move model to CPU after sampling to free VRAM."}),
            },
            "optional": {
                "ref_image": ("IMAGE", {"tooltip": "Reference image for I2V (must match width x height)."}),
                "wan_vae": ("DAVINCI_WAN_VAE", {"tooltip": "Wan2.2 VAE for encoding ref image. Required for I2V."}),
            }
        }

    RETURN_TYPES = ("DAVINCI_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "DaVinci-MagiHuman"

    def sample(
        self, model, text_embeds, width, height, num_frames, steps, shift, seed,
        force_offload=True, ref_image=None, wan_vae=None,
    ):
        dit = model["model"]
        swap_manager = model["swap_manager"]
        dtype = model["dtype"]

        # Encode reference image if provided
        latent_image = None
        if ref_image is not None:
            if wan_vae is None:
                raise ValueError("Connect a DaVinci Wan2.2 VAE Loader for I2V mode.")
            img = ref_image[0:1]
            if img.shape[1] != height or img.shape[2] != width:
                raise ValueError(
                    f"Reference image size ({img.shape[2]}x{img.shape[1]}) doesn't match "
                    f"target video size ({width}x{height}). "
                    f"Use a Resize/Crop node to match dimensions."
                )
            latent_image = _encode_ref_image(img, height, width, device, dtype, wan_vae["vae"])
            print(f"[DaVinci] Encoded ref image: {latent_image.shape}")

        embeds = text_embeds["embeds"]
        text_len = (embeds.abs().sum(-1) > 0).sum(-1).item()
        if text_len == 0:
            text_len = 1

        data_proxy = create_data_proxy()
        pbar = ProgressBar(steps)

        def step_callback(idx, total, latent_video, latent_audio):
            pbar.update_absolute(idx + 1, total)

        result = run_distill_sampling(
            model=dit,
            swap_manager=swap_manager,
            data_proxy=data_proxy,
            text_embeds=embeds,
            text_len=text_len,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            shift=shift,
            seed=seed,
            device=device,
            dtype=dtype,
            callback=step_callback,
            latent_image=latent_image,
        )

        if force_offload:
            swap_manager.cleanup()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return (result,)


class DaVinciSuperResolution:
    """Upscale 256p latent to 1080p using the SR model."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sr_model": ("DAVINCI_MODEL",),
                "latent": ("DAVINCI_LATENT",),
                "text_embeds": ("DAVINCI_TEXT_EMBEDS",),
                "target_width": ("INT", {"default": 1920, "min": 448, "max": 1920, "step": 16}),
                "target_height": ("INT", {"default": 1088, "min": 256, "max": 1920, "step": 16}),
                "sr_steps": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "noise_value": ("INT", {"default": 220, "min": 0, "max": 999, "step": 1,
                                         "tooltip": "Re-noise level for SR. 220 is default."}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "ref_image": ("IMAGE", {"tooltip": "Same ref image, re-encoded at SR resolution."}),
                "wan_vae": ("DAVINCI_WAN_VAE", {"tooltip": "Wan2.2 VAE for encoding ref image at SR resolution."}),
            },
        }

    RETURN_TYPES = ("DAVINCI_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "upscale"
    CATEGORY = "DaVinci-MagiHuman"

    def upscale(self, sr_model, latent, text_embeds, target_width, target_height,
                sr_steps, noise_value, shift, seed, force_offload=True, ref_image=None, wan_vae=None):
        dit = sr_model["model"]
        swap_manager = sr_model["swap_manager"]
        dtype = sr_model["dtype"]

        # Encode ref image at SR resolution if provided
        sr_latent_image = None
        if ref_image is not None and wan_vae is not None:
            img = ref_image[0:1]
            # Resize to SR dimensions
            if img.shape[1] != target_height or img.shape[2] != target_width:
                img_t = img.permute(0, 3, 1, 2)
                img_t = F.interpolate(img_t, size=(target_height, target_width), mode='bilinear', align_corners=False)
                img = img_t.permute(0, 2, 3, 1)
            sr_latent_image = _encode_ref_image(img, target_height, target_width, device, dtype, wan_vae["vae"])
            print(f"[DaVinci SR] Encoded SR ref image: {sr_latent_image.shape}")

        embeds = text_embeds["embeds"]
        text_len = max(1, (embeds.abs().sum(-1) > 0).sum(-1).item())

        sr_data_proxy = create_sr_data_proxy()
        pbar = ProgressBar(sr_steps)

        def step_callback(idx, total, latent_video, latent_audio):
            pbar.update_absolute(idx + 1, total)

        result = run_sr_sampling(
            sr_model=dit,
            swap_manager=swap_manager,
            sr_data_proxy=sr_data_proxy,
            text_embeds=embeds,
            text_len=text_len,
            br_latent_video=latent["video_latent"],
            br_latent_audio=latent["audio_tokens"],
            sr_width=target_width,
            sr_height=target_height,
            num_frames=latent["num_frames"],
            sr_steps=sr_steps,
            noise_value=noise_value,
            shift=shift,
            seed=seed,
            latent_image=sr_latent_image,
            device=device,
            dtype=dtype,
            callback=step_callback,
        )

        if force_offload:
            swap_manager.cleanup()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return (result,)


class DaVinciDecode:
    """Decode video latent to frames using TurboVAE."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "turbo_vae": ("DAVINCI_VAE",),
                "latent": ("DAVINCI_LATENT",),
                "output_offload": ("BOOLEAN", {"default": True,
                                                "tooltip": "Offload decoded chunks to CPU (needed for 1080p)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "decode"
    CATEGORY = "DaVinci-MagiHuman"

    def decode(self, turbo_vae, latent, output_offload=True):
        vae = turbo_vae["vae"]
        dtype = turbo_vae["dtype"]

        video_latent = latent["video_latent"].to(dtype=dtype)

        print(f"[DaVinci] Decoding latent {video_latent.shape}...")

        vae = vae.to(device)

        pbar = ProgressBar(1)

        with torch.no_grad():
            video = vae.decode(video_latent.to(device), output_offload=output_offload).float()

        pbar.update(1)

        vae.to(offload_device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        video = video.cpu()
        video.mul_(0.5).add_(0.5).clamp_(0, 1)

        # [B, 3, T, H, W] -> [T, H, W, 3]
        video = video[0].permute(1, 2, 3, 0).contiguous()

        print(f"[DaVinci] Decoded {video.shape[0]} frames at {video.shape[1]}x{video.shape[2]}")

        return (video,)


class DaVinciAudioVAELoader:
    """Load Stable Audio Open 1.0 VAE for audio decoding.

    Point to a local directory containing model_config.json and model.safetensors
    from stabilityai/stable-audio-open-1.0.
    """

    @classmethod
    def INPUT_TYPES(s):
        # Scan for stable audio directories
        audio_dirs = []
        search_dirs = [
            os.path.join(folder_paths.models_dir, "stable_audio"),
            os.path.join(folder_paths.models_dir, "audio"),
            folder_paths.models_dir,
        ]
        for d in search_dirs:
            if os.path.isdir(d):
                for sub in os.listdir(d):
                    full = os.path.join(d, sub)
                    if os.path.isdir(full) and os.path.exists(os.path.join(full, "model_config.json")):
                        audio_dirs.append(full)
                if os.path.exists(os.path.join(d, "model_config.json")):
                    audio_dirs.append(d)
        # Also check HF cache structure
        for d in search_dirs:
            if os.path.isdir(d):
                for root, dirs, files in os.walk(d):
                    if "model_config.json" in files and "model.safetensors" in files:
                        audio_dirs.append(root)

        if not audio_dirs:
            audio_dirs = ["models/stable_audio"]

        # Deduplicate
        audio_dirs = list(dict.fromkeys(audio_dirs))

        return {
            "required": {
                "model_dir": (audio_dirs, {"tooltip": "Directory with model_config.json and model.safetensors"}),
            },
        }

    RETURN_TYPES = ("DAVINCI_AUDIO_VAE",)
    RETURN_NAMES = ("audio_vae",)
    FUNCTION = "load"
    CATEGORY = "DaVinci-MagiHuman"

    def load(self, model_dir):
        import sys
        ref_path = os.path.join(os.path.dirname(__file__), "davinci_ref", "inference", "model", "sa_audio")
        if ref_path not in sys.path:
            sys.path.insert(0, ref_path)
        from sa_audio_module import create_model_from_config

        config_path = os.path.join(model_dir, "model_config.json")
        weights_path = os.path.join(model_dir, "model.safetensors")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"model_config.json not found in {model_dir}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"model.safetensors not found in {model_dir}")

        print(f"[DaVinci] Loading Stable Audio VAE from {model_dir}...")

        with open(config_path) as f:
            full_config = json.load(f)

        vae_config = full_config["model"]["pretransform"]["config"]
        sample_rate = full_config["sample_rate"]

        autoencoder_config = {
            "model_type": "autoencoder",
            "sample_rate": sample_rate,
            "model": vae_config,
        }

        vae_model = create_model_from_config(autoencoder_config)

        from safetensors.torch import load_file
        full_sd = load_file(weights_path, device="cpu")
        vae_sd = {}
        for key, value in full_sd.items():
            if key.startswith("pretransform.model."):
                vae_sd[key[len("pretransform.model."):]] = value
        del full_sd

        vae_model.load_state_dict(vae_sd)
        vae_model.eval()

        print(f"[DaVinci] Stable Audio VAE loaded (sample_rate={sample_rate}).")

        return ({"vae": vae_model, "sample_rate": sample_rate},)


class DaVinciAudioDecode:
    """Decode audio latents to waveform using Stable Audio VAE."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_vae": ("DAVINCI_AUDIO_VAE",),
                "latent": ("DAVINCI_LATENT",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "DaVinci-MagiHuman"

    def decode(self, audio_vae, latent):
        vae = audio_vae["vae"]
        sample_rate = audio_vae["sample_rate"]

        audio_tokens = latent["audio_tokens"]
        if audio_tokens is None:
            print("[DaVinci] No audio tokens to decode.")
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": sample_rate},)

        print(f"[DaVinci] Decoding audio latents {audio_tokens.shape}...")

        audio_latent = audio_tokens.float().permute(0, 2, 1)

        vae = vae.to(device)

        with torch.no_grad():
            waveform = vae.decode(audio_latent.to(device))

        vae.to(offload_device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        waveform = waveform.cpu().float()
        print(f"[DaVinci] Audio decoded: {waveform.shape}, sample_rate={sample_rate}")

        return ({"waveform": waveform, "sample_rate": sample_rate},)


class DaVinciVideoOutput:
    """Save generated video frames to file with optional audio."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("INT", {"default": 25, "min": 1, "max": 60}),
                "filename_prefix": ("STRING", {"default": "davinci"}),
                "format": (["mp4", "webm"], {"default": "mp4"}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Audio from DaVinci Audio Decode node."}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "DaVinci-MagiHuman"

    def save(self, frames, fps, filename_prefix, format="mp4", audio=None):
        import subprocess
        import soundfile as sf

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir
        )

        output_path = os.path.join(full_output_folder, f"{filename}_{counter:05d}.{format}")

        T, H, W, C = frames.shape

        print(f"[DaVinci] Saving {T} frames ({W}x{H}) at {fps}fps to {output_path}")

        if format == "mp4":
            codec = "libx264"
            pix_fmt = "yuv420p"
        else:
            codec = "libvpx-vp9"
            pix_fmt = "yuv420p"

        audio_tmp = None
        if audio is not None and "waveform" in audio:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            if waveform.numel() > 1:
                audio_tmp = os.path.join(full_output_folder, f"{filename}_{counter:05d}_audio.wav")
                wav_np = waveform[0].permute(1, 0).numpy()
                sf.write(audio_tmp, wav_np, sample_rate)

        try:
            if audio_tmp:
                video_tmp = os.path.join(full_output_folder, f"{filename}_{counter:05d}_video_tmp.{format}")
                cmd_video = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(fps),
                    "-i", "-",
                    "-c:v", codec, "-pix_fmt", pix_fmt, "-crf", "18",
                    video_tmp,
                ]
                proc = subprocess.Popen(cmd_video, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                pbar = ProgressBar(T)
                for i in range(T):
                    frame = (frames[i].numpy() * 255).astype(np.uint8)
                    proc.stdin.write(frame.tobytes())
                    pbar.update(1)
                proc.stdin.close()
                proc.wait()

                cmd_mux = [
                    "ffmpeg", "-y",
                    "-i", video_tmp, "-i", audio_tmp,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", output_path,
                ]
                subprocess.run(cmd_mux, stderr=subprocess.PIPE)
                os.remove(video_tmp)
                os.remove(audio_tmp)
                print(f"[DaVinci] Video+audio saved: {output_path}")
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(fps),
                    "-i", "-",
                    "-c:v", codec, "-pix_fmt", pix_fmt, "-crf", "18",
                    output_path,
                ]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                pbar = ProgressBar(T)
                for i in range(T):
                    frame = (frames[i].numpy() * 255).astype(np.uint8)
                    proc.stdin.write(frame.tobytes())
                    pbar.update(1)
                proc.stdin.close()
                proc.wait()

                if proc.returncode != 0:
                    stderr = proc.stderr.read().decode()
                    print(f"[DaVinci] FFmpeg error: {stderr}")
                else:
                    print(f"[DaVinci] Video saved: {output_path}")
        except FileNotFoundError:
            print("[DaVinci] ERROR: ffmpeg not found. Install ffmpeg to save videos.")

        return {"ui": {"videos": [{"filename": f"{filename}_{counter:05d}.{format}",
                                    "subfolder": subfolder, "type": "output"}]}}


# Node registration
NODE_CLASS_MAPPINGS = {
    "DaVinciModelLoader": DaVinciModelLoader,
    "DaVinciTurboVAELoader": DaVinciTurboVAELoader,
    "DaVinciWan22VAELoader": DaVinciWan22VAELoader,
    "DaVinciT5GemmaLoader": DaVinciT5GemmaLoader,
    "DaVinciTextEncode": DaVinciTextEncode,
    "DaVinciSampler": DaVinciSampler,
    "DaVinciSuperResolution": DaVinciSuperResolution,
    "DaVinciDecode": DaVinciDecode,
    "DaVinciAudioVAELoader": DaVinciAudioVAELoader,
    "DaVinciAudioDecode": DaVinciAudioDecode,
    "DaVinciVideoOutput": DaVinciVideoOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaVinciModelLoader": "DaVinci Model Loader",
    "DaVinciTurboVAELoader": "DaVinci TurboVAE Loader",
    "DaVinciWan22VAELoader": "DaVinci Wan2.2 VAE Loader",
    "DaVinciT5GemmaLoader": "DaVinci T5Gemma Loader",
    "DaVinciTextEncode": "DaVinci Text Encode",
    "DaVinciSampler": "DaVinci Sampler",
    "DaVinciSuperResolution": "DaVinci Super Resolution",
    "DaVinciDecode": "DaVinci Decode (TurboVAE)",
    "DaVinciAudioVAELoader": "DaVinci Audio VAE Loader",
    "DaVinciAudioDecode": "DaVinci Audio Decode",
    "DaVinciVideoOutput": "DaVinci Video Output",
}
