"""
ComfyUI nodes for daVinci-MagiHuman.
Optimized for RTX 5090 (32GB VRAM) with block-level CPU offloading.
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
    load_ref_model, RefBlockSwapManager, create_data_proxy,
    run_distill_sampling, EvalInput,
)
# Use reference TurboVAE implementation
import sys as _sys
_ref_path = os.path.join(os.path.dirname(__file__), "davinci_ref")
if _ref_path not in _sys.path:
    _sys.path.insert(0, _ref_path)
from inference.model.turbo_vaed.turbo_vaed_model import get_turbo_vaed
from .preview import send_preview

# Register model paths
DAVINCI_MODELS_DIR = os.path.join(folder_paths.models_dir, "daVinci-MagiHuman")

device = mm.get_torch_device()
offload_device = mm.unet_offload_device()


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

        # Check if safetensors files exist (not just LFS pointers)
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        shard_files = set(index["weight_map"].values())
        for sf in shard_files:
            sp = os.path.join(model_dir, sf)
            if not os.path.exists(sp):
                raise FileNotFoundError(f"Model shard not found: {sp}. Run git lfs pull to download.")
            # Check it's not an LFS pointer (< 1KB = pointer file)
            if os.path.getsize(sp) < 1024:
                raise FileNotFoundError(f"Model shard is an LFS pointer, not downloaded: {sp}. Run git lfs pull.")

        pbar = ProgressBar(len(shard_files) + 1)

        model = load_ref_model(model_dir, dtype=torch_dtype)
        pbar.update(len(shard_files))

        # Set up block swap manager (reference model: model.block.layers)
        swap_manager = RefBlockSwapManager(
            model=model,
            blocks_on_gpu=blocks_on_gpu,
            device=device,
            offload_device=offload_device,
        )
        swap_manager.setup()
        pbar.update(1)

        print(f"[DaVinci] Ref model loaded. {blocks_on_gpu}/{swap_manager.num_layers} blocks on GPU.")

        model_data = {
            "model": model,
            "swap_manager": swap_manager,
            "dtype": torch_dtype,
            "variant": model_variant,
            "is_distill": model_variant == "distill",
        }

        return (model_data,)


class DaVinciTurboVAELoader:
    """Load TurboVAE decoder for fast video decoding."""

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

        # Find config json and checkpoint in turbo_vae dir
        config_path = None
        ckpt_path = None
        for f in os.listdir(vae_dir):
            if f.endswith('.json') and 'index' not in f:
                config_path = os.path.join(vae_dir, f)
            if f.endswith('.ckpt'):
                ckpt_path = os.path.join(vae_dir, f)

        if config_path is None or ckpt_path is None:
            raise FileNotFoundError(f"TurboVAE config or checkpoint not found in {vae_dir}")

        print(f"[DaVinci] Loading TurboVAE from {vae_dir} (reference code)...")
        vae = get_turbo_vaed(config_path, ckpt_path, device="cpu", weight_dtype=torch_dtype)
        print(f"[DaVinci] TurboVAE loaded.")

        return ({"vae": vae, "dtype": torch_dtype},)


class DaVinciT5GemmaLoader:
    """Load the T5Gemma-9B text encoder for daVinci-MagiHuman.

    Downloads google/t5gemma-9b-9b-ul2 from HuggingFace (~18GB in bf16).
    The encoder is offloaded to CPU after encoding to free VRAM.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("DAVINCI_T5GEMMA",)
    RETURN_NAMES = ("t5gemma",)
    FUNCTION = "load"
    CATEGORY = "DaVinci-MagiHuman"

    def load(self, dtype="bf16"):
        from transformers import AutoTokenizer
        from transformers.models.t5gemma import T5GemmaEncoderModel

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = dtype_map[dtype]

        model_id = "google/t5gemma-9b-9b-ul2"
        cache_dir = os.path.join(folder_paths.models_dir, "t5gemma")
        print(f"[DaVinci] Loading T5Gemma from {model_id} -> {cache_dir} ({dtype})...")

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        model = T5GemmaEncoderModel.from_pretrained(
            model_id,
            is_encoder_decoder=False,
            dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        # Keep on CPU until needed
        model = model.to("cpu")
        model.eval()

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
        embed_dim = 3584  # T5Gemma output dimension

        if t5gemma is not None:
            # Use the dedicated T5Gemma encoder
            model = t5gemma["model"]
            tokenizer = t5gemma["tokenizer"]
            torch_dtype = t5gemma["dtype"]

            print(f"[DaVinci] Encoding prompt with T5Gemma: {prompt[:80]}...")

            # Move to GPU for encoding
            model = model.to(device)

            with torch.no_grad():
                inputs = tokenizer([prompt], return_tensors="pt").to(device)
                outputs = model(**inputs)
                embeds = outputs["last_hidden_state"].float()  # [1, seq_len, 3584]

            # Move back to CPU to free VRAM
            model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Pad or truncate to max_tokens
            if embeds.shape[1] < max_tokens:
                pad = torch.zeros(1, max_tokens - embeds.shape[1], embed_dim,
                                   device=embeds.device, dtype=embeds.dtype)
                embeds = torch.cat([embeds, pad], dim=1)
            elif embeds.shape[1] > max_tokens:
                embeds = embeds[:, :max_tokens]

            print(f"[DaVinci] T5Gemma embeddings: {embeds.shape}")
            return ({"embeds": embeds.cpu(), "prompt": prompt},)

        # No T5Gemma: create deterministic prompt-seeded embeddings
        # This produces unique embeddings per prompt (not random noise)
        print(f"[DaVinci] Generating built-in text embeddings (connect T5 encoder for best quality)")

        # Hash prompt to seed for deterministic results
        prompt_hash = hash(prompt) & 0xFFFFFFFF
        gen = torch.Generator().manual_seed(prompt_hash)

        # Simple word-level encoding: each word gets a distinct embedding
        words = prompt.lower().split()
        num_words = min(len(words), max_tokens)

        embeds = torch.zeros(1, max_tokens, embed_dim, dtype=torch.float32)
        for i, word in enumerate(words[:num_words]):
            word_hash = hash(word) & 0xFFFFFFFF
            word_gen = torch.Generator().manual_seed(word_hash)
            embeds[0, i] = torch.randn(embed_dim, generator=word_gen) * 0.5

        # Add positional signal
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
                                   "tooltip": "256p base: 448. Only used for base/distill generation."}),
                "height": ("INT", {"default": 256, "min": 64, "max": 1088, "step": 16,
                                    "tooltip": "256p base: 256. Only used for base/distill generation."}),
                "num_frames": ("INT", {"default": 125, "min": 5, "max": 250, "step": 1,
                                        "tooltip": "Number of video frames. 125 = 5 seconds at 25fps."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50, "step": 1,
                                   "tooltip": "Distill: 8 steps. Base: 32 steps."}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True,
                                               "tooltip": "Move model to CPU after sampling to free VRAM."}),
            },
            "optional": {
                "turbo_vae": ("DAVINCI_VAE", {"tooltip": "Connect TurboVAE for real-time decoded preview during sampling."}),
                "ref_image": ("IMAGE", {"tooltip": "Reference image for I2V. First frame will match this image."}),
                "vae": ("VAE", {"tooltip": "ComfyUI VAE (Wan2.2) to encode reference image. Required for I2V."}),
            }
        }

    RETURN_TYPES = ("DAVINCI_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "DaVinci-MagiHuman"

    def sample(
        self, model, text_embeds, width, height, num_frames, steps, shift, seed,
        force_offload=True, turbo_vae=None, ref_image=None, vae=None,
    ):
        dit = model["model"]
        swap_manager = model["swap_manager"]
        dtype = model["dtype"]

        # Encode reference image if provided
        latent_image = None
        if ref_image is not None and vae is not None:
            print(f"[DaVinci] Encoding reference image {ref_image.shape}...")
            # Resize image to match target video dimensions
            img = ref_image[0:1]  # [1, H, W, 3]
            if img.shape[1] != height or img.shape[2] != width:
                img = img.permute(0, 3, 1, 2)  # [1, 3, H, W]
                img = F.interpolate(img, size=(height, width), mode='bilinear', align_corners=False)
                img = img.permute(0, 2, 3, 1)  # [1, H, W, 3]
                print(f"[DaVinci] Resized ref image to {width}x{height}")
            # ComfyUI VAE.encode expects [B, H, W, C] float32 0-1
            latent_image = vae.encode(img)  # Returns [B, C, H, W]
            latent_image = latent_image.unsqueeze(2)  # [1, C, 1, latH, latW] - single frame
            latent_image = latent_image.to(torch.float32)
            print(f"[DaVinci] Encoded ref image: {latent_image.shape}")
        elif ref_image is not None:
            print("[DaVinci] WARNING: ref_image provided but no VAE connected. Ignoring ref_image.")

        # Get text embedding info
        embeds = text_embeds["embeds"]
        # Compute actual text length (non-zero tokens)
        text_len = (embeds.abs().sum(-1) > 0).sum(-1).item()
        if text_len == 0:
            text_len = 1

        # Create data proxy using reference code
        data_proxy = create_data_proxy()

        pbar = ProgressBar(steps)

        # Preview VAE: decode a single frame for real preview
        preview_vae = turbo_vae["vae"] if turbo_vae is not None else None
        preview_dtype = turbo_vae["dtype"] if turbo_vae is not None else torch.bfloat16

        def step_callback(idx, total, latent_video, latent_audio):
            preview_img = None
            if preview_vae is not None:
                try:
                    from PIL import Image
                    mid_t = latent_video.shape[2] // 2
                    frame_latent = latent_video[:, :, mid_t:mid_t+1].to(dtype=preview_dtype, device=device)
                    preview_vae.to(device)
                    with torch.no_grad():
                        frame_decoded = preview_vae.decode(frame_latent, output_offload=False).float()
                    preview_vae.to(offload_device)
                    frame_decoded = frame_decoded[0, :, 0].mul(0.5).add(0.5).clamp(0, 1)
                    frame_decoded = frame_decoded.permute(1, 2, 0)
                    rgb_uint8 = (frame_decoded * 255).to(torch.uint8).cpu().numpy()
                    img = Image.fromarray(rgb_uint8)
                    w, h = img.size
                    if max(w, h) > 256:
                        scale = 256 / max(w, h)
                        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
                    preview_img = ("JPEG", img, 256)
                except Exception as e:
                    print(f"[DaVinci] Preview error: {e}")
            pbar.update_absolute(idx + 1, total, preview_img)

        # Run the reference distill sampling loop
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
                "target_height": ("INT", {"default": 1088, "min": 256, "max": 1088, "step": 16}),
                "sr_steps": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "noise_value": ("INT", {"default": 220, "min": 0, "max": 999, "step": 1,
                                         "tooltip": "Re-noise level for SR. 220 is default."}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("DAVINCI_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "upscale"
    CATEGORY = "DaVinci-MagiHuman"

    def upscale(self, sr_model, latent, text_embeds, target_width, target_height,
                sr_steps, noise_value, shift, seed, force_offload=True):
        dit = sr_model["model"]
        swap_manager = sr_model["swap_manager"]
        dtype = sr_model["dtype"]

        torch.manual_seed(seed)

        proxy = MagiDataProxy()
        scheduler = FlowMatchingScheduler(shift=shift)

        video_latent = latent["video_latent"].to(dtype=dtype)
        num_frames = latent["num_frames"]

        # Compute target latent dimensions
        _, _, sr_t, sr_h, sr_w = proxy.get_latent_shape(target_height, target_width, num_frames)

        print(f"[DaVinci SR] Upscaling to {target_width}x{target_height}, "
              f"SR latent: {sr_t}x{sr_h}x{sr_w}")

        # Trilinear interpolation of base latent to SR size
        sr_latent = F.interpolate(
            video_latent.float(),
            size=(sr_t, sr_h, sr_w),
            mode='trilinear',
            align_corners=False,
        ).to(dtype)

        # Re-noise the interpolated latent
        sigma = scheduler.get_noise_level_sigma(noise_value)
        noise = torch.randn_like(sr_latent)
        sr_latent = scheduler.add_noise(sr_latent, noise, sigma)

        # Patchify
        video_tokens, video_coords = proxy.patchify_video(sr_latent.to(device))

        # Audio (re-noise audio too)
        audio_tokens = latent["audio_tokens"].to(device=device, dtype=dtype)
        audio_noise = torch.randn_like(audio_tokens)
        sr_audio_noise_scale = 0.7
        audio_tokens = audio_noise * sr_audio_noise_scale + audio_tokens * (1 - sr_audio_noise_scale)
        audio_coords = torch.zeros(1, audio_tokens.shape[1], 3, device=device, dtype=dtype)
        for i in range(audio_tokens.shape[1]):
            audio_coords[:, i, 0] = i

        # Text
        text_tokens = text_embeds["embeds"].to(device=device, dtype=dtype)
        text_coords = proxy.prepare_text_coords(text_tokens)

        # Build sequence
        seq_data = proxy.build_sequence(
            video_tokens, video_coords,
            audio_tokens, audio_coords,
            text_tokens, text_coords,
        )

        # SR scheduler
        scheduler.set_timesteps(sr_steps)

        # Keep modalities separate (different channel dims)
        sv = seq_data["video_shape"][0]
        sa = seq_data["audio_shape"][0]
        x_video = video_tokens  # [B, sv, 192]
        x_audio = audio_tokens  # [B, sa, 64]
        x_text = text_tokens    # [B, st, 3584]

        coords = torch.cat([
            seq_data["video_coords"],
            seq_data["audio_coords"],
            seq_data["text_coords"],
        ], dim=1)

        pbar = ProgressBar(sr_steps)

        for i in range(sr_steps):
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                rope_cos, rope_sin = dit.rope(coords)

                v_emb = dit.video_embedder(x_video)
                a_emb = dit.audio_embedder(x_audio)
                t_emb = dit.text_embedder(x_text)
                h = torch.cat([v_emb, a_emb, t_emb], dim=1)

                h = swap_manager.forward_with_swap(
                    h, rope_cos, rope_sin,
                    seq_data["modality_ids"],
                )

                video_vel = dit.final_linear_video(_rms_norm(h[:, :sv], dit.final_norm_video))
                audio_vel = dit.final_linear_audio(_rms_norm(h[:, sv:sv + sa], dit.final_norm_audio))

            x_video = scheduler.step_ddim(video_vel, i, x_video)
            x_audio = scheduler.step_ddim(audio_vel, i, x_audio)

            pbar.update(1)

        # Unpatchify
        final_video_tokens = x_video.cpu()
        final_audio_tokens = x_audio.cpu()
        video_latent_sr = proxy.unpatchify_video(final_video_tokens, sr_t, sr_h, sr_w)

        if force_offload:
            swap_manager.cleanup()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(f"[DaVinci SR] Done. Output latent: {video_latent_sr.shape}")

        return ({
            "video_latent": video_latent_sr,
            "audio_tokens": final_audio_tokens,
            "width": target_width,
            "height": target_height,
            "num_frames": num_frames,
        },)


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

        # Move VAE to GPU
        vae = vae.to(device)

        pbar = ProgressBar(1)

        with torch.no_grad():
            # Reference TurboVAED.decode handles normalization internally
            video = vae.decode(video_latent.to(device), output_offload=output_offload).float()

        pbar.update(1)

        # Move VAE back to CPU
        vae.to(offload_device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Reference output is [-1, 1], convert to [0, 1] (matching reference post_process)
        video = video.cpu()
        video.mul_(0.5).add_(0.5).clamp_(0, 1)

        # [B, 3, T, H, W] -> [T, H, W, 3]
        video = video[0].permute(1, 2, 3, 0).contiguous()

        print(f"[DaVinci] Decoded {video.shape[0]} frames at {video.shape[1]}x{video.shape[2]}")

        return (video,)


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

        # frames: [T, H, W, 3] float32 0-1
        T, H, W, C = frames.shape

        print(f"[DaVinci] Saving {T} frames ({W}x{H}) at {fps}fps to {output_path}")

        if format == "mp4":
            codec = "libx264"
            pix_fmt = "yuv420p"
        else:
            codec = "libvpx-vp9"
            pix_fmt = "yuv420p"

        # If audio provided, save audio to temp file then mux
        audio_tmp = None
        if audio is not None and "waveform" in audio:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            if waveform.numel() > 1:
                audio_tmp = os.path.join(full_output_folder, f"{filename}_{counter:05d}_audio.wav")
                # [B, channels, samples] -> [samples, channels]
                wav_np = waveform[0].permute(1, 0).numpy()
                sf.write(audio_tmp, wav_np, sample_rate)
                print(f"[DaVinci] Audio saved to temp: {audio_tmp}")

        try:
            if audio_tmp:
                # Video + audio mux
                video_tmp = os.path.join(full_output_folder, f"{filename}_{counter:05d}_video_tmp.{format}")
                # First encode video
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

                # Mux video + audio
                cmd_mux = [
                    "ffmpeg", "-y",
                    "-i", video_tmp, "-i", audio_tmp,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", output_path,
                ]
                subprocess.run(cmd_mux, stderr=subprocess.PIPE)
                # Cleanup temps
                os.remove(video_tmp)
                os.remove(audio_tmp)
                print(f"[DaVinci] Video+audio saved: {output_path}")
            else:
                # Video only
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


class DaVinciAudioVAELoader:
    """Load Stable Audio Open 1.0 VAE for audio decoding."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("DAVINCI_AUDIO_VAE",)
    RETURN_NAMES = ("audio_vae",)
    FUNCTION = "load"
    CATEGORY = "DaVinci-MagiHuman"

    def load(self):
        import sys
        # Add reference code to path for sa_audio_module
        ref_path = os.path.join(os.path.dirname(__file__), "davinci_ref", "inference", "model", "sa_audio")
        if ref_path not in sys.path:
            sys.path.insert(0, ref_path)
        from sa_audio_module import create_model_from_config

        model_id = "stabilityai/stable-audio-open-1.0"
        cache_dir = os.path.join(folder_paths.models_dir, "stable_audio")

        print(f"[DaVinci] Loading Stable Audio VAE from {model_id}...")

        # Download model files
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_id, "model_config.json", cache_dir=cache_dir)
        weights_path = hf_hub_download(model_id, "model.safetensors", cache_dir=cache_dir)

        # Load config and build VAE only
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

        # Load only VAE weights from full checkpoint
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

        audio_tokens = latent["audio_tokens"]  # [B, num_frames, 64]
        if audio_tokens is None:
            print("[DaVinci] No audio tokens to decode.")
            return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": sample_rate},)

        print(f"[DaVinci] Decoding audio latents {audio_tokens.shape}...")

        # Audio latent: [B, T, 64] -> [B, 64, T] for 1D conv decoder
        audio_latent = audio_tokens.float().permute(0, 2, 1)

        vae = vae.to(device)

        with torch.no_grad():
            waveform = vae.decode(audio_latent.to(device))  # [B, channels, samples]

        vae.to(offload_device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        waveform = waveform.cpu().float()
        print(f"[DaVinci] Audio decoded: {waveform.shape}, sample_rate={sample_rate}")

        # ComfyUI AUDIO format: dict with waveform [B, channels, samples] and sample_rate
        return ({"waveform": waveform, "sample_rate": sample_rate},)


# Node registration
NODE_CLASS_MAPPINGS = {
    "DaVinciModelLoader": DaVinciModelLoader,
    "DaVinciTurboVAELoader": DaVinciTurboVAELoader,
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
    "DaVinciT5GemmaLoader": "DaVinci T5Gemma Loader",
    "DaVinciTextEncode": "DaVinci Text Encode",
    "DaVinciSampler": "DaVinci Sampler",
    "DaVinciSuperResolution": "DaVinci Super Resolution",
    "DaVinciDecode": "DaVinci Decode (TurboVAE)",
    "DaVinciAudioVAELoader": "DaVinci Audio VAE Loader",
    "DaVinciAudioDecode": "DaVinci Audio Decode",
    "DaVinciVideoOutput": "DaVinci Video Output",
}
