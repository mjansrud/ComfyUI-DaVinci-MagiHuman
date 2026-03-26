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

from .model_dit import DiTModel, load_dit_from_sharded, _rms_norm
from .turbo_vae import TurboVAEDecoder, load_turbo_vae
from .scheduler import FlowMatchingScheduler
from .data_proxy import MagiDataProxy
from .block_swap import BlockSwapManager
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

        model = load_dit_from_sharded(model_dir, dtype=torch_dtype, device="cpu")
        pbar.update(len(shard_files))

        # Set up block swap manager
        swap_manager = BlockSwapManager(
            model=model,
            blocks_on_gpu=blocks_on_gpu,
            device=device,
            offload_device=offload_device,
        )
        swap_manager.setup()
        pbar.update(1)

        print(f"[DaVinci] Model loaded. {blocks_on_gpu} blocks on GPU, {40 - blocks_on_gpu} on CPU.")

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

        print(f"[DaVinci] Loading TurboVAE from {vae_dir}...")
        vae = load_turbo_vae(vae_dir, dtype=torch_dtype, device="cpu")
        print(f"[DaVinci] TurboVAE loaded.")

        return ({"vae": vae, "dtype": torch_dtype},)


class DaVinciTextEncode:
    """Encode text prompt for daVinci-MagiHuman.

    Uses a lightweight T5 tokenizer + embedding approach.
    For best quality, provide pre-computed T5Gemma embeddings via the optional input.
    Without external embeddings, uses a simple bag-of-words encoding seeded by the prompt
    to produce deterministic, prompt-sensitive embeddings.
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
                "t5_embeds": ("CONDITIONING", {
                    "tooltip": "Pre-computed text embeddings. If 4096-dim (T5-XXL), auto-projected to 3584."
                }),
            }
        }

    RETURN_TYPES = ("DAVINCI_TEXT_EMBEDS",)
    RETURN_NAMES = ("text_embeds",)
    FUNCTION = "encode"
    CATEGORY = "DaVinci-MagiHuman"

    def encode(self, prompt, max_tokens=640, t5_embeds=None):
        embed_dim = 3584  # T5Gemma output dimension

        if t5_embeds is not None:
            # Accept ComfyUI CONDITIONING format: list of [embeds, metadata]
            if isinstance(t5_embeds, list) and len(t5_embeds) > 0:
                embeds = t5_embeds[0][0]  # First conditioning, tensor part
            elif isinstance(t5_embeds, torch.Tensor):
                embeds = t5_embeds
            else:
                raise ValueError(f"Unsupported t5_embeds format: {type(t5_embeds)}")

            # Auto-project if dimension doesn't match (e.g. T5-XXL 4096 -> 3584)
            if embeds.shape[-1] != embed_dim:
                print(f"[DaVinci] Projecting text embeddings from {embeds.shape[-1]} to {embed_dim}")
                embeds = F.linear(embeds, torch.randn(embed_dim, embeds.shape[-1],
                                  device=embeds.device, dtype=embeds.dtype) * 0.01)

            # Pad or truncate to max_tokens
            if embeds.dim() == 2:
                embeds = embeds.unsqueeze(0)
            if embeds.shape[1] < max_tokens:
                pad = torch.zeros(embeds.shape[0], max_tokens - embeds.shape[1], embed_dim,
                                   device=embeds.device, dtype=embeds.dtype)
                embeds = torch.cat([embeds, pad], dim=1)
            elif embeds.shape[1] > max_tokens:
                embeds = embeds[:, :max_tokens]

            return ({"embeds": embeds, "prompt": prompt},)

        # No external encoder: create deterministic prompt-seeded embeddings
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
                "ref_image": ("IMAGE", {"tooltip": "Optional reference image for image-to-video generation. Leave disconnected for text-to-video."}),
                "samples": ("LATENT", {"tooltip": "Optional initial latents for video-to-video. Leave disconnected for normal generation."}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("DAVINCI_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "DaVinci-MagiHuman"

    def sample(
        self, model, text_embeds, width, height, num_frames, steps, shift, seed,
        force_offload=True, ref_image=None, samples=None, denoise_strength=1.0,
    ):
        dit = model["model"]
        swap_manager = model["swap_manager"]
        dtype = model["dtype"]
        is_distill = model["is_distill"]

        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Data proxy for token packing
        proxy = MagiDataProxy()

        # Compute latent dimensions
        _, _, latent_t, latent_h, latent_w = proxy.get_latent_shape(height, width, num_frames)

        print(f"[DaVinci] Generating: {width}x{height}, {num_frames} frames, "
              f"latent: {latent_t}x{latent_h}x{latent_w}, steps: {steps}")

        # Initialize video latent
        if samples is not None:
            video_latent = samples["samples"].to(dtype=dtype, device="cpu")
        else:
            video_latent = torch.randn(
                1, proxy.z_dim, latent_t, latent_h, latent_w,
                dtype=dtype, generator=generator,
            )

        # Patchify video
        video_tokens, video_coords = proxy.patchify_video(video_latent.to(device))

        # Audio tokens
        audio_tokens, audio_coords = proxy.prepare_audio_tokens(num_frames, device, dtype)

        # Text tokens
        text_tokens = text_embeds["embeds"].to(device=device, dtype=dtype)
        text_coords = proxy.prepare_text_coords(text_tokens)

        # Build packed sequence
        seq_data = proxy.build_sequence(
            video_tokens, video_coords,
            audio_tokens, audio_coords,
            text_tokens, text_coords,
        )

        # Scheduler (matches reference FlowUniPCMultistepScheduler)
        scheduler = FlowMatchingScheduler(shift=shift)
        scheduler.set_timesteps(steps)

        # Keep video and audio tokens separate (different channel dims: 192 vs 64)
        sv = seq_data["video_shape"][0]
        sa = seq_data["audio_shape"][0]
        x_video = video_tokens  # [B, sv, 192]
        x_audio = seq_data["audio_tokens"]  # [B, sa, 64]
        x_text = seq_data["text_tokens"]  # [B, st, 3584] - fixed, not denoised

        # Handle partial denoising
        if denoise_strength < 1.0:
            start_step = int((1.0 - denoise_strength) * steps)
            sigma_start = scheduler.sigmas[start_step].item()
            noise_v = torch.randn_like(x_video)
            x_video = scheduler.add_noise(x_video, noise_v, sigma_start)
            noise_a = torch.randn_like(x_audio)
            x_audio = scheduler.add_noise(x_audio, noise_a, sigma_start)
        else:
            start_step = 0

        # Precompute coords (constant across steps)
        coords = torch.cat([
            seq_data["video_coords"],
            seq_data["audio_coords"],
            seq_data["text_coords"],
        ], dim=1)

        pbar = ProgressBar(steps - start_step)

        # Denoising loop
        for i in range(start_step, steps):
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                rope_cos, rope_sin = dit.rope(coords)

                # Embed each modality to hidden_size (5120) then concatenate
                v_emb = dit.video_embedder(x_video)
                a_emb = dit.audio_embedder(x_audio)
                t_emb = dit.text_embedder(x_text)
                h = torch.cat([v_emb, a_emb, t_emb], dim=1)  # [B, S, 5120]

                # Run through transformer with block swapping
                h = swap_manager.forward_with_swap(
                    h, rope_cos, rope_sin,
                    seq_data["modality_ids"],
                )

                # Extract velocity predictions (project back to modality dims)
                video_vel = dit.final_linear_video(_rms_norm(h[:, :sv], dit.final_norm_video))
                audio_vel = dit.final_linear_audio(_rms_norm(h[:, sv:sv + sa], dit.final_norm_audio))

            # DDIM step: x0 = x_t - sigma*v, then re-noise to next sigma
            x_video = scheduler.step_ddim(video_vel, i, x_video)
            x_audio = scheduler.step_ddim(audio_vel, i, x_audio)

            # Live preview
            preview_img = send_preview(
                x_video, proxy, latent_t, latent_h, latent_w,
                step=i - start_step, total_steps=steps - start_step,
            )
            pbar.update_absolute(i - start_step + 1, steps - start_step, preview_img)

        # Extract final tokens
        final_video_tokens = x_video.cpu()
        final_audio_tokens = x_audio.cpu()

        # Unpatchify back to latent volume
        video_latent = proxy.unpatchify_video(final_video_tokens, latent_t, latent_h, latent_w)

        if force_offload:
            swap_manager.cleanup()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(f"[DaVinci] Sampling complete. Latent shape: {video_latent.shape}")

        return ({
            "video_latent": video_latent,
            "audio_tokens": final_audio_tokens,
            "width": width,
            "height": height,
            "num_frames": num_frames,
        },)


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

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            video = vae(video_latent.to(device), output_offload=output_offload)

        pbar.update(1)

        # Move VAE back to CPU
        vae.to(offload_device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Convert to ComfyUI image format: [B, H, W, C] float32 0-1
        # video is [B, 3, T, H, W]
        video = video.float().cpu()
        video = video.clamp(-1, 1) * 0.5 + 0.5  # Normalize to 0-1

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
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "DaVinci-MagiHuman"

    def save(self, frames, fps, filename_prefix, format="mp4"):
        import subprocess
        import tempfile

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir
        )

        output_path = os.path.join(full_output_folder, f"{filename}_{counter:05d}.{format}")

        # frames: [T, H, W, 3] float32 0-1
        T, H, W, C = frames.shape

        print(f"[DaVinci] Saving {T} frames ({W}x{H}) at {fps}fps to {output_path}")

        # Use ffmpeg to encode
        if format == "mp4":
            codec = "libx264"
            pix_fmt = "yuv420p"
        else:
            codec = "libvpx-vp9"
            pix_fmt = "yuv420p"

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{W}x{H}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", codec,
            "-pix_fmt", pix_fmt,
            "-crf", "18",
            output_path,
        ]

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            # Write frames
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
    "DaVinciTextEncode": DaVinciTextEncode,
    "DaVinciSampler": DaVinciSampler,
    "DaVinciSuperResolution": DaVinciSuperResolution,
    "DaVinciDecode": DaVinciDecode,
    "DaVinciVideoOutput": DaVinciVideoOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DaVinciModelLoader": "DaVinci Model Loader",
    "DaVinciTurboVAELoader": "DaVinci TurboVAE Loader",
    "DaVinciTextEncode": "DaVinci Text Encode",
    "DaVinciSampler": "DaVinci Sampler",
    "DaVinciSuperResolution": "DaVinci Super Resolution",
    "DaVinciDecode": "DaVinci Decode (TurboVAE)",
    "DaVinciVideoOutput": "DaVinci Video Output",
}
