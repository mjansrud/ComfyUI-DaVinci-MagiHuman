"""
Latent preview for daVinci-MagiHuman.
Projects 48-channel latent to RGB using a simple linear transform.
No extra model needed — uses approximate learned factors.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import io

MAX_PREVIEW_SIZE = 256


def latent_to_rgb(latent_frame):
    """Convert a single latent frame [C, H, W] to RGB image [H, W, 3].

    Uses a random but deterministic projection from 48 latent channels to RGB.
    This gives a rough but consistent visual preview during denoising.
    """
    C, H, W = latent_frame.shape

    # Deterministic projection matrix (seeded for consistency across steps)
    gen = torch.Generator(device=latent_frame.device).manual_seed(42)
    proj = torch.randn(3, C, generator=gen, device=latent_frame.device, dtype=latent_frame.dtype) * 0.1

    # [C, H, W] -> [H, W, C]
    x = latent_frame.permute(1, 2, 0).float()
    # [H, W, C] @ [C, 3] -> [H, W, 3]
    rgb = F.linear(x, proj)

    # Normalize to 0-1
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return rgb


def send_preview(x_video, proxy, latent_t, latent_h, latent_w, step, total_steps):
    """Generate a preview image from current video tokens.

    Args:
        x_video: [B, num_tokens, 192] current video tokens
        proxy: MagiDataProxy for unpatchification
        latent_t, latent_h, latent_w: latent dimensions
        step: current step index
        total_steps: total number of steps

    Returns:
        ("JPEG", PIL.Image, max_size) tuple for ComfyUI preview, or None
    """
    try:
        with torch.no_grad():
            # Unpatchify to latent volume: [B, 48, T, H, W]
            latent = proxy.unpatchify_video(x_video.cpu().float(), latent_t, latent_h, latent_w)

            # Take middle frame for preview
            mid_t = latent.shape[2] // 2
            frame = latent[0, :, mid_t]  # [48, H, W]

            # Project to RGB
            rgb = latent_to_rgb(frame)  # [H, W, 3]

            # Convert to uint8
            rgb_uint8 = (rgb.clamp(0, 1) * 255).to(torch.uint8).numpy()

            # Resize if too large
            img = Image.fromarray(rgb_uint8)
            w, h = img.size
            if max(w, h) > MAX_PREVIEW_SIZE:
                scale = MAX_PREVIEW_SIZE / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

            return ("JPEG", img, MAX_PREVIEW_SIZE)
    except Exception:
        return None
