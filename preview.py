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

    Uses the mean of latent channel groups mapped to R/G/B.
    Channel 0-15 -> R, 16-31 -> G, 32-47 -> B.
    """
    C, H, W = latent_frame.shape
    latent_frame = latent_frame.float()

    # Split 48 channels into 3 groups of 16, average each for R/G/B
    group_size = C // 3
    r = latent_frame[:group_size].mean(dim=0)
    g = latent_frame[group_size:2*group_size].mean(dim=0)
    b = latent_frame[2*group_size:3*group_size].mean(dim=0)

    rgb = torch.stack([r, g, b], dim=-1)  # [H, W, 3]

    # Normalize per-frame to 0-1
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
