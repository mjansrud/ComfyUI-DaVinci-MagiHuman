"""
Flow-matching scheduler for daVinci-MagiHuman.
Matches the reference FlowUniPCMultistepScheduler for distill mode (step_ddim).

Flow matching formulation:
    x_t = (1 - sigma) * x_0 + sigma * noise
    Model predicts velocity v where: x_0 = x_t - sigma * v

DDIM step (distill mode):
    x0_pred = x_t - sigma_t * velocity
    x_{t-1} = sigma_{t-1} * fresh_noise + (1 - sigma_{t-1}) * x0_pred
"""

import torch
import numpy as np
from typing import Optional


class FlowMatchingScheduler:
    """Flow matching scheduler matching the reference implementation."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        # Initialize base sigmas (same as reference __init__)
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        base_sigmas = 1.0 - alphas
        # Apply shift once at init
        base_sigmas = shift * base_sigmas / (1 + (shift - 1) * base_sigmas)
        self.sigma_max = base_sigmas[0]
        self.sigma_min = base_sigmas[-1]

    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """Compute sigma schedule for inference. Matches reference set_timesteps."""
        # Linspace from sigma_max to sigma_min (already shifted once)
        sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1)[:-1]

        # Apply shift again (reference does this in set_timesteps)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        # Append 0 as final sigma
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        if device is not None:
            self.sigmas = self.sigmas.to(device)

        self.num_inference_steps = num_inference_steps
        return self.sigmas

    def step_ddim(
        self,
        velocity: torch.Tensor,
        step_index: int,
        curr_state: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """DDIM step for flow matching (distill mode).

        Reference implementation:
            curr_t = self.sigmas[t]
            prev_t = self.sigmas[t + 1]
            variance_noise = randn_tensor(...)
            cur_clean_ = curr_state - curr_t * velocity
            prev_state = prev_t * variance_noise + (1 - prev_t) * cur_clean_
        """
        curr_t = self.sigmas[step_index]
        prev_t = self.sigmas[step_index + 1]

        # Predict clean sample
        x0_pred = curr_state - curr_t * velocity

        # Re-noise with fresh random noise to next sigma level
        if prev_t > 0:
            noise = torch.randn_like(curr_state, generator=generator)
            prev_state = prev_t * noise + (1 - prev_t) * x0_pred
        else:
            # Final step: no noise
            prev_state = x0_pred

        return prev_state

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Add noise: x_t = (1 - sigma) * x_0 + sigma * noise"""
        return (1 - sigma) * original + sigma * noise

    def get_noise_level_sigma(self, noise_value: int) -> float:
        """Convert noise_value index to sigma for SR re-noising."""
        t = noise_value / self.num_train_timesteps
        return self.shift * t / (1 + (self.shift - 1) * t)
