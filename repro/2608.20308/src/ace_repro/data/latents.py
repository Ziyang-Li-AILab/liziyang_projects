"""VAE latent precompute (gaps_filled G/J: latents are computed once per recording).

Thin wrapper over the released ``ace_ego_hand.video_vae`` so training and
inference share one encoder path (clean latent, ``dist.mode()``, bf16 VAE).
"""
from __future__ import annotations

import os

import torch

from ace_ego_hand import video_vae


class LatentEncoder:
    def __init__(self, device: torch.device):
        self.device = device
        self.vae = video_vae.load_vae(device)

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """(T, 3, H, W) float in [-1, 1] -> (48, F_lat, H/16, W/16) float16 (cpu)."""
        return video_vae.encode(self.vae, frames, self.device)

    @torch.no_grad()
    def encode_uint8(self, frames_u8) -> torch.Tensor:
        """(T, H, W, 3) uint8 RGB numpy -> latent."""
        x = torch.from_numpy(frames_u8).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        return self.encode_frames(x)


def cached(path: str, compute):
    """Load ``path`` if present, else compute, save and return."""
    if os.path.isfile(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    obj = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)
    return obj
