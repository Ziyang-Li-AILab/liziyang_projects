"""On-disk clip store: one ``.pt`` per recording/image in the shared format.

Converters (``ace_repro.data.converters.*``) write recording-level ``Clip``
objects (full-length latent + per-frame GT). At train time a random
21-latent-frame window (81 RGB frames) is drawn per recording (gaps_filled G);
static sources are returned as-is (5-frame clips).

Latent <-> frame alignment of the causal Wan VAE (4x temporal): latent 0 covers
frame 0, latent l >= 1 covers frames 4l-3 .. 4l. A window of latents [a, a+21)
therefore covers RGB frames [max(0, 4a-3), 4a+78) for a >= 1 and [0, 81) for a = 0.
"""
from __future__ import annotations

import glob
import os
import random

import torch
from torch.utils.data import Dataset

from ace_repro.data.schema import Clip, _TENSOR_FIELDS

LATENT_WINDOW = 21
VIDEO_WINDOW = 81


def window_frames(a: int, n_frames: int) -> tuple[int, int]:
    f0 = 0 if a == 0 else 4 * a - 3
    return f0, min(f0 + VIDEO_WINDOW, n_frames)


def save_clip(clip: Clip, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(clip.__dict__, path)


def load_clip(path: str) -> Clip:
    return Clip(**torch.load(path, map_location="cpu", weights_only=False))


class ClipStore(Dataset):
    def __init__(self, root: str, window: bool = True, seed: int = 0):
        self.paths = sorted(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True))
        self.window = window
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int) -> Clip:
        clip = load_clip(self.paths[i])
        if clip.is_static or not self.window or clip.latent.shape[1] <= LATENT_WINDOW:
            return clip
        F = clip.latent.shape[1]
        a = self.rng.randint(0, F - LATENT_WINDOW)
        f0, f1 = window_frames(a, clip.n_video_frames)
        d = dict(clip.__dict__)
        d["latent"] = clip.latent[:, a:a + LATENT_WINDOW].contiguous()
        for k in _TENSOR_FIELDS:
            if k in ("latent", "K", "betas"):
                continue
            d[k] = getattr(clip, k)[f0:f1].contiguous()
        d["n_video_frames"] = f1 - f0
        d["clip_id"] = f"{clip.clip_id}@{f0}"
        return Clip(**d)
