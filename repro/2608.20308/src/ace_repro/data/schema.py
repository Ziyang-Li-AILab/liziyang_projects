"""Shared clip format (gaps_filled G "Shared MANO format") and batch collation.

Every data source — real converters and the synthetic smoke source — emits
``Clip`` objects in this format. Conventions:

* frame count ``T`` = RGB frames of the clip (81 for video, 5 for static);
* slot 0 = left hand, slot 1 = right hand (released projector convention);
* MANO = smplx MANO with ``flat_hand_mean=False``; ``joints_cam = MANO(go,hp,betas).joints21 + trans``;
* 2D coordinates are normalised to [0, 1] by the encode resolution ``(W, H)``;
* ``K = [fx, fy, cx, cy, W, H]`` in encode pixels;
* ``exists`` = the hand has GT this frame (out-of-sight hands included, §3.4);
  ``visible`` = App. A.1 on-screen gate; ``has_mano`` = rotation/shape GT available
  (False for RHD, whose rotation and shape terms are held at zero).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import torch

N_JOINTS_OP = 21
N_SLOTS = 2


@dataclass
class Clip:
    dataset: str
    clip_id: str
    latent: torch.Tensor          # (48, F_lat, Hl, Wl) float16 — VAE latent of the clip
    n_video_frames: int           # T
    K: torch.Tensor               # (6,) float32
    go: torch.Tensor              # (T, 2, 3, 3)
    hp: torch.Tensor              # (T, 2, 15, 3, 3)
    betas: torch.Tensor           # (2, 10)
    trans: torch.Tensor           # (T, 2, 3)
    joints_cam: torch.Tensor      # (T, 2, 21, 3) metres, camera frame
    joints2d: torch.Tensor        # (T, 2, 21, 2) normalised [0,1]
    joint_vis: torch.Tensor       # (T, 2, 21) bool
    has_mano: torch.Tensor        # (T, 2) bool
    exists: torch.Tensor          # (T, 2) bool
    visible: torch.Tensor         # (T, 2) bool
    is_static: bool = False
    camera: str = "pinhole"       # "pinhole" | "fisheye" (fisheye -> no pinhole fit, ray-mode decode)
    meta: dict = field(default_factory=dict)


_TENSOR_FIELDS = [f.name for f in fields(Clip) if f.type == "torch.Tensor"]


def visibility_gate(joints_cam: torch.Tensor, joints2d: torch.Tensor, z_min: float = 0.01):
    """App. A.1 on-screen gate. Returns ``(joint_vis (..., 21), visible (...))``."""
    inside = (joints2d[..., 0] >= 0) & (joints2d[..., 0] < 1) & (joints2d[..., 1] >= 0) & (joints2d[..., 1] < 1)
    joint_vis = inside & (joints_cam[..., 2] > z_min)
    return joint_vis, joint_vis.any(-1)


def collate(clips: list[Clip]) -> dict:
    """Stack a *homogeneous* list of clips (same dataset, same shapes) into a batch dict."""
    assert len({c.dataset for c in clips}) == 1, "batches must be single-source (App. A.2)"
    assert len({c.n_video_frames for c in clips}) == 1
    out = {k: torch.stack([getattr(c, k) for c in clips]) for k in _TENSOR_FIELDS}
    out["dataset"] = clips[0].dataset
    out["clip_ids"] = [c.clip_id for c in clips]
    out["n_video_frames"] = clips[0].n_video_frames
    out["is_static"] = clips[0].is_static
    out["camera"] = clips[0].camera
    return out


def batch_to(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
