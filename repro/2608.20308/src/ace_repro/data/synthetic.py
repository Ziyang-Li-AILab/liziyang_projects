"""Procedural clips for the smoke tiers (stage 6) — no licensed dataset needed.

Random smooth two-hand trajectories are posed with the hand model, projected
with a random pinhole camera, rendered as a colour-coded skeleton video and
encoded with the real Wan VAE. Everything the loss / metric stack needs is
therefore self-consistent; the only thing that is *not* real is the imagery.
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch

from ace_ego_hand.mano_utils import mano_forward_batch_full

from ace_repro.camera import project
from ace_repro.data.latents import LatentEncoder, cached
from ace_repro.data.schema import Clip, visibility_gate

OP_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
            (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
            (18, 19), (19, 20)]


def _smooth(g: torch.Generator, T: int, dim: int, amp: float, base_amp: float = 0.0) -> torch.Tensor:
    """Sum of two random sinusoids per channel -> (T, dim), amplitude ``amp``."""
    t = torch.linspace(0, 1, T)[:, None]
    f1 = torch.rand(1, dim, generator=g) * 1.5 + 0.25
    f2 = torch.rand(1, dim, generator=g) * 3.0 + 0.5
    p1, p2 = torch.rand(2, 1, dim, generator=g) * 2 * math.pi
    base = (torch.rand(1, dim, generator=g) * 2 - 1) * base_amp
    return base + amp * (0.7 * torch.sin(2 * math.pi * f1 * t + p1) + 0.3 * torch.sin(2 * math.pi * f2 * t + p2))


def _aa_to_mat(aa: torch.Tensor) -> torch.Tensor:
    from ace_repro.hand_model import axis_angle_to_matrix
    return axis_angle_to_matrix(aa)


def _render(joints2d_px: np.ndarray, vis: np.ndarray, depth: np.ndarray, fx: float, W: int, H: int):
    """joints2d_px (2,21,2), vis (2,21), depth (2,21) -> (H, W, 3) uint8 RGB frame."""
    img = np.zeros((H, W, 3), np.uint8)
    for s in range(2):
        if not vis[s].any():
            continue
        pts = np.clip(joints2d_px[s], -4 * max(W, H), 4 * max(W, H)).astype(int)
        col_base = np.array([230, 120, 40]) if s == 0 else np.array([40, 140, 230])
        for a, b in OP_BONES:
            if vis[s, a] and vis[s, b]:
                cv2.line(img, (int(pts[a, 0]), int(pts[a, 1])), (int(pts[b, 0]), int(pts[b, 1])),
                         tuple(int(c) for c in col_base * 0.6), 2, cv2.LINE_AA)
        for j in range(21):
            if vis[s, j]:
                r = int(np.clip(fx * 0.006 / max(depth[s, j], 0.05), 2, 14))
                col = col_base * (0.5 + 0.5 * j / 20)
                cv2.circle(img, (int(pts[j, 0]), int(pts[j, 1])), r, tuple(int(c) for c in col), -1, cv2.LINE_AA)
    return img


def make_clip_gt(hand_models, g: torch.Generator, T: int, W: int, H: int, clip_id: str) -> dict:
    """Sample one clip's GT (all tensors on cpu)."""
    dev = next(hand_models[True].buffers()).device
    f = float(W) * (0.8 + 0.4 * torch.rand(1, generator=g).item())
    K = torch.tensor([f, f, W / 2, H / 2, W, H], dtype=torch.float32)
    exists_hand = torch.rand(2, generator=g) < 0.85
    if not exists_hand.any():
        exists_hand[1] = True
    betas = torch.randn(2, 10, generator=g) * 0.5
    go = torch.zeros(T, 2, 3, 3)
    hp = torch.zeros(T, 2, 15, 3, 3)
    trans = torch.zeros(T, 2, 3)
    joints = torch.zeros(T, 2, 21, 3)
    for s in range(2):
        go_aa = _smooth(g, T, 3, 0.35, base_amp=math.pi * 0.8)
        hp_aa = _smooth(g, T, 45, 0.25, base_amp=0.35)
        centre = torch.tensor([(torch.rand(1, generator=g).item() - 0.5) * 0.3 + (0.08 if s else -0.08),
                               (torch.rand(1, generator=g).item() - 0.5) * 0.2,
                               0.35 + 0.35 * torch.rand(1, generator=g).item()])
        tr = centre + _smooth(g, T, 3, 1.0) * torch.tensor([0.10, 0.06, 0.08])
        if torch.rand(1, generator=g).item() < 0.35:                 # out-of-sight excursion
            tr[:, 0] += torch.linspace(0, 0.5, T) * (1 if s else -1)
        go_m = _aa_to_mat(go_aa)
        hp_m = _aa_to_mat(hp_aa.reshape(T, 15, 3))
        with torch.no_grad():
            j21, _ = mano_forward_batch_full(go_m.to(dev), hp_m.to(dev),
                                             betas[s].expand(T, 10).to(dev), hand_models[bool(s)])
        go[:, s], hp[:, s], trans[:, s] = go_m, hp_m, tr
        joints[:, s] = j21.cpu() + tr[:, None]
    joints2d = project(joints.reshape(1, -1, 3), K[None]).reshape(T, 2, 21, 2)
    joint_vis, visible = visibility_gate(joints, joints2d)
    exists = exists_hand[None].expand(T, 2).clone()
    joint_vis &= exists[..., None]
    visible &= exists
    return dict(clip_id=clip_id, K=K, go=go, hp=hp, betas=betas, trans=trans, joints_cam=joints,
                joints2d=joints2d, joint_vis=joint_vis, has_mano=exists.clone(), exists=exists,
                visible=visible)


def render_clip(gt: dict, W: int, H: int) -> np.ndarray:
    T = gt["go"].shape[0]
    px = (gt["joints2d"] * torch.tensor([W, H])).numpy()
    vis = gt["joint_vis"].numpy()
    depth = gt["joints_cam"][..., 2].numpy()
    return np.stack([_render(px[t], vis[t], depth[t], float(gt["K"][0]), W, H) for t in range(T)])


class SyntheticClips(torch.utils.data.Dataset):
    """``n`` procedural clips, rendered + VAE-encoded once and cached on disk."""

    def __init__(self, hand_models, encoder: LatentEncoder | None, n: int, res=(480, 480),
                 video_frames: int = 21, seed: int = 0, cache_dir: str | None = None,
                 name: str = "synthetic"):
        self.W, self.H = res
        self.T = video_frames
        self.name = name
        g = torch.Generator().manual_seed(seed)
        self.clips: list[Clip] = []
        for i in range(n):
            gt = make_clip_gt(hand_models, g, self.T, self.W, self.H, f"{name}_{seed}_{i:04d}")

            def _enc(gt=gt):
                if encoder is None:
                    raise RuntimeError("no cached latent and no VAE encoder given")
                return encoder.encode_uint8(render_clip(gt, self.W, self.H))
            path = os.path.join(cache_dir or "/tmp/ace_repro_latents", name,
                                f"{gt['clip_id']}_{self.W}x{self.H}_T{self.T}.pt")
            lat = cached(path, _enc)
            self.clips.append(Clip(dataset=name, latent=lat, n_video_frames=self.T, is_static=False,
                                   camera="pinhole", **gt))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return self.clips[i]
