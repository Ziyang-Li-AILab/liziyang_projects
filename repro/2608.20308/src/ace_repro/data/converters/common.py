"""Shared conversion helpers: MANO convention, resizing, K rescale, VAE encode."""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from ace_ego_hand.mano_utils import mano_forward_batch_full

from ace_repro.camera import project
from ace_repro.data.latents import LatentEncoder
from ace_repro.data.schema import Clip, visibility_gate
from ace_repro.hand_model import axis_angle_to_matrix

# smplx MANO hands_mean is added to the articulation when flat_hand_mean=False.
# Sources fitted with flat_hand_mean=True (OakInk2 / manotorch) store
# articulation = theta + hands_mean; converting means subtracting hands_mean.
def hands_mean(hand_models, is_right: bool) -> torch.Tensor | None:
    m = hand_models[is_right]
    return getattr(m, "hands_mean", None)


def snap32(x: float) -> int:
    return int(max(32, round(x / 32) * 32))


def encode_size(w: int, h: int, target_w: int | None) -> tuple[int, int]:
    """Paper App. A.2 resize rule: width -> target (aspect kept), both axes snapped to 32."""
    if target_w is None:
        return snap32(w), snap32(h)
    s = target_w / w
    return snap32(w * s), snap32(h * s)


def rescale_K(K3: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> torch.Tensor:
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    return torch.tensor([K3[0, 0] * sx, K3[1, 1] * sy, K3[0, 2] * sx, K3[1, 2] * sy, dst_wh[0], dst_wh[1]],
                        dtype=torch.float32)


def resize_frames(frames_u8: np.ndarray, dst_wh: tuple[int, int]) -> np.ndarray:
    if frames_u8.shape[2] == dst_wh[0] and frames_u8.shape[1] == dst_wh[1]:
        return frames_u8
    return np.stack([cv2.resize(f, dst_wh, interpolation=cv2.INTER_AREA) for f in frames_u8])


def encode_recording(enc: LatentEncoder, frames_u8: np.ndarray, chunk: int = 81) -> torch.Tensor:
    """VAE-encode a long recording in 81-frame chunks (the inference tiling), concat along time."""
    lats = []
    for s in range(0, len(frames_u8), chunk):
        piece = frames_u8[s:s + chunk]
        if len(piece) % 4 != 1 and s > 0:            # causal VAE wants 4k+1 frames per chunk
            piece = piece[: (len(piece) - 1) // 4 * 4 + 1]
            if len(piece) == 0:
                break
        lats.append(enc.encode_uint8(piece))
    return torch.cat(lats, dim=1)


def mano_joints_cam(hand_models, go: torch.Tensor, hp: torch.Tensor, betas: torch.Tensor,
                    trans: torch.Tensor) -> torch.Tensor:
    """(T,2,3,3),(T,2,15,3,3),(2,10),(T,2,3) -> camera-frame OpenPose-21 joints (T,2,21,3)."""
    T = go.shape[0]
    out = torch.zeros(T, 2, 21, 3)
    for s in range(2):
        j, _ = mano_forward_batch_full(go[:, s], hp[:, s], betas[s].expand(T, 10), hand_models[bool(s)])
        out[:, s] = j.cpu() + trans[:, s, None]
    return out


def build_clip(*, dataset: str, clip_id: str, frames_u8: np.ndarray, K3: np.ndarray, target_w: int | None,
               enc: LatentEncoder, hand_models, go_aa: np.ndarray | None, hp_aa: np.ndarray | None,
               betas: np.ndarray | None, trans: np.ndarray | None, exists: np.ndarray,
               joints_cam: np.ndarray | None = None, joint_vis_extra: np.ndarray | None = None,
               is_static: bool = False, camera: str = "pinhole", flat_hand_mean: bool = False,
               meta: dict | None = None) -> Clip:
    """Assemble one Clip from raw annotations.

    go_aa (T,2,3), hp_aa (T,2,45) axis-angle, betas (2,10), trans (T,2,3) camera frame;
    any of them None -> no MANO (has_mano False, joints_cam required).
    exists (T,2) bool. joints_cam (T,2,21,3) optional override / non-MANO GT.
    """
    T, H, W = frames_u8.shape[:3]
    dst = encode_size(W, H, target_w)
    K = rescale_K(np.asarray(K3, dtype=np.float64), (W, H), dst)
    frames = resize_frames(frames_u8, dst)
    latent = encode_recording(enc, frames) if not is_static else enc.encode_uint8(frames)
    ex = torch.as_tensor(exists, dtype=torch.bool)
    has_mano_src = go_aa is not None and hp_aa is not None and betas is not None and trans is not None
    if has_mano_src:
        go = axis_angle_to_matrix(torch.as_tensor(go_aa, dtype=torch.float32))
        hp_t = torch.as_tensor(hp_aa, dtype=torch.float32).reshape(T, 2, 15, 3)
        if flat_hand_mean:                                    # -> smplx flat_hand_mean=False convention
            for s in range(2):
                hm = hands_mean(hand_models, bool(s))
                if hm is not None:
                    hp_t[:, s] = hp_t[:, s] - hm.reshape(15, 3).cpu()
        hp = axis_angle_to_matrix(hp_t)
        be = torch.as_tensor(betas, dtype=torch.float32)
        tr = torch.as_tensor(trans, dtype=torch.float32)
        j_mano = mano_joints_cam(hand_models, go, hp, be, tr)
        jc = torch.as_tensor(joints_cam, dtype=torch.float32) if joints_cam is not None else j_mano
    else:
        assert joints_cam is not None, "need joints_cam when no MANO parameters are given"
        go = torch.eye(3).expand(T, 2, 3, 3).clone()
        hp = torch.eye(3).expand(T, 2, 15, 3, 3).clone()
        be = torch.zeros(2, 10)
        jc = torch.as_tensor(joints_cam, dtype=torch.float32)
        tr = jc[:, :, 0].clone()                              # wrist position stands in for tau (unused)
    j2d = project(jc.reshape(1, -1, 3), K[None]).reshape(T, 2, 21, 2)
    joint_vis, visible = visibility_gate(jc, j2d)
    if joint_vis_extra is not None:
        joint_vis &= torch.as_tensor(joint_vis_extra, dtype=torch.bool)
    joint_vis &= ex[..., None]
    visible &= ex
    has_mano = ex & has_mano_src
    return Clip(dataset=dataset, clip_id=clip_id, latent=latent, n_video_frames=T, K=K, go=go, hp=hp,
                betas=be, trans=tr, joints_cam=jc, joints2d=j2d, joint_vis=joint_vis, has_mano=has_mano,
                exists=ex, visible=visible, is_static=is_static, camera=camera, meta=meta or {})
