"""Hand models for training: the licensed MANO (via the released loader) or a
stand-in kinematic hand used ONLY when the MANO pkls are not on disk.

The stand-in exists so that the plumbing (losses, mixed-PnP decode, metrics)
can be smoke-tested without redistributing MANO. Any number produced with it is
labelled ``hand_model=fake`` and is not comparable to the paper.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from ace_ego_hand.mano_utils import MANO_CHECKPOINT_DIR, _MANO_TIP_IDS  # noqa: F401

# MANO kinematic tree (smplx order: 0 wrist, 1-3 index, 4-6 middle, 7-9 pinky,
# 10-12 ring, 13-15 thumb).
MANO_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
# Rest-pose bone offsets (metres, right hand, fingers along +x, palm normal +z).
_REST = {
    1: (0.095, 0.020, 0.0), 2: (0.040, 0.0, 0.0), 3: (0.025, 0.0, 0.0),        # index
    4: (0.095, 0.000, 0.0), 5: (0.045, 0.0, 0.0), 6: (0.028, 0.0, 0.0),        # middle
    7: (0.080, -0.040, 0.0), 8: (0.035, 0.0, 0.0), 9: (0.020, 0.0, 0.0),       # pinky
    10: (0.090, -0.020, 0.0), 11: (0.040, 0.0, 0.0), 12: (0.026, 0.0, 0.0),    # ring
    13: (0.030, 0.030, 0.010), 14: (0.030, 0.030, 0.0), 15: (0.025, 0.025, 0.0),  # thumb
}
# tip offsets appended after the last joint of [thumb, index, middle, ring, pinky]
_TIPS = {15: (0.020, 0.020, 0.0), 3: (0.020, 0.0, 0.0), 6: (0.022, 0.0, 0.0),
         12: (0.020, 0.0, 0.0), 9: (0.018, 0.0, 0.0)}
_TIP_PARENT = [15, 3, 6, 12, 9]


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 3, 3) via Rodrigues."""
    theta = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / theta
    K = torch.zeros(aa.shape[:-1] + (3, 3), device=aa.device, dtype=aa.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return eye + s * K + (1 - c) * (K @ K)


class FakeMANO(nn.Module):
    """Minimal articulated hand exposing the smplx MANO call signature.

    ``forward(global_orient=(B,3), hand_pose=(B,45), betas=(B,10))`` returns an
    object with ``joints`` (B,16,3) and ``vertices`` (B,778,3); the 5 fingertip
    vertices ``_MANO_TIP_IDS`` are placed at the fingertips so that the released
    ``mano_forward_batch_full`` OpenPose-21 remap works unchanged.
    """

    NUM_VERTS = 778

    def __init__(self, is_rhand: bool = True):
        super().__init__()
        rest = torch.zeros(16, 3)
        for j, o in _REST.items():
            rest[j] = torch.tensor(o)
        tips = torch.stack([torch.tensor(_TIPS[p]) for p in _TIP_PARENT])
        if not is_rhand:                       # mirror across the yz-plane
            rest[:, 0] *= -1
            tips[:, 0] *= -1
        self.register_buffer("rest", rest)
        self.register_buffer("tips", tips)
        g = torch.Generator().manual_seed(0)
        self.register_buffer("vert_joint", torch.randint(0, 16, (self.NUM_VERTS,), generator=g))
        self.register_buffer("vert_off", (torch.rand(self.NUM_VERTS, 3, generator=g) - 0.5) * 0.02)
        self.is_rhand = is_rhand

    def forward(self, global_orient, hand_pose, betas, **_):
        B = global_orient.shape[0]
        scale = (1.0 + 0.05 * betas[:, :1]).clamp(0.7, 1.3)            # (B,1)
        R_loc = axis_angle_to_matrix(torch.cat([global_orient[:, None], hand_pose.reshape(B, 15, 3)], 1))
        R_glob = [R_loc[:, 0]]
        pos = [torch.zeros(B, 3, device=global_orient.device, dtype=global_orient.dtype)]
        for j in range(1, 16):
            p = MANO_PARENTS[j]
            R_glob.append(R_glob[p] @ R_loc[:, j])
            pos.append(pos[p] + (R_glob[p] @ (self.rest[j] * scale)[..., None])[..., 0])
        joints = torch.stack(pos, 1)                                     # (B,16,3)
        R_glob = torch.stack(R_glob, 1)                                  # (B,16,3,3)
        verts = joints[:, self.vert_joint] + (R_glob[:, self.vert_joint] @ self.vert_off[None, ..., None])[..., 0]
        verts = verts.clone()
        for k, (vid, par) in enumerate(zip(_MANO_TIP_IDS, _TIP_PARENT)):
            verts[:, vid] = joints[:, par] + (R_glob[:, par] @ (self.tips[k] * scale)[..., None])[..., 0]
        return SimpleNamespace(joints=joints, vertices=verts)


def mano_available(mano_dir: str | None = None) -> bool:
    d = mano_dir or MANO_CHECKPOINT_DIR
    return (os.path.isfile(os.path.join(d, "mano", "MANO_RIGHT.pkl"))
            or os.path.isfile(os.path.join(d, "MANO_RIGHT.pkl")))


def _shapedir_x_gap(models: dict) -> float | None:
    left, right = models.get(False), models.get(True)
    if left is None or not hasattr(left, "shapedirs") or not hasattr(right, "shapedirs"):
        return None
    return torch.sum(torch.abs(left.shapedirs[:, 0, :] - right.shapedirs[:, 0, :])).item()


def mirror_left_shapedirs(models: dict) -> None:
    """Undo the smplx left-hand shapedirs bug (smplx issue 48).

    HOT3D's ``MANOHandModel`` applies the same correction: if the left model's
    x-component of ``shapedirs`` matches the right model, mirror it. FakeMANO
    has no shapedirs and is left alone. ARCTIC was fit in the uncorrected
    space; call ``restore_smplx_left_shapedirs`` for those parameters.
    """
    if (gap := _shapedir_x_gap(models)) is not None and gap < 1.0:
        models[False].shapedirs[:, 0, :] *= -1


def restore_smplx_left_shapedirs(models: dict) -> None:
    """Back to the smplx default left shapedirs.

    ARCTIC's authors fit the raw MANO in that space and did not rerun MoSh
    after smplx issue 48. Drawing those parameters through the HOT3D mirror
    lifts the left fingertips off the hand.
    """
    if (gap := _shapedir_x_gap(models)) is not None and gap >= 1.0:
        models[False].shapedirs[:, 0, :] *= -1


def build_hand_models(device: torch.device, mano_dir: str | None = None, allow_fake: bool = False):
    """Return ``({False: left, True: right}, kind)`` with kind in {"mano", "fake"}."""
    if mano_dir:
        os.environ["ACE_EGO_HAND_MANO_DIR"] = mano_dir
    if mano_available(mano_dir):
        import ace_ego_hand.mano_utils as mu
        mu.MANO_CHECKPOINT_DIR = mano_dir or mu.MANO_CHECKPOINT_DIR
        models = mu.setup_mano_models({}, device)
        mirror_left_shapedirs(models)
        return models, "mano"
    if not allow_fake:
        raise FileNotFoundError(
            f"MANO pkls not found under {mano_dir or MANO_CHECKPOINT_DIR}; register at "
            "https://mano.is.tue.mpg.de and follow code/README.md, or pass --allow-fake-mano "
            "for a plumbing-only smoke run.")
    models = {s: FakeMANO(is_rhand=s).to(device).eval() for s in (False, True)}
    for m in models.values():
        for p in m.parameters():
            p.requires_grad_(False)
    print("[hand_model] WARNING: using FakeMANO stand-in (no licensed MANO on disk)")
    return models, "fake"
