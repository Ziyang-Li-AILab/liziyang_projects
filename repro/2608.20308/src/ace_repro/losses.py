"""Training objective — paper Eq. 6 with App. B.3 weights (gaps_filled H).

L = L_rot + L_joint + L_img + L_cam + L_pres + L_tmp + L_ray (+ L_fit, K-free)

Routing masks (all [guess]-tagged in gaps_filled H unless the paper states them):
  exists    — hand has GT this frame (OOS frames included: §3.4 "fully supervised")
  has_mano  — rotation/shape GT available (False for RHD: rot & betas held at zero, §3.4)
  joint_vis — App. A.1 on-screen gate, gates every image-space term
  cam terms need MANO (tau is defined w.r.t. canonical MANO joints) -> off for RHD
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ace_ego_hand.mano_utils import mano_forward_batch_full

from ace_repro.camera import fit_loss, project, rays_from_intrinsics

PRED_KEYS = ["global_orient", "hand_pose", "betas", "cam_trans", "exists_2d", "exists_3d",
             "direct_joints_rootrel", "direct_wrist_cam", "direct_joints_cam", "direct_joints2d",
             "segment_betas"]


def stack_preds(pred_list: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([p[k] for p in pred_list]) for k in PRED_KEYS}


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over entries where ``mask`` (broadcastable, leading dims) is 1."""
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.to(x.dtype).expand_as(x)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def geodesic(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Angle between rotation matrices, (..., 3, 3) -> (...)."""
    tr = (R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))


def mano_joints(hand_models, go, hp, betas) -> torch.Tensor:
    """(B,T,2,3,3),(B,T,2,15,3,3),(B,T,2,10) -> canonical OpenPose-21 joints (B,T,2,21,3)."""
    B, T = go.shape[:2]
    out = []
    for s in range(2):
        j, _ = mano_forward_batch_full(go[:, :, s].reshape(-1, 3, 3).float(),
                                       hp[:, :, s].reshape(-1, 15, 3, 3).float(),
                                       betas[:, :, s].reshape(-1, 10).float(), hand_models[bool(s)])
        out.append(j.reshape(B, T, 21, 3))
    return torch.stack(out, dim=2)


def compute_losses(out: dict, batch: dict, hand_models, weights: dict, *, step: int,
                   fit: dict | None = None, fit_warmup_steps: int = 500) -> tuple[torch.Tensor, dict]:
    """Return ``(total, {term: value})`` for one batch (all taps averaged)."""
    K = batch["K"].float()
    B, T = batch["exists"].shape[:2]
    m_ex = batch["exists"].float()                         # (B,T,2)
    m_mano = (batch["has_mano"] & batch["exists"]).float()
    m_vis = batch["visible"].float()
    m_jv = batch["joint_vis"].float()                      # (B,T,2,21)
    m_cam = m_mano                                         # tau needs canonical MANO joints
    J_gt = batch["joints_cam"].float()
    J_gt_rr = J_gt - J_gt[..., :1, :]
    wrist_gt = J_gt[..., 0, :]
    j2d_gt = batch["joints2d"].float()

    terms: dict[str, torch.Tensor] = {}
    taps = list(out["preds"].keys())
    for layer in taps:
        p = stack_preds(out["preds"][layer])
        t = {}
        # -- L_rot ---------------------------------------------------------
        t["rot_geodesic"] = (masked_mean(geodesic(p["global_orient"], batch["go"].float()), m_mano)
                             + masked_mean(geodesic(p["hand_pose"], batch["hp"].float()), m_mano)) / 2
        t["rot_mse"] = (masked_mean((p["global_orient"] - batch["go"].float()) ** 2, m_mano)
                        + masked_mean((p["hand_pose"] - batch["hp"].float()) ** 2, m_mano)) / 2
        m_beta = m_mano.amax(dim=1)                                          # (B,2) any frame with MANO
        t["betas_l1"] = masked_mean((p["segment_betas"] - batch["betas"].float()).abs(), m_beta)
        # -- MANO read-out of the prediction ------------------------------
        J_can = mano_joints(hand_models, p["global_orient"], p["hand_pose"], p["betas"])
        J_cam = J_can + p["cam_trans"][..., None, :]
        J_rr = J_can - J_can[..., :1, :]
        # -- L_joint -------------------------------------------------------
        t["joint_rootrel"] = (masked_mean((p["direct_joints_rootrel"] - J_gt_rr).abs(), m_ex)
                              + masked_mean((J_rr - J_gt_rr).abs(), m_ex)) / 2
        t["joint_cam"] = (masked_mean((J_cam - J_gt).abs(), m_cam)
                          + masked_mean((p["direct_joints_cam"] - J_gt).abs(), m_ex)) / 2
        t["wrist_cam"] = (masked_mean((p["direct_wrist_cam"] - wrist_gt).abs(), m_ex)
                          + masked_mean((J_cam[..., 0, :] - wrist_gt).abs(), m_cam)) / 2
        # -- L_img (normalised image units, visible joints only) -----------
        t["img_joints2d"] = masked_mean((p["direct_joints2d"] - j2d_gt).abs(), m_jv)
        t["img_reproj"] = masked_mean((project(J_cam, K) - j2d_gt).abs(), m_jv * m_cam[..., None])
        t["img_wrist2d"] = masked_mean((project(p["direct_wrist_cam"], K) - j2d_gt[..., 0, :]).abs(),
                                       m_jv[..., 0])
        # -- L_cam (gradient flows through the mixed-PnP decode) -----------
        t["cam_trans"] = masked_mean((p["cam_trans"] - batch["trans"].float()).abs(), m_cam)
        # -- L_pres --------------------------------------------------------
        t["presence_3d"] = F.binary_cross_entropy(p["exists_3d"].clamp(1e-5, 1 - 1e-5), m_ex)
        t["presence_2d"] = F.binary_cross_entropy(p["exists_2d"].clamp(1e-5, 1 - 1e-5), m_vis)
        # -- L_tmp: App. B.3 penalises ||Jhat_{t+1}-2 Jhat_t+Jhat_{t-1}||_1.
        # The formula contains only the prediction, so this is a smoothness
        # penalty toward zero acceleration, not a match to the GT trajectory.
        if T >= 3:
            acc_p = J_cam[:, 2:] - 2 * J_cam[:, 1:-1] + J_cam[:, :-2]
            t["temporal_accel"] = acc_p.abs().mean()
        else:
            t["temporal_accel"] = J_cam.sum() * 0
        # -- L_ray ---------------------------------------------------------
        rm = out["raymap"].get(layer)
        if rm is not None and batch.get("camera", "pinhole") == "pinhole":
            rays_gt = rays_from_intrinsics(K, rm.shape[-2], rm.shape[-1])          # (B,3,H,W)
            cos = F.cosine_similarity(rm.float(), rays_gt[:, :, None], dim=1)     # (B,F,H,W)
            t["ray_cosine"] = (1 - cos).mean()
        elif rm is not None and "rays_gt" in batch:
            cos = F.cosine_similarity(rm.float(), batch["rays_gt"].float()[:, :, None], dim=1)
            t["ray_cosine"] = (1 - cos).mean()
        else:
            t["ray_cosine"] = p["cam_trans"].sum() * 0
        # -- L_fit (K-free, pinhole sources, linear warm-up) ---------------
        if fit is not None and weights.get("fit_bearing", 0) > 0 and batch.get("camera") == "pinhole":
            Ht, Wt = rm.shape[-2], rm.shape[-1]
            ramp = min(1.0, step / max(1, fit_warmup_steps))
            t["fit_bearing"] = ramp * fit_loss(fit, K, Ht, Wt)
        for k, v in t.items():
            terms[k] = terms.get(k, 0) + v / len(taps)

    total = sum(weights.get(k, 0.0) * v for k, v in terms.items())
    return total, terms
