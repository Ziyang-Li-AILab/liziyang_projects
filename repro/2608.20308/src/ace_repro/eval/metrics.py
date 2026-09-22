"""Evaluation protocol of paper App. A.1 (gaps_filled J).

Two scoring passes per segment:
1. detection-gated: on-screen GT hands matched to active predictions (existence
   > 0.5, same side, strictly positive IoU of projected boxes with GT dilated
   10%); MPJPE-p / PA-p / EPE2D-p / GO-p / CT-p charge every FN a canonical MANO
   placeholder (identity R, zero pose, mean shape, tau = 0; EPE2D: image diagonal);
   FAcc / Recall / F1 from per-side TP/FP/FN; Jitter on matched runs >= 3 frames.
2. ground-truth-gated: wrist-aligned error of the slot output on every GT
   hand-frame, split into in-view / out-of-sight strata (MPJPE^OOS, MPJPE^+OOS).

Boxes come from the projected MANO *mesh* when a hand model is available (the
paper's convention) and from the projected joints otherwise.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ace_ego_hand.mano_utils import mano_forward_batch_full

from ace_repro.camera import project


def procrustes(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Similarity-align X to Y (both (N,3)); returns aligned X."""
    mx, my = X.mean(0), Y.mean(0)
    X0, Y0 = X - mx, Y - my
    U, S, Vt = torch.linalg.svd(X0.T @ Y0)
    d = torch.sign(torch.det(U @ Vt))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=X.device))
    R = U @ D @ Vt
    s = (S * torch.tensor([1.0, 1.0, d], device=X.device)).sum() / (X0 ** 2).sum().clamp_min(1e-9)
    return s * X0 @ R + my


def _box(pts2d_px: torch.Tensor):
    return torch.stack([pts2d_px[:, 0].min(), pts2d_px[:, 1].min(), pts2d_px[:, 0].max(), pts2d_px[:, 1].max()])


def _iou(a, b) -> float:
    ix = (torch.min(a[2], b[2]) - torch.max(a[0], b[0])).clamp_min(0)
    iy = (torch.min(a[3], b[3]) - torch.max(a[1], b[1])).clamp_min(0)
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua.clamp_min(1e-9))


def _dilate(b, f=0.10):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    w, h = (b[2] - b[0]) * (1 + f), (b[3] - b[1]) * (1 + f)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def _geo_deg(R1, R2) -> float:
    tr = (R1.T @ R2).diagonal().sum()
    return float(torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1))))


def _mano(hand_models, go, hp, betas, slot: int):
    """(T,3,3),(T,15,3,3),(T,10) -> joints (T,21,3), verts (T,V,3)."""
    j, v = mano_forward_batch_full(go.float(), hp.float(), betas.float(), hand_models[bool(slot)])
    return j, v


class SegmentScorer:
    """Accumulates per-hand-frame records over segments; ``summary()`` aggregates."""

    def __init__(self, hand_models, penalty_canonical: bool = True):
        self.hm = hand_models
        self.tp, self.fp, self.fn = 0, 0, 0
        self.frames, self.frames_ok = 0, 0
        self.e = {k: [] for k in ("mpjpe", "pa", "epe2d", "go", "ct")}     # TP + FN placeholder values
        self.jitter_runs: list[tuple[float, int]] = []
        self.eps_iv: list[float] = []
        self.eps_oos: list[float] = []
        dev = next(hand_models[True].buffers()).device
        eye = torch.eye(3, device=dev)[None]
        self.canon = {s: _mano(hand_models, eye, eye.expand(1, 15, 3, 3).clone(), torch.zeros(1, 10, device=dev), s)[0][0]
                      for s in (0, 1)}                                     # canonical placeholder joints

    @torch.no_grad()
    def add_segment(self, pred: dict, gt: dict, K: torch.Tensor) -> None:
        """pred/gt tensors of one clip: (T,2,...) as in ace_repro.data.schema; K (6,)."""
        T = gt["joints_cam"].shape[0]
        dev = K.device
        W, H = float(K[4]), float(K[5])
        diag = math.hypot(W, H)
        Kb = K[None]
        matched = torch.zeros(T, 2, dtype=torch.bool)
        active_all = torch.zeros(T, 2, dtype=torch.bool)
        Jp_cam_all = torch.zeros(T, 2, 21, 3, device=dev)
        for s in range(2):
            Jp, Vp = _mano(self.hm, pred["global_orient"][:, s], pred["hand_pose"][:, s], pred["betas"][:, s], s)
            tau = pred["cam_trans"][:, s].float()
            Jp_cam, Vp_cam = Jp + tau[:, None], Vp + tau[:, None]
            Jp_cam_all[:, s] = Jp_cam
            Jg_cam = gt["joints_cam"][:, s].float()
            Vg_cam = None
            if gt["has_mano"][:, s].any():
                Jg_c, Vg_c = _mano(self.hm, gt["go"][:, s], gt["hp"][:, s], gt["betas"][s][None].expand(T, 10), s)
                Vg_cam = Vg_c + gt["trans"][:, s].float()[:, None]
            p2d_pred_px = project(Jp_cam[None], Kb)[0] * torch.tensor([W, H], device=dev)
            v2d_pred_px = project(Vp_cam[None], Kb)[0] * torch.tensor([W, H], device=dev)
            # "on-screen gate applied to both sides": an active prediction must itself
            # project inside the frame (any joint, z > 1 cm) to enter detection scoring.
            p_norm = project(Jp_cam[None], Kb)[0]
            p_on = ((p_norm[..., 0] >= 0) & (p_norm[..., 0] < 1) & (p_norm[..., 1] >= 0) & (p_norm[..., 1] < 1)
                    & (Jp_cam[..., 2] > 0.01)).any(-1)
            active = (pred["exists_3d"][:, s] > 0.5) & p_on
            active_all[:, s] = active.cpu()
            for t in range(T):
                gt_on = bool(gt["exists"][t, s] and gt["visible"][t, s])
                if gt["exists"][t, s]:                                   # ground-truth-gated pass
                    eps = float(((Jp[t] - Jp[t, 0]) - (Jg_cam[t] - Jg_cam[t, 0])).norm(dim=-1).mean()) * 1000
                    (self.eps_iv if gt["visible"][t, s] else self.eps_oos).append(eps)
                is_tp = False
                if active[t] and gt_on:
                    if Vg_cam is not None:
                        gbox = _box(project(Vg_cam[t][None], Kb)[0] * torch.tensor([W, H], device=dev))
                    else:
                        gbox = _box(gt["joints2d"][t, s].float() * torch.tensor([W, H], device=dev))
                    is_tp = _iou(_box(v2d_pred_px[t]), _dilate(gbox)) > 0
                if is_tp:
                    self.tp += 1
                    matched[t, s] = True
                    Jb_p, Jb_g = Jp[t] - Jp[t, 0], Jg_cam[t] - Jg_cam[t, 0]
                    self.e["mpjpe"].append(float((Jb_p - Jb_g).norm(dim=-1).mean()) * 1000)
                    self.e["pa"].append(float((procrustes(Jp_cam[t], Jg_cam[t]) - Jg_cam[t]).norm(dim=-1).mean()) * 1000)
                    vis = gt["joint_vis"][t, s]
                    gt_px = gt["joints2d"][t, s].float() * torch.tensor([W, H], device=dev)
                    self.e["epe2d"].append(float((p2d_pred_px[t][vis] - gt_px[vis]).norm(dim=-1).mean()) if vis.any() else 0.0)
                    self.e["go"].append(_geo_deg(pred["global_orient"][t, s].float(), gt["go"][t, s].float()))
                    self.e["ct"].append(float((tau[t] - gt["trans"][t, s].float()).norm()))
                else:
                    if active[t]:
                        self.fp += 1
                    if gt_on:                                             # FN -> canonical placeholder
                        self.fn += 1
                        Jc = self.canon[s]
                        Jb_g = Jg_cam[t] - Jg_cam[t, 0]
                        self.e["mpjpe"].append(float(((Jc - Jc[0]) - Jb_g).norm(dim=-1).mean()) * 1000)
                        self.e["pa"].append(float((procrustes(Jc, Jg_cam[t]) - Jg_cam[t]).norm(dim=-1).mean()) * 1000)
                        self.e["epe2d"].append(diag)
                        self.e["go"].append(_geo_deg(torch.eye(3, device=dev), gt["go"][t, s].float()))
                        self.e["ct"].append(float(gt["trans"][t, s].float().norm()))
            # Jitter over maximal matched runs (>= 3 frames), mm/frame^2
            t = 0
            while t < T:
                if not matched[t, s]:
                    t += 1
                    continue
                t0 = t
                while t < T and matched[t, s]:
                    t += 1
                if t - t0 >= 3:
                    J = Jp_cam_all[t0:t, s] * 1000
                    acc = (J[2:] - 2 * J[1:-1] + J[:-2]).norm(dim=-1).mean(-1)
                    self.jitter_runs.append((float(acc.sum()), int(acc.numel())))
        # frame accuracy: no FP and no FN on either side
        for t in range(T):
            ok = True
            for s in range(2):
                gt_on = bool(gt["exists"][t, s] and gt["visible"][t, s])
                act = bool(active_all[t, s])
                if (act and not matched[t, s]) or (gt_on and not matched[t, s]):
                    ok = False
            self.frames += 1
            self.frames_ok += int(ok)

    def summary(self) -> dict:
        P = self.tp / max(1, self.tp + self.fp)
        R = self.tp / max(1, self.tp + self.fn)
        out = {"FAcc": self.frames_ok / max(1, self.frames), "Recall": R,
               "F1": 2 * P * R / max(1e-9, P + R), "TP": self.tp, "FP": self.fp, "FN": self.fn}
        for k, v in self.e.items():
            out[f"{k.upper()}-p"] = float(np.mean(v)) if v else float("nan")
        num = sum(a for a, _ in self.jitter_runs)
        den = sum(n for _, n in self.jitter_runs)
        out["Jitter"] = num / den if den else float("nan")
        out["MPJPE_OOS"] = float(np.mean(self.eps_oos)) if self.eps_oos else float("nan")
        out["MPJPE_IV"] = float(np.mean(self.eps_iv)) if self.eps_iv else float("nan")
        allv = self.eps_iv + self.eps_oos
        out["MPJPE_+OOS"] = float(np.mean(allv)) if allv else float("nan")
        out["n_IV"], out["n_OOS"] = len(self.eps_iv), len(self.eps_oos)
        return out
