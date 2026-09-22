#!/usr/bin/env python
"""Train -> checkpoint -> reload -> eval, without the Wan 5B encoder.

``closed_loop.py`` is the full path: it builds GeoDiT, encodes clips with the
Wan VAE, and hard-requires CUDA plus the DiT/VAE weights. This machine has
neither, so this script keeps every module *after* the tap and replaces the
tapped feature grid with a small random tensor of the real width (3072).

What is real: MemorySegmentBetasTransformer (paper §3.2), the 1x1 ray head,
mixed-PnP (§3.3), FakeMANO, and Eq. 6 (``compute_losses``).
What is stubbed: the Wan DiT and VAE. Numbers are ``hand_model=fake`` and are
not Table 1.

    python debug_loop.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ace_repro  # noqa: E402,F401  — puts code/ on sys.path before ace_ego_hand

from ace_ego_hand.archs.projector.memory_projector import MemorySegmentBetasTransformer  # noqa: E402
from ace_ego_hand.mano_utils import mano_forward_batch_full  # noqa: E402

from ace_repro.camera import intrinsics_to_list, project  # noqa: E402
from ace_repro.config import dump_config, load_config  # noqa: E402
from ace_repro.data.schema import visibility_gate  # noqa: E402
from ace_repro.hand_model import axis_angle_to_matrix, build_hand_models  # noqa: E402
from ace_repro.losses import compute_losses  # noqa: E402
from ace_repro.optim import build_optimizer, build_scheduler  # noqa: E402

# GeoDiT tap width. The paper text quotes a 16px grid; the released code taps
# a 32px token grid whose channel count is the Wan hidden size, 3072.
TAP_CHANNELS = 3072
TAP_LAYER = 15
# F_lat, H, W. A real 480px clip is about 21 x 15 x 15; this grid is only so
# a CPU step finishes in seconds.
GRID = (2, 4, 4)
CLIP_T = 9
RES = (160, 160)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class TapStub(nn.Module):
    """GeoDiT.forward_emode after the backbone tap.

    ``feat`` is ``(B, 3072, F, H, W)``, the tensor ``fold(taps[layer])`` would
    have produced. The ray head and projector are the released modules.
    """

    def __init__(self, projector_kwargs: dict) -> None:
        super().__init__()
        kw = dict(projector_kwargs)
        kw["latent_channels"] = TAP_CHANNELS
        self.projector = MemorySegmentBetasTransformer(**kw)
        self.ray_head = nn.Conv3d(TAP_CHANNELS, 3, kernel_size=1)
        # Released GeoDiT zero-inits this conv (App. B.1). atan2 clamps |dz|,
        # so an all-zero field stays finite on the first step.
        nn.init.zeros_(self.ray_head.weight)
        nn.init.zeros_(self.ray_head.bias)

    def forward(self, feat: torch.Tensor, n_video_frames: int, intrinsics_list: list[dict]) -> dict:
        # feat: (B, 3072, F, H, W)
        rays = self.ray_head(feat)
        pred_rays = rays.mean(dim=2)
        preds = self.projector.forward_batched(
            feat, n_video_frames=n_video_frames,
            intrinsics_list=intrinsics_list, pred_rays=pred_rays)
        return {
            "preds": {TAP_LAYER: preds},
            "raymap": {TAP_LAYER: rays},
            "head_out": None,
            "aux_geo": {},
            "gen_render": {},
        }


def _release_state(stub: TapStub) -> dict:
    """Same keys as ReproModel.state_for_release, minus the Wan tensors."""
    prefix = f"l{TAP_LAYER}."
    return {
        "projectors": {prefix + k: v.detach().cpu() for k, v in stub.projector.state_dict().items()},
        "raymap_heads": {prefix + k: v.detach().cpu() for k, v in stub.ray_head.state_dict().items()},
        "backbone_trainable": {},
        "debug_loop": {"tap_channels": TAP_CHANNELS, "grid": list(GRID), "backbone": "stub"},
    }


def _load_release(stub: TapStub, path: str) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    prefix = f"l{TAP_LAYER}."
    stub.projector.load_state_dict({k[len(prefix):]: v for k, v in ck["projectors"].items()})
    stub.ray_head.load_state_dict({k[len(prefix):]: v for k, v in ck["raymap_heads"].items()})


def make_batch(hand_models, device: torch.device) -> dict:
    """One synthetic clip of GT. Imagery is not rendered; the loss never sees pixels."""
    T, (W, H) = CLIP_T, RES
    g = torch.Generator().manual_seed(0)
    focal = 0.9 * W
    K = torch.tensor([focal, focal, W / 2, H / 2, float(W), float(H)])
    go = torch.eye(3).view(1, 1, 3, 3).expand(T, 2, 3, 3).contiguous()
    aa = torch.zeros(T, 2, 15, 3)
    aa[..., 1] = 0.35
    hp = axis_angle_to_matrix(aa)
    betas = torch.randn(2, 10, generator=g) * 0.3
    trans = torch.zeros(T, 2, 3)
    trans[:, 0, 0], trans[:, 1, 0] = -0.06, 0.06
    trans[:, :, 1] = 0.02 * torch.linspace(-1, 1, T)[:, None]
    trans[:, :, 2] = 0.45
    joints = torch.zeros(T, 2, 21, 3)
    for slot, is_right in enumerate((False, True)):
        j21, _ = mano_forward_batch_full(
            go[:, slot], hp[:, slot], betas[slot].expand(T, 10), hand_models[is_right])
        joints[:, slot] = j21.cpu() + trans[:, slot, None]
    joints2d = project(joints.reshape(1, -1, 3), K[None]).reshape(T, 2, 21, 2)
    joint_vis, visible = visibility_gate(joints, joints2d)
    exists = torch.ones(T, 2, dtype=torch.bool)

    def batch(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0).to(device)

    return {
        "K": batch(K), "go": batch(go), "hp": batch(hp), "betas": batch(betas),
        "trans": batch(trans), "joints_cam": batch(joints), "joints2d": batch(joints2d),
        "joint_vis": batch(joint_vis), "has_mano": batch(exists), "exists": batch(exists),
        "visible": batch(visible), "n_video_frames": T, "camera": "pinhole",
        "dataset": "debug", "clip_ids": ["debug_0000"], "is_static": False,
    }


def feature_grid(device: torch.device) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    return torch.randn(1, TAP_CHANNELS, *GRID, generator=g).to(device)


def _one_step(stub, feat, batch, hand_models, opt, sch, cfg, step: int) -> tuple[float, dict]:
    opt.zero_grad(set_to_none=True)
    out = stub(feat, batch["n_video_frames"], intrinsics_to_list(batch["K"]))
    total, terms = compute_losses(
        out, batch, hand_models, cfg["loss"], step=step,
        fit=None, fit_warmup_steps=cfg["schedule"].get("fit_warmup_steps", 500))
    if not torch.isfinite(total):
        raise FloatingPointError(f"non-finite loss at step {step}: { {k: float(v) for k, v in terms.items()} }")
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for g in opt.param_groups for p in g["params"]], cfg["optimizer"]["grad_clip"])
    opt.step()
    sch.step()
    return float(total.detach()), {k: float(v.detach()) for k, v in terms.items()}


def _fresh(cfg, device, hand_models) -> TapStub:
    stub = TapStub(cfg["model"]["projector"]).to(device)
    stub.projector.set_mano_models(hand_models)
    return stub


def train_and_save(cfg, device, hand_models, steps: int, run_dir: str) -> tuple[str, list[float]]:
    stub = _fresh(cfg, device, hand_models)
    feat = feature_grid(device)
    batch = make_batch(hand_models, device)
    opt = build_optimizer(
        [{"params": list(stub.parameters()), "lr": cfg["optimizer"]["lr_decoder"]}],
        cfg["optimizer"])
    sch_cfg = dict(cfg["schedule"])
    sch_cfg["total_steps"] = steps
    sch_cfg["warmup_steps"] = min(1, steps)
    sch = build_scheduler(opt, sch_cfg)
    losses = []
    for step in range(steps):
        loss, terms = _one_step(stub, feat, batch, hand_models, opt, sch, cfg, step)
        losses.append(loss)
        print(f"[train {step + 1}/{steps}] loss={loss:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in terms.items()))
    path = os.path.join(run_dir, "ckpt_infer.pt")
    torch.save(_release_state(stub), path)
    return path, losses


@torch.no_grad()
def infer_reloaded(cfg, device, hand_models, kind: str, ckpt: str, run_dir: str) -> dict:
    stub = _fresh(cfg, device, hand_models)
    _load_release(stub, ckpt)
    stub.eval()
    batch = make_batch(hand_models, device)
    out = stub(feature_grid(device), batch["n_video_frames"], intrinsics_to_list(batch["K"]))
    pred = out["preds"][TAP_LAYER][0]
    cpu = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in pred.items()}
    bad = [k for k, v in cpu.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
    if bad:
        raise FloatingPointError(f"non-finite inference tensors: {bad}")
    t = batch["n_video_frames"]
    if tuple(cpu["cam_trans"].shape) != (t, 2, 3):
        raise RuntimeError(f"cam_trans shape {tuple(cpu['cam_trans'].shape)} != {(t, 2, 3)}")
    if tuple(cpu["direct_joints2d"].shape) != (t, 2, 21, 2):
        raise RuntimeError(f"direct_joints2d shape {tuple(cpu['direct_joints2d'].shape)}")
    pred_path = os.path.join(run_dir, "infer_pred.pt")
    torch.save(cpu, pred_path)
    summary = {
        "hand_model": kind,
        "backbone": "stub",
        "tap_layer": TAP_LAYER,
        "feature_grid": [1, TAP_CHANNELS, *GRID],
        "ckpt": ckpt,
        "pred": pred_path,
        "clip_id": batch["clip_ids"][0],
        "shapes": {k: list(v.shape) for k, v in cpu.items() if torch.is_tensor(v)},
        "cam_trans_frame0_right": [round(x, 4) for x in cpu["cam_trans"][0, 1].tolist()],
    }
    print(f"[infer] clip={summary['clip_id']} hand_model={kind} backbone=stub "
          f"cam_trans[0,right]={summary['cam_trans_frame0_right']} -> {pred_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "repro-default.yaml"))
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    cfg = load_config(args.config)
    cfg["name"] = "debug-loop"
    seed_all(cfg["seed"])
    device = torch.device(args.device)
    hand_models, kind = build_hand_models(device, allow_fake=True)
    run_dir = os.path.join(cfg["paths"]["out_root"], cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    dump_config(cfg, os.path.join(run_dir, "config.yaml"))
    ckpt, losses = train_and_save(cfg, device, hand_models, args.steps, run_dir)
    summary = infer_reloaded(cfg, device, hand_models, kind, ckpt, run_dir)
    summary["train_losses"] = [round(x, 4) for x in losses]
    with open(os.path.join(run_dir, "loop_result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[debug-loop] ok  losses={summary['train_losses']}  {run_dir}")


if __name__ == "__main__":
    main()
