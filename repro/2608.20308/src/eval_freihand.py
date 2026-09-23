#!/usr/bin/env python
"""Score a FreiHAND checkpoint with the paper's segment metrics and draw a few hands.

The clips are the training images. This run did not hold out a FreiHAND test
split, and Table 1 is ARCTIC / HOT3D / HOI4D after the seven-source mixture.
The metrics use the same definitions (App. A.1) so the gap is interpretable,
not so the numbers can be lined up with that table.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from ace_repro.config import load_config
from ace_repro.data.schema import batch_to, collate
from ace_repro.data.store import ClipStore
from ace_repro.eval.metrics import SegmentScorer
from ace_repro.model import ReproModel

# OpenPose-21 edges. Slot 1 (right) is the only FreiHAND hand.
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
          (0, 5), (5, 6), (6, 7), (7, 8),
          (0, 9), (9, 10), (10, 11), (11, 12),
          (0, 13), (13, 14), (14, 15), (15, 16),
          (0, 17), (17, 18), (18, 19), (19, 20)]


def _indices(n_clips: int, n: int) -> list[int]:
    if n >= n_clips:
        return list(range(n_clips))
    return np.linspace(0, n_clips - 1, n).astype(int).tolist()


def _score(model: ReproModel, store: ClipStore, ids: list[int], device) -> dict:
    model.set_eval_mode()
    scorer = SegmentScorer(model.hand_models)
    for i in ids:
        batch = batch_to(collate([store[i]]), device)
        out, _ = model(batch)
        layer = list(out["preds"].keys())[0]
        pred = out["preds"][layer][0]
        gt = {k: batch[k][0] for k in ("joints_cam", "joints2d", "joint_vis", "go", "hp", "betas", "trans",
                                       "exists", "visible", "has_mano")}
        scorer.add_segment(pred, gt, batch["K"][0])
    return scorer.summary()


def _draw_hand(img, joints_uv, color, exists: float) -> None:
    """joints_uv: (21, 2) normalised. Skip the hand when the presence head is off."""
    if exists < 0.5:
        return
    h, w = img.shape[:2]
    pts = np.stack([joints_uv[:, 0] * w, joints_uv[:, 1] * h], axis=-1)
    for a, b in _EDGES:
        cv2.line(img, tuple(np.rint(pts[a]).astype(int)), tuple(np.rint(pts[b]).astype(int)), color, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(np.rint(p).astype(int)), 2, color, -1, cv2.LINE_AA)


def _visualize(model: ReproModel, store: ClipStore, ids: list[int], rgb_dir: str, out_dir: str, device) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.set_eval_mode()
    for i in ids:
        clip = store[i]
        batch = batch_to(collate([clip]), device)
        out, _ = model(batch)
        layer = list(out["preds"].keys())[0]
        pred = out["preds"][layer][0]
        # Static clips repeat one photo for 5 frames; the middle frame is the one we draw.
        t = clip.n_video_frames // 2
        idx = int(clip.clip_id.split("_")[-1])
        bgr = cv2.imread(os.path.join(rgb_dir, f"{idx:08d}.jpg"))
        canvas = np.concatenate([bgr, bgr.copy()], axis=1)
        gt = batch["joints2d"][0, t, 1].detach().cpu().numpy()
        pr = pred["direct_joints2d"][t, 1].detach().cpu().numpy()
        ex = float(pred["exists_3d"][t, 1])
        _draw_hand(canvas[:, :bgr.shape[1]], gt, (80, 220, 80), 1.0)
        _draw_hand(canvas[:, bgr.shape[1]:], pr, (180, 105, 255), ex)
        cv2.putText(canvas, "GT", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"pred exists={ex:.2f}", (bgr.shape[1] + 8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 105, 255), 1, cv2.LINE_AA)
        path = os.path.join(out_dir, f"{clip.clip_id}.jpg")
        cv2.imwrite(path, canvas)
        print(f"[viz] {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/repro-freihand.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=256, help="evenly spaced training clips to score")
    ap.add_argument("--viz", type=int, default=6)
    ap.add_argument("--out", default="../runs/repro-freihand-k")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda")
    model = ReproModel(cfg, device, allow_fake_mano=False)
    model.load_release(args.ckpt)
    root = os.path.join(cfg["paths"]["data_root"], "freihand", "train")
    store = ClipStore(root, window=False, seed=cfg["seed"])
    ids = _indices(len(store), args.n)
    summary = _score(model, store, ids, device)
    summary["n_clips"] = len(ids)
    summary["ckpt"] = os.path.abspath(args.ckpt)
    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "eval_freihand.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[eval]", json.dumps({k: summary[k] for k in summary if not isinstance(summary[k], str)}))
    viz_ids = _indices(len(store), args.viz)
    rgb = os.path.join(cfg["paths"]["data_root"], "freihand", "raw", "training", "rgb")
    _visualize(model, store, viz_ids, rgb, os.path.join(args.out, "viz"), device)


if __name__ == "__main__":
    main()
