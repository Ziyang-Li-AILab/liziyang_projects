"""Reproduce the K-free failure at infer_video.py's default encode width.

The released path keeps ``self_ray_decode`` on, so translation is solved from
the predicted rays. The mesh is then drawn through the pinhole fitted
afterwards (``pred_intrinsics``), never through the clip's real K. At 832 that
fit is anisotropic and the depth is too close, which is the floating hand.

The right panel is the clip's ground truth, projected with the real K scaled
to the same encode grid.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from ace_ego_hand.inference import _fit_pred_K  # noqa: E402
from ace_ego_hand.video_vae import decode_video, encode, load_vae  # noqa: E402
from infer_video import _encode_size  # noqa: E402

from ace_repro.config import load_config
from ace_repro.data.schema import batch_to, collate
from ace_repro.data.store import load_clip
from ace_repro.eval.metrics import SegmentScorer
from ace_repro.model import ReproModel
from viz_pair import _as_intr, _metric_line, _render, _to_numpy

_ROOT = Path(__file__).resolve().parents[1]


def _placeholder(width: int, height: int) -> dict:
    """60-degree stand-in. K-free does not read fx/fy; only the image size is used."""
    focal = 0.5 * width / math.tan(math.radians(30.0))
    return {"fx": focal, "fy": focal, "cx": width / 2.0, "cy": height / 2.0,
            "image_width": width, "image_height": height,
            "source": "nominal 60deg placeholder; K-free does not read it"}


def _scaled_K(clip_K: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Clip K lives at its encode size. Scale it onto the 832 grid."""
    sx, sy = width / float(clip_K[4]), height / float(clip_K[5])
    return torch.tensor(
        [float(clip_K[0]) * sx, float(clip_K[1]) * sy,
         float(clip_K[2]) * sx, float(clip_K[3]) * sy, width, height],
        dtype=torch.float32)


def _bgr_frames(pixels: torch.Tensor) -> list[np.ndarray]:
    """(T,3,H,W) float RGB in [-1,1] -> BGR uint8 frames."""
    rgb = ((pixels.permute(0, 2, 3, 1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return [frame[:, :, ::-1].copy() for frame in rgb]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", type=Path, default=_ROOT / "data/hot3d/pinhole/clip-001849.mp4")
    ap.add_argument("--clip", type=Path, default=_ROOT / "data/hot3d/train/clip-001849.pt")
    ap.add_argument("--config", type=Path, default=_ROOT / "src/configs/repro-kfree.yaml")
    ap.add_argument("--ckpt", type=Path, default=_ROOT / "code/checkpoints/ace_ego_hand_kfree.pt")
    ap.add_argument("--encode_w", type=int, default=832)
    ap.add_argument("--out", type=Path, default=_ROOT / "code/results")
    args = ap.parse_args()

    clip = load_clip(str(args.clip))
    n_frames = int(clip.n_video_frames)
    probe = cv2.VideoCapture(str(args.video))
    src_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    enc_w, enc_h = _encode_size(src_w, src_h, args.encode_w)
    print(f"[video] {src_w}x{src_h} -> {enc_w}x{enc_h}, {n_frames} frames")

    device = torch.device("cuda")
    pixels = decode_video(str(args.video), n_frames, resize_hw=(enc_w, enc_h))
    frames = _bgr_frames(pixels)
    vae = load_vae(device)
    latent = encode(vae, pixels, device).float()
    del vae, pixels
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = ReproModel(load_config(str(args.config)), device, allow_fake_mano=False)
    model.load_release(str(args.ckpt))
    model.set_eval_mode()
    for projector in model.net.projectors.values():
        projector.self_ray_decode = True  # released infer_video path: rays, not fitted K
    stand_in = _placeholder(enc_w, enc_h)
    with torch.no_grad():
        out = model.net.forward_emode(latent.unsqueeze(0).to(device), n_frames, [stand_in])
    layer = 15 if 15 in out["preds"] else next(iter(out["preds"]))
    pred = out["preds"][layer][0]
    fitted = _fit_pred_K(out["raymap"][layer][0], stand_in)
    if fitted is None:
        raise RuntimeError("ray field produced no fitted camera")
    pred_K = torch.tensor(
        [fitted["fx"], fitted["fy"], fitted["cx"], fitted["cy"], enc_w, enc_h],
        dtype=torch.float32, device=device)
    gt_K = _scaled_K(clip.K, enc_w, enc_h).to(device)
    print(f"[camera] fitted fx={fitted['fx']:.1f} fy={fitted['fy']:.1f} "
          f"fy/fx={fitted['fy'] / fitted['fx']:.3f} "
          f"cx={fitted['cx']:.1f} cy={fitted['cy']:.1f}")
    print(f"[camera] true   fx={float(gt_K[0]):.1f} fy={float(gt_K[1]):.1f} "
          f"cx={float(gt_K[2]):.1f} cy={float(gt_K[3]):.1f}")

    batch = batch_to(collate([clip]), device)
    gt = {key: batch[key][0] for key in (
        "joints_cam", "joints2d", "joint_vis", "go", "hp", "betas", "trans", "exists", "visible", "has_mano")}
    scorer = SegmentScorer(model.hand_models)
    scorer.add_segment(pred, gt, gt_K, pred_K=pred_K)
    summary = scorer.summary()
    print("[metrics]", {k: round(summary[k], 3) if isinstance(summary[k], float) else summary[k]
                         for k in ("F1", "MPJPE-p", "PA-p", "EPE2D-p", "GO-p", "CT-p")})

    gt_pack = {
        "go": clip.go.numpy(), "hp": clip.hp.numpy(),
        "betas": np.repeat(clip.betas.numpy()[None], n_frames, axis=0),
        "cam_trans": clip.trans.numpy(),
        "exists": clip.exists.numpy().astype(np.float32),
    }
    lines = [
        f"左：预测（832 射线解码 + 拟合相机）    右：真值    {clip.clip_id}",
        _metric_line("kfree-832", summary),
    ]
    out_path = args.out / f"{clip.clip_id}_kfree832_pred_gt.mp4"
    _render(frames, _to_numpy(pred), gt_pack, {
        "models": model.hand_models,
        "faces": {slot: np.asarray(model.hand_models[bool(slot)].faces, np.int64) for slot in (0, 1)},
    }, _as_intr(pred_K), _as_intr(gt_K), lines, out_path)


if __name__ == "__main__":
    main()
