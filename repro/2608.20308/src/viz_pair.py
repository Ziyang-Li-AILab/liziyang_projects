"""Side-by-side mesh video: prediction on the left, ground truth on the right.

The banner states the App. A.1 numbers of this clip only. HOT3D clip-001849 is
the upright pinhole recording (cw90); the mp4 is the pre-resize 1408 frame and
is scaled to the 480 encode size stored in the clip.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from scripts.viz_preds import draw_meshes, mano_cam_joints, project  # noqa: E402

from ace_repro.camera import fit_to_intrinsics
from ace_repro.config import load_config
from ace_repro.data.schema import batch_to, collate
from ace_repro.data.store import load_clip
from ace_repro.eval.metrics import SegmentScorer
from ace_repro.hand_model import restore_smplx_left_shapedirs
from ace_repro.model import ReproModel

_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
_FONT_INDEX = 2  # Noto Sans CJK SC
_GT_KEYS = ("joints_cam", "joints2d", "joint_vis", "go", "hp", "betas", "trans", "exists", "visible", "has_mano")
_VIEW = 2


def _rgb_480(path: str, wh: tuple[int, int]) -> list[np.ndarray]:
    """BGR frames resized to the encode size ``wh`` = (W, H)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, wh, interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames


def _pack(go, hp, betas, trans) -> dict:
    """Rotmats plus per-frame betas (T, 2, 10) for ``mano_cam_joints``."""
    t = trans.shape[0]
    if betas.ndim == 2:
        betas = np.repeat(betas[None], t, axis=0)
    return {"go": go, "hp": hp, "betas": betas, "cam_trans": trans}


def _to_numpy(pred: dict) -> dict:
    packed = _pack(
        pred["global_orient"].detach().float().cpu().numpy(),
        pred["hand_pose"].detach().float().cpu().numpy(),
        pred["betas"].detach().float().cpu().numpy(),
        pred["cam_trans"].detach().float().cpu().numpy(),
    )
    packed["exists"] = pred["exists_3d"].detach().float().cpu().numpy()
    return packed


def _banner(width: int, lines: list[str]) -> np.ndarray:
    font = ImageFont.truetype(_FONT, 32, index=_FONT_INDEX)
    line_h = 44
    bar = Image.new("RGB", (width, 16 + line_h * len(lines)), (0, 0, 0))
    draw = ImageDraw.Draw(bar)
    for i, line in enumerate(lines):
        draw.text((16, 8 + i * line_h), line, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(bar), cv2.COLOR_RGB2BGR)


def _stamp(panel: np.ndarray, text: str) -> np.ndarray:
    font = ImageFont.truetype(_FONT, 40, index=_FONT_INDEX)
    img = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text((16, 12), text, font=font, fill=(0, 0, 0), stroke_width=4, stroke_fill=(0, 0, 0))
    draw.text((16, 12), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _metric_line(tag: str, summary: dict) -> str:
    return (
        f"{tag}    F1 {summary['F1']:.3f}    "
        f"MPJPE-p {summary['MPJPE-p']:.1f} mm    "
        f"PA-p {summary['PA-p']:.1f} mm    "
        f"EPE2D-p {summary['EPE2D-p']:.1f} px    "
        f"GO-p {summary['GO-p']:.1f} deg    "
        f"CT-p {summary['CT-p']:.3f} m"
    )


def _as_intr(row) -> dict:
    """(6,) ``fx, fy, cx, cy, W, H`` -> the dict ``project`` expects."""
    fx, fy, cx, cy, w, h = (float(v) for v in row[:6])
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "image_width": w, "image_height": h}


def _forward(model: ReproModel, clip, device) -> tuple[dict, dict, torch.Tensor | None]:
    model.set_eval_mode()
    batch = batch_to(collate([clip]), device)
    with torch.no_grad():
        out, fit = model(batch)
    pred = out["preds"][list(out["preds"].keys())[0]][0]
    gt = {k: batch[k][0] for k in _GT_KEYS}
    pred_K = None
    if model.kfree:
        # Image width and height only; focal length and principal point stay fitted.
        if fit is None or not bool(fit["valid"].all()):
            raise RuntimeError("K-free camera fit failed; refusing to draw with the ground-truth K")
        width, height = float(batch["K"][0, 4]), float(batch["K"][0, 5])
        pred_K = fit_to_intrinsics({k: v.detach() for k, v in fit.items()}, width, height)[0]
    scorer = SegmentScorer(model.hand_models)
    scorer.add_segment(pred, gt, batch["K"][0], pred_K=pred_K)
    return pred, scorer.summary(), pred_K


def _render(frames: list[np.ndarray], pred: dict, gt_pack: dict, faces: dict,
            pred_intr: dict, gt_intr: dict, lines: list[str], out_path: Path) -> None:
    t_count = min(len(frames), pred["cam_trans"].shape[0], gt_pack["cam_trans"].shape[0])
    _, pverts = mano_cam_joints(pred, faces["models"], "cuda", want_verts=True)
    _, gverts = mano_cam_joints(gt_pack, faces["models"], "cuda", want_verts=True)
    h, w = frames[0].shape[:2]
    sample = cv2.resize(frames[0], (w * _VIEW, h * _VIEW), interpolation=cv2.INTER_CUBIC)
    sample = np.hstack([sample, sample])
    bar = _banner(sample.shape[1], lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 30,
                             (sample.shape[1], sample.shape[0] + bar.shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {out_path}")
    show_gt = (gt_pack["exists"] > 0.5)
    for t in range(t_count):
        view = cv2.resize(frames[t], (w * _VIEW, h * _VIEW), interpolation=cv2.INTER_CUBIC)
        pv2d, pz = project(pverts[t], pred_intr, view.shape[1], view.shape[0])
        gv2d, gz = project(gverts[t], gt_intr, view.shape[1], view.shape[0])
        p_show = (pred["exists"][t] > 0.5) & (pz.mean(-1) > 0.05)
        g_show = show_gt[t] & (gz.mean(-1) > 0.05)
        left = _stamp(draw_meshes(view.copy(), pv2d, pverts[t], faces["faces"], p_show), "预测")
        right = _stamp(draw_meshes(view.copy(), gv2d, gverts[t], faces["faces"], g_show), "真值")
        writer.write(np.vstack([bar, np.hstack([left, right])]))
    writer.release()
    print(f"[viz] {t_count} frames -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/repro-freihand.yaml")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--ckpt", action="append", nargs=2, metavar=("TAG", "PATH"), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clip = load_clip(args.clip)
    wh = (int(clip.K[4]), int(clip.K[5]))
    frames = _rgb_480(args.video, wh)
    if len(frames) != clip.n_video_frames:
        raise SystemExit(f"video has {len(frames)} frames, clip has {clip.n_video_frames}")
    gt_intr = _as_intr(clip.K)
    gt_pack = _pack(clip.go.numpy(), clip.hp.numpy(), clip.betas.numpy(), clip.trans.numpy())
    gt_pack["exists"] = clip.exists.numpy().astype(np.float32)

    device = torch.device("cuda")
    cfg = load_config(args.config)
    model = ReproModel(cfg, device, allow_fake_mano=False)
    if clip.dataset == "arctic":
        # Same modules the projector already holds; flip the buffer in place.
        restore_smplx_left_shapedirs(model.hand_models)
    faces = {
        "models": model.hand_models,
        "faces": {slot: np.asarray(model.hand_models[bool(slot)].faces, np.int64) for slot in (0, 1)},
    }
    out_dir = Path(args.out)
    for tag, path in args.ckpt:
        model.load_release(path)
        pred, summary, pred_K = _forward(model, clip, device)
        pred_intr = gt_intr if pred_K is None else _as_intr(pred_K)
        print(f"[metrics] {tag}", {k: round(summary[k], 3) if isinstance(summary[k], float) else summary[k]
                                   for k in ("F1", "MPJPE-p", "PA-p", "EPE2D-p", "GO-p", "CT-p", "TP", "FP", "FN")})
        if pred_K is not None:
            print(f"[camera] {tag} fitted", {k: round(pred_intr[k], 1) for k in ("fx", "fy", "cx", "cy")})
        lines = [f"左：预测    右：真值    {clip.clip_id}", _metric_line(tag, summary)]
        _render(frames, _to_numpy(pred), gt_pack, faces, pred_intr, gt_intr, lines,
                out_dir / f"{clip.clip_id}_{tag}_pred_gt.mp4")


if __name__ == "__main__":
    main()
