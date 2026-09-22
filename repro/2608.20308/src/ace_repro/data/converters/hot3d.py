#!/usr/bin/env python
"""HOT3D-Clips (Aria RGB) -> shared-format clips.

Raw layout (Hugging Face ``bop-benchmark/hot3d``):
    <raw>/train_aria/clip-*.tar
    <raw>/test_aria/clip-*.tar

Each tar is one 150-frame clip. Stream ``214-1`` is the RGB fisheye
(``CameraModelType.FISHEYE624``). Paper App. A.2 feeds HOT3D at 480x480,
30 fps, in 81-frame windows. The public test split drops ``hands.json``,
so only clips that still have MANO poses are written.

Undistort follows ``hand_tracking_toolkit`` / ``clip_util.convert_to_pinhole_camera``
with ``focal_scale=1`` (the toolkit default; the paper does not name another).
MANO ``thetas`` are 15 PCA coefficients. smplx 0.1.28 (and this repo's MANO
loader) use ``flat_hand_mean=False``, so the axis-angle residual is
``thetas @ hands_components[:15]`` and the model adds ``hands_mean``.
``wrist_xform`` is a world-frame axis-angle plus translation. The camera-frame
translation passed to the shared clip subtracts the shaped rest-pose wrist
``J0``: smplx leaves that joint at ``J0`` and only rotates the bones around it,
so a rigid change of camera does not move ``J0`` by itself.

    python -m ace_repro.data.converters.hot3d --raw ../data/hot3d/raw --out ../data/hot3d

An output ``.pt`` that already exists is left alone, so a rerun only encodes
the clips that are still missing.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import tarfile

import cv2
import numpy as np
import torch

from ace_ego_hand.mano_utils import MANO_CHECKPOINT_DIR, mano_forward_batch_full, rotmat_to_axis_angle

from ace_repro.data.store import save_clip
from ace_repro.hand_model import axis_angle_to_matrix, build_hand_models

STREAM = "214-1"
ENCODE_FRAMES = 81          # paper App. A.2 window; 4*20+1 for the causal VAE
TARGET_W = 480              # paper App. A.2; diagonal is 679 px
FOCAL_SCALE = 1.0           # clip_util.convert_to_pinhole_camera default
N_PCA = 15
_EPS = np.float32(2.0 ** -128)


def quat_wxyz_to_mat(q) -> np.ndarray:
    """Unit quaternion (w, x, y, z) -> 3x3 rotation. Matches the HOT3D extrinsics."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ], dtype=np.float32)


def fisheye624_project(pts: np.ndarray, fxy, cxy, coeff) -> np.ndarray:
    """Camera-frame points -> FISHEYE624 pixels. ``pts`` is (N, 3).

    Projection is the toolkit's arctan model plus ``OVR624Distortion``
    (6 radial, 2 tangential, 4 thin-prism). Tangential ``p1``/``p2`` are
    crossed relative to OpenCV; that is the toolkit's order.
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x * x + y * y)
    s = np.arctan2(r, z) / np.maximum(r, _EPS)
    p = np.stack((x * s, y * s), axis=-1)
    k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4 = coeff
    r2 = np.clip((p * p).sum(axis=-1, keepdims=True), -(np.pi ** 2), np.pi ** 2)
    r4 = r2 * r2
    r6 = r2 * r4
    r8 = r4 * r4
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r4 * r6 + k6 * r6 * r6
    uv = p * radial
    x, y = uv[:, 0], uv[:, 1]
    x2, y2, xy = x * x, y * y, x * y
    rr = x2 + y2
    x = x + (2 * p2 * xy + p1 * (rr + 2 * x2))
    y = y + (2 * p1 * xy + p2 * (rr + 2 * y2))
    r4 = rr * rr
    # Thin prism is applied in normalized uv, then scaled by focal length.
    x = x + s1 * rr + s2 * r4
    y = y + s3 * rr + s4 * r4
    fx, fy = fxy
    cx, cy = cxy
    return np.stack((x * fx + cx, y * fy + cy), axis=-1)


def undistort_map(h: int, w: int, f: float, cxy, coeff, focal_scale: float):
    """Pinhole pixels -> fisheye source pixels, plus the pinhole 3x3 K.

    Both cameras share extrinsics, so a pinhole ray is projected by the fisheye
    (``dataset.warp_image`` with a ``PinholePlaneCameraModel``).
    """
    fx = fy = np.float32(f * focal_scale)
    cx, cy = np.float32(cxy[0]), np.float32(cxy[1])
    vs, us = np.mgrid[0:h, 0:w].astype(np.float32)
    rays = np.stack(((us - cx) / fx, (vs - cy) / fy, np.ones_like(us)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    # The source camera keeps its own focal length. focal_scale only changes the pinhole K.
    pix = fisheye624_project(rays.reshape(-1, 3), (np.float32(f), np.float32(f)), (cx, cy), coeff).reshape(h, w, 2)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return pix[..., 0].astype(np.float32), pix[..., 1].astype(np.float32), K


def _jpeg(blob: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("failed to decode a HOT3D JPEG")
    return bgr[:, :, ::-1]


def _components(mano_root: str) -> dict[bool, np.ndarray]:
    out = {}
    for is_right, name in ((False, "MANO_LEFT.pkl"), (True, "MANO_RIGHT.pkl")):
        path = os.path.join(mano_root, "mano", name)
        if not os.path.isfile(path):
            path = os.path.join(mano_root, name)
        with open(path, "rb") as f:
            data = pickle.load(f)
        out[is_right] = np.asarray(data["hands_components"][:N_PCA], dtype=np.float32)
    return out


def _rest_wrist(model, betas: torch.Tensor) -> np.ndarray:
    """Shaped rest-pose wrist. Pose does not move it (smplx root translation is ``J0``)."""
    z3, z45 = torch.zeros(1, 3), torch.zeros(1, 45)
    joints = model(betas=betas[None], global_orient=z3, hand_pose=z45, transl=z3).joints
    return joints[0, 0].detach().cpu().numpy().astype(np.float32)


def _pose_in_camera(R_wc, t_wc, wrist_xform, J0) -> tuple[np.ndarray, np.ndarray]:
    """World wrist -> camera-frame global rotation and the translation the clip stores."""
    R_w = axis_angle_to_matrix(torch.tensor(wrist_xform[:3], dtype=torch.float32)).numpy()
    R_c = R_wc.T @ R_w
    t_w = np.asarray(wrist_xform[3:], dtype=np.float32)
    tau = R_wc.T @ (J0 + t_w - t_wc) - J0
    go = rotmat_to_axis_angle(torch.tensor(R_c)[None])[0].numpy().astype(np.float32)
    return go, tau.astype(np.float32)


def _box_contains(pix, box, margin: float = 30.0) -> bool:
    x1, y1, x2, y2 = box
    return (x1 - margin) <= pix[0] <= (x2 + margin) and (y1 - margin) <= pix[1] <= (y2 + margin)


def _check_frame(joints_cam, fisheye_px, box, map_x, map_y, K) -> None:
    """Visible wrist must sit in the fisheye amodal box.

    When that wrist also lands inside the pinhole, the undistort map has to
    agree with the fisheye projection. A hand can be inside the fisheye and
    outside the narrower pinhole; that frame stays an out-of-sight label.
    """
    if not _box_contains(fisheye_px[0], box):
        raise RuntimeError(
            f"projected wrist {fisheye_px[0].tolist()} is outside the amodal box {box}")
    z = float(joints_cam[0, 2])
    u = float(K[0, 0] * joints_cam[0, 0] / z + K[0, 2])
    v = float(K[1, 1] * joints_cam[0, 1] / z + K[1, 2])
    h, w = map_x.shape
    if not (0.0 <= u < w and 0.0 <= v < h):
        return
    ui = int(np.clip(np.rint(u), 0, w - 1))
    vi = int(np.clip(np.rint(v), 0, h - 1))
    mapped = np.array([map_x[vi, ui], map_y[vi, ui]], dtype=np.float32)
    if abs(float(mapped[0] - fisheye_px[0, 0])) > 3 or abs(float(mapped[1] - fisheye_px[0, 1])) > 3:
        raise RuntimeError(
            f"undistort map disagrees with the fisheye projection "
            f"({mapped.tolist()} vs {fisheye_px[0].tolist()})")


def convert_tar(path: str, hand_models, comps, n_frames: int, focal_scale: float) -> dict:
    """Read one clip tar into the arrays ``build_clip`` consumes, and check frame 0's pose."""
    tar = tarfile.open(path)
    try:
        names = set(tar.getnames())
        if "__hand_shapes.json__" not in names or "000000.hands.json" not in names:
            return {}
        betas_np = np.asarray(json.load(tar.extractfile("__hand_shapes.json__"))["mano"], dtype=np.float32)
        betas = torch.tensor(betas_np)
        J0 = {is_right: _rest_wrist(hand_models[is_right], betas) for is_right in (False, True)}
        cam0 = json.load(tar.extractfile("000000.cameras.json"))[STREAM]
        params = cam0["calibration"]["projection_params"]
        f, cx, cy = params[:3]
        coeff = np.asarray(params[3:], dtype=np.float32)
        h = int(cam0["calibration"]["image_height"])
        w = int(cam0["calibration"]["image_width"])
        map_x, map_y, K = undistort_map(h, w, f, (cx, cy), coeff, focal_scale)

        T = n_frames
        frames = np.empty((T, h, w, 3), dtype=np.uint8)
        go = np.zeros((T, 2, 3), np.float32)
        hp = np.zeros((T, 2, 45), np.float32)
        trans = np.zeros((T, 2, 3), np.float32)
        exists = np.zeros((T, 2), bool)
        checked = False
        info = json.load(tar.extractfile("000000.info.json"))
        for t in range(T):
            key = f"{t:06d}"
            frames[t] = _jpeg(tar.extractfile(f"{key}.image_{STREAM}.jpg").read())
            cam = json.load(tar.extractfile(f"{key}.cameras.json"))[STREAM]
            R_wc = quat_wxyz_to_mat(cam["T_world_from_camera"]["quaternion_wxyz"])
            t_wc = np.asarray(cam["T_world_from_camera"]["translation_xyz"], dtype=np.float32)
            hands = json.load(tar.extractfile(f"{key}.hands.json"))
            for slot, is_right, side in ((0, False, "left"), (1, True, "right")):
                ann = hands.get(side) or {}
                pose = ann.get("mano_pose")
                if not pose:
                    continue
                exists[t, slot] = True
                theta = np.asarray(pose["thetas"], dtype=np.float32)
                residual = theta @ comps[is_right]
                go[t, slot], trans[t, slot] = _pose_in_camera(R_wc, t_wc, pose["wrist_xform"], J0[is_right])
                hp[t, slot] = residual
                vis = (ann.get("visibilities_modeled") or {}).get(STREAM, 0.0)
                box = (ann.get("boxes_amodal") or {}).get(STREAM)
                if checked or vis < 0.5 or not box:
                    continue
                joints = _joints_cam(hand_models[is_right], go[t, slot], residual, betas, trans[t, slot])
                if float(joints[0, 2]) < 0.02:
                    continue
                fisheye_px = fisheye624_project(joints[:1], (f, f), (cx, cy), coeff)
                _check_frame(joints, fisheye_px, box, map_x, map_y, K)
                checked = True
        warped = np.stack([cv2.remap(frames[t], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                           for t in range(T)])
    finally:
        tar.close()
    return {
        "frames": warped, "K": K, "go": go, "hp": hp, "betas": np.stack([betas_np, betas_np]),
        "trans": trans, "exists": exists, "checked": checked,
        "meta": {"sequence": info.get("sequence_id"), "participant": info.get("participant_id"),
                 "stream": STREAM, "focal_scale": focal_scale, "src": os.path.basename(path)},
    }


def _joints_cam(model, go_aa, residual, betas, tau) -> np.ndarray:
    go = axis_angle_to_matrix(torch.tensor(go_aa)[None])
    hp = axis_angle_to_matrix(torch.tensor(residual, dtype=torch.float32).reshape(1, 15, 3))
    joints, _ = mano_forward_batch_full(go, hp, betas[None], model)
    return (joints[0].detach().cpu().numpy() + tau).astype(np.float32)


def _tars(raw: str) -> list[tuple[str, str]]:
    found = []
    for split, folder in (("train", "train_aria"), ("test", "test_aria")):
        d = os.path.join(raw, folder)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".tar"):
                found.append((split, os.path.join(d, name)))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mano-dir", default=None)
    ap.add_argument("--frames", type=int, default=ENCODE_FRAMES)
    ap.add_argument("--limit", type=int, default=None, help="convert at most this many clips that have MANO")
    ap.add_argument("--focal-scale", type=float, default=FOCAL_SCALE)
    ap.add_argument("--check-only", action="store_true", help="validate projection, do not VAE-encode")
    args = ap.parse_args()
    if args.frames % 4 != 1:
        raise SystemExit(f"--frames must be 4k+1 for the causal VAE, got {args.frames}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hand_models, kind = build_hand_models(device, args.mano_dir, allow_fake=False)
    if kind != "mano":
        raise SystemExit("HOT3D PCA pose needs the licensed MANO pkls")
    torch.set_grad_enabled(False)
    for model in hand_models.values():
        model.eval()
    mano_root = args.mano_dir or MANO_CHECKPOINT_DIR
    comps = _components(mano_root)
    enc = None
    build_clip = None
    if not args.check_only:
        from ace_repro.data.converters.common import build_clip
        from ace_repro.data.latents import LatentEncoder
        enc = LatentEncoder(device)

    n_ok = 0
    for split, path in _tars(args.raw):
        if args.limit and n_ok >= args.limit:
            break
        clip_id = os.path.splitext(os.path.basename(path))[0]
        dest = os.path.join(args.out, split, f"{clip_id}.pt")
        if not args.check_only and os.path.isfile(dest):
            print(f"[hot3d] skip {clip_id}: already written")
            n_ok += 1
            continue
        packed = convert_tar(path, hand_models, comps, args.frames, args.focal_scale)
        if not packed:
            print(f"[hot3d] skip {clip_id}: no public MANO pose")
            continue
        tag = "box-ok" if packed["checked"] else "no-visible-hand"
        print(f"[hot3d] {clip_id} {split} {tag} exists={int(packed['exists'].sum())}", flush=True)
        if args.check_only:
            n_ok += 1
            continue
        clip = build_clip(
            dataset="hot3d", clip_id=clip_id, frames_u8=packed["frames"], K3=packed["K"],
            target_w=TARGET_W, enc=enc, hand_models=hand_models, go_aa=packed["go"], hp_aa=packed["hp"],
            betas=packed["betas"], trans=packed["trans"], exists=packed["exists"], camera="pinhole",
            flat_hand_mean=False, meta=packed["meta"])
        save_clip(clip, dest)
        n_ok += 1
    print(f"[hot3d] {'checked' if args.check_only else 'wrote'} {n_ok} clips -> {args.out}")


if __name__ == "__main__":
    main()
