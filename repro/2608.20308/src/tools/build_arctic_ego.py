"""Build an 81-frame ARCTIC clip from the helmet camera (image folder 0).

Folder 0 is the 2800×2000 helmet frame scaled by 0.3. Frames are undistorted
with ``egocam.dist.npy`` so the stored K is a pinhole. MANO is smplx's
original left shape space: ARCTIC was fit there (smplx issue 48). Do not route
this script through ``build_hand_models``, which mirrors the left hand for HOT3D.

    cd repro/2608.20308/src
    python tools/build_arctic_ego.py --seq s01/espressomachine_use_01
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
_REPO = _SRC.parent
sys.path[:0] = [str(_SRC), str(_REPO / "code")]

import ace_ego_hand.mano_utils as mu  # noqa: E402
from ace_repro.data.converters.common import build_clip  # noqa: E402
from ace_repro.data.latents import LatentEncoder  # noqa: E402
from ace_repro.data.store import save_clip  # noqa: E402


def _axis_angle(rot: np.ndarray) -> np.ndarray:
    aa, _ = cv2.Rodrigues(rot)
    return aa.reshape(3).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seq", default="s01/espressomachine_use_01",
                    help="subject/sequence inside the ARCTIC zips")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--arctic", type=Path, default=_REPO / "data" / "arctic",
                    help="directory with raw_seqs.zip, meta is unused, and cropped_images/")
    ap.add_argument("--out", type=Path, default=None, help="defaults to <arctic>/clips")
    args = ap.parse_args()
    subject, sequence = args.seq.split("/")
    out_dir = args.out or (args.arctic / "clips")
    n_frames = args.frames

    raw = zipfile.ZipFile(args.arctic / "raw_seqs.zip")
    ego = np.load(raw.open(f"raw_seqs/{subject}/{sequence}.egocam.dist.npy"), allow_pickle=True).item()
    mano = np.load(raw.open(f"raw_seqs/{subject}/{sequence}.mano.npy"), allow_pickle=True).item()
    k_full = np.array(ego["intrinsics"], np.float64)
    k_img = k_full.copy()
    k_img[:2] *= 0.3
    dist = np.array(ego["dist8"], np.float64)
    images = zipfile.ZipFile(args.arctic / "cropped_images" / subject / f"{sequence}.zip")

    sample = cv2.imdecode(np.frombuffer(images.read("0/00001.jpg"), np.uint8), cv2.IMREAD_COLOR)
    height, width = sample.shape[:2]
    frames = np.empty((n_frames, height, width, 3), np.uint8)
    for t in range(n_frames):
        bgr = cv2.imdecode(np.frombuffer(images.read(f"0/{t + 1:05d}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        frames[t] = cv2.undistort(bgr, k_img, dist)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hand_models = mu.setup_mano_models({}, device)
    go = np.zeros((n_frames, 2, 3), np.float32)
    hp = np.zeros((n_frames, 2, 45), np.float32)
    trans = np.zeros((n_frames, 2, 3), np.float32)
    betas = np.zeros((2, 10), np.float32)
    for slot, side in ((0, "left"), (1, "right")):
        betas[slot] = np.asarray(mano[side]["shape"], np.float32)
        hp[:, slot] = np.asarray(mano[side]["pose"][:n_frames], np.float32)
        model = hand_models[bool(slot)]
        beta = torch.tensor(betas[slot], dtype=torch.float32, device=device).view(1, 10)
        for t in range(n_frames):
            rot_w = np.asarray(mano[side]["rot"][t], np.float64)
            trans_w = np.asarray(mano[side]["trans"][t], np.float64)
            rc = np.asarray(ego["R_k_cam_np"][t], np.float64)
            tc = np.asarray(ego["T_k_cam_np"][t], np.float64).reshape(3)
            aa_c = _axis_angle(rc @ cv2.Rodrigues(rot_w)[0])
            go[t, slot] = aa_c
            pose = torch.tensor(hp[t, slot], dtype=torch.float32, device=device).view(1, 45)
            j_world = model(
                global_orient=torch.tensor(rot_w, dtype=torch.float32, device=device).view(1, 3),
                hand_pose=pose, betas=beta).joints[0, 0].detach().cpu().numpy()
            j_cam = model(
                global_orient=torch.tensor(aa_c, dtype=torch.float32, device=device).view(1, 3),
                hand_pose=pose, betas=beta).joints[0, 0].detach().cpu().numpy()
            # smplx rotates around the wrist, so camera-frame transl is not R @ t + t_c.
            trans[t, slot] = (rc @ (j_world + trans_w) + tc - j_cam).astype(np.float32)

    enc = LatentEncoder(device)
    clip_id = f"arctic-{subject}-{sequence}-ego"
    clip = build_clip(
        dataset="arctic", clip_id=clip_id,
        frames_u8=frames[:, :, :, ::-1].copy(), K3=k_img, target_w=672,
        enc=enc, hand_models=hand_models, go_aa=go, hp_aa=hp, betas=betas, trans=trans,
        exists=np.ones((n_frames, 2), bool), camera="pinhole", flat_hand_mean=False,
        meta={"view": "ego", "sequence": args.seq})
    out_dir.mkdir(parents=True, exist_ok=True)
    save_clip(clip, str(out_dir / f"{sequence}_ego.pt"))
    video = out_dir / f"{sequence}_ego.mp4"
    wh = (int(clip.K[4]), int(clip.K[5]))
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, wh)
    for frame in frames:
        writer.write(cv2.resize(frame, wh, interpolation=cv2.INTER_AREA))
    writer.release()
    print("clip", tuple(clip.latent.shape), "K", [round(float(v), 2) for v in clip.K], "video", video, flush=True)


if __name__ == "__main__":
    main()
