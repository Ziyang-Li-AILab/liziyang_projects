"""Build an 81-frame ARCTIC clip from one allocentric pinhole camera.

Image folders 1–8 are moving crops. Each crop is pasted back onto the full
view so the clip keeps that camera's full-frame K. Camera 1 is folder ``1/``
and ``world2cam[0]`` (index 0 of ``misc.json`` is the first allocentric view;
the helmet camera is separate). This is the side view, not the paper's ego view.

Left MANO uses smplx's original shapedirs. See ``build_arctic_ego.py``.

    cd repro/2608.20308/src
    python tools/build_arctic_allocentric.py --seq s01/espressomachine_use_01 --camera 1
"""
from __future__ import annotations

import argparse
import json
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


def _loose_box(pix: np.ndarray) -> tuple[int, int, int, int]:
    """Allocentric crop window: vertex bbox, then the 1.5× box used at save time."""
    ul, lr = pix.min(0), pix.max(0)
    w, h = float(lr[0] - ul[0]), float(lr[1] - ul[1])
    side = max(w, h) * 1.1
    scale = max((side + max(w, h) * 0.6) / 200.0, 3.0)
    cx, cy = float(ul[0] + w / 2.0), float(ul[1] + h / 2.0)
    dim = scale * 1.5 * 200.0
    box = [cx - dim / 2.0, cy - dim / 2.0, cx + dim / 2.0, cy + dim / 2.0]
    return tuple(int(round(v)) for v in box)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seq", default="s01/espressomachine_use_01")
    ap.add_argument("--camera", type=int, default=1, choices=range(1, 9),
                    help="allocentric camera index, 1–8 (image folder name)")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--arctic", type=Path, default=_REPO / "data" / "arctic")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    subject, sequence = args.seq.split("/")
    out_dir = args.out or (args.arctic / "clips")
    n_frames = args.frames
    cam = args.camera - 1  # world2cam[0] is image folder 1

    meta_zip = zipfile.ZipFile(args.arctic / "meta.zip")
    misc = json.loads(meta_zip.read("meta/misc.json"))[subject]
    w2c = np.array(misc["world2cam"][cam], np.float64)
    rot, trans_c = w2c[:3, :3], w2c[:3, 3]
    k_full = np.array(misc["intris_mat"][cam], np.float64)
    width, height = (int(v) for v in misc["image_size"][args.camera])
    raw = zipfile.ZipFile(args.arctic / "raw_seqs.zip")
    mano = np.load(raw.open(f"raw_seqs/{subject}/{sequence}.mano.npy"), allow_pickle=True).item()
    obj = np.load(raw.open(f"raw_seqs/{subject}/{sequence}.object.npy"))
    object_name = sequence.split("_")[0]
    parts = np.array(json.loads(meta_zip.read(f"meta/object_vtemplates/{object_name}/parts.json")))
    verts = []
    for line in meta_zip.read(f"meta/object_vtemplates/{object_name}/mesh.obj").decode().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
    canon = np.array(verts, np.float64) / 1000.0
    articulated = parts == 0  # parts_ids == 1 after the official +1; these rotate with the lever

    images = zipfile.ZipFile(args.arctic / "cropped_images" / subject / f"{sequence}.zip")
    frames = np.empty((n_frames, height, width, 3), np.uint8)
    for t in range(n_frames):
        row = obj[t]
        posed = canon.copy()
        r_arti, _ = cv2.Rodrigues(np.array([0.0, 0.0, -float(row[0])], np.float64))
        posed[articulated] = canon[articulated] @ r_arti.T
        rw, _ = cv2.Rodrigues(row[1:4].astype(np.float64))
        world = posed @ rw.T + row[4:7] / 1000.0
        cam_pts = (rot @ world.T).T + trans_c
        z = np.clip(cam_pts[:, 2], 1e-4, None)
        pix = np.stack([k_full[0, 0] * cam_pts[:, 0] / z + k_full[0, 2],
                        k_full[1, 1] * cam_pts[:, 1] / z + k_full[1, 2]], -1)
        x0, y0, x1, y1 = _loose_box(pix)
        crop = cv2.imdecode(np.frombuffer(images.read(f"{args.camera}/{t + 1:05d}.jpg"), np.uint8),
                            cv2.IMREAD_COLOR)
        resized = cv2.resize(crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((height, width, 3), np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(width, x1), min(height, y1)
        canvas[sy0:sy1, sx0:sx1] = resized[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
        frames[t] = canvas

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
            aa_c, _ = cv2.Rodrigues(rot @ cv2.Rodrigues(rot_w)[0])
            aa_c = aa_c.reshape(3).astype(np.float32)
            go[t, slot] = aa_c
            pose = torch.tensor(hp[t, slot], dtype=torch.float32, device=device).view(1, 45)
            j_world = model(
                global_orient=torch.tensor(rot_w, dtype=torch.float32, device=device).view(1, 3),
                hand_pose=pose, betas=beta).joints[0, 0].detach().cpu().numpy()
            j_cam = model(
                global_orient=torch.tensor(aa_c, dtype=torch.float32, device=device).view(1, 3),
                hand_pose=pose, betas=beta).joints[0, 0].detach().cpu().numpy()
            trans[t, slot] = (rot @ (j_world + trans_w) + trans_c - j_cam).astype(np.float32)

    enc = LatentEncoder(device)
    tag = f"cam{args.camera}"
    clip = build_clip(
        dataset="arctic", clip_id=f"arctic-{subject}-{sequence}-{tag}",
        frames_u8=frames[:, :, :, ::-1].copy(), K3=k_full, target_w=672,
        enc=enc, hand_models=hand_models, go_aa=go, hp_aa=hp, betas=betas, trans=trans,
        exists=np.ones((n_frames, 2), bool), camera="pinhole", flat_hand_mean=False,
        meta={"view": f"allocentric-{args.camera}", "sequence": args.seq})
    out_dir.mkdir(parents=True, exist_ok=True)
    save_clip(clip, str(out_dir / f"{sequence}_{tag}.pt"))
    video = out_dir / f"{sequence}_{tag}.mp4"
    wh = (int(clip.K[4]), int(clip.K[5]))
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, wh)
    for frame in frames:
        writer.write(cv2.resize(frame, wh, interpolation=cv2.INTER_AREA))
    writer.release()
    print("clip", tuple(clip.latent.shape), "K", [round(float(v), 2) for v in clip.K], "video", video, flush=True)


if __name__ == "__main__":
    main()
