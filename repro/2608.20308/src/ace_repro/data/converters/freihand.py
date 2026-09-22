#!/usr/bin/env python
"""FreiHAND (static, right hand only, full MANO fit) -> shared-format clips.

Expected raw layout (official FreiHAND_pub_v2 or the same files re-assembled):
    <raw>/training_K.json      32560 x 3x3
    <raw>/training_xyz.json    32560 x 21x3   (metres, camera frame, OpenPose order)
    <raw>/training_mano.json   32560 x 1x61   [pose48 | betas10 | uv_root2 | scale1]
    <raw>/training/rgb/%08d.jpg  (224x224; index // 32560 selects the background version)

Each image becomes a 5-frame static clip (paper App. A.2). tau is recovered by
wrist-aligning MANO(pose, betas) to xyz; the flat_hand_mean convention of the
fit is resolved empirically (the one with the smaller wrist-aligned residual).
With the FakeMANO stand-in the MANO terms are disabled (has_mano=False) since
the stand-in cannot reproduce the fits.

    python -m ace_repro.data.converters.freihand --raw data/freihand/raw --out data/freihand --limit 9000
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from ace_ego_hand.mano_utils import mano_forward_batch_full

from ace_repro.data.converters.common import build_clip
from ace_repro.data.latents import LatentEncoder
from ace_repro.data.store import save_clip
from ace_repro.hand_model import axis_angle_to_matrix, build_hand_models

N_UNIQUE = 32560
STATIC_T = 5


def resolve_convention(hand_models, mano61: np.ndarray, xyz: np.ndarray, n: int = 200) -> bool:
    """Return flat_hand_mean flag whose MANO joints best match xyz (wrist-aligned)."""
    hm = getattr(hand_models[True], "hands_mean", None)
    if hm is None:
        return False
    idx = np.linspace(0, len(xyz) - 1, n).astype(int)
    pose = torch.tensor(mano61[idx, :48], dtype=torch.float32)
    betas = torch.tensor(mano61[idx, 48:58], dtype=torch.float32)
    gt = torch.tensor(xyz[idx], dtype=torch.float32)
    errs = {}
    for flat in (False, True):
        art = pose[:, 3:].clone()
        if flat:
            art = art - hm.reshape(1, 45).cpu()
        go = axis_angle_to_matrix(pose[:, :3])
        hp = axis_angle_to_matrix(art.reshape(-1, 15, 3))
        j, _ = mano_forward_batch_full(go, hp, betas, hand_models[True])
        j = j.cpu()
        errs[flat] = float(((j - j[:, :1]) - (gt - gt[:, :1])).norm(dim=-1).mean()) * 1000
    print(f"[freihand] wrist-aligned residual vs xyz: flat_hand_mean=False {errs[False]:.2f} mm, "
          f"True {errs[True]:.2f} mm")
    return errs[True] < errs[False]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N image indices")
    ap.add_argument("--val-every", type=int, default=50, help="every k-th unique sample -> test split")
    ap.add_argument("--allow-fake-mano", action="store_true")
    ap.add_argument("--mano-dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    hand_models, kind = build_hand_models(device, args.mano_dir, allow_fake=args.allow_fake_mano)
    enc = LatentEncoder(device)
    K_all = np.array(json.load(open(os.path.join(args.raw, "training_K.json"))), dtype=np.float64)
    xyz_all = np.array(json.load(open(os.path.join(args.raw, "training_xyz.json"))), dtype=np.float32)
    mano_all = np.array(json.load(open(os.path.join(args.raw, "training_mano.json"))), dtype=np.float32).reshape(-1, 61)
    flat = resolve_convention(hand_models, mano_all, xyz_all) if kind == "mano" else False
    rgb_dir = os.path.join(args.raw, "training", "rgb")
    ids = [i for i in range(len(K_all) * 4) if os.path.isfile(os.path.join(rgb_dir, f"{i:08d}.jpg"))]
    if args.limit:
        ids = ids[: args.limit]
    print(f"[freihand] {len(ids)} images, hand_model={kind}, flat_hand_mean={flat}")
    n_ok = 0
    for i in ids:
        u = i % N_UNIQUE
        img = cv2.imread(os.path.join(rgb_dir, f"{i:08d}.jpg"))[:, :, ::-1]
        frames = np.repeat(img[None], STATIC_T, axis=0)
        xyz = xyz_all[u]                                                # (21,3) right hand
        jc = np.zeros((STATIC_T, 2, 21, 3), np.float32)
        jc[:, 1] = xyz
        exists = np.zeros((STATIC_T, 2), bool)
        exists[:, 1] = True
        kw = dict(go_aa=None, hp_aa=None, betas=None, trans=None)
        if kind == "mano":
            pose, betas = mano_all[u, :48], mano_all[u, 48:58]
            art = pose[3:].copy()
            go = axis_angle_to_matrix(torch.tensor(pose[None, :3]))
            hp = axis_angle_to_matrix(torch.tensor(art.reshape(1, 15, 3)) - (hand_models[True].hands_mean.reshape(1, 15, 3).cpu() if flat else 0))
            j, _ = mano_forward_batch_full(go, hp, torch.tensor(betas[None]), hand_models[True])
            tau = xyz[0] - j[0, 0].cpu().numpy()                       # wrist-align MANO to xyz
            go_aa = np.zeros((STATIC_T, 2, 3), np.float32)
            hp_aa = np.zeros((STATIC_T, 2, 45), np.float32)
            go_aa[:, 1] = pose[:3]
            hp_aa[:, 1] = art
            be = np.zeros((2, 10), np.float32)
            be[1] = betas
            tr = np.zeros((STATIC_T, 2, 3), np.float32)
            tr[:, 1] = tau
            kw = dict(go_aa=go_aa, hp_aa=hp_aa, betas=be, trans=tr, flat_hand_mean=flat)
        clip = build_clip(dataset="freihand", clip_id=f"freihand_{i:08d}", frames_u8=frames, K3=K_all[u],
                          target_w=None, enc=enc, hand_models=hand_models, exists=exists, joints_cam=jc,
                          is_static=True, meta={"unique_id": int(u)}, **kw)
        split = "test" if (u % args.val_every == 0) else "train"
        save_clip(clip, os.path.join(args.out, split, f"{clip.clip_id}.pt"))
        n_ok += 1
        if n_ok % 500 == 0:
            print(f"  {n_ok}/{len(ids)}", flush=True)
    print(f"[freihand] wrote {n_ok} clips -> {args.out}")


if __name__ == "__main__":
    main()
