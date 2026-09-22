"""Clip-level inference: VDM control latents -> per-frame hand predictions.

``predict_video`` runs the model over one clip and returns the per-frame arrays
listed in ``DUMP_KEYS``. Two modes:

  full   ONE forward over the whole clip (RoPE length extrapolation) — what the
         long-sequence result uses.
  tiled  22-latent windows aligned to cover each 81-frame segment; this matches
         the training window and is what the benchmark numbers use.
"""
from __future__ import annotations

import numpy as np
import torch

# `direct_joints_cam` / `direct_wrist_cam` are the projector's direct 3D joint
# regression — root-relative joints plus a directly regressed camera-space wrist.
# They bypass MANO and the cam_trans decode entirely, so they are an independent
# second read of the same hands.
DUMP_KEYS = ("global_orient", "hand_pose", "betas", "cam_trans", "direct_joints2d",
             "direct_joints_cam", "direct_wrist_cam", "exists_2d", "exists_3d")


def _fit_pred_K(rm, intr: dict):
    """Fit an effective pinhole (fx,fy,cx,cy) from the model's PREDICTED ray field
    (K-free deployment: the camera the model read from the image). rm: (3,F,H,W).
    Per-axis least squares over u=cx+fx*(dx/dz), v=cy+fy*(dy/dz)."""
    if rm is None:
        return None
    iw = float(intr["image_width"]); ih = float(intr["image_height"])
    pdir = torch.nn.functional.normalize(rm.float().cpu(), dim=0)  # (3,F,H,W) on CPU
    C, F, H, W = pdir.shape
    dz = pdir[2].clamp_min(1e-6)
    tanx = (pdir[0] / dz).reshape(-1); tany = (pdir[1] / dz).reshape(-1)
    gi, gj = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    u = ((gj + 0.5) * iw / W).reshape(1, H, W).expand(F, H, W).reshape(-1)
    v = ((gi + 0.5) * ih / H).reshape(1, H, W).expand(F, H, W).reshape(-1)
    sx = torch.linalg.lstsq(torch.stack([torch.ones_like(tanx), tanx], 1),
                            u.unsqueeze(1)).solution.squeeze()
    sy = torch.linalg.lstsq(torch.stack([torch.ones_like(tany), tany], 1),
                            v.unsqueeze(1)).solution.squeeze()
    return {"fx": float(sx[1]), "fy": float(sy[1]),
            "cx": float(sx[0]), "cy": float(sy[0]),
            "image_width": iw, "image_height": ih, "source": "kf_predicted_rays"}


@torch.no_grad()
def predict_video(model, tap: int, ctrl: torch.Tensor, intr: dict, mode: str,
                  device, tile_w: int = 22) -> dict[str, np.ndarray]:
    """Per-frame predictions over the whole clip -> {key: (F_px, ...) np}."""
    f_lat = ctrl.shape[1]
    n_px = 4 * (f_lat - 1) + 1

    def run(lat_slice: torch.Tensor, n_video: int) -> dict:
        ls = lat_slice.unsqueeze(0).float().to(device)
        out = model.net.forward_emode(ls, n_video, [intr])
        rm = out.get("raymap", {})
        rm = rm.get(tap) if isinstance(rm, dict) else None
        return out["preds"][tap][0], (rm[0] if rm is not None else None)

    if mode == "full":
        p, rm = run(ctrl, n_px)
        res = {k: p[k].cpu().numpy() for k in DUMP_KEYS}
        res["_pred_K"] = _fit_pred_K(rm, intr)      # model-predicted camera (K-free)
        return res

    # tiled: w-latent windows (default 22), outputs indexed by absolute px frame 4a+i
    w = int(tile_w)
    acc = {k: None for k in DUMP_KEYS}
    # Tiled used to DROP the raymap, so `pred_intrinsics` was only ever available
    # under --mode full. That matters for uncalibrated footage, where the ray head
    # is the only camera estimate available: a hand is too small and too planar to
    # constrain a focal by itself (measured: per-hand PnP focal fits scatter over
    # 28-108 deg HFOV on the same clip). Collect the per-window ray fields and fit
    # ONE effective pinhole over all of them; window overlaps just add LSQ samples.
    rms: list[torch.Tensor] = []
    afs: list[float] = []
    n_seg = n_px // 81
    anchors = [(min((81 * i) // 4, f_lat - w) if f_lat >= w else 0, None)
               for i in range(max(n_seg, 1))]
    covered = 4 * anchors[-1][0] + 4 * (min(w, f_lat) - 1) + 1
    if covered < n_px:
        # px frames past the last 81f-aligned window were left zero before;
        # cover them with an end-anchored window, writing only the gap so all
        # previously covered frames stay byte-identical.
        anchors.append((f_lat - w, covered))
    for a, wlo in anchors:
        ww = min(w, f_lat)
        nv = 4 * (ww - 1) + 1
        p, rm = run(ctrl[:, a:a + ww], nv)
        if rm is not None:
            rms.append(rm.float().cpu())
        pj = getattr(getattr(model, "net", None), "projectors", None)
        af = (getattr(pj[f"l{tap}"], "_pnp_accept_frac", None)
              if pj is not None and f"l{tap}" in pj else None)
        if af is not None:
            afs.append(float(af))
        for k in DUMP_KEYS:
            arr = p[k].cpu().numpy()
            if acc[k] is None:
                shape = (n_px,) + arr.shape[1:]
                acc[k] = np.zeros(shape, arr.dtype)
            lo, hi = 4 * a, min(4 * a + nv, n_px)
            if wlo is not None:
                lo = max(lo, wlo)
            acc[k][lo:hi] = arr[lo - 4 * a: hi - 4 * a]
    acc["_pred_K"] = _fit_pred_K(torch.cat(rms, dim=1), intr) if rms else None
    # fraction of (frame, hand) rows the mixed_pnp LSQ accepted rather than
    # silently falling back to inv_proj — the telemetry that exposes a broken decode
    acc["_pnp_accept_frac"] = float(np.mean(afs)) if afs else None
    return acc
