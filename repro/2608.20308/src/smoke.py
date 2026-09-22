#!/usr/bin/env python
"""Stage-6 smoke tiers (reproduce skill, 06-smoke.md) + a tier-0 component check.

    python smoke.py --config configs/repro-default.yaml --tier 0|1|2|3 [--allow-fake-mano]

Tier 0: VAE round trip, camera-fit round trip, loss==0 and metrics==perfect on GT.
Tier 1: one batch forward in train + eval mode; finite loss; grad norm > 0.
Tier 2: forward -> backward -> step -> forward on the same batch; loss decreases.
Tier 3: 20 real iterations; 5-step rolling mean trends down; no NaN.
Logs go to ../smoke_logs/tier<N>.log.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ace_repro import REPRO_ROOT  # noqa: E402
from ace_repro.camera import fit_pinhole, fit_to_intrinsics, rays_from_intrinsics  # noqa: E402
from ace_repro.config import load_config  # noqa: E402
from ace_repro.data.latents import LatentEncoder  # noqa: E402
from ace_repro.data.schema import batch_to, collate  # noqa: E402
from ace_repro.data.synthetic import SyntheticClips, make_clip_gt, render_clip  # noqa: E402
from ace_repro.eval.metrics import SegmentScorer  # noqa: E402
from ace_repro.hand_model import build_hand_models  # noqa: E402
from ace_repro.losses import compute_losses  # noqa: E402
from ace_repro.optim import build_optimizer, build_scheduler  # noqa: E402

LOG_DIR = os.path.join(REPRO_ROOT, "smoke_logs")


class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def gt_as_pred(batch: dict) -> dict:
    """Build a 'perfect' prediction dict (per sample) from the GT batch."""
    B = batch["go"].shape[0]
    outs = []
    for b in range(B):
        ex = batch["exists"][b].float()
        outs.append({"global_orient": batch["go"][b], "hand_pose": batch["hp"][b],
                     "betas": batch["betas"][b][None].expand(batch["go"].shape[1], 2, 10),
                     "segment_betas": batch["betas"][b], "cam_trans": batch["trans"][b],
                     "exists_3d": ex * 0.98 + 0.01, "exists_2d": batch["visible"][b].float() * 0.98 + 0.01,
                     "direct_joints_rootrel": batch["joints_cam"][b] - batch["joints_cam"][b][..., :1, :],
                     "direct_wrist_cam": batch["joints_cam"][b][..., 0, :],
                     "direct_joints_cam": batch["joints_cam"][b], "direct_joints2d": batch["joints2d"][b]})
    return outs


def tier0(cfg, device, log, allow_fake):
    hm, kind = build_hand_models(device, cfg["paths"].get("mano_dir"), allow_fake=allow_fake)
    log(f"hand model: {kind}")
    # camera fit round trip
    K = torch.tensor([[520.0, 520.0, 240.0, 240.0, 480.0, 480.0], [700.0, 690.0, 340.0, 230.0, 672.0, 480.0]], device=device)
    rays = rays_from_intrinsics(K, 15, 21)
    fit = fit_pinhole(rays)
    K_hat = fit_to_intrinsics(fit, 1.0, 1.0)
    K_hat = torch.cat([K_hat[:, :2] * K[:, 4:6], K_hat[:, 2:4] * K[:, 4:6]], -1)
    err = (K_hat - K[:, :4]).abs().max().item()
    log(f"camera fit round trip: max |K_hat-K| = {err:.4f} px, valid={fit['valid'].tolist()}")
    assert err < 1e-2 and fit["valid"].all(), "pinhole fit does not invert the ray field"
    # VAE + synthetic clip
    enc = LatentEncoder(device)
    g = torch.Generator().manual_seed(0)
    gt = make_clip_gt(hm, g, 21, 480, 480, "t0")
    frames = render_clip(gt, 480, 480)
    t = time.time()
    lat = enc.encode_uint8(frames)
    log(f"VAE encode 21x480x480 -> latent {tuple(lat.shape)} {lat.dtype} in {time.time() - t:.1f}s; "
        f"std={lat.float().std():.3f} finite={torch.isfinite(lat.float()).all().item()}")
    assert lat.shape[0] == 48 and lat.shape[1] == 6 and lat.shape[2] == 30 and lat.shape[3] == 30
    ds = SyntheticClips(hm, enc, 2, (480, 480), 21, seed=123, cache_dir="/tmp/ace_repro_smoke", name="t0")
    batch = batch_to(collate([ds[0], ds[1]]), device)
    # loss on GT-as-prediction must be ~0 (except presence BCE at 0.98 confidence)
    fake_out = {"preds": {15: gt_as_pred(batch)}, "raymap": {15: rays_from_intrinsics(batch["K"], 15, 15)[:, :, None].expand(-1, -1, 6, -1, -1)}}
    total, terms = compute_losses(fake_out, batch, hm, cfg["loss"], step=1000)
    log("loss on GT-as-prediction:", json.dumps({k: round(float(v), 5) for k, v in terms.items()}))
    for k, v in terms.items():
        if k.startswith("presence"):
            assert float(v) < 0.05, k
        elif k == "rot_geodesic":          # acos clamp floor: acos(1-1e-6) = 1.4e-3 rad
            assert float(v) < 2e-3, k
        else:
            assert float(v) < 1e-3, f"{k} = {float(v)} should be ~0 on GT"
    # metrics on GT-as-prediction must be perfect
    sc = SegmentScorer(hm)
    for b in range(2):
        p = gt_as_pred(batch)[b]
        gtb = {k: batch[k][b] for k in ("joints_cam", "joints2d", "joint_vis", "go", "hp", "betas", "trans", "exists", "visible", "has_mano")}
        sc.add_segment(p, gtb, batch["K"][b])
    m = sc.summary()
    log("metrics on GT-as-prediction:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}))
    assert m["F1"] > 0.999 and m["MPJPE-p"] < 1e-3 and m["CT-p"] < 1e-5 and m["GO-p"] < 2e-2
    log("TIER 0 PASS")


def _setup(cfg, device, allow_fake, n_clips=2, T=21):
    from ace_repro.model import ReproModel
    model = ReproModel(cfg, device, allow_fake_mano=allow_fake)
    enc = LatentEncoder(device)
    syn = cfg["data"]["synthetic"]
    ds = SyntheticClips(model.hand_models, enc, max(n_clips, 8), tuple(syn["res"]), T, seed=cfg["seed"],
                        cache_dir=os.path.join(cfg["paths"]["latent_cache"], "synthetic"), name="synthetic")
    del enc
    torch.cuda.empty_cache()
    return model, ds


def tier1(cfg, device, log, allow_fake):
    model, ds = _setup(cfg, device, allow_fake)
    batch = batch_to(collate([ds[0], ds[1]]), device)
    model.set_train_mode()
    t = time.time()
    out, fit = model(batch)
    total, terms = compute_losses(out, batch, model.hand_models, cfg["loss"], step=1000, fit=fit)
    log(f"train forward {time.time() - t:.1f}s  loss={float(total):.4f}  "
        + json.dumps({k: round(float(v), 4) for k, v in terms.items()}))
    assert torch.isfinite(total), "train loss is NaN/Inf"
    t = time.time()
    total.backward()
    params = [(n, p) for n, p in model.net.named_parameters() if p.requires_grad]
    gn = {"decoder": 0.0, "lora": 0.0, "patch_embed": 0.0}
    for n, p in params:
        if p.grad is None:
            continue
        key = "lora" if "lora_" in n else ("patch_embed" if "patch_embedding" in n else "decoder")
        gn[key] += float(p.grad.norm()) ** 2
    gn = {k: v ** 0.5 for k, v in gn.items()}
    log(f"backward {time.time() - t:.1f}s  grad norms per group: {json.dumps({k: round(v, 4) for k, v in gn.items()})}  "
        f"max mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    assert gn["decoder"] > 0 and gn["lora"] > 0 and gn["patch_embed"] > 0, "a parameter group received no gradient"
    model.set_eval_mode()
    with torch.no_grad():
        out_e, _ = model(batch)
    p = out_e["preds"][list(out_e["preds"])[0]][0]
    shapes = {k: tuple(v.shape) for k, v in p.items() if torch.is_tensor(v)}
    log("eval output shapes:", json.dumps(shapes))
    T = batch["n_video_frames"]
    assert shapes["cam_trans"] == (T, 2, 3) and shapes["direct_joints2d"] == (T, 2, 21, 2)
    assert all(torch.isfinite(v).all() for v in p.values() if torch.is_tensor(v))
    log(f"pnp accept frac: {getattr(list(model.net.projectors.values())[0], '_pnp_accept_frac', None)}")
    log("TIER 1 PASS")


def tier2(cfg, device, log, allow_fake):
    model, ds = _setup(cfg, device, allow_fake)
    batch = batch_to(collate([ds[0], ds[1]]), device)
    opt = build_optimizer(model.param_groups(cfg["optimizer"]), cfg["optimizer"])
    sch = build_scheduler(opt, {"total_steps": 100, "warmup_steps": 1})
    out, fit = model(batch)
    before, _ = compute_losses(out, batch, model.hand_models, cfg["loss"], step=1000, fit=fit)
    opt.zero_grad()
    before.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], cfg["optimizer"]["grad_clip"])
    opt.step()
    sch.step()
    n_state = sum(1 for g in opt.param_groups for p in g["params"] if p in opt.state and "exp_avg" in opt.state[p])
    n_grad = sum(1 for g in opt.param_groups for p in g["params"] if p.grad is not None)
    bad = [n for n, p in model.net.named_parameters() if p.requires_grad and not torch.isfinite(p).all()]
    with torch.no_grad():
        out2, fit2 = model(batch)
        after, _ = compute_losses(out2, batch, model.hand_models, cfg["loss"], step=1000, fit=fit2)
    log(f"loss: {float(before):.5f} -> {float(after):.5f}   optimizer state populated {n_state}/{n_grad} params with grads   "
        f"non-finite params: {len(bad)}")
    assert not bad and n_state == n_grad
    assert float(after) < float(before), "loss did not decrease after one step"
    log("TIER 2 PASS")


def tier3(cfg, device, log, allow_fake, n_iter=20):
    model, ds = _setup(cfg, device, allow_fake, n_clips=8)
    opt = build_optimizer(model.param_groups(cfg["optimizer"]), cfg["optimizer"])
    sch = build_scheduler(opt, {"total_steps": 200, "warmup_steps": 5})
    losses = []
    g = torch.Generator().manual_seed(0)
    for i in range(n_iter):
        idx = torch.randint(0, len(ds), (2,), generator=g).tolist()
        batch = batch_to(collate([ds[j] for j in idx]), device)
        out, fit = model(batch)
        total, terms = compute_losses(out, batch, model.hand_models, cfg["loss"], step=i, fit=fit)
        assert torch.isfinite(total), f"NaN at iter {i}"
        opt.zero_grad()
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for g_ in opt.param_groups for p in g_["params"]], cfg["optimizer"]["grad_clip"])
        opt.step()
        sch.step()
        losses.append(float(total))
        log(f"iter {i:2d} loss={float(total):.4f} gnorm={float(gn):.2f} " + " ".join(f"{k}={float(v):.3f}" for k, v in terms.items()))
    roll = [sum(losses[i:i + 5]) / 5 for i in range(len(losses) - 4)]
    log("rolling-5:", [round(r, 4) for r in roll])
    trend = all(roll[i + 1] <= roll[i] * 1.05 for i in range(len(roll) - 1))
    log(f"trend down (<=5% bumps): {trend}; first {losses[0]:.4f} -> last {losses[-1]:.4f}")
    assert trend and losses[-1] < losses[0]
    log("TIER 3 PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "repro-default.yaml"))
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--allow-fake-mano", action="store_true")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda")
    log = Tee(os.path.join(LOG_DIR, f"tier{args.tier}.log"))
    log(f"# smoke tier {args.tier}  config={args.config} kfree={cfg.get('kfree')} {time.strftime('%Y-%m-%d %H:%M:%S')}")
    {0: tier0, 1: tier1, 2: tier2, 3: tier3}[args.tier](cfg, device, log, args.allow_fake_mano)


if __name__ == "__main__":
    main()
