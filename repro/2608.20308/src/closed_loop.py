#!/usr/bin/env python
"""Minimal train -> checkpoint -> reload -> infer loop.

The network is the released GeoDiT (paper §3.1–3.3). The loss is Eq. 6.
Clips are the synthetic stand-in, so this runs without the licensed MANO
pkls. Pass --mano-dir once those pkls are converted (see REPRODUCTION_NOTES.md).

    python closed_loop.py --steps 2 --allow-fake-mano
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ace_repro.config import dump_config, load_config  # noqa: E402
from ace_repro.data.mixture import MixtureLoader  # noqa: E402
from ace_repro.data.schema import batch_to, collate  # noqa: E402
from ace_repro.losses import compute_losses  # noqa: E402
from ace_repro.model import ReproModel  # noqa: E402
from ace_repro.optim import build_optimizer, build_scheduler  # noqa: E402
from train import build_sources, seed_all  # noqa: E402


def _one_step(model, batch, opt, sch, cfg, step):
    opt.zero_grad(set_to_none=True)
    out, fit = model(batch)
    total, terms = compute_losses(
        out, batch, model.hand_models, cfg["loss"], step=step,
        fit=fit, fit_warmup_steps=cfg["schedule"].get("fit_warmup_steps", 500))
    if not torch.isfinite(total):
        raise FloatingPointError(f"non-finite loss at step {step}")
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for g in opt.param_groups for p in g["params"]], cfg["optimizer"]["grad_clip"])
    opt.step()
    sch.step()
    return float(total), {k: float(v) for k, v in terms.items()}


def train_and_save(cfg, device, steps: int, allow_fake: bool, run_dir: str) -> tuple[str, list[float]]:
    """A few optimizer steps, then the inference-format checkpoint (no Adam state)."""
    model = ReproModel(cfg, device, allow_fake_mano=allow_fake)
    sources, _val = build_sources(cfg, model, device, debug=True)
    loader = MixtureLoader(sources, cfg["batch"]["clips_per_gpu"], seed=cfg["seed"])
    opt = build_optimizer(model.param_groups(cfg["optimizer"]), cfg["optimizer"])
    sch = build_scheduler(opt, cfg["schedule"])
    losses = []
    for step in range(steps):
        batch = batch_to(loader.next_batch(), device)
        loss, terms = _one_step(model, batch, opt, sch, cfg, step)
        losses.append(loss)
        print(f"[train {step + 1}/{steps}] loss={loss:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in terms.items()))
    path = os.path.join(run_dir, "ckpt_infer.pt")
    torch.save(model.state_for_release(), path)
    del model
    torch.cuda.empty_cache()
    return path, losses


def infer_reloaded(cfg, device, ckpt: str, allow_fake: bool, run_dir: str) -> dict:
    """Load the checkpoint into a new module and run one eval forward."""
    model = ReproModel(cfg, device, allow_fake_mano=allow_fake)
    model.load_release(ckpt)
    model.set_eval_mode()
    _sources, val = build_sources(cfg, model, device, debug=True)
    batch = batch_to(collate([val[0]]), device)
    with torch.no_grad():
        out, _fit = model(batch)
    layer = list(out["preds"])[0]
    pred = out["preds"][layer][0]
    cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in pred.items()}
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
        "hand_model": model.hand_kind,
        "ckpt": ckpt,
        "pred": pred_path,
        "clip_id": batch["clip_ids"][0],
        "shapes": {k: list(v.shape) for k, v in cpu.items() if torch.is_tensor(v)},
        "cam_trans_wrist0": [round(x, 4) for x in cpu["cam_trans"][0, 1].tolist()],
    }
    print(f"[infer] clip={summary['clip_id']} hand_model={model.hand_kind} "
          f"cam_trans[0,right]={summary['cam_trans_wrist0']} -> {pred_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "repro-default.yaml"))
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--allow-fake-mano", action="store_true")
    ap.add_argument("--mano-dir", default=None, help="parent of mano/MANO_{LEFT,RIGHT}.pkl")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["name"] = "closed-loop"
    cfg["batch"]["clips_per_gpu"] = 2
    cfg["schedule"]["total_steps"] = args.steps
    cfg["schedule"]["warmup_steps"] = min(1, args.steps)
    if args.mano_dir:
        cfg["paths"]["mano_dir"] = os.path.abspath(args.mano_dir)
    allow_fake = args.allow_fake_mano or not os.path.isdir(cfg["paths"].get("mano_dir", ""))
    seed_all(cfg["seed"])
    device = torch.device("cuda")
    run_dir = os.path.join(cfg["paths"]["out_root"], cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    dump_config(cfg, os.path.join(run_dir, "config.yaml"))
    ckpt, losses = train_and_save(cfg, device, args.steps, allow_fake, run_dir)
    summary = infer_reloaded(cfg, device, ckpt, allow_fake, run_dir)
    summary["train_losses"] = [round(x, 4) for x in losses]
    with open(os.path.join(run_dir, "loop_result.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[closed-loop] ok  losses={summary['train_losses']}  {run_dir}")


if __name__ == "__main__":
    main()
