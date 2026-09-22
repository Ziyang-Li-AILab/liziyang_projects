#!/usr/bin/env python
"""ACE-Ego-Hand training launch (the piece the release omits).

    python train.py --config configs/repro-default.yaml [--debug-mode] [--allow-fake-mano]

--debug-mode: 10 iterations on the synthetic source, then exit (stage-4 gate).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ace_repro.config import dump_config, load_config  # noqa: E402
from ace_repro.data.mixture import MixtureLoader  # noqa: E402
from ace_repro.data.schema import batch_to, collate  # noqa: E402
from ace_repro.data.store import ClipStore  # noqa: E402
from ace_repro.eval.metrics import SegmentScorer  # noqa: E402
from ace_repro.losses import compute_losses  # noqa: E402
from ace_repro.model import ReproModel  # noqa: E402
from ace_repro.optim import build_optimizer, build_scheduler  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_sources(cfg: dict, model: ReproModel, device, debug: bool):
    """Real sources found under data_root (+ synthetic when asked / nothing else exists)."""
    sources, val = {}, None
    root = cfg["paths"]["data_root"]
    if not debug:
        for name, spec in cfg["data"]["sources"].items():
            d = os.path.join(root, name, "train")
            if os.path.isdir(d):
                ds = ClipStore(d, window=(spec["kind"] == "video"), seed=cfg["seed"])
                if len(ds):
                    sources[name] = (ds, float(spec["weight"]))
                    print(f"[data] {name:<12s} {len(ds):6d} clips  weight={spec['weight']}")
        vdir = os.path.join(root, "arctic", "test")
        if os.path.isdir(vdir):
            val = ClipStore(vdir, window=False)
    if debug or not sources:
        from ace_repro.data.latents import LatentEncoder
        from ace_repro.data.synthetic import SyntheticClips
        syn = cfg["data"]["synthetic"]
        enc = LatentEncoder(device)
        cache = os.path.join(cfg["paths"]["latent_cache"], "synthetic")
        n_train = 16 if debug else syn["n_train"]
        train = SyntheticClips(model.hand_models, enc, n_train, tuple(syn["res"]), syn["video_frames"],
                               seed=cfg["seed"], cache_dir=cache, name="synthetic")
        val = SyntheticClips(model.hand_models, enc, syn["n_val"], tuple(syn["res"]), syn["video_frames"],
                             seed=cfg["seed"] + 1, cache_dir=cache, name="synthetic_val")
        sources["synthetic"] = (train, 1.0)
        del enc
        torch.cuda.empty_cache()
        print(f"[data] synthetic {len(train)} train / {len(val)} val clips (hand_model={model.hand_kind})")
    return sources, val


@torch.no_grad()
def evaluate(model: ReproModel, val, device, max_clips: int | None = None) -> dict:
    model.set_eval_mode()
    scorer = SegmentScorer(model.hand_models)
    n = len(val) if max_clips is None else min(len(val), max_clips)
    for i in range(n):
        batch = batch_to(collate([val[i]]), device)
        out, _ = model(batch)
        layer = list(out["preds"].keys())[0]
        p = out["preds"][layer][0]
        gt = {k: batch[k][0] for k in ("joints_cam", "joints2d", "joint_vis", "go", "hp", "betas", "trans",
                                       "exists", "visible", "has_mano")}
        scorer.add_segment(p, gt, batch["K"][0])
    model.set_train_mode()
    return scorer.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--debug-mode", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="override schedule.total_steps")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None, help="released .pt to warm-start from")
    ap.add_argument("--allow-fake-mano", action="store_true")
    ap.add_argument("--override", nargs="*", default=[], help="dotted.key=value overrides")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    if args.run_name:
        cfg["name"] = args.run_name
    if args.steps:
        cfg["schedule"]["total_steps"] = args.steps
    if args.debug_mode:
        cfg["schedule"].update(total_steps=10, warmup_steps=2, eval_every=10, save_every=10, log_every=1)
        cfg["batch"]["clips_per_gpu"] = 2
        cfg["name"] = cfg["name"] + "-debug"
    seed_all(cfg["seed"])
    device = torch.device("cuda")
    run_dir = os.path.join(cfg["paths"]["out_root"], cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    dump_config(cfg, os.path.join(run_dir, "config.yaml"))

    model = ReproModel(cfg, device, allow_fake_mano=args.allow_fake_mano)
    if args.init_from:
        model.load_release(args.init_from)
    sources, val = build_sources(cfg, model, device, args.debug_mode)
    loader = MixtureLoader(sources, cfg["batch"]["clips_per_gpu"], seed=cfg["seed"])
    optimizer = build_optimizer(model.param_groups(cfg["optimizer"]), cfg["optimizer"])
    scheduler = build_scheduler(optimizer, cfg["schedule"])
    accum = int(cfg["batch"].get("grad_accum", 1))
    sch = cfg["schedule"]
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_release(args.resume)
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        step = loader.step = ck["step"]
        print(f"[train] resumed at step {step}")

    log = open(os.path.join(run_dir, "train_log.jsonl"), "a")
    t0 = time.time()
    while step < sch["total_steps"]:
        optimizer.zero_grad(set_to_none=True)
        acc_terms, acc_total = {}, 0.0
        for _ in range(accum):
            batch = batch_to(loader.next_batch(), device)
            out, fit = model(batch)
            total, terms = compute_losses(out, batch, model.hand_models, cfg["loss"], step=step, fit=fit,
                                          fit_warmup_steps=sch.get("fit_warmup_steps", 500))
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: "
                                         f"{ {k: float(v) for k, v in terms.items()} }")
            (total / accum).backward()
            acc_total += float(total) / accum
            for k, v in terms.items():
                acc_terms[k] = acc_terms.get(k, 0.0) + float(v) / accum
        gnorm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]],
                                               cfg["optimizer"]["grad_clip"])
        optimizer.step()
        scheduler.step()
        step += 1
        if step % sch["log_every"] == 0 or step == 1:
            rec = {"step": step, "loss": acc_total, "grad_norm": float(gnorm), "dataset": batch["dataset"],
                   "lr_decoder": optimizer.param_groups[0]["lr"], "time": time.time() - t0,
                   "pnp_accept": getattr(list(model.net.projectors.values())[0], "_pnp_accept_frac", None),
                   **{f"L_{k}": v for k, v in acc_terms.items()}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{step:6d}] loss={acc_total:.4f} gnorm={float(gnorm):.2f} src={batch['dataset']} "
                  + " ".join(f"{k}={v:.3f}" for k, v in acc_terms.items()))
        if val is not None and step % sch["eval_every"] == 0:
            m = evaluate(model, val, device, max_clips=32 if args.debug_mode else None)
            m.update(step=step, hand_model=model.hand_kind)
            with open(os.path.join(run_dir, "eval_log.jsonl"), "a") as f:
                f.write(json.dumps(m) + "\n")
            print(f"[eval @ {step}] " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float)))
        if step % sch["save_every"] == 0 or step == sch["total_steps"]:
            ck = model.state_for_release()
            ck.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), step=step,
                      config=cfg["_config_path"])
            torch.save(ck, os.path.join(run_dir, f"ckpt_step{step:06d}.pt"))
            torch.save(ck, os.path.join(run_dir, "ckpt_last.pt"))
    print(f"[train] done: {step} steps in {(time.time() - t0) / 60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
