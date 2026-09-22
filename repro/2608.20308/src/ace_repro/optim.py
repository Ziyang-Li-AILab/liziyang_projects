"""AdamW + linear warm-up + cosine decay (gaps_filled D/F)."""
from __future__ import annotations

import math

import torch


def build_optimizer(param_groups: list[dict], opt_cfg: dict) -> torch.optim.Optimizer:
    assert opt_cfg.get("name", "AdamW") == "AdamW"
    return torch.optim.AdamW(param_groups, betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
                             eps=float(opt_cfg.get("eps", 1e-8)),
                             weight_decay=float(opt_cfg.get("weight_decay", 0.01)))


def lr_multiplier(step: int, total: int, warmup: int, min_ratio: float = 0.0) -> float:
    """Linear warm-up to 1 over ``warmup`` steps, then cosine to ``min_ratio`` at ``total``."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))


def build_scheduler(optimizer: torch.optim.Optimizer, sch_cfg: dict) -> torch.optim.lr_scheduler.LambdaLR:
    total, warm = int(sch_cfg["total_steps"]), int(sch_cfg["warmup_steps"])
    floor = float(sch_cfg.get("min_lr_ratio", 0.0))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_multiplier(s, total, warm, floor))
