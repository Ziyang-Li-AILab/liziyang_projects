"""Config loading: repro YAML + the released model options it points at."""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

from ace_repro import CODE_ROOT


def _abspath(base: str, p: str | None) -> str | None:
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def load_config(path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a repro config and attach the released model options under ``model``.

    ``overrides`` are ``a.b.c=value`` strings (YAML-parsed values).
    Relative paths in the repro config's ``paths`` block and in the released
    options' ``paths`` block are resolved against ``code/`` so that the released
    inference code (``GeoDitModel``) works unchanged.
    """
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(val)

    cfg_dir = os.path.dirname(path)
    opt_path = _abspath(cfg_dir, cfg["base_options"])
    with open(opt_path, encoding="utf-8") as f:
        model_opt = yaml.safe_load(f)
    for k, v in list(model_opt.get("paths", {}).items()):
        if isinstance(v, str):
            model_opt["paths"][k] = _abspath(CODE_ROOT, v)
    for k, v in list(cfg.get("paths", {}).items()):
        if isinstance(v, str):
            cfg["paths"][k] = _abspath(CODE_ROOT, v)
    if cfg.get("kfree"):
        model_opt.setdefault("backbone", {})["self_ray_decode"] = True
    cfg["model"] = model_opt
    cfg["_config_path"] = path
    return cfg


def dump_config(cfg: dict[str, Any], path: str) -> None:
    c = copy.deepcopy(cfg)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(c, f, sort_keys=False)
