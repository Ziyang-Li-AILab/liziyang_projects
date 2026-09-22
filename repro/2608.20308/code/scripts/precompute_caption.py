#!/usr/bin/env python3
"""Precompute the fixed caption embedding (run once after downloading the
Wan2.2-Fun-5B-Control release -- see the README, "Download checkpoints").

The backbone consumes one constant caption; its umT5 embedding is computed here
and cached to ``cache/caption_embed.pt`` so the 11 GB text encoder never has to
be loaded again.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))

MODEL_ROOT = ROOT / "ckpt" / "Wan2.2-Fun-5B-Control"
VIDEOX_CFG = ROOT / "third_party" / "config" / "wan2.2" / "wan_civitai_5b.yaml"

CAPTION = ("Three geometry renders of two hands on black background: "
           "color-coded depth, joint skeleton, surface normals.")


def cache_caption(device) -> None:
    out = ROOT / "cache" / "caption_embed.pt"
    if out.exists():
        print(f"caption embed exists: {out}")
        return
    from transformers import AutoTokenizer
    from videox_fun.models import WanT5EncoderModel
    config = OmegaConf.load(str(VIDEOX_CFG))
    tok = AutoTokenizer.from_pretrained(
        os.path.join(str(MODEL_ROOT), config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")))
    te = WanT5EncoderModel.from_pretrained(
        os.path.join(str(MODEL_ROOT), config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
    ).eval().to(device)
    ids = tok(CAPTION, padding="max_length", max_length=512, truncation=True,
              add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        emb = te(ids.input_ids.to(device), attention_mask=ids.attention_mask.to(device))[0]
    seq_len = int(ids.attention_mask.gt(0).sum())
    emb = emb[0, :seq_len].to(torch.float32).cpu()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embed": emb, "caption": CAPTION}, out)
    print(f"caption embed ({tuple(emb.shape)}) -> {out}")
    del te
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    cache_caption(torch.device(ap.parse_args().device))
