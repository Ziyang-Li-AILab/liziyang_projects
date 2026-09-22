"""Trainable ACE-Ego-Hand: the released ``GeoDiT`` plus what training needs.

* builds the network from the released options (architecture untouched);
* attaches the hand models the mixed-PnP decode requires;
* K-free: adds the paper's fitted-camera bearing path (§3.5) on top of the
  released ray-mode decode (see inventory.md N3) without editing the release;
* exposes the three optimizer parameter groups of App. B.2 and a checkpoint
  format ``infer_video.py --ckpt`` can load.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ace_ego_hand.archs.geodit_arch import GeoDiT
from ace_ego_hand.archs.wan_backbone import fold_tokens, fold_tokens_pixelshuffle

from ace_repro.camera import fit_pinhole, fit_to_intrinsics, intrinsics_to_list
from ace_repro.hand_model import build_hand_models


def build_geodit(opt: dict, device: torch.device) -> GeoDiT:
    """Mirror of ``GeoDitModel.__init__``'s constructor call (kept 1:1 with the release)."""
    p, bb = opt["paths"], opt.get("backbone", {})
    net = GeoDiT(
        model_root=p["model_root"], tap_layers=bb.get("tap_layers", [16]),
        projector_kwargs=opt["projector"], caption_embed_path=p["caption_embed"],
        videox_config=p.get("videox_config"), mh_expand=bb.get("mh_expand", True),
        train_backbone=bb.get("train_backbone", False),
        gradient_checkpointing=bb.get("gradient_checkpointing", False),
        projector_warmstart=p.get("projector_warmstart"), lora=bb.get("lora"),
        emode_t=float(bb.get("emode_t", 1000.0)), tap_grid=bb.get("tap_grid", "patch"),
        aux_geo=bb.get("aux_geo", False), feature_mode=bb.get("feature_mode", "emode"),
        nv_sigma=float(bb.get("nv_sigma", 0.5)),
        geo_control_mods=tuple(bb.get("geo_control_mods", ())),
        geo_control_source=str(bb.get("geo_control_source", "render3")),
        raymap=bool(bb.get("raymap", False)), self_ray_pe=bool(bb.get("self_ray_pe", False)),
        self_ray_decode=bool(bb.get("self_ray_decode", False)),
        mixed_pnp_detach_mano=bool(bb.get("mixed_pnp_detach_mano", False)),
        gen_render=bool(bb.get("gen_render", False)),
        gen_render_detach=bool(bb.get("gen_render_detach", False)),
        gen_concat_gt=bool(bb.get("gen_concat_gt", False)),
    ).to(device)
    return net


class ReproModel(nn.Module):
    def __init__(self, cfg: dict, device: torch.device, allow_fake_mano: bool = False):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.kfree = bool(cfg.get("kfree", False))
        self.net = build_geodit(cfg["model"], device)
        self.hand_models, self.hand_kind = build_hand_models(
            device, cfg["paths"].get("mano_dir"), allow_fake=allow_fake_mano)
        if str(cfg["model"]["projector"].get("cam_trans_decode", "inv_proj")) == "mixed_pnp":
            for proj in self.net.projectors.values():
                proj.set_mano_models(self.hand_models)
        self.set_train_mode()

    # ------------------------------------------------------------ modes
    def set_train_mode(self):
        """Frozen bf16 backbone stays in eval (no dropout anyway); trainables train."""
        self.net.projectors.train()
        if self.net.raymap:
            self.net.raymap_heads.train()
        self.net.backbone.eval()

    def set_eval_mode(self):
        self.net.eval()

    # ------------------------------------------------------------ params
    def param_groups(self, opt_cfg: dict) -> list[dict]:
        """App. B.2: decoder/heads 2e-4, LoRA 1e-4, patch embedding 2e-5; one weight decay."""
        groups = {"decoder": [], "lora": [], "patch_embed": []}
        for name, p in self.net.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in name:
                groups["lora"].append(p)
            elif "patch_embedding" in name:
                groups["patch_embed"].append(p)
            else:
                groups["decoder"].append(p)
        lrs = {"decoder": opt_cfg["lr_decoder"], "lora": opt_cfg["lr_lora"],
               "patch_embed": opt_cfg["lr_patch_embed"]}
        out = [{"params": v, "lr": lrs[k], "name": k, "initial_lr": lrs[k]} for k, v in groups.items() if v]
        for g in out:
            print(f"[model] group {g['name']:<12s} {sum(p.numel() for p in g['params'])/1e6:8.3f}M  lr={g['lr']:.1e}")
        return out

    # ------------------------------------------------------------ forward
    def forward(self, batch: dict) -> tuple[dict, dict | None]:
        """Returns ``(forward_emode-style dict, fit-or-None)``."""
        ctrl = batch["latent"].to(self.device, torch.float32)
        T = int(batch["n_video_frames"])
        intr = intrinsics_to_list(batch["K"])
        if not self.kfree:
            return self.net.forward_emode(ctrl, T, intr), None
        return self._forward_kfree(ctrl, T, batch)

    def _forward_kfree(self, ctrl: torch.Tensor, T: int, batch: dict):
        """K-free forward = released ray-mode forward + the paper's fitted-camera bearings.

        The predicted ray field is fitted with the closed-form per-axis regression
        (paper §3.5). When the fit passes its guards for every row, the mixed-PnP
        decode runs its standard (K-given) branch on the *fitted* intrinsics; when
        it fails (fisheye sources, degenerate fields) the batch falls back to the
        released ray-mode decode, which samples the field at the joint pixels.
        """
        net = self.net
        taps, head_out, grid = net.extract_features(ctrl)
        fold = fold_tokens_pixelshuffle if net.tap_grid == "pixelshuffle" else fold_tokens
        preds, raymap = {}, {}
        fit_out = None
        W, H = float(batch["K"][0, 4]), float(batch["K"][0, 5])
        for layer in net.tap_layers:
            feat = fold(taps[layer], grid).float()
            rm = net.raymap_heads[f"l{layer}"](feat)                       # (B,3,F,H,W)
            raymap[layer] = rm
            pr_rays = rm.mean(dim=2)                                       # camera constant per clip
            fit = fit_pinhole(pr_rays)
            fit_out = fit if fit_out is None else fit_out
            use_fit = bool(fit["valid"].all()) and batch.get("camera", "pinhole") == "pinhole"
            proj = net.projectors[f"l{layer}"]
            if use_fit:
                K_fit = fit_to_intrinsics({k: v.detach() for k, v in fit.items()}, W, H)
                proj.self_ray_decode = False                               # K-branch on fitted K
                preds[layer] = proj.forward_batched(feat, n_video_frames=T,
                                                    intrinsics_list=intrinsics_to_list(K_fit),
                                                    pred_rays=pr_rays)
                proj.self_ray_decode = True
            else:                                                          # released ray-mode fallback
                preds[layer] = proj.forward_batched(feat, n_video_frames=T,
                                                    intrinsics_list=intrinsics_to_list(batch["K"]),
                                                    pred_rays=pr_rays)
        out = {"preds": preds, "head_out": head_out, "aux_geo": {}, "raymap": raymap, "gen_render": {}}
        return out, fit_out

    # ------------------------------------------------------------ ckpt
    def state_for_release(self) -> dict:
        """Checkpoint payload in the released ``--ckpt`` format (see GeoDitModel.load_inference)."""
        ck = {"projectors": self.net.projectors.state_dict()}
        if self.net.raymap:
            ck["raymap_heads"] = self.net.raymap_heads.state_dict()
        ck["backbone_trainable"] = {n: p.detach().cpu() for n, p in self.net.backbone.dit.named_parameters()
                                    if p.requires_grad}
        return ck

    def load_release(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net.projectors.load_state_dict(ck["projectors"])
        if self.net.raymap and "raymap_heads" in ck:
            self.net.raymap_heads.load_state_dict(ck["raymap_heads"])
        own = dict(self.net.backbone.dit.named_parameters())
        n = 0
        for k, v in (ck.get("backbone_trainable") or {}).items():
            if k in own and own[k].shape == v.shape:
                own[k].data.copy_(v.to(own[k].dtype))
                n += 1
        print(f"[model] loaded {n} backbone trainable tensors from {path}")
