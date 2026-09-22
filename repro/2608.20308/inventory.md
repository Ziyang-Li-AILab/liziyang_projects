# Inventory — ACE-Ego-Hand official code vs. what reproduction needs

**Repo**: https://github.com/ggxxii/ACE-Ego-Hand — MIT. Snapshot in `code/` from tarball of `main` @ `9757868` (2026-09-10; `git clone` timed out, `codeload.github.com` tarball used instead, so `code/` has no `.git`). Created 2026-08-21; 105 stars, 5 forks, 6 open / 0 closed issues (2026-09-21). Repo was named `dreamhand` at v1 of the paper.
**Checkpoints**: https://huggingface.co/acerobotics2025/ACE-Ego-Hand — `ace_ego_hand_k.pt`, `ace_ego_hand_kfree.pt` (CC BY-NC 4.0; sizes not retrievable from this network). Backbone `Wan2.2-Fun-5B-Control` from VideoX-Fun (separate download, ~10 GB bf16 DiT + VAE + umT5-xxl text encoder ~11 GB, needed once for the caption cache).
**README claim**: "This repository ships inference only; training code is not included yet." TODO list: evaluation code, fisheye support, training code and data-prep scripts all unchecked.
**Local environment**: no `torch` installed; nothing here was executed. Every statement below is from static reading.

## 1. Component table

| Component | Present? | Path | Notes |
|---|---|---|---|
| Model definition — backbone wrapper | **y** | `code/ace_ego_hand/archs/wan_backbone.py` (298 lines) | Wraps `Wan2_2Transformer3DModel` imported from **VideoX-Fun** (`third_party/`, not vendored). Loads the **low-noise** expert (`transformer_low_noise_model_subpath`). LoRA via `peft.inject_adapter_in_model` on 10 linears/block. `forward_features` re-implements the stock forward with block taps and optional early exit (`run_head=False`). Contains a multi-head "render" expansion (`mh_expand_dit`, 148->244 in-ch, 48->144 out-ch) that the released configs disable (`mh_expand: false`). |
| Model definition — composite | **y** | `code/ace_ego_hand/archs/geodit_arch.py` (494) | `GeoDiT`: input assembly for `feature_mode in {emode, noised_video}`, ray head (`raymap: true`), fixed caption buffer, `forward_emode` → projector. Paper runs = `noised_video` with `nv_sigma=0.0` (docstring: "the regime used by the paper runs"). Also contains a flow-matching sampler (`_sample_flow_noise`) for a generative "dmode"/`gen_overlay` loss path that the paper does not describe. |
| Model definition — decoder | **y** | `code/ace_ego_hand/archs/projector/memory_projector.py` (1449), `memory_encoder.py` (638), `spatial_readout.py` (109) | `MemorySegmentBetasTransformer` = paper's Bidirectional Spatiotemporal Decoder when `temporal_arch: alternating`. `MemoryAlternatingEncoder` self-describes as a port of VGGT-Omega's register-token alternating attention. Heavily parameterised for ablations (`pooled` arch, `camera_tokens`, `alt_mem_tokens`, `mano_readout`, `joints2d_readout`, `mixed_pnp_detach_mano`, `shape_ema_logit`...). ~40% of the module is dead for the released configs (paper B.1: 20.347M decoder params registered, 12.287M used). |
| Mixed-PnP translation solver | **y** | `memory_projector.py::_decode_cam_trans_mixed_pnp` (L1259–1421) | Exact Eq. 5 with gates `_PNP_Z_NEAR=0.05`, `_PNP_INTERIOR_MARGIN=0.02`, `_PNP_MIN_INTERIOR=6`, `_PNP_RESID_FRAC=0.25`, `_PNP_RESID_MIN_PX=15` (= App. B.1). Differentiable; `torch.where` selects branch. |
| K-free camera fit (paper §3.3, App. B.1) | **partial / mismatch** | `code/ace_ego_hand/inference.py::_fit_pred_K` (post-hoc only) | The paper's fitted pinhole `(f_hat, c_hat)` feeding the bearings and the wrist fallback is **not** in the model decode. `self_ray_decode: true` samples the ray field bilinearly at the 2D anchors (`grid_sample`, `align_corners=True`) — the paper's *fallback* path. `_fit_pred_K` fits `(fx,fy,cx,cy)` by per-axis `lstsq` on `u = cx + fx*(dx/dz)` after the forward and only writes it to the pickle as `pred_intrinsics`. No variance floor / focal bracket guards anywhere. |
| Ray head | **y** | `geodit_arch.py` (`raymap: true`, L223 area) | 1x1 conv on tapped features → unit rays; temporal mean in projector. Zero-init per paper — verify at Stage 4 smoke. |
| MANO layer + conversions | **y** | `code/ace_ego_hand/mano_utils.py` (146), `code/scripts/convert_mano_pkls.py` (127) | smplx MANO, `flat_hand_mean=False`, 16 joints + 5 fingertip vertices → OpenPose-21 remap (`mano_forward_batch_full`). Converter strips chumpy from official pickles. |
| Video I/O + VAE | **y** | `code/ace_ego_hand/video_vae.py` (83), `code/infer_video.py` (220) | OpenCV decode → resize to `--encode_w 832` snapped to multiples of 32 → `AutoencoderKLWan3_8` (Wan2.2 VAE, 48 ch, x16 spatial, causal 4x+1 temporal). Output latent fp16 on CPU. |
| Inference driver | **y** | `code/infer_video.py`, `code/ace_ego_hand/inference.py` (112) | `tiled` (22-latent windows anchored per 81-frame segment; "what the benchmark numbers use") and `full` modes. Camera: `--camera cam.json` / `--intrinsics`; K-free uses a 60° placeholder that the model "does not read". Dumps `global_orient, hand_pose, betas, cam_trans, direct_joints2d, direct_joints_cam, direct_wrist_cam, exists_2d, exists_3d` per frame. |
| Model wrapper / checkpoint loading | **y** | `code/ace_ego_hand/models/geodit_model.py` (111) | Builds `GeoDiT`, attaches MANO models to projector, loads `.pt`. |
| Caption cache | **y** | `code/scripts/precompute_caption.py` (61) | Fixed caption `"Three geometry renders of two hands on black background: color-coded depth, joint skeleton, surface normals."` → umT5-xxl (512 tokens, padded) → `cache/caption_embed.pt`. Leftover of the abandoned render-generation design; still a model input. |
| Visualisation | **y** | `code/scripts/viz_preds.py` (334) | 2D overlay, CPU mesh rasteriser. Uses `exists_2d > 0.5` (2D) and `exists_3d > 0.5 & mean z > 5 cm` (3D) as display gates. |
| Configs | **y** | `code/options/ace_ego_hand_k.yml`, `ace_ego_hand_kfree.yml` | Identical except `self_ray_decode: true`. Key values below. |
| Data loaders | **n** | — | No dataset code at all. Needed: ARCTIC, HOT3D, H2O, OakInk2, Re:InterHand, FreiHAND, RHD readers → shared MANO format; HOI4D test reader. |
| Augmentation pipeline | **n** (hooks only) | `geodit_arch.py::extract_features(span_mask=...)`, `wan_backbone.py::build_control(visible_mask_lat)` | Span-dropout hook exists (zero latent frames + set mask channels `[48:52]=1`). Whether the paper's runs used it is **unknown**; paper mentions no augmentation. |
| Loss functions | **n** | — | Zero loss code. Only hints: docstrings reference `_mano_geometry_losses`, "joints-only cam_trans loss", "render loss", `gen_overlay`; `mixed_pnp_detach_mano` flag; direct heads (`direct_joints_cam`, `direct_wrist_cam`) exist as loss targets. All 8 terms of Eq. 6 (+ L_fit) must be written. |
| Training loop | **n** | — | Docstrings mention a trainer with `emode/dmode` routing under DDP (`forward(mode=...)`), `t2v_flag`, wandb. Not shipped. |
| Optimizer + schedule | **n** | — | Paper B.2 gives AdamW/wd/clip/cosine/warmup/3 LR groups; betas/eps/min-LR unspecified. |
| Eval scripts | **n** | — | Metric definitions in paper App. A.1 only. Segment lists for HOT3D (custom 126/72 recording split, 437 segments), OakInk2 (202 subset), HOI4D (166 recordings / 498 segments) unreleased (issue #5). EPE2D-p penalty details questioned in issue #6. |
| Pretrained checkpoints | **y** | HF `acerobotics2025/ACE-Ego-Hand` | Both configs. Pinhole only. Checkpoint schema covers all ablation variants ("single checkpoint schema covers every ablation", App. B.1). |
| Reproduction instructions | **partial** | `code/README.md` | Inference setup only (VideoX-Fun clone, MANO conversion, backbone download, caption cache). `requirements.txt` pins: torch 2.5.1, diffusers 0.36.0, transformers 4.57.3, peft 0.18.1, smplx 0.1.28, numpy 1.24.4, CUDA 12.1, Python 3.10. |

### Released config values (`options/ace_ego_hand_k.yml`)

```yaml
seed: 42
backbone:
  tap_layers: [15]            # block index, output taken after block 15 (16 blocks run)
  mh_expand: false            # no render-head expansion: 148 in / 48 out channels kept
  train_backbone: true        # gradients flow into LoRA + patch_embedding
  gradient_checkpointing: true
  feature_mode: noised_video  # x = (1-sigma) z + sigma eps, control = 0
  nv_sigma: 0.0               # => clean latent, t = 0
  lora: {lora_rank: 64, lora_alpha: 64, lora_dropout: 0.0}
  raymap: true                # ray head on
  self_ray_pe: true           # ray PE from predicted rays (not GT K)
  # kfree only: self_ray_decode: true
projector:
  hidden_dim: 384, num_temporal_layers: 4, num_heads: 8, ffn_mult: 4, dropout: 0.0
  max_spatial_tokens: 4096, max_video_frames: 256
  cam_trans_decode: mixed_pnp
  spatial_pe_mode: interp2d   # learned 16x16 grid, bilinear resize
  betas_scope: per_hand, per_hand_betas_source: slot_mean
  spatial_pool_mode: slot, num_direct_joints: 21, joints2d_readout: pooled
  temporal_arch: alternating, alt_num_register: 4, alt_num_layers: 4
  alt_query_mode: joint, alt_temporal_rope: true, alt_mem_tokens: 0
  use_ray_pe: true, ray_pe_n_freqs: 8
```

## 2. Sanity check: code vs paper architecture

| Paper claim | Code | Verdict |
|---|---|---|
| Feature grid `42x30` cells of `16x16` px, `D=3072` (§3.1) | `tap_grid` defaults to `patch`: DiT tokens after `(1,2,2)` patchify of the x16 VAE latent → `21x15` cells of **32 px**, `D=3072`. `pixelshuffle` mode (`D/4=768` at `42x30`) exists but is off. Timing in App. C.4 (28% of 1.28 s for DiT+decoder) is only consistent with the 32-px grid (see notes §3.5). | **Mismatch — paper text wrong or describes VAE grid.** Reproduce with 32-px tokens. |
| Truncate at block 15 of 30, run 16 blocks (Eq. 1) | `tap_layers: [15]`, `run_head=False` returns after block 15 | match |
| σ=0 clean latent, "keep released input/output channel counts" | `nv_sigma=0.0`, `x=z`, `y=zeros(100)`, `t=0`, 148-ch patch embed | match; paper omits that control channels are zero and that `t=0` |
| Text conditioning | fixed caption embedding in every cross-attn | **paper silent** |
| LoRA r=64, α=64, no dropout, 10 linears/block, all 30 blocks | `LORA_TARGETS` = self q,k,v,o + cross q,k,v,o + ffn.0, ffn.2; injected into every block | match (5.37M/block) |
| Patch embedding fully trainable; diffusion head registered but unreachable | `enable_training(train_patch_embed=True, train_head=True)`; both cast fp32 | match |
| 48 queries: 2 hand + 42 joint + 4 register | `hand_token(2)`, `joint_token(42)`, `register_token(4)` in `MemoryAlternatingEncoder` | match |
| 4 alternating layers, RoPE temporal, no causal mask | `alt_num_layers: 4`, `alt_temporal_rope: true`, `_RoPETemporalLayer` | match |
| soft-argmax over "normalized grid-cell center coordinates" | `linspace(0,1,W)` inclusive endpoints (`memory_encoder.py` L633) vs ray PE pixel centers `(j+0.5)/W` (`memory_projector.py` L137) | minor convention mismatch (learned around) |
| Camera Head predicts only log-depth | `head_cam_trans` outputs 3 values: `(u_norm, v_norm, log z)`; `u,v` used only by the wrist-ray fallback | paper under-describes |
| Fallback: "wrist placed on its own inverse-projected ray at depth t_z" | inverse projection of the Camera Head's own `(u,v)` (not the soft-argmax wrist anchor) | resolves open Q16 |
| K-free bearings from fitted pinhole | bearings sampled from ray field at anchors; no fit in decode | **Mismatch — released K-free decode = paper's no-fit fallback** |
| Existence + visibility scores | `exists_3d` (= "presence", existence) and `exists_2d` (visibility); zero-weight heads with bias −1 | match; naming map recorded |
| Clip-level β by temporal pooling of hand tokens | `betas_scope: per_hand`, `per_hand_betas_source: slot_mean`; also `shape_ema_logit` carry parameter | match (+ extra EMA knob) |

### Flags in code not mentioned in the paper
`feature_mode: emode` (render-generation regime, `t_emode=1000`), `mh_expand` multi-modality heads (depth/skeleton/normals renders), `gen_overlay` / `dmode` flow-matching loss, `span_mask` span-dropout, `geo_control_mods` (GT geometry renders as control — "DETACHED oracle"), `mixed_pnp_detach_mano`, `shape_ema_logit`, `camera_tokens`, `alt_mem_tokens` (cross-window memory), `tap_grid: pixelshuffle`, `joints2d_readout: spatial` (`SpatialJointReadout`), `cam_trans_decode: hamer`. These are the fossil record of the "DreamHand" design space; the released configs select one point in it.

## 3. Open-question triage (numbers = `paper.md` § Open questions)

| # | Question | Status | Resolution / where |
|---|---|---|---|
| 1 | Text conditioning of DiT | **resolved by code** | Fixed caption (`scripts/precompute_caption.py`), umT5-xxl, 512 tokens, fed to all cross-attn (`geodit_arch.py` L319) |
| 2 | Control channels at σ=0 | **resolved by code** | `y = zeros(100 ch)`; span-dropout sets `y[:,48:52]=1` on dropped frames (`geodit_arch.py` L298–313) |
| 3 | Timestep at σ=0 | **resolved by code** | `t = nv_sigma * 1000 = 0` (L314); adaLN via `time_embedding(sinusoidal(t))` in fp32 |
| 4 | LoRA on patch embedding? | **resolved by code** | No; patch embedding fully trained in fp32 (`wan_backbone.py` L155) |
| 5 | Compute dtype | **resolved by code** | Frozen base bf16; LoRA / patch embed / head fp32 params; `autocast(bf16)` around backbone; time-embedding fp32 island; projector receives `.float()` features |
| 6 | Gradient checkpointing | **resolved by code** | `gradient_checkpointing: true`, per-block `torch.utils.checkpoint` (non-reentrant) |
| 7 | Which Wan2.2 sub-model | **resolved by code** | `transformer_low_noise_model_subpath` from VideoX `wan_civitai_5b.yaml` |
| 8 | Decoder heads / FFN / dropout / activation | **resolved by code** | 8 heads, FFN ×4, dropout 0, GELU, pre-LayerNorm, `nn.MultiheadAttention` |
| 9 | Init of tokens / heads | **resolved by code** | tokens `N(0, 0.02)`; go/hp/cam_trans/betas/exists/direct-wrist heads zero-weight; exists bias −1; direct wrist bias `(0,0,0.5)`; cam head bias `(0,0,-0.693)` → initial `t_z = exp(-0.693) = 0.5 m` at image center (`memory_projector.py` L580–581); cross-attn readout `out_proj` zero ("step-0 no-op") |
| 10 | Soft-argmax coordinate convention | **resolved by code** | `linspace(0,1,W)` inclusive (`memory_encoder.py` L633–634) |
| 11 | Ray PE at step 0 | **resolved by code** | `RayPositionalEncoding.ray_proj[2]` zero-init (`memory_projector.py` L116); ray head zero-init per paper → ray PE is a no-op at init |
| 12 | Wrist for direct 3D | **resolved by code** | Separate `head_direct_wrist_cam` (camera-frame wrist, bias z=0.5) + root-relative `direct_joints_cam`; MANO path gives the other wrist |
| 13 | Existence vs visibility | **resolved by code** | `exists_3d` = existence/presence (detection gate), `exists_2d` = visibility; 0.5 threshold in viz |
| 14 | K-free camera fit (regression targets, focal bracket) | **carried forward** | Only post-hoc `_fit_pred_K` (regress `u` on `dx/dz`, pixel centers); in-model fit, variance floor `1e-4`, focal bracket **absent** |
| 15 | L_fit exact form | **carried forward** | no loss code |
| 16 | Fallback in K-free mode | **resolved by code** | Camera Head `(u,v)` → sample ray field (`_decode_cam_trans`, `pred_rays` branch) → `t_x = r_x/r_z · t_z` |
| 17 | Fisheye / no-fit bearing sampling | **resolved by code** | bilinear `grid_sample`, `align_corners=True`, per joint anchor |
| 18 | L_img norm and units | **carried forward** | hint: anchors in `[0,1]`, `direct_joints2d` sigmoid output |
| 19 | L_rot routing for RHD | **carried forward** | — |
| 20 | Presence targets for OOS / FreiHAND left slot | **carried forward** | hint: two heads (2D vis, 3D exist) → OOS = (exist 1, vis 0) is the natural reading |
| 21 | L_tmp operand / rate | **carried forward** | hint: hand-token *features* are linearly interpolated 21→T before the MANO/camera heads (`_interp_feat_t`, L1103), 2D/3D joint *coordinates* are interpolated directly (`_interp_coords_t`, L1134–1135), so all per-frame outputs exist at the 81-frame rate and L_tmp can run there |
| 22 | L_cam on OOS frames | **carried forward** | hint: `torch.where(accept, pnp, inv)` keeps a differentiable τ for every row, so a loss *can* apply on OOS rows |
| 23 | Geodesic / MSE reduction over 15 joints | **carried forward** | — |
| 24 | Loss masking for static 5-frame clips | **carried forward** | — |
| 25 | AdamW betas / eps | **carried forward** | — |
| 26 | Cosine floor / warmup shape | **carried forward** | — |
| 27 | EMA | **carried forward** | `shape_ema_logit` is a shape-carry knob, not weight EMA |
| 28 | Mixed precision | **resolved by code** (see 5) | bf16 autocast, fp32 trainables |
| 29 | Seed | **partially resolved** | inference configs `seed: 42`; training seed unknown |
| 30 | Validation protocol | **carried forward** | — |
| 31 | Augmentations | **carried forward** | span-dropout hook exists; usage unknown; no flip/color/crop code |
| 32 | Window sampling | **carried forward** | inference tiles are 22 latent = 85 px frames; training window "21 latent = 81 frames" per paper |
| 33 | Image-dataset resolutions / canvas | **carried forward** | — |
| 34 | Pseudo-GT estimator for HOI4D/H2O | **carried forward** | — |
| 35 | Shared MANO convention | **resolved by code** | `flat_hand_mean=False`, smplx MANO, OpenPose-21 remap via 5 tip vertices (`mano_utils.py`) |
| 36 | Per-batch dataset draw across GPUs | **carried forward** | — |
| 37 | HOT3D custom split / 437 segments | **carried forward — unrecoverable** | issue #5 unanswered; plan: official HOT3D split + own segments, report separately |
| 38 | OakInk2 202 subset / HOI4D 498 segments | **carried forward — unrecoverable** | inherited from ViDiHand (no release) |
| 39 | Evaluation code | **carried forward** | write from App. A.1; issue #6 shows even the authors' EPE2D-p needs clarification |
| 40 | Tiled vs full decoding | **resolved by code** | benchmark = `tiled`, `tile_w=22`, windows anchored at `(81 i)//4`, end-anchored gap fill; `full` = single pass |
| 41 | Inference resolution | **resolved by code** | `--encode_w 832`, aspect kept, both axes snapped to 32 |

**New items surfaced by the code** (added to `gaps_filled.md`): N1 fixed caption text; N2 feature grid 32 px; N3 released K-free decode lacks camera fit; N4 accept-gate pseudo-pixel scaling with `f_ref = img_w` in ray mode; N5 `mixed_pnp_detach_mano` default (off → gradients reach MANO heads through PnP, consistent with paper "gradients flowing through the PnP solver"); N6 span-dropout usage; N7 `exists` head bias −1 init.

## 4. Where the missing pieces will live (Stage 4 plan)

```
repro/2608.20308/
├── code/                          # official inference code, untouched until a number lands
└── src/                           # to be created in Stage 4
    ├── data/
    │   ├── mano_format.py         # shared MANO record (Q35 convention) + per-dataset converters
    │   ├── arctic.py hot3d.py h2o.py oakink2.py reinterhand.py freihand.py rhd.py hoi4d.py
    │   ├── windows.py             # random 21-latent (81-frame) window sampling (Q32), 5-frame static clips
    │   └── mixture.py             # per-batch single-source sampler with Table 2 weights (Q36)
    ├── losses/
    │   ├── rot.py joint.py img.py cam.py pres.py tmp.py ray.py fit.py   # Eq. 6 terms, App. B.3 weights
    │   └── routing.py             # RHD / FreiHAND / OOS masks (Q19, Q20, Q24)
    ├── camera/
    │   └── kfree_fit.py           # differentiable per-axis pinhole fit + guards, bearing source (Q14, N3)
    ├── train.py                   # DDP loop, 3 LR groups, AdamW, cosine+warmup, grad ckpt (Q25–Q30)
    ├── eval/
    │   ├── metrics.py             # App. A.1 (MPJPE-p, PA-p, EPE2D-p, GO-p, CT-p, Jitter, OOS strata)
    │   └── segments.py            # our own 81-frame segment lists (Q37–Q39), documented as non-official
    └── configs/                   # train yml extending options/ace_ego_hand_*.yml
```

Model code itself is reused from `code/` (import `ace_ego_hand.*`); the only model-side addition is the fitted-camera bearing path for K-free (N3), added as a new `cam_trans_decode` option rather than by editing the released path.
