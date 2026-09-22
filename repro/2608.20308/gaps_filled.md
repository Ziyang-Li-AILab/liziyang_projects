# Gaps filled — everything needed to launch a training run of ACE-Ego-Hand

Provenance tags: `[paper §X]` explicit in the paper (`paper.md` / `source/main.tex`); `[code: path:line]` from the released inference repo (`code/`); `[framework default]`; `[venue convention]`; `[guess: reason]`. `[code-vs-paper]` marks values where the two disagree and says which one is chosen for reproduction. Open-question numbers (Q1–Q41) refer to `paper.md` § Open questions; N1–N7 are the code-surfaced items from `inventory.md` §3.

Two configurations are reproduced: **standard** (K-given) and **K-free**. Values are shared unless a row says otherwise.

---

## A. Architecture — Deterministic Clean-Latent Encoder

| Knob | Value | Provenance |
|---|---|---|
| Backbone release | `Wan2.2-Fun-5B-Control` (VideoX-Fun), low-noise DiT expert | `[paper App. B.1]`, `[code: ace_ego_hand/archs/wan_backbone.py:108-111]` (Q7) |
| DiT geometry | 30 blocks, width 3072, FFN 14336, 148 in-ch (48 latent + 100 control), 48 out-ch, patch (1,2,2) | `[paper App. B.1]`; patch size `[code: wan_backbone.py:223]` |
| VAE | Wan 2.2 VAE (`AutoencoderKLWan3_8`), 48 ch, ×16 spatial, causal ×4+1 temporal, frozen | `[paper §3.1, App. B.1]`, `[code: ace_ego_hand/video_vae.py:70]` |
| Input assembly at σ=0 | `x = z` (48 ch clean latent), `y = zeros(100 ch)`, concat → 148 ch | σ=0 `[paper §3.1 Eq. 1]`; 148 in-ch `[paper App. B.1]`; control = 0 over the remaining 148−48 = 100 ch `[code: geodit_arch.py:310; wan_backbone.py:123]` (Q2) |
| Timestep conditioning | `t = 0` (σ·1000) | `[code: geodit_arch.py:314]` (Q3); consistent with GenCeption's t=0 |
| Text conditioning | fixed caption "Three geometry renders of two hands on black background: color-coded depth, joint skeleton, surface normals." → umT5-xxl, padded to 512 tokens, cached; fed to every block's cross-attention | `[code: scripts/precompute_caption.py:26-27; geodit_arch.py:319]` (Q1, N1). Paper silent. |
| Tap | output of block index 15 (16 blocks executed), rest skipped | `[paper §3.1, App. D]`, `[code: options/*.yml tap_layers: [15]; wan_backbone.py:257-260]` |
| Feature grid | `21 × (H/32) × (W/32)` tokens, D=3072 (ARCTIC 672×480 → 21×15) | `[code: wan_backbone.py:223, 271-280; geodit_arch.py:103 tap_grid='patch']` — **`[code-vs-paper]`**: paper §3.1 says 42×30 @16 px; code + App. C.4 timing support 32 px (notes §3.5). Reproduce with 32 px. (N2) |
| LoRA | r=64, α=64, dropout 0, on `self_attn.{q,k,v,o}`, `cross_attn.{q,k,v,o}`, `ffn.0`, `ffn.2` of all 30 blocks (5.37M/block, 161.219M total); `B` zero-init | `[paper App. B.1, B.2]`, `[code: wan_backbone.py:134-148]` |
| Trainable non-LoRA backbone params | patch embedding (1.822M) fully trained from released weights; diffusion head registered but unreachable (0.596M) | `[paper App. B.1]`, `[code: wan_backbone.py:155-158]` (Q4) |
| Precision | frozen base bf16; LoRA / patch-embed / head fp32 params; `autocast(bf16)` around backbone; time embedding fp32; features cast to fp32 before decoder | `[code: wan_backbone.py:100,121,141-143,232; geodit_arch.py:320,396]` (Q5, Q28) |
| Gradient checkpointing | on, per block, non-reentrant | `[code: options/*.yml gradient_checkpointing: true; wan_backbone.py:249-252]` (Q6) |
| Multi-head render expansion | off (`mh_expand: false`) | `[code: options/*.yml]` |

## B. Architecture — Bidirectional Spatiotemporal Decoder

| Knob | Value | Provenance |
|---|---|---|
| Width / heads / FFN / dropout / act / norm | d=384, 8 heads, FFN ×4, dropout 0.0, GELU, pre-LayerNorm, `nn.MultiheadAttention` | d `[paper §3.2]`; rest `[code: options/*.yml; memory_encoder.py:77-89]` (Q8) |
| Queries per latent frame | 2 hand + 42 joint + 4 register = 48 | `[paper §3.2]`, `[code: memory_encoder.py:409-420]` |
| Layers | 4 alternating rounds: spatial cross-attn (queries → frame tokens) then temporal self-attn (RoPE), no causal mask | `[paper §3.2]`, `[code: alt_num_layers: 4, alt_temporal_rope: true]` |
| Token projection | `LN(W_F F)`, 3072→384 | `[paper §3.2 Eq. 2]` |
| Spatial PE | learned 16×16 grid, bilinear resize to token grid (`spatial_pe_mode: interp2d`) | `[paper App. B.1]`, `[code: options/*.yml; memory_projector.py:400]` |
| Ray PE | Fourier(azimuth, elevation) with 8 doubling frequencies (sin+cos) → zero-init MLP → decoder width 384; source = predicted rays (`self_ray_pe: true`) | encoding `[paper App. B.1]`; width `[paper §3.2]`; `[code: ray_pe_n_freqs: 8; memory_projector.py:107-117]` (Q11) |
| Soft-argmax grid | `linspace(0, 1, W)` / `linspace(0, 1, H)` inclusive endpoints over the 32-px token grid; output in [0,1]² → pixels by × image size | `[code: memory_encoder.py:633-634]` (Q10). Paper says "grid-cell centers"; ray PE uses `(j+0.5)/W` `[code: memory_projector.py:137]` — tolerated inconsistency, keep as code. |
| Wrist-relative 3D MLP | 21 root-relative joints (`num_direct_joints: 21`) + separate camera-frame direct wrist head | MLP `[paper §3.2]`; 21 joints `[paper §3 intro]`, `[code: options/*.yml num_direct_joints: 21; memory_projector.py:590-591]` (Q12) |
| Pose Head | 6D → Gram–Schmidt → global orient (1) + articulation (15) rotmats | `[paper §3.2]`, `[code: memory_projector.py:20]` |
| Camera Head | 3 outputs `(u_norm, v_norm, log z)`; `t_z = exp(log z)`; `(u,v)` used only by wrist-ray fallback | `[paper §3.2]` (log-depth), `[code: memory_projector.py:1170,1200-1201]` |
| Shape Head | per hand per clip, temporal mean of hand tokens (`betas_scope: per_hand`, `per_hand_betas_source: slot_mean`) | `[paper §3.2]`, `[code: options/*.yml]` |
| Presence heads | `exists_3d` = existence, `exists_2d` = visibility; sigmoid; threshold 0.5 at eval | `[paper §3.2, App. A.1]`, `[code: memory_projector.py:802-803,957; scripts/viz_preds.py:208-210]` (Q13) |
| Temporal upsampling | hand-token features linearly interpolated T'=21 → T=81 before MANO/camera heads; 2D/3D joint coordinates interpolated directly | interpolation `[paper §3.2]`; 21/81 `[paper §3.1]`; `[code: memory_projector.py:1103-1135]` |
| Init | tokens `N(0, 0.02)`; go/hp/betas/exists/cam/direct-wrist output layers zero-weight; exists bias −1; cam bias `(0, 0, −0.693)` (t_z₀ = 0.5 m, image center); direct wrist bias `(0, 0, 0.5)`; readout cross-attn `out_proj` zero | `[code: memory_projector.py:65-81,396-438,554-591; memory_encoder.py:293-294]` (Q9) |
| Slot convention | slot 0 = left, slot 1 = right (fixed; no Hungarian matching) | `[paper §3.2]`, `[code: memory_projector.py:1229-1230]` |

## C. Architecture — Ray-Based Camera Solver

| Knob | Value | Provenance |
|---|---|---|
| Ray Head | zero-init 1×1 conv on F → 3 ch → L2-normalize → temporal mean over 21 latent frames | `[paper §3.3]`, `[code: geodit_arch.py raymap: true]` |
| Ray supervision | `L_ray = mean_u (1 − ⟨r̂(u), r_K(u)⟩)` over the token grid, r_K unprojected from cell centers under the calibrated model (pinhole or fisheye) | `[paper §3.3 Eq. 4]` |
| Bearings, standard | `((u_j − c_x)/f_x, (v_j − c_y)/f_y)` at soft-argmax anchors in pixels | `[paper §3.3]`, `[code: memory_projector.py:1379-1382]` |
| Bearings, K-free (paper) | from fitted effective pinhole `(f̂, ĉ)`: `b_j = (p̂_j − ĉ)/f̂`, also used for the wrist fallback | `[paper §3.3, App. B.1]` — **missing in code** (N3) |
| Bearings, K-free (released code) | bilinear `grid_sample` of ray field at anchors, `b = (r_x/r_z, r_y/r_z)` | `[code: memory_projector.py:1341-1355]` — this is the paper's *fallback* path; keep as fallback only |
| K-free camera fit — implementation to write | per axis, closed-form OLS of normalized pixel coordinate on `tan = r_x/r_z` (resp. `r_y/r_z`) over all grid cells: `f̂ = cov(u, tan)/var(tan)`, `ĉ = mean(u) − f̂·mean(tan)`; differentiable | form `[paper App. B.1]` ("closed-form, differentiable per-axis linear regression over the token grid"); regression direction (pixel on tan) `[code: inference.py:39-42]` (post-hoc `_fit_pred_K` uses the same direction) (Q14) |
| Fit guards | `var(tan) ≥ 1e-4` else fallback; focal bracket `f̂_norm ∈ [0.35, 3.0]` (image-width units ⇒ HFOV ≈ 19°–110°) else fallback | floor `[paper App. B.1]`; bracket **`[guess: paper gives "a bracket" without numbers; chosen to include the ≈80° training FOV family with margin]`** |
| Mixed-PnP | Eq. 5 weighted LS for `(t_x, t_y)`, `t_z` from Camera Head; per-joint vote `m_j`: `z_j ≥ 0.05 m` and anchor inside frame by ≥ 2% margin | `[paper §3.3 Eq. 5, App. B.1]`, `[code: memory_projector.py:1253-1257,1364-1368]` |
| Row acceptance | `≥ 6` votes and refit RMS `≤ max(15 px, 0.25 × 2D bbox diagonal)`; else fallback | `[paper App. B.1]`, `[code: memory_projector.py:1411-1415]` |
| Fallback | wrist on inverse-projected ray of the Camera Head's own `(u,v)` at depth `t_z` (standard: via K; K-free: via fitted camera per paper / via ray field per code) | `[paper §3.3]`, `[code: memory_projector.py:1173-1204]` (Q16) |
| Ray-mode gate scaling | pseudo-pixels with nominal focal `f_ref = image width` for the RMS threshold | `[code: memory_projector.py:1391-1398]` (N4). Paper silent. |
| Gradient routing through PnP | `torch.where(accept, pnp, fallback)`; `mixed_pnp_detach_mano` off ⇒ L_cam gradients reach MANO heads, log-z, anchors | `[paper §3.4]` ("gradients flowing through the PnP solver"), `[code: memory_projector.py:1325-1328,1421]` (N5) |

## D. Optimizer

| Knob | Value | Provenance |
|---|---|---|
| Optimizer | AdamW | `[paper App. B.2]` |
| Peak LR — decoder + readout heads + Ray Head + diffusion head | 2e-4 | `[paper App. B.2]` |
| Peak LR — LoRA adapters | 1e-4 | `[paper App. B.2]` |
| Peak LR — patch embedding | 2e-5 | `[paper App. B.2]` |
| Weight decay | 1e-2 (all three groups) | `[paper App. B.2]`; applied uniformly **`[guess: paper gives one value; no mention of excluding norms/biases]`** |
| Betas | (0.9, 0.999) | `[framework default]` (Q25) |
| Epsilon | 1e-8 | `[framework default]` (Q25) |
| Gradient clipping | global norm 1.0 | `[paper App. B.2]` |
| Parameters registered | 183.99M: LoRA 161.219M, decoder+heads 20.347M, patch embed 1.822M, diffusion head 0.596M, Ray Head 9,219 | `[paper App. B.1]` |
| Layer-wise LR decay | none | `[paper App. B.2]` (three flat groups) |

## E. Batch

| Knob | Value | Provenance |
|---|---|---|
| Clips per GPU | 4 | `[paper App. B.2]` |
| GPUs | 16 A100 (standard) / 8 A100 (K-free); Table 3 pooled-query and absolute-PE rows: half | `[paper App. B.2]` |
| Effective batch | 64 clips (standard) / 32 (K-free) | `[paper App. B.2]` |
| Gradient accumulation | 1 | **`[guess: effective batch = GPUs × 4 exactly]`** |
| Clip length | 81 frames (21 latent) for video sources; 5 frames (2 latent) for FreiHAND / RHD | `[paper App. B.2, A.2]` |
| Batch homogeneity | every batch from a single dataset; **same dataset on all ranks per step** | single-dataset `[paper App. A.2]`; cross-rank sync **`[guess: RHD batches skip the MANO heads, so DDP's reducer needs identical parameter participation across ranks]`** (Q36) |
| Source sampling weights | ARCTIC 14, HOT3D 18, H2O 10, OakInk2 14, Re:InterHand 16, FreiHAND 14, RHD 14 (%) | `[paper Table 2]` |

## F. Schedule

| Knob | Value | Provenance |
|---|---|---|
| Total steps | 20,000 | `[paper App. B.2]` |
| LR schedule | cosine decay | `[paper App. B.2]` |
| Warmup | 200 steps | `[paper App. B.2]` |
| Warmup shape / cosine floor | linear warmup; cosine to 0 | **`[guess: diffusers get_cosine_schedule_with_warmup convention; the repo pins diffusers and imports diffusers.training_utils]`** (Q26) |
| Validation cadence | every 500 steps from 5k to 20k (31 checkpoints) | computed from `[paper App. D]` ("31 validation checkpoints logged between step 5k and step 20k") (Q30) |
| Validation set | ARCTIC test segments (291) | **`[guess: App. D reports the 31 checkpoints on wrist-aligned ARCTIC error; paper never names a separate val split]`** |
| Checkpoint saving | every 500 steps, keep final | **`[guess: matches validation cadence]`** |
| Reported checkpoint | step 20k | `[paper App. B.1]` ("after 20k steps") |
| L_fit warmup (K-free) | linear over first 500 steps | `[paper App. B.3]` |
| Ablation recipes | `tab:vdm`: ARCTIC only, 10k steps. `tab:tap`: ARCTIC+HOT3D, eff. batch 16, 20k, inverse-projection decode. Table 3: K-free recipe without L_fit, bearings read from ray field | `[paper §4.3, App. D, Table 3 caption]` |

## G. Data pipeline and augmentation

| Knob | Value | Provenance |
|---|---|---|
| Video sources / splits | ARCTIC (test subject s05), HOT3D (custom recording split 126/72), H2O (test subject 4), OakInk2 (sequence-level, 202-segment eval subset), HOI4D (held out, zero-shot, 166 rec.) | `[paper App. A.2, Table 2]`; HOT3D/OakInk2/HOI4D segment IDs **unrecoverable** (Q37, Q38) — reproduce on official HOT3D split + own segments, report separately |
| Frame rate | 30 fps, no subsampling | `[paper App. A.2]`. Flag: HOI4D is natively 15 fps per ViDiHand App.; ACE paper does not address this. |
| Input resolution | ARCTIC 672×480; HOT3D 480×480; H2O, OakInk2 resized to width 832 with intrinsics rescaled | `[paper App. A.2]` |
| H2O / OakInk2 height | pad/resize height to a multiple of 32 (H2O 1280×720 → 832×468 → **480**) | **`[guess: VAE needs multiples of 32 (code snaps to 32); paper only says "even latent grid"]`** `[code: infer_video.py:45-53]` (Q33) |
| Re:InterHand | egocentric fisheye renders, 10 fps, 21-latent windows, full MANO; resolution unspecified → encode at native size snapped to 32 | `[paper App. A.2]`; resolution **`[guess]`** (Q33) |
| FreiHAND / RHD | static images replicated to 5-frame clips; encode at native 224×224 / 320×320 (both multiples of 32); use provided K | replication `[paper App. A.2]`; resolution **`[guess: native sizes already satisfy the VAE grid; paper gives no resize]`** (Q33) |
| VAE latents | precomputed once per full recording; windows drawn in latent space | **`[guess: App. A.2 "random 21-latent-frame window", App. D "21 or 22 latent frames depending on where the segment starts" both imply recording-level latents]`** |
| Window sampling | uniform random 21-latent-frame window per recording per step; recordings shorter than 21 latents padded/ skipped | uniform **`[guess]`**; short-clip handling **`[guess: skip]`** (Q32) |
| Shared MANO format | smplx MANO, `flat_hand_mean=False`, 16 joints + 5 fingertip vertices → OpenPose-21; OakInk2 (`flat_hand_mean=True`) converted by subtracting `hands_mean` from articulation | `[code: ace_ego_hand/mano_utils.py; memory_projector.py:1211,1228]` (Q35); OakInk2 conversion **`[guess: standard smplx identity]`** |
| Pseudo-GT MANO for HOI4D and partially H2O | per-frame estimator, unnamed | `[paper App. A.2]`; estimator **unknown** (Q34) — use HaMeR-style fits only if regenerating labels, and document |
| OOS frames | fully supervised with GT MANO (no masking) | `[paper §3.4]` |
| Augmentation | none (no flip / color / crop); optional span-dropout hook (zero latent frames + mask channels [48:52]=1) exists in code, **off by default** | `[paper: none stated]`; hook `[code: geodit_arch.py:287-313]`; usage in paper runs **unknown → off** **`[guess]`** (Q31, N6) |
| Horizontal flip | none (would swap left/right slots and MANO handedness) | **`[guess: incompatible with fixed slot convention]`** |

## H. Loss formulation (Eq. 6; weights App. B.3)

```yaml
loss:                                  # L = sum of all terms, unit weight on each group
  rot:
    - {name: geodesic, target: [global_orient, articulation], weight: 1.0}      # [paper App. B.3]
    - {name: rotmat_mse, target: [global_orient, articulation], weight: 1.0}    # [paper App. B.3]
    - {name: l1, target: betas, weight: 0.1}                                    # [paper App. B.3]
    reduction: mean over hands, frames and the 16 rotations                     # [guess: unstated; mean keeps weights scale-free]
  joint:
    - {name: l1, target: root_relative_3d, weight: 10.0,                        # [paper App. B.3] "dominant term"
       applied_to: [direct_mlp_joints, mano_joints]}                            # [guess: §3.2 MLP "predicts wrist-relative 3D joints" and §3.4 L_joint "root-relative"; supervise both reads]
    - {name: l1, target: camera_frame_3d, weight: 5.0,                          # [paper App. B.3]
       applied_to: mano_joints + tau}                                           # [guess]
    - {name: l1, target: wrist_3d, weight: 2.0,                                 # [paper App. B.3]
       applied_to: [direct_wrist_cam, mano_wrist + tau]}                        # [guess: direct wrist head exists in code]
  img:
    - {name: l1, target: softargmax_2d_anchors, weight: 1.0, units: normalized [0,1]}   # weight [paper App. B.3]; norm+units [guess: anchors live in [0,1] in code] (Q18)
    - {name: l1, target: reprojected_mano_joints_under_training_K, weight: 1.0}         # [paper App. B.3]
    - {name: l1, target: wrist_2d, weight: 0.5}                                          # [paper App. B.3]
    visibility_mask: only joints whose GT projects inside the image                     # [guess: mirrors EPE2D "on-screen joints"]
  cam:
    - {name: l1, target: tau_assembled, weight: 1.0, grad_through_pnp: true}    # [paper App. B.3, §3.4]
      apply_on_oos_frames: true                                                 # [guess: torch.where keeps tau differentiable on every row; OOS "fully supervised"]
  pres:
    - {name: bce, target: existence (exists_3d), weight: 0.5}                   # [paper App. B.3]
    - {name: bce, target: visibility (exists_2d), weight: 0.25}                 # [paper App. B.3]
    targets:
      existence: 1 if the hand has GT for the frame (incl. OOS), else 0          # [guess: OOS hands must count as existing for §3.4 "fully supervise out-of-sight hands"]
      visibility: on-screen gate of App. A.1 (a joint projects inside image and z > 1 cm)   # [guess: reuse the evaluation gate]
      freihand_left_slot: existence 0, visibility 0                             # [guess: FreiHAND is right-hand only]
  tmp:
    - {name: l1_acceleration, target: camera_frame_mano_joints_at_81fps, weight: 0.5}   # weight+form [paper App. B.3]; operand [guess: eval Jitter uses J_wrist_rel + tau] (Q21)
      mask: frames t-1,t,t+1 all with GT; zero for 5-frame static clips by construction  # [guess] (Q24)
  ray:
    - {name: cosine, target: unit_rays_from_calibration_at_cell_centers, weight: 1.0}   # [paper §3.3 Eq. 4, App. B.3]
  fit:                                   # K-free only
    - {name: l1_bearing_error, target: calibrated_bearings, weight: 5.0, warmup_linear_steps: 500,   # [paper §3.4, App. B.3]
       evaluated_at: all token-grid cells, gradient only via (f_x, f_y, c_x, c_y)}       # point set + norm [guess: grid is the natural domain of the fit] (Q15)
routing:
  RHD: {rot: 0, betas: 0, joint: on, img: on, pres: on, cam: off, tmp: on}      # [paper §3.4, App. A.2] "rotation and shape terms held at zero"; cam off [guess: no MANO ⇒ no canonical joints for PnP]
  FreiHAND: all terms, right slot only                                          # [paper App. A.2]
  Re:InterHand (fisheye): fit term off (pinhole fit fails → fallback)           # [paper App. B.1]
  OOS frames: rot, joint(root-rel), pres, tmp, cam on; img off                  # [paper §3.4] full supervision; img off [guess: no in-frame 2D target]
```

## I. Tricks

| Trick | Value | Provenance |
|---|---|---|
| LoRA `B` zero-init; Ray Head zero-init; ray-PE MLP zero-init ⇒ step-0 network = unmodified backbone | yes | `[paper App. B.1, B.2]`, `[code: memory_projector.py:116-117]` |
| Decoder output heads zero-init; presence bias −1; depth bias log(0.5) | yes | `[code: memory_projector.py:554-591]` |
| Mixed precision | bf16 autocast backbone, fp32 trainables and decoder | `[code]` (see A) |
| Gradient checkpointing | on | `[code: options/*.yml]` |
| EMA of weights | none | **`[guess: not in paper, not in code; shape_ema_logit is a per-clip shape carry, not weight EMA]`** (Q27) |
| Dropout / drop-path | 0 / none | `[code: options/*.yml dropout: 0.0, lora_dropout: 0.0]` |
| Label smoothing / mixup | none | `[paper: none stated]` |
| Seed | 42 | `[code: options/*.yml seed: 42]` for inference; training seed **`[guess: reuse 42]`** (Q29); paper: single run per setting `[paper App. C, D]` |
| Weight decay exclusions | none | **`[guess]`** |
| Clip-level shape carry (`shape_ema_logit`, init 0.5) | present in released checkpoint schema; keep default | `[code: memory_projector.py:367]` |

## J. Evaluation protocol (to implement; App. A.1)

| Item | Value | Provenance |
|---|---|---|
| Segments | fixed 81-frame test segments: ARCTIC 291, HOT3D 437, H2O 355, OakInk2 202, HOI4D 498 | `[paper Table 2]`; IDs unrecoverable except ARCTIC/H2O by subject (Q37–Q39) |
| Decoding | `tiled`, 22-latent windows anchored at `(81·i)//4`, end-anchored gap fill | `[code: ace_ego_hand/inference.py:78-107]` (Q40); paper: 21 or 22 latent frames `[paper App. D]` |
| Detection | existence > 0.5; match by strictly positive IoU of projected-mesh boxes (GT dilated 10%), same side | `[paper App. A.1]` |
| On-screen gate | any GT joint projects inside `[0,W)×[0,H)` with `z > 1 cm` | `[paper App. A.1]` |
| Penalty (-p) | FN charged with canonical MANO (identity R, zero θ, mean β, τ=0); EPE2D FN charged image diagonal (826 px ARCTIC, 679 px HOT3D) | `[paper App. A.1]` |
| Metrics | MPJPE / PA (wrist-aligned, 21 joints, mm); EPE2D over visible joints (px); GO (deg); CT (m); Jitter (mm/frame², on `J̄ + τ`); FAcc; Recall; F1 | `[paper App. A.1, Table tab:metrics]` |
| OOS strata | MPJPE^OOS wrist-aligned, GT-gated, no penalty; MPJPE^+OOS by Eq. oosmix | `[paper App. A.1]` |
| Efficiency timing | one 81-frame ARCTIC clip, one A100, median of ≥3 passes after warmup | `[paper App. C.4]` |

## K. Inference defaults

| Knob | Value | Provenance |
|---|---|---|
| Encode width | 832, aspect kept, both axes snapped to multiples of 32 | `[code: infer_video.py:45-53,77]` (Q41) |
| Tile width | 22 latent frames | `[code: infer_video.py:76]` |
| K-free placeholder intrinsics | nominal 60° HFOV pinhole, unused by the model | `[code: infer_video.py:149-156]` |
| Output | per-frame `global_orient, hand_pose, betas, cam_trans, direct_joints2d, direct_joints_cam, direct_wrist_cam, exists_2d, exists_3d`, plus `pred_intrinsics` (post-hoc fit) | `[code: inference.py:20-21; infer_video.py:194-198]` |

---

## L. Tag tally and round-trip check

Tally (scripted, `roundtrip_check.txt`) over the 129 tagged rows in sections A–K: rows citing `[paper]` 87, `[code]` 51 (30 rows carry both), `[framework default]` 2, rows containing a `[guess]` **30 (23%)**, 35 individual guess tags — under the 30% threshold, but the guesses cluster in the loss routing (H) and data pipeline (G), which is exactly where a from-scratch trainer can silently diverge. Two `[code-vs-paper]` conflicts (feature-grid resolution N2, K-free bearing source N3) are resolved in favour of the code for architecture and in favour of the paper for the K-free fit (implemented as a new option).

Round-trip against the paper (performed 2026-09-21; log in `roundtrip_check.txt`): the 150 numeric tokens in `[paper §X]`-tagged rows were searched inside the cited section of `source/main.tex` (not just anywhere in the paper). First pass flagged 24; four were genuine attribution errors and were fixed (148 input channels → App. B.1; decoder width 384 → §3.2; 21 joints → §3 intro + code; 21/81 frames → §3.1). The remaining flags are LaTeX-normalisation artifacts, unit conversions (5 cm ↔ 0.05 m), computed values (500-step validation spacing) or `[guess]` values sharing a row with a paper citation; each is dispositioned in the log. No contradiction between `gaps_filled.md` and the paper remains. The `paper-verification` skill referenced by the stage guide is not installed here; the scripted check stands in for it.

### Highest-risk unknowns for Stage 4 (ranked)
1. **Loss routing on OOS frames and RHD** (H) — paper gives principles, not masks; wrong masks change the headline OOS result.
2. **K-free fitted-camera bearing path + L_fit** (C, H) — absent from code; the released K-free checkpoint's decode differs from the paper's description, so even the released numbers may not be reproducible with the released decode.
3. **Segment lists** (G, J) — HOT3D/OakInk2/HOI4D protocol cannot be recovered; expect a protocol offset versus Table 1 and report our own segments as such.
4. **Pseudo-GT source for HOI4D/H2O** (G) — different estimator ⇒ different label distribution on H2O training.
5. **Feature-grid resolution** (A) — if the paper is right and the code default wrong, soft-argmax granularity and compute change 4×; settle with one smoke on the released checkpoint (token count at tap must equal `21×(H/32)×(W/32)`).
