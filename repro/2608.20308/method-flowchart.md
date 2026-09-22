# ACE-Ego-Hand — Method Flowcharts

Source of truth for each box: paper section / equation IDs in brackets (`main.tex` v2), or `[code]` when the released repository (`ggxxii/ACE-Ego-Hand` @ `9757868`) is the only or the more precise source. Shapes assume the ARCTIC setting (81 frames, 672x480). Where paper and code disagree, the code value is drawn and the disagreement is flagged (see `ace-ego-hand-notes.md` §6.2 and `gaps_filled.md`).

Shape legend: `T=81` RGB frames, `T'=21` latent frames, `Hl x Wl = 42x30` VAE latent grid (16 px), `Hp x Wp = 21x15` DiT token grid (32 px, `[code]` default `tap_grid=patch`), `D=3072` backbone width, `d=384` decoder width.

---

## Figure 1 — Inference pipeline (one deterministic pass)

```mermaid
flowchart TB
    subgraph Input["Input (Sec. 3, App. A.2)"]
        V["RGB clip V<br/>T=81 x 3 x 480 x 672, 30 fps"]
        K["Camera intrinsics K<br/>(standard config only)"]
    end

    subgraph Enc["Deterministic Clean-Latent Encoder (Sec. 3.1, Eq. 1)"]
        VAE["Frozen Wan2.2 VAE encoder E<br/>x16 spatial, x4 temporal, 48 ch"]
        DIT["Wan2.2-Fun-5B-Control DiT, low-noise expert<br/>blocks 0..15 of 30, LoRA r=64 + trainable patch embed<br/>sigma=0, t=0, control=0, fixed caption ctx [code]"]
    end

    subgraph Cam["Ray-Based Camera Solver (Sec. 3.3)"]
        RAY["Ray Head: zero-init 1x1 conv<br/>unit ray per cell, temporal mean"]
        FIT["K-free camera fit<br/>closed-form per-axis regression -> (f_hat, c_hat)<br/>(paper only; not in released decode path [code])"]
    end

    subgraph Dec["Bidirectional Spatiotemporal Decoder (Sec. 3.2)"]
        TOK["Tokenize Eq. 2<br/>LN(W_F F) + spatial PE + ray PE"]
        ALT["4 x [spatial cross-attn -> temporal self-attn (RoPE)]<br/>48 queries/frame: 2 hand, 42 joint, 4 register"]
        HEADS["Readout heads<br/>2D anchors (soft-argmax, Eq. 3), wrist-rel 3D,<br/>6D rot -> R,theta, log-depth -> t_z,<br/>clip-level beta, existence, visibility"]
    end

    subgraph Place["Metric placement (Sec. 3.3, Eq. 5)"]
        MANO["MANO layer M(R, theta, beta)<br/>J_can: 21 joints, oriented, unplaced"]
        PNP["Mixed-PnP weighted LS -> (t_x, t_y)<br/>gates: at least 6 votes, RMS within max(15px, 1/4 bbox diag)<br/>else wrist-ray fallback at t_z"]
        UP["Linear interp T'=21 -> T=81"]
    end

    OUT["Per frame, per hand:<br/>R_t, theta_t, tau_t, exist_t, vis_t; beta per clip"]

    V -->|"T x 3 x H x W in [-1,1]"| VAE
    VAE -->|"z: 48 x 21 x 42 x 30"| DIT
    DIT -->|"F: 3072 x 21 x 21 x 15 (block-15 tap)"| RAY
    DIT -->|"F"| TOK
    RAY -->|"r_hat: 3 x 21 x 15 (unit)"| TOK
    RAY -->|"r_hat"| FIT
    FIT -->|"bearings b_j (K-free)"| PNP
    K -->|"bearings ((u-c_x)/f_x, (v-c_y)/f_y) (standard)"| PNP
    TOK -->|"X_l: 315 tokens x 384 per frame"| ALT
    ALT -->|"hand/joint tokens 21 x 48 x 384"| HEADS
    HEADS -->|"R, theta, beta"| MANO
    HEADS -->|"p_hat_j 2D anchors, t_z"| PNP
    MANO -->|"J_can 21 x 3"| PNP
    PNP -->|"tau = (t_x, t_y, t_z) per latent frame"| UP
    HEADS -->|"R, theta, exist, vis per latent frame"| UP
    UP --> OUT
```

Note: the paper describes the feature grid as `42x30` cells of `16x16` px with `D=3072` (Sec. 3.1). The released code folds DiT tokens without undoing the `(1,2,2)` patchify, giving `21x15` cells of `32x32` px; the pixelshuffle variant (`tap_grid=pixelshuffle`, `D/4=768` ch at `42x30`) exists in code but is not enabled by the released configs.

---

## Figure 2 — Encoder input assembly at sigma = 0 (`[code]` `geodit_arch.py::extract_features`, `feature_mode=noised_video`)

```mermaid
flowchart LR
    Z["clean latent z<br/>48 x 21 x 42 x 30 (bf16)"]
    EPS["eps ~ N(0, I)"]
    MIX["x = (1 - sigma) z + sigma eps<br/>sigma = nv_sigma = 0.0 -> x = z"]
    CTRL["control y = zeros<br/>100 x 21 x 42 x 30<br/>(span_mask off at inference)"]
    CAT["concat -> 148 ch<br/>(released channel count kept, App. B.1)"]
    PE["patch_embedding Conv3d (1,2,2)<br/>148 -> 3072, trainable (1.822M)"]
    T["timestep t = sigma * 1000 = 0<br/>adaLN conditioning"]
    CTX["fixed caption embedding (umT5, 512 tok)<br/>'Three geometry renders of two hands ...'<br/>cached in cache/caption_embed.pt"]
    BLK["blocks 0..15<br/>self-attn + cross-attn(ctx) + FFN<br/>LoRA r=64, alpha=64 on q,k,v,o x2 and ffn.0, ffn.2"]
    TAP["tap after block 15<br/>tokens 6615 x 3072 -> fold -> F"]

    Z --> MIX
    EPS --> MIX
    MIX -->|"48 ch"| CAT
    CTRL -->|"100 ch"| CAT
    CAT -->|"148 x 21 x 42 x 30"| PE
    PE -->|"6615 tokens x 3072"| BLK
    T -->|"adaLN modulation"| BLK
    CTX -->|"cross-attn K,V"| BLK
    BLK --> TAP
```

Blocks 16..29 and the diffusion output head are never executed (App. B.1: their LoRA `B` matrices stay exactly zero).

---

## Figure 3 — Decoder: spatially grounded queries and alternating attention (Sec. 3.2, App. B.1)

```mermaid
flowchart TB
    F["F_l per latent frame<br/>3072 x 21 x 15"]
    WF["W_F linear 3072 -> 384 + LayerNorm"]
    PSP["spatial PE P_sp<br/>learned 16x16 grid, bilinear -> 21x15"]
    PRAY["ray PE g(Gamma(r_hat))<br/>Fourier(azimuth, elevation) 8 octaves -> zero-init MLP"]
    X["X_l: 315 tokens x 384"]

    Q0["learned queries per frame<br/>2 hand + 42 joint + 4 register = 48"]
    L1["Layer k (k=1..4)"]
    SCA["spatial cross-attn<br/>queries of frame l -> X_l"]
    TSA["temporal self-attn<br/>same query slot across 21 frames, RoPE, no causal mask"]
    QT["refined queries 21 x 48 x 384"]

    A["cross-attn weights of joint token j<br/>A_j over 315 cells"]
    SOFT["soft-argmax Eq. 3<br/>grid = linspace(0,1) [code]"]
    P2D["2D anchor p_hat_j in [0,1]^2 -> pixels"]
    J3D["MLP -> wrist-relative 3D joints (m)"]
    ROT["Pose Head: 6D -> Gram-Schmidt -> R_hat, theta_hat"]
    DEP["Camera Head: zeta_hat -> t_z = exp(zeta_hat)"]
    BETA["Shape Head: temporal mean of hand token -> beta_hat (per hand, per clip)"]
    PRES["existence / visibility logits (no Hungarian matching)"]

    F --> WF --> X
    PSP -->|"add"| X
    PRAY -->|"add"| X
    Q0 --> L1
    X -->|"keys/values"| SCA
    L1 --> SCA --> TSA -->|"x4"| QT
    QT -->|"joint tokens"| A --> SOFT --> P2D
    QT -->|"joint tokens"| J3D
    QT -->|"hand tokens"| ROT
    QT -->|"hand tokens"| DEP
    QT -->|"hand tokens, all frames"| BETA
    QT -->|"hand tokens"| PRES
```

Config anchors `[code]` `options/ace_ego_hand_k.yml`: `hidden_dim: 384`, `num_heads: 8`, `ffn_mult: 4`, `dropout: 0.0`, `alt_num_layers: 4`, `alt_num_register: 4`, `alt_query_mode: joint`, `alt_temporal_rope: true`, `betas_scope: per_hand`, `per_hand_betas_source: slot_mean`, `use_ray_pe: true`, `ray_pe_n_freqs: 8`.

---

## Figure 4 — Ray-Based Camera Solver and mixed-PnP decision flow (Sec. 3.3, Eq. 4-5, App. B.1)

```mermaid
flowchart TB
    F["F: 3072 x 21 x 21 x 15"]
    RH["Ray Head 1x1 conv (zero-init)<br/>-> 3 x 21 x 21 x 15, L2-normalize"]
    TM["temporal mean -> ray field r_hat(u)<br/>3 x 21 x 15 (intrinsics constant per clip)"]
    LRAY["train: L_ray = mean(1 - dot(r_hat, r_K)) Eq. 4"]

    CFG{"config?"}
    KG["standard: bearings from K<br/>b_j = ((u_j - c_x)/f_x, (v_j - c_y)/f_y), (u_j,v_j) = p_hat_j"]
    KF["K-free: fit pinhole (f_hat, c_hat) to r_hat<br/>closed-form per-axis regression,<br/>variance floor 1e-4, focal bracket"]
    KFOK{"fit passes guards?"}
    BFIT["b_j = (p_hat_j - c_hat) / f_hat<br/>(extrapolates out-of-frame anchors)"]
    BRAW["fallback: read r_hat at anchors<br/>(fisheye Re:InterHand; released code path [code])"]

    JC["J_can = M(R_hat, theta_hat, beta_hat)<br/>z_j = J_can_z,j + t_z"]
    MASK["m_j = 1 if z_j >= 5 cm and anchor inside frame by >= 2% margin"]
    GATE{"at least 6 votes and<br/>refit RMS within max(15 px, 1/4 bbox diag)?"}
    LS["t_x = sum m_j z_j^-1 (b_j^x - J_can_x,j / z_j) / sum m_j z_j^-2 (Eq. 5)<br/>t_y symmetric"]
    FB["wrist on its own inverse-projected ray at depth t_z"]
    TAU["tau_hat = (t_x, t_y, t_z)"]

    F --> RH --> TM
    TM -.-> LRAY
    TM --> CFG
    CFG -->|"K given"| KG
    CFG -->|"no K"| KF --> KFOK
    KFOK -->|"yes"| BFIT
    KFOK -->|"no"| BRAW
    KG --> MASK
    BFIT --> MASK
    BRAW --> MASK
    JC --> MASK --> GATE
    GATE -->|"yes"| LS --> TAU
    GATE -->|"no"| FB --> TAU
```

Each configuration is a separately trained model with its own bearing source (Sec. 3.3); the K-free rows in Table 1 are not a test-time solver switch.

---

## Figure 5 — Training objective routing (Sec. 3.4, Eq. 6, App. B.3)

```mermaid
flowchart LR
    subgraph Pred["Predictions (per frame, per hand)"]
        PR["R_hat, theta_hat"]
        PB["beta_hat (per clip)"]
        PJ["J_hat: root-rel / cam-frame / wrist 3D"]
        P2["p_hat 2D anchors + reprojected MANO joints"]
        PT["tau_hat via mixed-PnP (grad flows through solver)"]
        PP["exist / vis logits"]
        PRF["ray field r_hat; fitted camera (K-free)"]
    end

    subgraph Loss["L = sum of terms (Eq. 6)"]
        LROT["L_rot: geodesic (1) + rotmat MSE (1) + beta l1 (0.1)"]
        LJ["L_joint: l1 root-rel (10), cam-frame (5), wrist (2)"]
        LIMG["L_img: 2D anchors + reprojections under training K (1), wrist (0.5)"]
        LCAM["L_cam: l1 on tau_hat (1)"]
        LPRES["L_pres: BCE exist (0.5) / vis (0.25)"]
        LTMP["L_tmp: l1 acceleration of J_hat (0.5)"]
        LRAY["L_ray: cosine, Eq. 4 (1)"]
        LFIT["L_fit (K-free only): bearing error of fitted cam vs calibrated (5, 500-step warmup)"]
    end

    subgraph Data["Per-batch data source (Table 2 weights)"]
        VID["ARCTIC 14 / HOT3D 18 / H2O 10 / OakInk2 14 / Re:InterHand 16<br/>81-frame windows, full MANO GT, OOS frames fully supervised"]
        FH["FreiHAND 14: 5-frame static clips, right hand, full MANO"]
        RHD["RHD 14: 5-frame static clips, 21 3D joints, no MANO"]
    end

    PR --> LROT
    PB --> LROT
    PJ --> LJ
    P2 --> LIMG
    PT --> LCAM
    PP --> LPRES
    PJ --> LTMP
    PRF --> LRAY
    PRF --> LFIT

    VID -->|"all terms"| Loss
    FH -->|"all terms (static: L_tmp trivially ~0)"| Loss
    RHD -->|"L_joint, L_img, L_pres only; L_rot held at 0"| Loss
```

Optimizer (App. B.2): AdamW, wd 1e-2, clip 1.0, cosine, 200 warmup, 20k steps; LR 2e-4 decoder+heads, 1e-4 LoRA, 2e-5 patch embed; 4 clips/GPU; 16 A100 (standard, eff. batch 64) or 8 A100 (K-free, eff. batch 32).

---

## Figure 6 — Evaluation protocol (App. A.1)

```mermaid
flowchart TB
    SEG["fixed 81-frame test segments shared with baselines<br/>ARCTIC 291, HOT3D 437, H2O 355, OakInk2 202, HOI4D 498"]
    RUN["single pass per segment (21 or 22 latent frames)"]
    DET["detection: existence > 0.5<br/>match by IoU > 0 of projected mesh boxes (GT dilated 10%), same side"]
    GATE["on-screen gate on GT: some joint projects inside image and z > 1 cm"]
    PEN["-p penalty: FN charged canonical MANO (identity R, zero theta, mean beta, tau = 0);<br/>EPE2D FN charged image diagonal"]
    IV["in-view metrics: MPJPE-p, PA-p, EPE2D-p, GO-p, CT-p, FAcc, Recall, F1, Jitter"]
    OOS["out-of-sight stratum: MPJPE^OOS wrist-aligned, GT-gated, no penalty"]
    MIX["MPJPE^+OOS = (n_IV e_IV + n_OOS e_OOS) / (n_IV + n_OOS) (eq:oosmix)"]

    SEG --> RUN --> DET
    RUN --> GATE
    DET --> PEN --> IV
    GATE -->|"visible hand-frames"| IV
    GATE -->|"OOS hand-frames"| OOS
    IV --> MIX
    OOS --> MIX
```

Not released: segment IDs (HOT3D custom 126/72 recording split, OakInk2 202-subset, HOI4D 498 segments) and the scorer itself (GitHub issues #5, #6).
