# ACE-Ego-Hand: Repurposing Video Diffusion Models for Occlusion-Robust Egocentric 3D Hand Motion Recovery

**arxiv**: 2608.20308 (v2; v1 2026-08-20; source fetched 2026-09-21, `source/main.tex` single-file CVPR-style)
**venue**: arXiv preprint (CVPR template `cvpr.sty`; no venue stated). Project originally named "DreamHand" (repo commit 3418f0eb, 2026-09-01).
**authors**: Yufei Liu (1,4), Xixi Wang (2), Hao Li (3,4), Ganlong Zhao (3,4), Kaitong Cai (4), Chengkai Jin (2,4), Chunxiao Liu (4), Jianbo Liu (4), Siyuan Huang (4, project leader), Xingang Pan (2), Hongsheng Li (3,4, corresponding). 1 = SJTU, 2 = NTU, 3 = CUHK, 4 = ACE Robotics.
**code**: https://github.com/ggxxii/ACE-Ego-Hand (MIT, inference only; commit `9757868`, 2026-09-10). Checkpoints: https://huggingface.co/acerobotics2025/ACE-Ego-Hand (`ace_ego_hand_k.pt`, `ace_ego_hand_kfree.pt`, CC BY-NC 4.0). Project page: https://ggxxii.github.io/ace-ego-hand/
**local sources**: `2608.20308.md` (pandoc conversion), `source/main.tex`, `pdf/2608.20308.pdf`, `figures/*.png`

**abstract** (verbatim):

> Egocentric video offers scalable manipulation data for embodied AI, yet recovering metric 3D hand trajectories remains challenging due to severe object occlusion and frequent out-of-sight gaps. Existing single-frame and windowed temporal regressors fail when a hand briefly leaves the frame, while recent video diffusion models (VDMs) rely on heavy, stochastic multi-step sampling as pixel-space renderers. We instead repurpose a VDM into a deterministic geometry encoder. A single forward pass over the clean latent exposes scene content beyond current observations, including occluded and out-of-sight hands. We introduce **ACE-Ego-Hand**, an offline clip-level framework that extracts features via a Deterministic Clean-Latent Encoder and decodes them with a Bidirectional Spatiotemporal Decoder. ACE-Ego-Hand recovers continuous bimanual trajectories with metric placement and no external detector, while a Ray-Based Camera Solver supports a second configuration that needs no test-time camera intrinsics. Across five egocentric benchmarks, ACE-Ego-Hand sets a new state of the art, cutting MPJPE-p by 30% on occlusion-heavy ARCTIC and 40% on HOT3D. These gains reach 46%–61% once out-of-sight hands are included in the evaluation, offering a scalable path from everyday human video to robot manipulation data.

---

## Method (verbatim, §3 "ACE-Ego-Hand")

> ACE-Ego-Hand processes an egocentric video clip $V=\{I_t\}_{t=1}^{T}$ to predict frame-wise 3D bimanual MANO parameters in a single deterministic pass: global orientation $\hat R_t$, articulation $\hat\theta_t$, camera-frame translation $\hat\tau_t$, and a per-clip shape $\hat\beta$. The MANO layer $\mathcal{M}$ maps the pose and shape parameters to hand meshes and 21 joints, and the network additionally predicts per-frame existence and visibility flags. Instead of iteratively sampling from generative models, we extract motion representations directly through three unified modules (Figure 2): a **Deterministic Clean-Latent Encoder** that reads features from a LoRA-adapted pretrained video diffusion model, a **Bidirectional Spatiotemporal Decoder** for sequence-wide trajectory estimation, and a **Ray-Based Camera Solver**. We consider two configurations: standard ACE-Ego-Hand and the intrinsics-free ACE-Ego-Hand.

### §3.1 From Generator to Encoder

> **Feedforward encoding.** The generator $\Phi$ is a Diffusion Transformer (DiT) pretrained via rectified flow on noisy latents $x_\sigma = (1{-}\sigma)z + \sigma\epsilon$, where $z$ is the clean video latent, $\epsilon$ is Gaussian noise, and $\sigma\in[0,1]$ is the noise level. This sequence-level pretraining is expected to equip $\Phi$ with spatiotemporal priors such as object permanence, 3D structural consistency, and occlusion reasoning. §4.3 probes this premise by swapping feature sources. We therefore use $\Phi$ strictly as an encoder. A frozen VAE encoder $\mathcal{E}$ compresses the clip into the clean latent $z=\mathcal{E}(V)$, and a single deterministic forward pass runs at zero noise ($\sigma=0$):
> $$F = \Phi_{0:L^\star}(z; \sigma=0), \tag{1, eq:encoder}$$
> where $\Phi_{0:L^\star}$ truncates execution at block $L^\star=15$ of the zero-indexed 30-block stack, running the first 16 blocks. Bypassing the remaining blocks and generation head cuts per-pass computation by approximately half. A tap-depth sweep in Appendix D shows that block 15 retains nearly all of the available accuracy. The resulting feature sequence $F=\{F_\ell\}_{\ell=1}^{T'}$ contains $T'=21$ latent frames for an input of $T=81$ frames. This setup yields $4\times$ temporal and $16\times$ spatial compression. Each latent frame $F_\ell$ forms a spatial grid of $D=3072$-channel feature cells. For a $672\times 480$ input, this corresponds to a $42\times 30$ grid of $16\times 16$ pixel patches. The latent features $F$ then feed both the Bidirectional Spatiotemporal Decoder (§3.2) and the Ray-Based Camera Solver (§3.3).
>
> **End-to-end adaptation.** We keep the encoder trainable within the optimization loop, utilizing LoRA on attention and feed-forward projections alongside a trainable patch embedding layer, so that backpropagated 3D supervision reshapes $F$ into geometry-aware features end to end. We compare alternative latent representations in §4.3. A matched comparison in Appendix D further shows that reading the clean latent at $\sigma=0$ outperforms reading noised latents.

### §3.2 Bidirectional Spatiotemporal Decoder

> **Spatially grounded queries.** We propose a lightweight Bidirectional Spatiotemporal Decoder with spatially grounded queries to extract representations from the tapped feature grid, as illustrated in Figure 1. The decoder processes 48 queries per latent frame: 2 hand tokens assigned to fixed left and right slots, 42 joint tokens, and 4 register tokens that serve as learned scratch space without updating $F$. Structuring joint queries as spatially grounded tokens binds features directly to physical keypoints, facilitating precise 3D hand pose estimation.
>
> Each latent frame is tokenized into patch tokens by integrating two additive positional encodings:
> $$X_\ell = \mathrm{LN}(W_F F_\ell) + P^{\mathrm{sp}} + g\big(\Gamma(\hat r)\big), \tag{2, eq:token}$$
> where $\mathrm{LN}$ denotes Layer Normalization and $W_F$ projects the $D=3072$ backbone channels to the 384-dimensional decoder space. The spatial PE $P^{\mathrm{sp}}$ provides explicit coordinates for attention, and the ray PE injects viewing geometry: $\Gamma$ Fourier-encodes the direction of each cell's predicted viewing ray $\hat r$, produced by the Ray Head (§3.3), and a zero-initialized MLP $g$ maps that encoding to the decoder dimension (implementation details in Appendix B).
>
> **Spatial readout heads.** The Joint Head estimates 2D joint locations directly from this spatial grid. For each joint $j$, the cross-attention weights from its corresponding token form a spatial heatmap $A_j$. A soft-argmax operation then aggregates these attention probabilities to derive 2D joint anchors $\hat p_j$:
> $$\hat p_j = \sum_{u} A_j(u)\,u, \qquad \sum_{u} A_j(u)=1, \tag{3, eq:softargmax}$$
> where $u$ iterates over normalized grid-cell center coordinates in $[0,1]^2$. This differentiable formulation anchors each keypoint with sub-cell precision, and an MLP then predicts wrist-relative 3D joint positions in meters. In parallel, the Pose Head regresses the global orientation $\hat R$ and articulation $\hat\theta$ as Gram–Schmidt-orthogonalized 6D rotations, and the Camera Head predicts a log-depth $\hat\zeta$ with $\hat t_z = \exp(\hat\zeta)$ guaranteeing strictly positive metric depth. Existence and visibility confidence scores complete the frame-level readout, avoiding the need for Hungarian matching.
>
> **Clip-level shape prior.** In egocentric videos, an individual hand maintains a constant physical shape and scale throughout a continuous recording. Estimating mesh parameters independently per frame, however, risks size flickering and shape drift under camera motion and local occlusions. To enforce this physical invariant, the Shape Head predicts hand shape parameters $\hat\beta$ once per hand per video clip by temporally pooling hand tokens across all $T$ frames, yielding a single mesh scale for the entire sequence.
>
> **Unconstrained bidirectional reasoning.** We process the spatially grounded queries with four alternating attention layers. Frame $\ell$'s queries first extract per-frame visual details through Spatial Cross-Attention over $X_\ell$, and queries from all frames then exchange motion information via Temporal Self-Attention. Rotary relative positional encodings on the temporal axis remove absolute sequence constraints, so a single forward pass generalizes to long recordings (Appendix D). Temporal attention runs at the downsampled latent frame rate $T'$, with outputs linearly interpolated back to the $T$ video frames, reducing whole-clip attention to roughly $(T'/T)^2 \approx 1/15$ of the full-resolution cost. We apply no causal mask: each frame conditions on both past and future context across the entire clip, so the decoder reconstructs occluded or out-of-sight hands rather than extrapolating unidirectionally.

### §3.3 Ray-Based Camera Solver

> **Intrinsics-free ray field prediction.** Camera geometry is needed at two points: the ray positional encoding in Eq. (2) and the metric projection of the predicted hand. Readout heads that memorize the pixel-to-metric mapping of one camera fail to generalize across intrinsics, so we instead predict a continuous viewing-ray field, following ray-based camera representations [RayDiffusion 2024; PerspectiveFields 2023]. The Ray Head, a zero-initialized $1\times 1$ convolution on $F$, predicts a per-cell direction normalized to a unit ray $\hat r = (\hat r_x, \hat r_y, \hat r_z)$ in the camera frame. Temporal pooling then averages the per-frame predictions into a single field, since intrinsics are constant within a clip.
>
> During training, ground-truth camera calibration supervises this ray field using a cosine distance loss:
> $$\mathcal{L}_{\mathrm{ray}} = \frac{1}{|\Omega|} \sum_{u\in\Omega} \big(1 - \langle \hat r(u),\, r_K(u) \rangle\big), \tag{4, eq:ray}$$
> where $\langle\cdot,\cdot\rangle$ denotes the inner product, $\Omega$ is the latent token grid, and $r_K(u)$ is the unit ray unprojected from the cell center under the calibrated camera model. One head thus accommodates both pinhole and fisheye camera models without intrinsics as input at test time.
>
> **Mixed-PnP translation.** To determine metric placement, we employ a mixed Perspective-n-Point (PnP) scheme, similar in spirit to ViDiHand but formulated for the calibration-free setting. Rather than predicting full 3D translations, the Camera Head regresses only the optical depth $\hat t_z = \exp(\hat\zeta)$, and we solve the in-plane translation $(t_x, t_y)$ directly against the predicted 2D joint anchors. Per frame and hand (indices suppressed), the MANO forward pass yields $J^{\mathrm{can}} = \mathcal{M}(\hat R, \hat\theta, \hat\beta)$, the 21 joints posed and camera-oriented but not yet placed, so the depth of joint $j$ along the optical axis is $z_j = J^{\mathrm{can}}_{z,j} + \hat t_z$. The bearing vector $b_j = (b^x_j, b^y_j)$, the dimensionless pair $(x/z, y/z)$ of the ray toward joint $j$, is evaluated analytically from the predicted ray field $\hat r$. A closed-form per-axis regression fits an effective pinhole camera $(\hat f, \hat c)$ to $\hat r$ in normalized pixel units, with no calibration input, and $b_j = (\hat p_j - \hat c)/\hat f$. The fitted camera extrapolates to out-of-frame anchors and denoises the per-token field. Standard ACE-Ego-Hand instead computes the bearings from the provided intrinsics as $\big((u_j{-}c_x)/f_x,\,(v_j{-}c_y)/f_y\big)$ with $(u_j, v_j) = \hat p_j$ in pixels. Each configuration is trained end to end with its own bearing source, the $K$-free rows in Table 1 report a separately trained model, not a test-time solver switch.
>
> Perspective projection is linear in the in-plane shift, so $t_x$ admits a closed-form weighted least-squares solution:
> $$b^x_j = \frac{J^{\mathrm{can}}_{x,j} + t_x}{z_j} \quad \Rightarrow \quad \hat t_x = \frac{\sum_j m_j z_j^{-1} \big(b^x_j - J^{\mathrm{can}}_{x,j} / z_j\big)}{\sum_j m_j z_j^{-2}}, \tag{5, eq:pnp}$$
> where $m_j \in \{0, 1\}$ selects joints that lie in front of the camera with anchors inside the frame. We solve for $\hat t_y$ symmetrically, yielding $\hat\tau = (\hat t_x, \hat t_y, \hat t_z)$. With too few valid joints or an excessive re-projection residual, the hand falls back to its inverse-projected wrist ray at depth $\hat t_z$ (thresholds in Appendix B).

### §3.4 Training Recipe

> We optimize all trainable components, namely the patch embedding, the LoRA modules, the Ray Head, and the Bidirectional Spatiotemporal Decoder, jointly under a unified objective:
> $$\mathcal{L} = \mathcal{L}_{\mathrm{rot}} + \mathcal{L}_{\mathrm{joint}} + \mathcal{L}_{\mathrm{img}} + \mathcal{L}_{\mathrm{cam}} + \mathcal{L}_{\mathrm{pres}} + \mathcal{L}_{\mathrm{tmp}} + \mathcal{L}_{\mathrm{ray}}. \tag{6, eq:loss}$$
> $\mathcal{L}_{\mathrm{rot}}$ supervises orientation, articulation, and shape. $\mathcal{L}_{\mathrm{joint}}$ constrains root-relative, camera-frame, and wrist 3D positions. $\mathcal{L}_{\mathrm{img}}$ penalizes 2D anchors and re-projected MANO keypoints under the training camera. $\mathcal{L}_{\mathrm{cam}}$ supervises camera-frame translation, with gradients flowing through the PnP solver. $\mathcal{L}_{\mathrm{pres}}$ trains existence and visibility scores, and $\mathcal{L}_{\mathrm{tmp}}$ penalizes 3D joint accelerations for temporal smoothness. The $K$-free configuration adds $\mathcal{L}_{\mathrm{fit}}$, the bearing error of its learned camera (§3.3) against the calibrated one, whose gradient reaches the ray field only through the four fitted parameters. Loss weights are in Appendix B.
>
> We highlight three key supervision strategies. First, we fully supervise out-of-sight hands rather than masking them out, forcing the bidirectional attention to reconstruct invisible hands from temporal context. Second, a source lacking 3D MANO annotations, RHD in our mixture, supervises only 2D anchors, 3D joints, and presence heads, a routing that adds appearance diversity while shielding the MANO parameter heads. Third, camera intrinsics never enter the encoder or decoder: they serve as training targets for the $\mathcal{L}_{\mathrm{img}}$ re-projections, $\mathcal{L}_{\mathrm{ray}}$, and $\mathcal{L}_{\mathrm{fit}}$, and in the standard configuration, they additionally supply the bearings of the translation solve.

---

## Implementation details (verbatim, Appendix B `sec:supp_impl`)

### B.1 Architecture

> The spatial PE $P^{\mathrm{sp}}$ is a learned $16\times16$ grid bilinearly resized to the token resolution, and the ray PE Fourier-encodes each ray's azimuth and elevation with sines and cosines at eight doubling frequencies, after which a zero-initialized MLP maps the resulting features to the decoder width, giving a smooth start to training. In the mixed-PnP solve, a joint votes ($m_j=1$) if it lies at least 5 cm in front of the camera and its anchor lies inside the frame by a margin of at least 2% of the image size. When fewer than six joints vote, or when the refit RMS anchor residual exceeds the larger of 15 px and a quarter of the hand's 2D bounding-box diagonal, the wrist is placed on its own inverse-projected ray at depth $\hat t_z$. The $K$-free camera fit is a closed-form, differentiable per-axis linear regression over the token grid, guarded by a variance floor of $10^{-4}$ and a bracket on the fitted focal length; a clip that fails either guard, including the fisheye Re:InterHand training arm, falls back to reading the ray field at the anchors, and both decode paths, the per-joint anchor bearings and the wrist fallback, use the fitted camera. The backbone is the Wan2.2-Fun-5B-Control release distributed with VideoX-Fun, loaded as its low-noise DiT submodel with 30 blocks of width 3072, feed-forward width 14336, 148 input latent channels, and 48 output channels, and paired with the Wan 2.2 VAE. We keep the released input and output channel counts unchanged. LoRA adapters of rank 64 with $\alpha=64$ and no dropout are injected into all ten linear layers of every block, namely the query, key, value, and output projections of both self-attention and cross-attention plus the two feed-forward layers. Adapters are matched by module name rather than by depth, so one set is instantiated in each of the 30 blocks, giving 5.37M adapter parameters per block and 161.219M in total. Three counts are worth keeping apart. Of the 5.002B pretrained weights the released DiT holds, only the 1.822M patch embedding is ever updated, and the 30 transformer blocks stay frozen throughout. With the adapters attached, the instantiated backbone holds 5.16B parameters. The optimizer is then handed 183.99M parameters, split by module into 161.219M in the LoRA adapters, 20.347M in the decoder and its readout heads, 1.822M in the patch embedding, 0.596M in the DiT's own diffusion output head, and 9,219 in the Ray Head. These fall into the three learning-rate groups of the next subsection: the LoRA group, the patch-embedding group, and one group holding the decoder, its readout heads, the Ray Head, and the diffusion output head. Since the forward pass stops at the tap, the parameters a gradient can actually reach are the 85.983M of LoRA inside the executed blocks, the 1.822M patch embedding, the 12.287M of decoder parameters that this configuration routes through, and the Ray Head, for 100.10M in all. The adapters in the bypassed blocks and the diffusion output head sit on the skipped path and therefore never leave their initialization. We confirmed this on the released checkpoint: after 20k steps every LoRA $B$ matrix past the tap is still exactly zero, which makes those adapters exact identity maps, and every tensor of the diffusion output head is bit-identical to the pretrained release. The decoder parameters outside the 12.287M belong to readout variants that this configuration does not select, and we keep them instantiated so that a single checkpoint schema covers every ablation.

### B.2 Optimization

> Only the 183.99M parameters listed above are registered with the optimizer. The decoder, the Ray Head, and the LoRA adapters are trained from scratch, with the Ray Head and the LoRA $B$ matrices starting at zero so that the network begins the run as the unmodified backbone, while the patch embedding is fine-tuned from its released weights. We run 20k AdamW steps (weight decay $10^{-2}$, gradient clip 1.0, cosine decay, 200 warmup steps) at three learning rates: $2{\times}10^{-4}$ for the decoder and heads, $1{\times}10^{-4}$ for the LoRA adapters, $2{\times}10^{-5}$ for the patch embedding. Each GPU holds four clips, where a clip is 81 frames for the video datasets and five frames for the two image datasets. The standard configuration runs on 16 A100 GPUs, for an effective batch of 64 clips, and the $K$-free configuration on 8, for an effective batch of 32. The ablations of Table 3 follow the $K$-free configuration, except that the hand-pooled-query and absolute-PE variants run on half as many GPUs at the same step count and the same number of clips per GPU, the halved-effective-batch caveat noted alongside Table 3.

### B.3 Loss Weights

> $\mathcal{L}_{\mathrm{rot}}$ supervises orientation and articulation with the geodesic distance $\arccos\big(\tfrac{1}{2}(\mathrm{tr}(\hat R^{\top}R)-1)\big)$ to the ground-truth rotation $R$, plus a rotation-matrix MSE (weight 1 each), and shape with an $\ell_1$ loss (0.1). $\mathcal{L}_{\mathrm{joint}}$ places $\ell_1$ losses on root-relative (weight 10, the dominant term), camera-frame (5), and wrist (2) 3D joints. $\mathcal{L}_{\mathrm{img}}$ supervises, under the *training* camera, the grounded soft-argmax 2D anchors and the re-projected MANO joints (weight 1, and 0.5 for the wrist). $\mathcal{L}_{\mathrm{cam}}$ is an $\ell_1$ loss on the assembled translation (weight 1), with the gradient flowing through the mixed-PnP solve. $\mathcal{L}_{\mathrm{pres}}$ applies binary cross-entropy to existence and visibility (0.5/0.25). $\mathcal{L}_{\mathrm{tmp}}$ penalizes the acceleration of the predicted 3D joints $\hat J_t$, $\|\hat J_{t+1}-2\hat J_t+\hat J_{t-1}\|_1$ (0.5). $\mathcal{L}_{\mathrm{ray}}$ carries weight 1, and the $K$-free configuration's $\mathcal{L}_{\mathrm{fit}}$ carries weight 5 with a linear warmup over the first 500 steps.

---

## Hyperparameter / dataset tables (verbatim)

### Table 2 (`tab:mix`) — Training and evaluation data

| Source | Train clips | Train frames | Test clips | Test frames | Eval (81-f segments) | Wt. (%) |
|---|---:|---:|---:|---:|---:|---:|
| ARCTIC | 267 | 184,373 | 34 | 24,863 | 291 | 14 |
| HOT3D | 126 | 444,649 | 72 | 256,870 | 437 | 18 |
| H2O | 114 | 68,929 | 46 | 30,724 | 355 | 10 |
| OakInk2 | 550 | 935,260 | 77 | 69,371 | 202 | 14 |
| Re:InterHand | 43 | 21,315 | -- | -- | -- | 16 |
| FreiHAND | 32,560 | 162,800 | -- | -- | -- | 14 |
| RHD | 41,251 | 206,255 | -- | -- | -- | 14 |
| HOI4D (held out) | -- | -- | 166 | 49,800 | 498 | 0 |

> Caption: Clip and frame counts are read from the dataset manifests. For the video sources one clip is one full-length recording, not an 81-frame window. FreiHAND and RHD are static image sources, so their clip column counts images. "Eval" is the number of 81-frame test segments shared with every baseline. For HOT3D and OakInk2 these segments cover only a subset of the available test video. "Wt." is the per-batch sampling weight in percent and is identical for both configurations. HOI4D is held out of training.

### Datasets, Splits, and Preprocessing (verbatim, Appendix A.2)

> The five evaluated video datasets (ARCTIC, HOT3D, H2O, OakInk2, and HOI4D) are processed at 30 fps without temporal subsampling. On the four of these that enter training, training operates on full-length recordings, drawing a random 21-latent-frame (81 RGB frame) window from each recording at every step, while evaluation decodes the fixed 81-frame test segments shared with all baselines. Three further datasets enter the training mixture only. [...] Input resolutions are $672\times480$ for ARCTIC and $480\times480$ for HOT3D, and H2O and OakInk2 are resized to a width of 832 with intrinsics rescaled accordingly, so that every dataset yields an even latent grid. Test splits are subject-disjoint on ARCTIC (test subject s05) and H2O (test subject 4), recording-level on HOT3D (126/72 recordings), and sequence-level on OakInk2 (evaluated on the 202-segment subset shared with the baselines), while HOI4D is excluded from training entirely and evaluated zero-shot. Each dataset ships its own hand annotation format, and we convert all of them into one shared MANO format so that a single loader and a single evaluator serve every dataset. As the main text notes, HOI4D and H2O rely partially on pseudo-ground-truth MANO annotation derived from a per-frame estimator [...]
>
> **Training mixture.** Both configurations train on the same seven-source mixture, with a dataset drawn per batch from the weights of Table 2, which sum to 100. [...] Re:InterHand contributes relit studio captures rendered under egocentric fisheye cameras at 10 fps, so it enters as video and receives the full 21-latent-frame window like the video datasets above. FreiHAND and RHD are two static image sources. Each image is replicated into a static five-frame clip. Those batches contain almost no temporal variation, so they supply appearance and hand-pose diversity rather than motion. FreiHAND is right-hand only and carries a full MANO fit, as does Re:InterHand. RHD provides 21 3D joints and no MANO fit. Those samples supervise the 2D anchor, 3D joint, existence, and visibility heads while the rotation and shape terms are held at zero, following the loss routing described in the main text. Every batch is drawn from a single dataset, so clips of different length and different latent grid are never stacked together.

### Table `tab:metrics` — Metric symbols

| Symbol | Averaged over | Pen. | Unit |
|---|---|:-:|---|
| FAcc | frames | -- | -- |
| Recall, F1 | on-screen hands | -- | -- |
| MPJPE-p | on-screen hands | yes | mm |
| PA-p | on-screen hands | yes | mm |
| EPE2D-p | on-screen joints | yes | px |
| GO-p | on-screen hands | yes | deg |
| CT-p | on-screen hands | yes | m |
| Jitter | matched runs | -- | mm/frame² |
| MPJPE^OOS | out-of-sight hand-frames | -- | mm |
| MPJPE^+OOS | all hand-frames | -- | mm |

---

## Algorithm boxes

The paper contains **no numbered algorithm / pseudocode block**. The closest procedural specification is the mixed-PnP solve (Eq. 5 + Appendix B.1 gates), reconstructed in `method-flowchart.md` and `gaps_filled.md`.

---

## Loss formulation and equations (with numbers)

| # | Label | Equation | Where |
|---|---|---|---|
| 1 | `eq:encoder` | $F = \Phi_{0:L^\star}(z;\sigma=0)$, $L^\star=15$ | §3.1 |
| 2 | `eq:token` | $X_\ell = \mathrm{LN}(W_F F_\ell) + P^{\mathrm{sp}} + g(\Gamma(\hat r))$ | §3.2 |
| 3 | `eq:softargmax` | $\hat p_j = \sum_u A_j(u)\,u$ | §3.2 |
| 4 | `eq:ray` | $\mathcal{L}_{\mathrm{ray}} = \frac{1}{|\Omega|}\sum_{u}(1-\langle \hat r(u), r_K(u)\rangle)$ | §3.3 |
| 5 | `eq:pnp` | $\hat t_x = \frac{\sum_j m_j z_j^{-1}(b^x_j - J^{\mathrm{can}}_{x,j}/z_j)}{\sum_j m_j z_j^{-2}}$ | §3.3 |
| 6 | `eq:loss` | $\mathcal{L}=\mathcal{L}_{\mathrm{rot}}+\mathcal{L}_{\mathrm{joint}}+\mathcal{L}_{\mathrm{img}}+\mathcal{L}_{\mathrm{cam}}+\mathcal{L}_{\mathrm{pres}}+\mathcal{L}_{\mathrm{tmp}}+\mathcal{L}_{\mathrm{ray}}$ (+ $\mathcal{L}_{\mathrm{fit}}$ for K-free) | §3.4 |
| 7 | `eq:oosmix` | $\mathrm{MPJPE}^{+\mathrm{OOS}} = \frac{n_{\mathrm{IV}}\bar\varepsilon_{\mathrm{IV}} + n_{\mathrm{OOS}}\bar\varepsilon_{\mathrm{OOS}}}{n_{\mathrm{IV}}+n_{\mathrm{OOS}}}$ | App. A.1 |

Per-term weights (Appendix B.3):

| Term | Sub-term | Loss | Weight |
|---|---|---|---:|
| $\mathcal{L}_{\mathrm{rot}}$ | global orient + articulation | geodesic $\arccos(\tfrac12(\mathrm{tr}(\hat R^\top R)-1))$ | 1 |
| | global orient + articulation | rotation-matrix MSE | 1 |
| | shape $\beta$ | $\ell_1$ | 0.1 |
| $\mathcal{L}_{\mathrm{joint}}$ | root-relative 3D joints | $\ell_1$ | 10 |
| | camera-frame 3D joints | $\ell_1$ | 5 |
| | wrist 3D | $\ell_1$ | 2 |
| $\mathcal{L}_{\mathrm{img}}$ | soft-argmax 2D anchors + reprojected MANO joints (training camera) | (unspecified norm) | 1 |
| | wrist 2D | (unspecified norm) | 0.5 |
| $\mathcal{L}_{\mathrm{cam}}$ | assembled translation $\hat\tau$ (grad through PnP) | $\ell_1$ | 1 |
| $\mathcal{L}_{\mathrm{pres}}$ | existence / visibility | BCE | 0.5 / 0.25 |
| $\mathcal{L}_{\mathrm{tmp}}$ | $\|\hat J_{t+1}-2\hat J_t+\hat J_{t-1}\|_1$ | $\ell_1$ | 0.5 |
| $\mathcal{L}_{\mathrm{ray}}$ | cosine distance, Eq. 4 | | 1 |
| $\mathcal{L}_{\mathrm{fit}}$ (K-free only) | bearing error of fitted camera vs calibrated | (unspecified) | 5, linear warmup 500 steps |

Evaluation metric definitions (Appendix A.1, verbatim formulas): $\bar J = J - J_0$ (wrist-relative), $\Lambda$ Procrustes,
$\mathrm{MPJPE}=\tfrac{1}{21}\sum_j\|\bar{\hat J}_j-\bar J_j\|_2$, $\mathrm{PA}=\tfrac{1}{21}\sum_j\|\Lambda(\hat J)_j-J_j\|_2$, $\mathrm{EPE_{2D}}=\tfrac{1}{|\mathcal V|}\sum_{j\in\mathcal V}\|\hat p_j-p_j\|_2$, $\mathrm{CT}=\|\hat\tau-\tau\|_2$, $\mathrm{GO}=\angle(\hat R,R)$, Jitter $=\tfrac{1}{T-2}\sum_t\|\tilde J_{t+1}-2\tilde J_t+\tilde J_{t-1}\|$ with $\tilde J=\bar{\hat J}+\hat\tau$. On-screen gate: $\exists j: \pi(J^{\mathrm{cam}}_j)\in[0,W)\times[0,H) \wedge J^{\mathrm{cam}}_{z,j}>z_{\min}=1$ cm. Detection: existence $>0.5$, matched by strictly-positive IoU of projected mesh boxes (GT boxes dilated 10%), same side. Penalty: FN charged with canonical MANO (identity $R$, zero $\theta$, mean $\beta$, $\tau=0$); EPE2D FN charged image diagonal (826 px ARCTIC, 679 px HOT3D).

---

## Results tables (verbatim)

### Table 1 (`tab:main`) — Main comparison on ARCTIC, HOT3D, held-out HOI4D

† zero-shot; ‡ causal Kalman filtering (EgoForce). Bold = best per column.

**ARCTIC**

| Method | FAcc↑ | Recall↑ | F1↑ | MPJPE-p↓ | PA-p↓ | MPJPE+OOS↓ | EPE2D-p↓ | GO-p↓ | CT-p↓ | Jitter↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| InterWild | 0.878 | 0.943 | 0.959 | 30.817 | 15.952 | 39.435 | 53.888 | 25.386 | 0.097 | 46.577 |
| HaMeR | 0.875 | 0.943 | 0.957 | 29.197 | 14.596 | 38.183 | 65.289 | 24.907 | 0.095 | 18.279 |
| Hamba | 0.833 | 0.912 | 0.941 | 31.233 | 17.168 | 40.039 | 87.047 | 27.822 | 0.110 | 15.357 |
| WildHands | 0.879 | 0.946 | 0.960 | 25.704 | 13.941 | 33.915 | 50.517 | 22.320 | 0.058 | 12.972 |
| WiLoR | 0.919 | 0.951 | 0.974 | 22.012 | 11.873 | 31.502 | 71.527 | 17.358 | 0.075 | 24.091 |
| EgoForce‡ | 0.882 | 0.930 | 0.963 | 22.388 | 14.322 | 32.193 | 61.761 | 21.073 | 0.069 | 24.158 |
| OmniHands | 0.866 | 0.949 | 0.954 | 29.674 | 14.203 | 38.191 | 51.505 | 24.580 | 0.087 | 45.312 |
| Dyn-HaMR | 0.842 | 0.918 | 0.951 | 27.904 | 17.017 | 37.445 | 85.723 | 25.951 | 0.121 | 12.840 |
| HaWoR | 0.700 | 0.817 | 0.895 | 45.357 | 26.375 | 53.443 | 158.062 | 43.325 | 0.149 | 19.789 |
| ViDiHand | 0.997 | 0.999 | 0.999 | 21.668 | 9.821 | 31.045 | 12.407 | 14.642 | 0.047 | 3.183 |
| **ACE-Ego-Hand** | **1.000** | **1.000** | **1.000** | **15.256** | **7.474** | **16.783** | **9.180** | **11.807** | **0.021** | **2.700** |
| ACE-Ego-Hand (K-free) | 0.999 | **1.000** | **1.000** | 16.615 | 8.756 | 18.061 | 11.014 | 12.208 | 0.028 | 2.770 |

**HOT3D**

| Method | FAcc↑ | Recall↑ | F1↑ | MPJPE-p↓ | PA-p↓ | MPJPE+OOS↓ | EPE2D-p↓ | GO-p↓ | CT-p↓ | Jitter↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| InterWild | 0.669 | 0.881 | 0.868 | 77.168 | 24.811 | 89.218 | 71.482 | 58.501 | 0.213 | 101.164 |
| HaMeR | 0.692 | 0.904 | 0.883 | 68.314 | 21.455 | 80.264 | 59.077 | 49.636 | 0.102 | 23.632 |
| Hamba | 0.632 | 0.828 | 0.853 | 71.732 | 29.620 | 83.438 | 107.625 | 56.525 | 0.128 | 18.507 |
| WildHands | 0.655 | 0.863 | 0.844 | 52.791 | 28.946 | 60.491 | 111.438 | 53.933 | 0.157 | 22.885 |
| WiLoR | 0.827 | 0.897 | 0.937 | 30.966 | 19.980 | 52.014 | 72.978 | 25.746 | 0.098 | 17.976 |
| EgoForce‡ | 0.769 | 0.856 | 0.916 | 43.960 | 25.709 | 63.085 | 83.521 | 37.144 | 0.130 | 38.342 |
| OmniHands | 0.649 | 0.895 | 0.868 | 63.281 | 22.682 | 73.503 | 68.437 | 49.120 | 0.133 | 69.510 |
| Dyn-HaMR | 0.614 | 0.811 | 0.802 | 74.214 | 38.201 | 80.952 | 171.617 | 43.851 | 0.571 | 44.942 |
| HaWoR | 0.348 | 0.499 | 0.654 | 71.396 | 66.031 | 84.733 | 327.294 | 79.350 | 0.262 | 23.872 |
| ViDiHand | 0.948 | 0.974 | 0.983 | 21.514 | 11.383 | 44.440 | 14.953 | 15.829 | 0.040 | 3.741 |
| **ACE-Ego-Hand** | **0.986** | 0.998 | **0.996** | **12.888** | **6.436** | **17.273** | 6.418 | **7.924** | **0.025** | **3.159** |
| ACE-Ego-Hand (K-free) | 0.984 | **0.999** | 0.995 | 13.535 | 6.768 | 18.703 | **6.067** | 8.362 | 0.026 | 3.426 |

**HOI4D† (zero-shot)**

| Method | FAcc↑ | Recall↑ | F1↑ | MPJPE-p↓ | PA-p↓ | MPJPE+OOS↓ | EPE2D-p↓ | GO-p↓ | CT-p↓ | Jitter↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| InterWild | 0.731 | 0.922 | 0.864 | 53.072 | 22.909 | -- | 80.549 | 41.743 | 0.228 | 98.866 |
| HaMeR | 0.731 | 0.923 | 0.864 | 44.481 | 21.580 | -- | 79.494 | 33.557 | 0.187 | 20.068 |
| Hamba | 0.710 | 0.885 | 0.849 | 47.161 | 25.924 | -- | 115.793 | 37.390 | 0.204 | 21.556 |
| WildHands | 0.730 | 0.924 | 0.864 | 45.623 | 23.601 | -- | 82.246 | 45.654 | 0.159 | 18.615 |
| WiLoR | 0.962 | 0.966 | 0.972 | 33.710 | 14.903 | -- | 41.579 | 25.527 | 0.115 | 17.449 |
| EgoForce‡ | 0.917 | 0.941 | 0.949 | 44.746 | 19.160 | -- | 75.821 | 38.861 | 0.126 | 83.452 |
| OmniHands | 0.655 | 0.937 | 0.834 | 44.255 | 18.689 | -- | 70.662 | 34.392 | 0.108 | 24.212 |
| Dyn-HaMR | 0.750 | 0.863 | 0.845 | 45.097 | 29.259 | -- | 144.643 | 40.176 | 0.258 | 17.947 |
| HaWoR | 0.869 | 0.864 | 0.919 | 47.329 | 28.851 | -- | 135.748 | 43.091 | 0.139 | 28.376 |
| ViDiHand | **0.984** | 0.991 | **0.990** | 30.090 | 13.960 | -- | 24.460 | 23.420 | 0.117 | 4.010 |
| **ACE-Ego-Hand** | 0.958 | 0.996 | 0.974 | 23.031 | 11.664 | -- | **14.987** | 18.426 | 0.057 | 2.393 |
| ACE-Ego-Hand (K-free) | 0.975 | **0.999** | 0.985 | **22.154** | **11.213** | -- | 18.004 | **17.242** | **0.055** | **2.317** |

### Table `tab:supp_extra` — H2O and OakInk2 (same protocol)

**H2O** (in-domain for ACE-Ego-Hand; partially pseudo-GT)

| Method | FAcc↑ | Recall↑ | F1↑ | MPJPE-p↓ | PA-p↓ | MPJPE+OOS↓ | EPE2D-p↓ | GO-p↓ | CT-p↓ | Jitter↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| InterWild | 0.981 | 0.990 | 0.994 | 21.526 | 8.979 | -- | 19.929 | 17.695 | 0.037 | 16.153 |
| HaMeR | 0.980 | 0.990 | 0.992 | 20.079 | 7.221 | -- | 21.786 | 17.971 | 0.034 | 9.307 |
| Hamba | 0.960 | 0.979 | 0.987 | 21.412 | 8.390 | -- | 32.978 | 19.510 | 0.038 | 8.285 |
| WildHands | 0.989 | 0.995 | 0.996 | 32.587 | 10.508 | -- | 26.297 | 22.033 | 0.082 | 10.894 |
| WiLoR | 0.994 | 0.998 | 0.998 | 16.390 | 5.633 | -- | 14.497 | 14.495 | **0.023** | 6.117 |
| EgoForce‡ | 0.990 | 0.996 | 0.997 | 16.219 | 6.020 | -- | 15.849 | 10.416 | 0.024 | 8.046 |
| OmniHands | 0.974 | 0.986 | 0.990 | 19.835 | 7.417 | -- | 25.665 | 18.032 | 0.038 | 6.321 |
| Dyn-HaMR | 0.988 | 0.999 | 0.997 | 17.213 | 6.390 | -- | 10.790 | 11.057 | 0.030 | 5.349 |
| HaWoR | 0.946 | 0.972 | 0.985 | 23.504 | 9.570 | -- | 39.735 | 17.080 | 0.045 | 14.162 |
| ViDiHand | 0.997 | 0.999 | 0.999 | 17.320 | 7.987 | -- | 16.544 | 12.159 | 0.040 | 1.990 |
| ACE-Ego-Hand | **0.999** | **1.000** | **1.000** | **9.223** | **4.894** | -- | **6.088** | **5.689** | 0.031 | **0.707** |
| ACE-Ego-Hand (K-free) | **0.999** | **1.000** | **1.000** | 10.310 | 5.522 | -- | 9.464 | 6.064 | 0.024 | 0.741 |

**OakInk2** (in-domain; native MANO GT; 202-segment subset)

| Method | FAcc↑ | Recall↑ | F1↑ | MPJPE-p↓ | PA-p↓ | MPJPE+OOS↓ | EPE2D-p↓ | GO-p↓ | CT-p↓ | Jitter↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| InterWild | 0.547 | 0.722 | 0.819 | 51.013 | 37.038 | 65.619 | 244.590 | 54.710 | 0.179 | 43.499 |
| HaMeR | 0.680 | 0.799 | 0.866 | 43.804 | 28.726 | 59.471 | 171.887 | 46.340 | 0.148 | 19.510 |
| Hamba | 0.634 | 0.755 | 0.840 | 46.517 | 32.488 | 61.863 | 213.576 | 51.295 | 0.159 | 13.811 |
| WildHands | 0.731 | 0.837 | 0.890 | 44.087 | 27.115 | 58.366 | 151.305 | 43.806 | 0.146 | 15.156 |
| WiLoR | 0.921 | 0.955 | 0.973 | 26.520 | 12.481 | 45.215 | 39.928 | 25.084 | 0.085 | 9.522 |
| EgoForce‡ | 0.846 | 0.912 | 0.949 | 36.725 | 18.198 | 55.095 | 81.356 | 35.671 | 0.101 | 30.476 |
| OmniHands | 0.530 | 0.688 | 0.787 | 54.712 | 40.373 | 68.024 | 283.245 | 60.157 | 0.174 | 22.562 |
| Dyn-HaMR | 0.882 | 0.941 | 0.947 | 28.781 | 14.945 | 47.434 | 46.902 | 25.711 | 0.102 | 7.311 |
| HaWoR | 0.818 | 0.891 | 0.935 | 35.428 | 20.670 | 52.775 | 96.442 | 32.043 | 0.120 | 13.780 |
| ViDiHand | 0.813 | 0.886 | 0.937 | 38.996 | 22.827 | 56.047 | 101.568 | 38.119 | 0.103 | 3.728 |
| ACE-Ego-Hand | **0.977** | **0.985** | **0.993** | **8.988** | **5.887** | **8.272** | **11.129** | **7.222** | **0.018** | **1.089** |
| ACE-Ego-Hand (K-free) | **0.977** | **0.985** | **0.993** | 9.813 | 6.362 | 9.308 | 12.046 | 7.366 | **0.018** | 1.150 |

### Table `tab:vdm` — Feature-source ablation on ARCTIC (ARCTIC-only, 10k steps; not comparable to Table 1)

| Feature source | MPJPE-p↓ | PA-p↓ | Jitter↓ | MPJPE^OOS↓ |
|---|---|---|---|---|
| **ACE-Ego-Hand** (Wan 2.2 + LoRA) | **16.95** | **8.16** | **2.69** | **41.5** |
| w/o LoRA (frozen Wan 2.2) | 23.10 | 10.83 | 3.35 | 50.1 |
| V-JEPA 2 (frozen) | 17.81 | 8.74 | 4.56 | 66.1 |
| VideoMAE (frozen) | 22.53 | 10.42 | 5.60 | 77.6 |
| VAE latent | 32.50 | 12.89 | 3.35 | 79.3 |
| raw RGB | 46.50 | 16.93 | 4.37 | 98.3 |

### Table 3 (`tab:decoder`) — Camera and decoder ablations on ARCTIC (all rows omit $\mathcal{L}_{\mathrm{fit}}$ and read bearings directly from the ray field; not comparable to Table 1)

| Variant | MPJPE-p↓ | PA-p↓ | EPE2D-p↓ | CT-p↓ | Jitter↓ |
|---|---|---|---|---|---|
| **ACE-Ego-Hand** (standard) | **15.256** | **7.474** | **9.180** | 0.021 | 2.700 |
| *Removing test-time intrinsics* | | | | | |
| ACE-Ego-Hand (K-free) | 15.264 | 7.770 | 13.168 | **0.020** | 2.704 |
| w/o mixed-PnP (inverse proj.) | 16.134 | 8.002 | 14.425 | 0.030 | 2.874 |
| w/o PnP solve (direct regr.) | 15.546 | 7.955 | 14.399 | 0.023 | **2.658** |
| *Decoder design (ablated from K-free)* | | | | | |
| w/o spatial PE | 15.348 | 8.101 | 13.141 | 0.023 | 2.686 |
| w/o joint queries (pooled) [half batch] | 17.144 | 8.916 | 14.849 | 0.029 | 2.727 |
| shape β from registers | 15.622 | 7.760 | 14.078 | **0.020** | 2.714 |
| w/o rotary PE (absolute PE) [half batch] | 17.787 | 8.671 | 13.915 | 0.033 | 2.948 |

### Table `tab:oos` — Out-of-sight stratification (wrist-aligned, GT-gated, no -p penalty)

| Method | ARCTIC in view | ARCTIC OOS | ARCTIC +OOS | HOT3D in view | HOT3D OOS | HOT3D +OOS | OakInk2 in view | OakInk2 OOS | OakInk2 +OOS |
|---|---|---|---|---|---|---|---|---|---|
| (counts IV / OOS) | 43,893 | 3,247 | | 57,985 | 12,394 | | 27,988 | 4,736 | |
| InterWild | 32.000 | 139.937 | 39.435 | 77.866 | 142.326 | 89.218 | 49.364 | 161.681 | 65.619 |
| HaMeR | 30.576 | 141.005 | 38.183 | 67.956 | 137.847 | 80.264 | 42.361 | 160.588 | 59.471 |
| Hamba | 32.513 | 141.774 | 40.039 | 71.277 | 140.333 | 83.438 | 45.119 | 160.812 | 61.863 |
| WildHands | 26.404 | 135.437 | 33.915 | 46.412 | 126.358 | 60.491 | 41.184 | 159.901 | 58.366 |
| WiLoR | 22.775 | 149.477 | 31.502 | 30.921 | 150.697 | 52.014 | 25.428 | 162.143 | 45.215 |
| EgoForce‡ | 23.533 | 149.264 | 32.193 | 44.225 | 151.322 | 63.085 | 36.690 | 163.858 | 55.095 |
| OmniHands | 30.706 | 139.371 | 38.191 | 61.152 | 131.289 | 73.503 | 52.382 | 160.461 | 68.024 |
| Dyn-HaMR | 29.084 | 150.464 | 37.445 | 67.786 | 142.547 | 80.952 | 28.127 | 161.532 | 47.434 |
| HaWoR | 46.406 | 148.565 | 53.443 | 70.767 | 150.074 | 84.733 | 34.343 | 161.700 | 52.775 |
| ViDiHand | 22.311 | 149.113 | 31.045 | 21.513 | 151.703 | 44.440 | 38.109 | 162.053 | 56.047 |
| ACE-Ego-Hand | 15.423 | 35.165 | 16.783 | **12.703** | **38.649** | **17.273** | **7.449** | **13.131** | **8.272** |

### Table `tab:radius` — Error vs distance from image center (in-view stratum; standard config vs K-free *without fit*)

| Dataset | Radius r | share | MPJPE (std) | CT (std) | MPJPE (no-fit) | CT (no-fit) |
|---|---|---|---|---|---|---|
| ARCTIC | 0.00–0.25 | 26.1% | 15.81 | 0.019 | 15.49 | 0.019 |
| ARCTIC | 0.25–0.50 | 57.7% | 14.96 | 0.021 | 14.97 | 0.020 |
| ARCTIC | 0.50–0.75 | 14.7% | 16.15 | 0.028 | 16.19 | 0.039 |
| ARCTIC | >0.75 | 1.5% | 19.17 | 0.048 | 19.00 | 0.165 |
| HOT3D | 0.00–0.25 | 7.8% | 12.62 | 0.022 | 12.97 | 0.023 |
| HOT3D | 0.25–0.50 | 59.3% | 12.28 | 0.024 | 12.57 | 0.025 |
| HOT3D | 0.50–0.75 | 28.5% | 12.68 | 0.025 | 13.12 | 0.058 |
| HOT3D | >0.75 | 4.4% | 18.80 | 0.035 | 18.19 | 0.190 |

### Table `tab:tap` — Tap depth (reduced recipe: ARCTIC+HOT3D, eff. batch 16, 20k steps, inverse-projection decode)

| Tap (blocks run) | MPJPE-p | PA-p | EPE2D-p | GO-p | CT-p | Jitter |
|---|---|---|---|---|---|---|
| Block 10 (11/30) | 17.77 | 8.70 | 11.63 | 12.58 | 0.030 | 2.74 |
| **Block 15 (16/30)** | 16.10 | 7.76 | 9.99 | **11.98** | 0.028 | 2.66 |
| Block 20 (21/30) | **15.93** | **7.68** | **9.87** | 12.15 | **0.021** | **2.59** |
| Block 24 (25/30) | 15.95 | 7.70 | 9.93 | 12.02 | 0.023 | 2.60 |

### Table `tab:sigma` — Clean vs noised latent (same 291 ARCTIC segments, 20k steps)

| Latent readout | MPJPE-p | PA-p | EPE2D-p | GO-p | CT-p | Jitter |
|---|---|---|---|---|---|---|
| clean, σ=0 | **15.26** | **7.47** | **9.18** | **11.81** | **0.021** | **2.70** |
| noised, σ=0.5 | 17.66 | 7.99 | 9.78 | 12.62 | 0.034 | 2.78 |

### Efficiency (Appendix C.4)

> ACE-Ego-Hand runs at 63.1 fps (63.3 fps in the K-free configuration) in a single deterministic pass whose runtime is 72% VAE encode, against 1.91 fps for ViDiHand, a 33× gap. [...] WiLoR reaches 15.5 fps. [Crop-based baselines with ViTDet-H+ViTPose+ cluster at 1.5–1.7 fps.] Timed on one 81-frame ARCTIC clip on one A100, median of ≥3 passes after warmup.

---

## Caveats / negative results / mentioned-but-not-shown

- **Ablation recipes are not comparable to Table 1.** Table `tab:vdm` = ARCTIC only, 10k steps. Table 3 = "one shared recipe (HOI4D held out)", all rows omit $\mathcal{L}_{\mathrm{fit}}$ and read bearings directly from the ray field; "hand-pooled queries and absolute PE run at half the data-parallel width". Table `tab:tap` = ARCTIC+HOT3D, eff. batch 16, inverse-projection decode. Table `tab:sigma` = single seed per setting.
- **Table 3 standard row (15.256/7.474/9.180/0.021/2.700) is numerically identical to the Table 1 standard ARCTIC row**, even though Table 3 claims a different shared recipe. Either the "shared recipe" for the standard row is the main run, or it is a transcription reuse. Ambiguous.
- "Direct regression does return the lowest Jitter of the table (2.658), a trade-off we do not adopt."
- **Tap depth**: block 20 and 24 are marginally better than 15 on 5/6 metrics ("we read the residual difference as run-to-run noise"). The choice of 15 was "fixed before the model reported in this paper was trained".
- "The K-free configuration trades a small pose margin for calibration freedom ... MPJPE-p runs 0.6–1.4 mm above the standard configuration ... consistent with the $\mathcal{L}_{\mathrm{fit}}$ term competing for shared capacity; annealing its weight is a natural next step."
- "The fit assumes a pinhole camera, so fisheye clips fall back to reading the ray field directly" (incl. Re:InterHand training arm).
- **Long single-pass decoding** degrades MPJPE-p 15.26 → 17.99 mm (+18%) at whole-recording length while Jitter improves 2.700 → 2.585; at 125 frames MPJPE-p is marginally *better* than at 81.
- **Out-of-sight metrics are wrist-aligned only**: "No table in this paper measures absolute out-of-sight placement, because none of the five benchmarks scores it." Absolute OOS placement "relies on depth and bearing regressed from temporal context through the solver fallback, and remains less constrained".
- **Without the camera fit**, K-free CT-p in the outermost radius bin is 0.165 m (ARCTIC) / 0.190 m (HOT3D) vs 0.048 / 0.035 for standard; HOT3D false negatives fall from 2,491 to 74 with the fit.
- "Each ACE-Ego-Hand entry comes from a single training run." (H2O/OakInk2; by extension all rows are single-seed.)
- **Label quality**: HOI4D and H2O labels are partially pseudo-GT from a per-frame estimator; on H2O the model *trains* on that label distribution. Only ARCTIC and H2O are subject-disjoint; HOT3D (recording-level) and OakInk2 (sequence-level) may have subject overlap.
- **Baseline handling**: HaWoR scored without SLAM/infilling (works against it); Dyn-HaMR gets GT extrinsics; WildHands/HaWoR get GT intrinsics; ViDiHand has no public release (authors' predictions scored on request); baselines not retrained on these splits.
- **HOT3D split** is a *custom recording-level* split (126/72), not the official subject split (GitHub issue #5 asks for split IDs; unanswered as of 2026-09-21).
- **Throughput**: 72% of the runtime is VAE encode (so the DiT half-pass + decoder is ~28%).
- Diffusion output head is registered with the optimizer but never receives gradient ("bit-identical to the pretrained release").
- "ACE-Ego-Hand predicts hands in the camera frame and does not estimate camera motion."
- Community failure report (GitHub issue #3): wide-FOV (>100°) user cameras yield systematically too-large depth; training FOV family ≈ 80°.

---

# Open questions (inputs to Stage 3)

Numbered for cross-reference from `inventory.md` and `gaps_filled.md`.

**Encoder / backbone**
1. **Text conditioning of the DiT.** Wan DiT blocks have text cross-attention. The paper never says what context is fed (empty prompt? fixed caption? dropped?). LoRA is applied to `cross_attn.{q,k,v,o}`, so this matters.
2. **Control channels at σ=0.** Wan2.2-Fun-5B-*Control* has 148 input channels = 48 latent + 100 control. Paper says "keep the released input and output channel counts unchanged" but not what fills the 100 control channels when the clean video latent is the input.
3. **Timestep conditioning at σ=0.** The DiT is adaLN-conditioned on $t$. Is $t=0$ fed, or the shifted-scheduler timestep corresponding to σ=0?
4. **LoRA on the patch-embedding vs full fine-tune.** Paper: patch embedding is "fully" updated (1.822M). Confirm no LoRA on it.
5. **Compute dtype**: bf16 backbone? fp32 LoRA? Autocast? Not stated.
6. **Gradient checkpointing** during training: not stated (would matter for a 5B backbone at 4 clips/GPU).
7. **Which Wan2.2 sub-model**: "low-noise DiT submodel" — confirm which of the two Wan2.2 MoE experts (`low_noise_model`).

**Decoder**
8. **Decoder attention heads / FFN width / dropout / activation**: only $d=384$ and "four alternating layers" stated.
9. **Initialization of decoder tokens / heads** (hand, joint, register tokens; head biases, e.g. initial depth).
10. **Normalized-coordinate convention for the soft-argmax** ("grid-cell center coordinates in $[0,1]^2$") — inclusive endpoints or $(j+0.5)/W$?
11. **Where the ray PE enters when there is no ray head yet / at step 0** — zero-init MLP, but the ray head is also zero-init, so both start as no-ops. Confirm.
12. **Wrist for `direct` 3D**: the wrist-relative 3D MLP gives root-relative joints; where does the camera-frame wrist for $\mathcal{L}_{\mathrm{joint}}$ (camera-frame term) come from — MANO+τ, or a separate direct wrist head?
13. **Existence vs visibility semantics**: which is "on-screen" and which is "exists in 3D (possibly OOS)"? Which one gates detection at 0.5?

**Camera solver**
14. **The K-free camera fit** (closed-form per-axis regression, variance floor $10^{-4}$, focal bracket): exact regression targets (pixel $u$ vs $\tan$ of ray) and the numeric bracket are unspecified.
15. **$\mathcal{L}_{\mathrm{fit}}$ exact form**: "bearing error of its learned camera against the calibrated one" — over which points (all grid cells? joint anchors?) and which norm?
16. **Fallback in K-free mode**: "wrist placed on its own inverse-projected ray at depth $\hat t_z$" — inverse-projected where? (a 2D wrist regressed by the Camera Head? the soft-argmax wrist anchor?)
17. **Per-joint gate under fisheye** (Re:InterHand): pinhole fit fails → "reading the ray field at the anchors" — bilinear sampling?

**Losses**
18. **$\mathcal{L}_{\mathrm{img}}$ norm** ($\ell_1$? $\ell_2$?) and whether 2D targets are normalized $[0,1]$ or pixels.
19. **$\mathcal{L}_{\mathrm{rot}}$ target for RHD** (no MANO): "rotation and shape terms are held at zero" — masked per-sample; confirm the presence/joint terms still fire.
20. **Existence/visibility BCE targets for OOS frames**: existence=1, visibility=0? Are the FreiHAND (right-hand-only) left slots negative examples for existence?
21. **$\mathcal{L}_{\mathrm{tmp}}$ operand**: MANO joints or direct joints? Camera-frame or root-relative? Computed at 81-frame or 21-latent rate?
22. **$\mathcal{L}_{\mathrm{cam}}$ on OOS frames**: does the translation loss apply when the hand is out of sight (fallback branch)?
23. **Geodesic + rotation-MSE for articulation**: per-joint mean over the 15 MANO joints, or summed?
24. **Loss masking for the FreiHAND static 5-frame clips**: temporal loss $\mathcal{L}_{\mathrm{tmp}}$ presumably 0 there; confirm.

**Optimization / schedule**
25. **AdamW betas / eps**: not stated.
26. **Cosine decay floor** (min LR) and whether warmup is linear.
27. **EMA of weights**: not mentioned.
28. **Mixed precision**: not stated.
29. **Random seed(s)**: single run each; seed unspecified (code uses `seed: 42` in inference configs).
30. **Validation protocol during training**: "31 validation checkpoints logged between step 5k and step 20k" ⇒ eval every 500 steps; the validation set is unspecified.

**Data pipeline**
31. **Augmentations**: none mentioned at all (flip? color jitter? crop? span-dropout exists in code).
32. **Window sampling**: "random 21-latent-frame window ... at every step" — uniform over the recording? Must the start align to a latent boundary?
33. **Resolutions for Re:InterHand / FreiHAND / RHD** (only ARCTIC 672×480, HOT3D 480×480, H2O/OakInk2 width 832 are stated). Image datasets: how are the 224×224 FreiHAND / 320×320 RHD images placed on a VAE-compatible (multiple-of-32) canvas, and what intrinsics are assumed?
34. **Pseudo-GT MANO for HOI4D/H2O**: which per-frame estimator produced them?
35. **Shared MANO format**: `flat_hand_mean=False` (HaMeR convention) vs OakInk2 `flat_hand_mean=True` — how is OakInk2 converted?
36. **Dataset sampling**: "a dataset drawn per batch" — is the per-GPU batch of 4 clips from one dataset, and do all 16 GPUs draw the same dataset in a step?
37. **HOT3D custom split** (126/72 recordings) and **437 evaluation segments**: IDs not released (issue #5).
38. **OakInk2 202-segment subset** and **HOI4D 166 recordings / 498 segments**: segment definitions inherited from ViDiHand (no public release).
39. **Evaluation code** not released (issue #6 asks how EPE2D-p penalty/gate is applied).

**Inference**
40. **Tiled vs full decoding at eval**: the paper decodes "21 or 22 latent frames depending on where the segment starts"; the code default is 22-latent tiles.
41. **Resolution at inference for in-the-wild video**: code default `--encode_w 832`, snapped to multiples of 32.
