# ACE-Ego-Hand 代码架构说明（对照论文）

论文：*ACE-Ego-Hand*，arXiv 2608.20308，正文以 `source/main.tex` v2 为准。  
代码分两层，不要混成一个仓库：

| 层 | 路径 | 角色 |
| --- | --- | --- |
| 官方推理 | `code/`（GitHub `ggxxii/ACE-Ego-Hand` @ `9757868`） | 论文里的网络。训练循环、损失、数据、评测脚本都没有。 |
| 复现训练 | `src/`（`ace_repro`） | 包住官方 `GeoDiT`，补上论文 Appendix B 的训练、损失、数据和评测。不改官方解码器内部。 |

下面每一节先写**现在代码实际在做什么**，再写**和论文哪一节对应、哪里不一致**。形状默认 ARCTIC：81 帧 RGB、672×480、30 fps。

---

## 1. 一句话

一段自我中心视频，一次前向，输出两只手在相机坐标系里的连续 3D 轨迹。

论文把视频扩散模型 Wan2.2-5B 截断成确定性编码器（§3.1，Eq. 1）：干净 VAE latent、噪声水平 σ=0，只跑前 16 个 DiT block，在 block 15 取出特征。后面接一个双向时空解码器（§3.2），读出 MANO 姿态、2D 关节点、深度和存在性。平移不用回归三个数，只回归深度 \(t_z\)，平面平移 \((t_x, t_y)\) 用 mixed-PnP 从 2D 锚点解出来（§3.3，Eq. 5）。K-free 配置再用射线场拟合一个针孔相机，测试时可以不给 \(K\)。

---

## 2. 目录地图

```
repro/2608.20308/
├── code/                              官方推理，论文的网络本体
│   ├── infer_video.py                 视频入口：解码、缩放、VAE、滑窗
│   ├── options/ace_ego_hand_k.yml     K 已知配置
│   ├── options/ace_ego_hand_kfree.yml 只多一行 self_ray_decode: true
│   ├── ace_ego_hand/
│   │   ├── models/geodit_model.py     组装 GeoDiT、挂 MANO、读 checkpoint
│   │   ├── inference.py               tiled / full 推理，K-free 的事后拟合
│   │   ├── video_vae.py               Wan2.2 VAE
│   │   ├── mano_utils.py              smplx MANO，OpenPose-21
│   │   └── archs/
│   │       ├── geodit_arch.py         输入拼装、射线头、forward_emode
│   │       ├── wan_backbone.py        DiT 加载、LoRA、block tap
│   │       └── projector/
│   │           ├── memory_projector.py   解码器外壳 + mixed-PnP
│   │           ├── memory_encoder.py     交替注意力（论文的 decoder 主体）
│   │           └── spatial_readout.py    另一种 2D 读出，发布配置未启用
│   └── third_party/                   VideoX-Fun：Wan DiT / VAE 定义
└── src/                               复现补上的训练侧
    ├── train.py                       训练循环
    ├── smoke.py                       四级冒烟（组件 / 前向 / 一步 / 20 步）
    ├── configs/repro-default.yaml     超参，指向 ace_ego_hand_k.yml
    └── ace_repro/
        ├── model.py                   ReproModel：官方网 + K-free 拟合分支
        ├── losses.py                  Eq. 6
        ├── optim.py                   AdamW + warmup + cosine
        ├── camera.py                  射线、针孔拟合、投影
        ├── hand_model.py              真 MANO，或没有 pkl 时的 FakeMANO
        ├── eval/metrics.py            Appendix A.1
        └── data/                      clip 格式、混合采样、VAE cache、FreiHAND / HOT3D 转换、合成数据
```

权重：`code/ckpt/Wan2.2-Fun-5B-Control/` 里是 DiT（约 9.4 GB）和 VAE。umT5 大权重已删，字幕向量缓存在 `code/cache/caption_embed.pt`。论文发布的 `ace_ego_hand_k.pt` / `ace_ego_hand_kfree.pt` 不在本机。

---

## 3. 推理主路径

入口是 `infer_video.py` → `GeoDitModel` → `GeoDiT.forward_emode`。一次前向，没有扩散采样。

### 3.1 输入

| 张量 | 形状（ARCTIC） | 论文 | 代码 |
| --- | --- | --- | --- |
| RGB 片段 | \(T{=}81\)，3×480×672，30 fps | §3 开头，App. A.2 | `infer_video.py` 用 OpenCV 解码，宽对齐到 32 的倍数，默认 `--encode_w 832` |
| 内参 \(K\) | \([f_x,f_y,c_x,c_y,W,H]\) | 标准配置必须有；K-free 不读 \(K\) | K-free 仍塞一个 60° 占位内参，模型在 `self_ray_decode` 下不用它做轴承 |
| 静态图（FreiHAND / RHD） | 单帧复制成 5 帧 → 2 个 latent 帧 | §4，App. A.2 | 训练侧 `schema.Clip.is_static`；官方推理脚本只吃视频 |

### 3.2 VAE：RGB → 干净 latent

`video_vae.py` 里的 `AutoencoderKLWan3_8`，冻结。

- 空间 ×16，时间因果 ×4（\(F_{\text{lat}}=(T-1)/4+1\)）。81 帧 → 21 个 latent 帧。
- 通道 48。ARCTIC latent：\(48 \times 21 \times 42 \times 30\)（672/16=42，480/16=30）。
- 论文 §3.1 把这个 42×30、每格 16×16 像素的格子写成特征图。**代码不是这样用的**，见 3.3。

### 3.3 编码器：截断的 Wan DiT（§3.1，Eq. 1）

实现：`wan_backbone.py` + `geodit_arch.py::extract_features`，配置 `feature_mode: noised_video`，`nv_sigma: 0.0`。

Eq. 1 写成 \(F=\Phi_{0:L^\star}(z;\sigma{=}0)\)，\(L^\star{=}15\)。代码的展开是：

1. 视频 latent 放在扩散模型的 **x** 上。\(\sigma{=}0\) 时 \(x=z\)，不加噪声。时间步 \(t=\sigma\cdot 1000=0\)，走 adaLN。
2. **control y** 全 0，100 通道。和 x 在通道维拼成 **148** 通道。这是 Wan2.2-Fun-Control 的原接口；论文只说“保留发布时的输入输出通道数”（App. B.1），没有写 control 为 0。
3. `patch_embedding`：Conv3d，核 `(1,2,2)`，148→3072，**整层可训练**（fp32，约 1.82M）。空间再缩一半。
4. token 网格因此是 **21×15**（时间仍是 21，因为时间核为 1），每格 **32×32** 像素，序列长度 \(21\times21\times15=6615\)，宽度 \(D{=}3072\)。
5. 30 个 block 里只跑 **0…15**（16 个 block），tap 取 block 15 的输出。block 16…29 和扩散 head 这次前向不执行。
6. 每个 block 的 self-attn 与 cross-attn 的 q,k,v,o，以及 FFN 的第 0、2 层，注入 LoRA，rank=α=64，dropout=0。30 个 block 都注了 LoRA（约 161M），但 16…29 的 LoRA 在这条前向里没有梯度。冻结底模 bf16，LoRA 参数 fp32，骨干 autocast bf16。
7. cross-attn 的文本是**固定字幕**，umT5-xxl，缓存在 `caption_embed.pt`。原文："Three geometry renders of two hands on black background: color-coded depth, joint skeleton, surface normals." 论文正文没有写文本条件。这是放弃掉的“几何渲染生成”设计留下来的输入。

**和论文的关键不一致：** §3.1 写特征图是 42×30、16×16 像素、\(D{=}3072\)。发布配置 `tap_grid: patch` 用的是 patchify 之后的 **21×15、32 像素**。代码里有 `pixelshuffle` 可以回到 42×30（通道变成 768），发布的 yml 没开。复现跟随发布配置，用 32 像素网格。

折叠：`fold_tokens` 把 6615 个 token 还原成 \(3072 \times 21 \times 21 \times 15\)，再 `.float()` 交给解码器。

### 3.4 射线头（§3.3）

`geodit_arch.py` 的 `raymap_heads`：在 tap 特征上做 1×1 卷积，输出 3 通道，L2 归一化成单位射线。论文要求零初始化。时间维做 mean，得到每段视频一张射线场 \(\hat r(u)\)，形状 \(3 \times 21 \times 15\)（相机在一个 clip 内当成不变）。

这张图有两个用途：

- 加到解码器 token 上，当射线位置编码（下面 3.5）。
- K-free 时替代内参，提供每个关节的轴承。

### 3.5 双向时空解码器（§3.2，Eq. 2–3）

类名是 `MemorySegmentBetasTransformer`（`memory_projector.py`），注意力在 `MemoryAlternatingEncoder`（`memory_encoder.py`）。配置 `temporal_arch: alternating`。论文没写这个模块来自 VGGT-Omega 的 register-token 交替注意力；代码注释写了。

Eq. 2：\(X_\ell=\mathrm{LN}(W_F F_\ell)+P^{\mathrm{sp}}+g(\Gamma(\hat r))\)。

| 项 | 论文 | 代码 |
| --- | --- | --- |
| \(W_F\) | 3072→384 的线性层 + LayerNorm | `hidden_dim: 384` |
| 空间 PE \(P^{\mathrm{sp}}\) | 学习到的位置编码 | `spatial_pe_mode: interp2d`：学一个 16×16 网格，双线性采样到 21×15 |
| 射线 PE | 方位角/仰角的傅里叶特征，再过 MLP | `ray_pe_n_freqs: 8`；MLP 最后一层零初始化，所以第 0 步射线 PE 是空操作 |
| 每帧 token 数 | — | \(21\times15=315\)，宽 384 |

然后是 4 层交替注意力（`alt_num_layers: 4`），8 头，FFN×4，dropout 0，GELU，Pre-LN：

1. **空间** cross-attn：这一帧的 query 只看这一帧的 315 个 cell。
2. **时间** self-attn：同一个 query 槽沿 21 个 latent 帧做注意力，RoPE，**没有因果掩码**（整段双向）。

每帧 query 共 **48** 个：2 个 hand token + 42 个 joint token（两只手 × 21 关节）+ 4 个 register token。`alt_mem_tokens: 0`，没有跨窗口记忆。

读出头（都在 hand / joint token 上）：

| 输出 | 论文 | 代码里的头 |
| --- | --- | --- |
| 2D 锚点 \(\hat p_j\) | Eq. 3 soft-argmax，\(\hat p_j=\sum_u A_j(u)\,u\) | joint token 对 315 个 cell 的注意力，坐标是 `linspace(0,1)` 的端点，再乘图像宽高。射线 PE 用的是格子中心 \((j+0.5)/W\)。两套坐标差半格，论文没写。 |
| 腕相对 3D | MLP，单位米 | `direct_joints_rootrel`，另有独立的相机系手腕 `head_direct_wrist_cam`（bias 的 z=0.5 m） |
| 旋转 | 6D → Gram-Schmidt → \(\hat R,\hat\theta\) | 全局朝向和 15 个关节各一条；这两颗头零初始化 |
| 深度 | Camera Head 只预测 \(\hat\zeta\)，\(t_z=\exp(\hat\zeta)\) | `head_cam_trans` 实际输出 3 个数 \((u, v, \log z)\)。\(u,v\) 只给腕部射线回退用。bias \((0,0,-0.693)\)，初始 \(t_z=0.5\) m，落在图像中心 |
| 形状 \(\beta\) | 整个 clip 一个，按手 | hand token 沿时间 mean（`betas_scope: per_hand`） |
| 存在 / 可见 | 两个分数，不做 Hungarian | `exists_3d` = 存在（检测），`exists_2d` = 可见。bias −1，阈值 0.5 |

时间上采样：latent 是 21 帧，输出要对齐 RGB 的 81 帧。hand-token 特征在进 MANO / 相机头之前线性插值（`_interp_feat_t`）；2D/3D 坐标直接插值（`_interp_coords_t`）。论文 §3.2 只说解码器工作在 latent 时间上，再对齐到视频帧。

发布配置下解码器注册参数约 20.3M，真正参与计算的约 12.3M（App. B.1）。`spatial_readout.py`、`joints2d_readout: spatial`、`camera_tokens` 等都是消融残留，当前 yml 不用。

### 3.6 度量摆放：MANO + mixed-PnP（§3.3，Eq. 5）

`mano_utils.py`：smplx，`flat_hand_mean=False`。16 个关节加 5 个指尖顶点，重排成 OpenPose-21。左手 slot 0，右手 slot 1。

\[
J^{\mathrm{can}}=\mathcal{M}(\hat R,\hat\theta,\hat\beta),\qquad z_j=J^{\mathrm{can}}_{z,j}+\hat t_z
\]

\(J^{\mathrm{can}}\) 已经转进相机朝向，但还没平移。mixed-PnP（`_decode_cam_trans_mixed_pnp`）用闭式加权最小二乘，把投影后的 \(J^{\mathrm{can}}\) 对齐到模型自己的 2D 锚点，只解 \((t_x,t_y)\)：

\[
\hat t_x=\frac{\sum_j m_j z_j^{-1}(b^x_j-J^{\mathrm{can}}_{x,j}/z_j)}{\sum_j m_j z_j^{-2}}
\]

\(t_y\) 对称。投票门 \(m_j\)（App. B.1，和代码常量一致）：

- 深度 \(z_j \ge 5\,\mathrm{cm}\)（`_PNP_Z_NEAR=0.05`）
- 锚点离图像边至少 2%（`_PNP_INTERIOR_MARGIN=0.02`）
- 整行至少 6 个有效关节（`_PNP_MIN_INTERIOR=6`）
- 解完之后的 RMS 不超过 \(\max(15\,\mathrm{px},\;0.25\times\text{手部框对角线})\)

门是 `torch.no_grad()` 算的。过门用 PnP 的 \((t_x,t_y)\)；不过门退回腕部反投影：在深度 \(t_z\) 上，沿 Camera Head 自己的 \((u,v)\) 那条射线放手腕。`torch.where` 选分支，梯度只走被选中的那条。`mixed_pnp_detach_mano` 默认关，所以 \(\mathcal{L}_{\mathrm{cam}}\) 的梯度会进 MANO 头和 2D 头，和论文 §3.4 “梯度穿过 PnP”一致。

轴承 \(b_j\) 从哪来，是两条配置的分叉：

| | 标准（`ace_ego_hand_k.yml`） | K-free（论文 §3.5） | K-free（发布代码） | K-free（`src` 复现） |
| --- | --- | --- | --- | --- |
| 轴承 | 用给定 \(K\)：\(((u-c_x)/f_x,(v-c_y)/f_y)\) | 先把 \(\hat r\) 拟合成 \((\hat f,\hat c)\)，再算轴承 | **不拟合**。在锚点上 `grid_sample` 射线场 | `ReproModel._forward_kfree`：拟合成功且是针孔，就用拟合的 \(K\) 走标准 PnP；失败（鱼眼、退化射线）退回发布代码的射线采样 |
| 拟合守卫 | — | 方差下限、焦距括号、主点在画面内、射线朝前 | 发布模型里没有；`inference.py::_fit_pred_K` 只在前向之后把 \((f,c)\) 写进 pickle | `camera.py`：方差下限 \(10^{-6}\)，焦距比图像边长在 \([0.35,3]\)，主点在 \([0,1]\)，至少 90% 射线 \(r_z>0\)。这些数论文没给，是复现时的假设 |

Re:InterHand 是鱼眼。论文和代码都不在鱼眼上做针孔拟合，直接读射线场。

### 3.7 一个 clip 的输出

每一帧、每一只手：\(R_t,\theta_t,\tau_t=(t_x,t_y,t_z)\)，存在分，可见分。\(\beta\) 每个 clip、每只手一个。官方 dump 的键：`global_orient, hand_pose, betas, cam_trans, direct_joints2d, direct_joints_cam, direct_wrist_cam, exists_2d, exists_3d`。

长视频用 `inference.py` 的 **tiled** 模式，这也是论文表格用的：22 个 latent 帧的窗口，锚在第 \(81i\) 帧上。`full` 是整段一次前向。

---

## 4. 训练路径（论文有公式，官方仓库没有代码）

`src/train.py` 调用 `ReproModel`，后者调用上面的 `GeoDiT`，不重写解码器。

### 4.1 数据

论文 Table 2 / App. A.2：每个 batch **只来自一个数据集**，按权重抽。权重复现进了 `repro-default.yaml`：

| 数据集 | 权重 | 种类 | 分辨率 | 相机 |
| --- | --- | --- | --- | --- |
| ARCTIC | 14 | 视频 | 672×480 | 针孔 |
| HOT3D | 18 | 视频 | 480×480 | 针孔 |
| H2O | 10 | 视频 | 832×480 | 针孔 |
| OakInk2 | 14 | 视频 | 832×480 | 针孔 |
| Re:InterHand | 16 | 视频 | 480×480 | 鱼眼 |
| FreiHAND | 14 | 静态，复制 5 帧 | 224×224 | 针孔 |
| RHD | 14 | 静态 | 320×320 | 针孔，**没有 MANO 旋转/形状真值** |

视频训练时从每条录像随机抽 21 个 latent 帧（81 个 RGB 帧）。HOI4D 不进训练，只做 zero-shot 测试。这台机器上有公开 HOT3D-Clips 的一个小子集：8 段训练 tar 里，`clip-001849` … `clip-001853` 已经写成 `data/hot3d/train/*.pt`，剩下 3 段没编码。其它数据集不在。进度和没做完的部分见 `HOT3D_STATUS.md`。没有这些 `.pt` 时，冒烟用 `data/synthetic.py`：FakeMANO 画彩色骨架，再用**真的** Wan VAE 编码。数字不能和 Table 1 比。

统一样本是 `schema.Clip`：latent、\(K\)、两只手的旋转/形状/平移、相机系 21 关节、归一化 2D、存在、可见、`has_mano`。约定：

- `exists`：这一帧有这只手的真值，出画也算（§3.4 出画手全监督）。
- `visible`：App. A.1 在画门（某个关节投影在画面内且 \(z>1\,\mathrm{cm}\)）。
- `has_mano`：RHD 为假，旋转和 \(\beta\) 损失关掉。

`data/mixture.py` 按 step 播种抽数据集，为的是多卡时每张卡选同一个数据集（RHD 不走 MANO 头，参与反传的参数必须一致）。`data/store.py` 读已经缓存好的 latent。FreiHAND 转换器在 `data/converters/freihand.py`，RGB 还缺，所以没跑过。HOT3D 转换器在 `data/converters/hot3d.py`，几何检查过 8 段训练 clip，VAE 编码停在 5/8。

Span-dropout（把一段 latent 涂黑，逼模型从上下文补出画的手）代码钩子在 `extract_features(span_mask=...)`。论文没写数据增强。复现配置 `span_dropout: 0.0`。

### 4.2 损失（§3.4，Eq. 6；权重 App. B.3）

`losses.py::compute_losses`。总损失是各项加权和，多个 tap 时取平均（发布配置只有 layer 15 一个 tap）。

| 论文的项 | 权重 | 代码里怎么算 | 谁被监督 |
| --- | --- | --- | --- |
| \(\mathcal{L}_{\mathrm{rot}}\) 测地线 | 1 | \(\arccos(\tfrac12(\mathrm{tr}(\hat R^\top R)-1))\)，全局朝向和 15 个关节各一份再平均 | `has_mano` 且 `exists` |
| \(\mathcal{L}_{\mathrm{rot}}\) 矩阵 MSE | 1 | 同上两套旋转 | 同上 |
| \(\beta\) 的 \(\ell_1\) | 0.1 | clip 级 \(\beta\) | 该手任一帧有 MANO |
| 腕相对 3D \(\ell_1\) | 10 | 直接头和 MANO 读出各一份 | `exists`（出画也算） |
| 相机系 3D \(\ell_1\) | 5 | 同上 | MANO 那份还要 `has_mano` |
| 手腕 3D \(\ell_1\) | 2 | 直接手腕 + MANO 手腕 | 同上 |
| 2D 锚点 \(\ell_1\) | 1 | 归一化图像坐标 | 在画关节 |
| 重投影 \(\ell_1\) | 1 | \(\pi(J^{\mathrm{can}}+\tau)\) 对真值 2D | 在画且有 MANO |
| 手腕 2D \(\ell_1\) | 0.5 | 直接手腕投影 | 手腕在画 |
| \(\mathcal{L}_{\mathrm{cam}}\) | 1 | \(\hat\tau\) 对真值平移，梯度穿过 PnP | `has_mano` |
| 存在 BCE | 0.5 | `exists_3d` | 全部帧 |
| 可见 BCE | 0.25 | `exists_2d` | 全部帧 |
| \(\mathcal{L}_{\mathrm{tmp}}\) | 0.5 | 相机系关节加速度的 \(\ell_1\)，在 81 帧率上 | 连续三帧都有 MANO |
| \(\mathcal{L}_{\mathrm{ray}}\) Eq. 4 | 1 | \(1-\langle\hat r, r_K\rangle\) 的均值 | 针孔；鱼眼若 batch 里带了 `rays_gt` 就用它 |
| \(\mathcal{L}_{\mathrm{fit}}\) | 5，前 500 步线性爬升 | 只 K-free。拟合相机的轴承对标定轴承 | 针孔且拟合开着 |

论文没写的、复现时定下来的路由：出画帧的 3D 和存在性仍监督；图像项只在在画关节上；RHD 的旋转、\(\beta\)、平移损失关闭。图像损失用的是 \(\ell_1\)（论文没写范数）。\(\mathcal{L}_{\mathrm{fit}}\) 的具体形式论文也没写，实现是 `camera.py::fit_loss`。

官方代码里还有一条论文没写的生成分支：`forward_dmode` / `gen_overlay` / `gen_render`，用扩散 head 预测深度-骨架-法线渲染。复现**没有**训这条，`run_head=False`。

### 4.3 优化（App. B.2）

三组学习率，都是 AdamW，weight decay 0.01，梯度裁剪 1.0：

| 组 | 学习率 | 参数 |
| --- | --- | --- |
| decoder | \(2\times10^{-4}\) | 解码器、射线头、存在头，以及代码里一并打开的扩散 head |
| LoRA | \(1\times10^{-4}\) | 全部 block 的 A/B |
| patch embedding | \(2\times10^{-5}\) | 148 通道卷积，fp32 |

20,000 step，前 200 步线性 warmup，然后 cosine 收到 0。论文没写 AdamW 的 \(\beta\)、\(\varepsilon\) 和 cosine 的下限；复现用 PyTorch 默认 \(\beta=(0.9,0.999)\)、\(\varepsilon=10^{-8}\)、下限 0。没有权重 EMA。batch：论文 16×A100、每卡 4 个 clip；本机配置 `clips_per_gpu: 4`，`grad_accum` 用来在单卡上凑全局 batch。

骨干 `gradient_checkpointing: true`，按 block 做非重入 checkpoint。

### 4.4 评测（App. A.1）

`eval/metrics.py`，和论文同一套定义，作用在 81 帧片段上：

- 检测：存在分 > 0.5，同一侧，投影框 IoU > 0（真值框膨胀 10%）。
- 漏检：3D 用一个标准 MANO（单位旋转、零姿态、平均 \(\beta\)、平移 0）填上，所以 MPJPE-p 会被拉高；2D 漏检记图像对角线。
- 指标：FAcc、Recall、F1、MPJPE-p、PA-p（带尺度的 Procrustes）、EPE2D-p、GO-p（旋转角）、CT-p（平移，米）、Jitter（匹配上的、至少 3 帧的轨迹，单位 mm/frame²）。
- 出画分层：MPJPE 在画面内 / 出画 / 两者按帧数混合（Eq. 7）。出画 = 手存在但 App. A.1 的在画门失败。

论文的 HOT3D 126/72 录像划分、OakInk2 的 202 段、HOI4D 的 498 段没有公开。评测脚本可以算指标，但没有这些名单就对不上表格里的分母。

---

## 5. 一次训练 step 的调用链

```
train.py
  MixtureLoader.next_batch()          抽一个数据集，取 batch 个 Clip
  ReproModel.forward(batch)
      K 已知:  GeoDiT.forward_emode(latent, T, K)
      K-free:  extract_features → ray head → fit_pinhole
               拟合通过: projector 走 K 分支（用拟合内参）
               否则:     发布代码的射线采样
  losses.compute_losses               Eq. 6
  AdamW.step                          三组学习率，clip=1
  每 500 step: SegmentScorer + 存 ckpt（官方 infer_video.py 能读的字典）
```

Checkpoint 字典（`ReproModel.state_for_release`）：`projectors`、`raymap_heads`、以及 DiT 里 `requires_grad` 的张量（LoRA + patch embedding + head）。这是官方 `GeoDitModel.load_inference` 的格式。

---

## 6. 论文写了、代码里还有、但当前配置关掉的东西

这些在 `geodit_arch.py` / projector 里，是仓库改名 DreamHand 之前的设计，**不是** Table 1 的模型：

- `feature_mode: emode`：x=0，RGB 放在 control 里，用来生成渲染图，时间步默认 1000。
- `mh_expand`：把输入扩到 244 通道、输出扩到 144 通道（深度 / 骨架 / 法线）。发布配置 `false`，所以保持 148 进、48 出。
- `gen_overlay` / `dmode`：跑满 30 个 block 加扩散 head。
- `span_mask`：训练时挖掉一段帧。
- `aux_geo`、`geo_control_mods`：把真值几何渲染灌进 control，做上界实验。

Table 1 的点是：`noised_video` + `nv_sigma=0` + `mh_expand=false` + tap 15 + mixed-PnP + 交替解码器 + 射线头。

---

## 7. 对照表（读代码时最容易看错的地方）

| 论文 | 当前代码 | 复现时以谁为准 |
| --- | --- | --- |
| 特征图 42×30，16 px，\(D{=}3072\)（§3.1） | patchify 后 21×15，32 px，\(D{=}3072\) | 发布配置，32 px |
| σ=0 的干净 latent（Eq. 1） | \(x{=}z\)，\(y{=}0\)（100 通道），\(t{=}0\)，外加固定字幕 | 代码。论文没写后三项 |
| 截断在 block 15 | `tap_layers: [15]`，`run_head=False` | 一致 |
| LoRA r=64 在全部 30 个 block，但只训到 15 | 30 个 block 都注入；前向在 15 停 | 一致 |
| 48 个 query，4 层交替，时间 RoPE，双向 | 2+42+4，`alt_num_layers: 4` | 一致 |
| Camera Head 只预测 \(\log t_z\) | 还预测腕部落点 \((u,v)\)，仅回退分支使用 | 代码补全了论文没写的回退坐标 |
| K-free：拟合 \((\hat f,\hat c)\) 再算轴承 | 发布推理：在锚点采样射线，拟合只写进 pickle。`src` 在 `kfree: true` 时把拟合接回 PnP | 要对齐论文 K-free 表格，用 `src` 的拟合分支，并标明守卫阈值是假设 |
| 损失八项 + K-free 的 \(\mathcal{L}_{\mathrm{fit}}\) | `src/ace_repro/losses.py` | 权重按 App. B.3；掩码是复现假设 |
| 训练 20k step，16×A100，全局 batch 64 | `train.py` 能跑。本机只有 5 段 HOT3D `.pt`，没有 DiT 权重 | 循环按论文。这 5 段不是 Table 2 的混合，也还没拿来训练 |
| 评测脚本和 HOT3D/OakInk2/HOI4D 的片段名单 | 指标实现了，名单没有 | 指标按 App. A.1；名单对不上就不能报 Table 1 |

---

## 8. 生图用的 prompt

下面两段是给文生图模型的，不是给读者的叙述。第一张是整条 pipeline，第二张是解码器和 PnP 的放大。一张图塞不下所有公式，所以拆开。标签用短英文：图里的中文很容易画错。

负向提示两张共用：

```
photorealistic hands, photographs of people, 3D render of a person, cluttered background, watermark, logo, page number, paragraph text, tiny unreadable captions, misspelled labels, extra arrows, arrows that skip a stage, decorative icons unrelated to the modules
```

### Prompt A — 全流程（横版 16:9）

```
A clean scientific architecture diagram, landscape 16:9, white background, flat vector infographic, thin gray arrows, rounded rectangles, no photographs, no 3D characters. Title at the top in black sans-serif: "ACE-Ego-Hand". Subtitle: "one forward pass, no diffusion sampling".

Five columns left to right, equal width, big stage numbers 1 to 5 in the corner of each column.

Column 1, blue header "1 Input". Two stacked boxes: "RGB clip" with small text "81 frames, 30 fps" and "camera K" with small text "used only in the K-given setup".

Column 2, slate header "2 Frozen VAE". One box: "Wan2.2 VAE". Arrow label from column 1: "RGB". Arrow label out: "clean latent z, 48 x 21 x 42 x 30".

Column 3, amber header "3 Encoder, trainable". A vertical stack inside the column: "patch embed 148 to 3072", then "Wan DiT blocks 0-15", then a small side note "blocks 16-29 not run". A thin arrow into the DiT labeled "fixed caption". A small badge on the DiT: "LoRA r=64". Arrow out of the column splits in two, both labeled "tap block 15, 3072 x 21 x 21 x 15".

Column 4, teal header "4 Decoder". Top small box "Ray head" outputting "unit ray field". Main box "4 alternating layers": line 1 "spatial cross-attn", line 2 "temporal self-attn, bidirectional". Under it four short chips in a row: "2D anchors", "rotation", "depth tz", "existence". Arrow from Ray head into the decoder labeled "ray positional encoding".

Column 5, green header "5 Placement". Three stacked boxes: "MANO", "mixed-PnP", "81-frame tracks". Arrow into MANO labeled "R, theta, beta". Arrow into mixed-PnP labeled "2D anchors + tz + rays". Arrow out labeled "tx, ty, tz". Final output box: "left and right hand, metric 3D".

A bottom strip across all columns, rose header "Training loss, not in the released code". Seven small equal boxes: "rotation", "joints", "image", "camera", "presence", "acceleration", "ray". A dashed arrow from column 5 up into this strip. Caption under the strip: "Eq. 6, weights from Appendix B.3".

Color key at the bottom left, four dots: slate "frozen", amber "trainable", teal "geometry", rose "loss".

Use only the English words written above. Do not add equations, do not add extra modules, do not draw human hands.
```

### Prompt B — 解码器与 mixed-PnP 放大（横版 16:9）

```
A clean scientific architecture diagram, landscape 16:9, white background, flat vector, two panels side by side with a narrow gap. No photographs.

Left panel title: "Decoder, section 3.2". Flow top to bottom.
Box "feature F, 3072 channels".
Arrow down to box "linear + layer norm, 384-d".
Two side arrows merge in: "spatial PE" and "ray PE".
Result box: "315 tokens per frame".
Box "48 queries": three chips "2 hand", "42 joint", "4 register".
A loop drawn as four stacked pairs, labeled "repeat x4": "spatial cross-attn" then "temporal self-attn, RoPE, no causal mask".
Then a row of readout boxes: "soft-argmax 2D", "6D rotation", "log-depth tz", "clip-level beta", "exist / visible".
Small note at the panel bottom: "upsample 21 latent frames to 81 RGB frames".

Right panel title: "Mixed-PnP, section 3.3". Flow top to bottom.
Box "MANO joints, camera-facing, not yet placed".
Box "depth zj = joint z + tz".
A diamond "gates": three lines inside, "zj > 5 cm", "anchor inside frame", "at least 6 joints".
Yes arrow to box "closed-form tx, ty".
No arrow to box "wrist on its own ray at depth tz".
Both arrows meet at box "translation tau".
A small left callout "bearing source" with two lines: "K-given: from intrinsics" and "K-free: from fitted ray field".
Dashed arrow from "translation tau" back to "6D rotation" labeled "gradient through PnP".

Bottom legend: amber "trainable head", green "closed-form geometry, not a network".
Use only these English labels. No people, no hand drawings, no extra text.
```
