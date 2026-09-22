# ACE-Ego-Hand：方法流程图

每个框的依据：括号里的论文章节 / 公式编号（`main.tex` v2），或者当已发布仓库（`ggxxii/ACE-Ego-Hand` @ `9757868`）是唯一来源、或比论文更精确时，标 `[code]`。形状按 ARCTIC 设定（81 帧，672×480）。论文和代码不一致时，图上画代码的值，并标出分歧（见 `ace-ego-hand-notes.md` §6.2 和 `gaps_filled.md`）。

形状图例：`T=81` 帧 RGB，`T'=21` 个潜变量帧，`Hl × Wl = 42×30` 的 VAE 潜变量网格（16 像素），`Hp × Wp = 21×15` 的 DiT token 网格（32 像素，`[code]` 默认 `tap_grid=patch`），骨干宽度 `D=3072`，解码器宽度 `d=384`。

---

## 图 1：推理管线（一次确定性前向）

```mermaid
flowchart TB
    subgraph Input["输入（§3，附录 A.2）"]
        V["RGB 片段 V<br/>T=81 x 3 x 480 x 672，30 fps"]
        K["相机内参 K<br/>（仅标准配置）"]
    end

    subgraph Enc["确定性干净潜变量编码器（§3.1，式 1）"]
        VAE["冻结的 Wan2.2 VAE 编码器 E<br/>空间 x16，时间 x4，48 通道"]
        DIT["Wan2.2-Fun-5B-Control DiT，低噪声专家<br/>30 个 block 中的 0..15，LoRA r=64 + 可训练 patch embed<br/>sigma=0，t=0，control=0，固定字幕上下文 [code]"]
    end

    subgraph Cam["基于射线的相机求解器（§3.3）"]
        RAY["射线头：零初始化 1x1 卷积<br/>每格一条单位射线，再做时间平均"]
        FIT["K-free 相机拟合<br/>闭式逐轴回归得到 (f_hat, c_hat)<br/>（只有论文有；发布的解码路径里没有 [code]）"]
    end

    subgraph Dec["双向时空解码器（§3.2）"]
        TOK["按式 2 变成 token<br/>LN(W_F F) + 空间位置编码 + 射线位置编码"]
        ALT["4 轮：空间交叉注意力，再时间自注意力（RoPE）<br/>每帧 48 个 query：2 手，42 关节，4 个 register"]
        HEADS["读出头<br/>2D 锚点（soft-argmax，式 3），腕部相对 3D，<br/>6D 旋转到 R、theta，对数深度到 t_z，<br/>片段级 beta，存在性，可见性"]
    end

    subgraph Place["米制放置（§3.3，式 5）"]
        MANO["MANO 层 M(R, theta, beta)<br/>J_can：21 个关节，已朝向，尚未平移"]
        PNP["Mixed-PnP 加权最小二乘得到 (t_x, t_y)<br/>门控：至少 6 票，RMS 不超过 max(15px, 1/4 框对角线)<br/>否则沿腕部射线在 t_z 处回退"]
        UP["线性插值 T'=21 到 T=81"]
    end

    OUT["每帧、每只手：<br/>R_t, theta_t, tau_t, exist_t, vis_t；beta 每段一个"]

    V -->|"T x 3 x H x W，范围 [-1,1]"| VAE
    VAE -->|"z: 48 x 21 x 42 x 30"| DIT
    DIT -->|"F: 3072 x 21 x 21 x 15（block 15 抽取）"| RAY
    DIT -->|"F"| TOK
    RAY -->|"r_hat: 3 x 21 x 15（单位向量）"| TOK
    RAY -->|"r_hat"| FIT
    FIT -->|"视线方向 b_j（K-free）"| PNP
    K -->|"视线方向 ((u-c_x)/f_x, (v-c_y)/f_y)（标准配置）"| PNP
    TOK -->|"X_l：每帧 315 个 token x 384"| ALT
    ALT -->|"手/关节 token 21 x 48 x 384"| HEADS
    HEADS -->|"R, theta, beta"| MANO
    HEADS -->|"p_hat_j 二维锚点，t_z"| PNP
    MANO -->|"J_can 21 x 3"| PNP
    PNP -->|"tau = (t_x, t_y, t_z)，每个潜变量帧"| UP
    HEADS -->|"每个潜变量帧的 R, theta, exist, vis"| UP
    UP --> OUT
```

说明：论文把特征网格写成 `42×30` 个 `16×16` 像素的格子、`D=3072`（§3.1）。发布代码在不做 pixelshuffle 的情况下直接折叠 DiT token，没有撤销 `(1,2,2)` 的 patch，得到的是 `21×15` 个 `32×32` 像素的格子；pixelshuffle 变体（`tap_grid=pixelshuffle`，在 `42×30` 上通道为 `D/4=768`）代码里有，但发布配置没开。

---

## 图 2：σ = 0 时编码器怎么拼输入（`[code]` `geodit_arch.py::extract_features`，`feature_mode=noised_video`）

```mermaid
flowchart LR
    Z["干净潜变量 z<br/>48 x 21 x 42 x 30（bf16）"]
    EPS["eps ~ N(0, I)"]
    MIX["x = (1 - sigma) z + sigma eps<br/>sigma = nv_sigma = 0.0，于是 x = z"]
    CTRL["控制量 y = 全零<br/>100 x 21 x 42 x 30<br/>（推理时 span_mask 关闭）"]
    CAT["拼接成 148 通道<br/>（保持发布时的通道数，附录 B.1）"]
    PE["patch_embedding，Conv3d (1,2,2)<br/>148 到 3072，可训练（1.822M）"]
    T["时间步 t = sigma * 1000 = 0<br/>adaLN 条件"]
    CTX["固定字幕嵌入（umT5，512 token）<br/>Three geometry renders of two hands ...<br/>缓存在 cache/caption_embed.pt"]
    BLK["block 0..15<br/>自注意力 + 对 ctx 的交叉注意力 + FFN<br/>LoRA r=64、alpha=64，加在两套 q,k,v,o 以及 ffn.0、ffn.2"]
    TAP["block 15 之后抽取<br/>6615 个 token x 3072，折叠成 F"]

    Z --> MIX
    EPS --> MIX
    MIX -->|"48 通道"| CAT
    CTRL -->|"100 通道"| CAT
    CAT -->|"148 x 21 x 42 x 30"| PE
    PE -->|"6615 个 token x 3072"| BLK
    T -->|"adaLN 调制"| BLK
    CTX -->|"交叉注意力的 K、V"| BLK
    BLK --> TAP
```

block 16..29 和扩散输出头从不执行（附录 B.1：它们的 LoRA `B` 矩阵始终精确为零）。

---

## 图 3：解码器：带空间锚的 query 与交替注意力（§3.2，附录 B.1）

```mermaid
flowchart TB
    F["每个潜变量帧的 F_l<br/>3072 x 21 x 15"]
    WF["线性层 W_F，3072 到 384，再 LayerNorm"]
    PSP["空间位置编码 P_sp<br/>可学习 16x16 网格，双线性到 21x15"]
    PRAY["射线位置编码 g(Gamma(r_hat))<br/>对方位角、俯仰角做傅里叶，8 个倍频，再过零初始化 MLP"]
    X["X_l：315 个 token x 384"]

    Q0["每帧的可学习 query<br/>2 手 + 42 关节 + 4 个 register = 48"]
    L1["第 k 层（k=1..4）"]
    SCA["空间交叉注意力<br/>第 l 帧的 query 去看 X_l"]
    TSA["时间自注意力<br/>同一个 query 槽跨 21 帧，RoPE，无因果掩码"]
    QT["细化后的 query，21 x 48 x 384"]

    A["关节 token j 的交叉注意力权重<br/>A_j 覆盖 315 个格子"]
    SOFT["soft-argmax，式 3<br/>网格 = linspace(0,1) [code]"]
    P2D["2D 锚点 p_hat_j，在 [0,1]^2，再换成像素"]
    J3D["MLP 得到腕部相对的 3D 关节（米）"]
    ROT["姿态头：6D 经 Gram-Schmidt 得到 R_hat、theta_hat"]
    DEP["相机头：zeta_hat，t_z = exp(zeta_hat)"]
    BETA["形状头：手部 token 的时间平均得到 beta_hat（每手、每段）"]
    PRES["存在性 / 可见性 logits（不做匈牙利匹配）"]

    F --> WF --> X
    PSP -->|"相加"| X
    PRAY -->|"相加"| X
    Q0 --> L1
    X -->|"键和值"| SCA
    L1 --> SCA --> TSA -->|"重复 4 次"| QT
    QT -->|"关节 token"| A --> SOFT --> P2D
    QT -->|"关节 token"| J3D
    QT -->|"手部 token"| ROT
    QT -->|"手部 token"| DEP
    QT -->|"手部 token，全部帧"| BETA
    QT -->|"手部 token"| PRES
```

配置锚点，`[code]` `options/ace_ego_hand_k.yml`：`hidden_dim: 384`，`num_heads: 8`，`ffn_mult: 4`，`dropout: 0.0`，`alt_num_layers: 4`，`alt_num_register: 4`，`alt_query_mode: joint`，`alt_temporal_rope: true`，`betas_scope: per_hand`，`per_hand_betas_source: slot_mean`，`use_ray_pe: true`，`ray_pe_n_freqs: 8`。

---

## 图 4：基于射线的相机求解与 mixed-PnP 的分支（§3.3，式 4–5，附录 B.1）

```mermaid
flowchart TB
    F["F: 3072 x 21 x 21 x 15"]
    RH["射线头，1x1 卷积（零初始化）<br/>得到 3 x 21 x 21 x 15，再 L2 归一化"]
    TM["时间平均得到射线场 r_hat(u)<br/>3 x 21 x 15（一段视频内内参不变）"]
    LRAY["训练：L_ray = mean(1 - dot(r_hat, r_K))，式 4"]

    CFG{"哪套配置？"}
    KG["标准：视线方向来自 K<br/>b_j = ((u_j - c_x)/f_x, (v_j - c_y)/f_y)，(u_j,v_j) = p_hat_j"]
    KF["K-free：用 r_hat 拟合针孔 (f_hat, c_hat)<br/>闭式逐轴回归，<br/>方差下限 1e-4，焦距有括号"]
    KFOK{"拟合通过保护条件？"}
    BFIT["b_j = (p_hat_j - c_hat) / f_hat<br/>（可以把出画的锚点外推回来）"]
    BRAW["回退：在锚点处直接读 r_hat<br/>（鱼眼的 Re:InterHand；也是发布代码的路径 [code]）"]

    JC["J_can = M(R_hat, theta_hat, beta_hat)<br/>z_j = J_can_z,j + t_z"]
    MASK["若 z_j 至少 5 cm，且锚点在画面内、边距至少 2%，则 m_j = 1"]
    GATE{"至少 6 票，且<br/>重拟合 RMS 不超过 max(15 px, 1/4 框对角线)？"}
    LS["t_x = sum m_j z_j^-1 (b_j^x - J_can_x,j / z_j) / sum m_j z_j^-2（式 5）<br/>t_y 对称"]
    FB["腕部放在它自己的反投影射线上，深度为 t_z"]
    TAU["tau_hat = (t_x, t_y, t_z)"]

    F --> RH --> TM
    TM -.-> LRAY
    TM --> CFG
    CFG -->|"给定 K"| KG
    CFG -->|"没有 K"| KF --> KFOK
    KFOK -->|"通过"| BFIT
    KFOK -->|"不通过"| BRAW
    KG --> MASK
    BFIT --> MASK
    BRAW --> MASK
    JC --> MASK --> GATE
    GATE -->|"是"| LS --> TAU
    GATE -->|"否"| FB --> TAU
```

两套配置是分开训练的模型，各自有自己的视线来源（§3.3）；Table 1 里的 K-free 行不是测试时把求解器换一下。

---

## 图 5：训练目标怎么路由（§3.4，式 6，附录 B.3）

```mermaid
flowchart LR
    subgraph Pred["预测（每帧、每只手）"]
        PR["R_hat, theta_hat"]
        PB["beta_hat（每段一个）"]
        PJ["J_hat：相对根 / 相机系 / 腕部 3D"]
        P2["p_hat 二维锚点 + 重投影的 MANO 关节"]
        PT["tau_hat，经 mixed-PnP（梯度穿过求解器）"]
        PP["存在 / 可见 logits"]
        PRF["射线场 r_hat；拟合相机（仅 K-free）"]
    end

    subgraph Loss["L = 各项之和（式 6）"]
        LROT["L_rot：测地距离（1）+ 旋转矩阵 MSE（1）+ beta 的 l1（0.1）"]
        LJ["L_joint：l1，相对根（10），相机系（5），腕部（2）"]
        LIMG["L_img：二维锚点 + 在训练相机下的重投影（1），腕部（0.5）"]
        LCAM["L_cam：对 tau_hat 的 l1（1）"]
        LPRES["L_pres：存在性 BCE（0.5）/ 可见性 BCE（0.25）"]
        LTMP["L_tmp：J_hat 加速度的 l1（0.5）"]
        LRAY["L_ray：余弦，式 4（1）"]
        LFIT["L_fit（仅 K-free）：拟合相机相对标定相机的视线误差（5，预热 500 步）"]
    end

    subgraph Data["每个 batch 的数据来源（Table 2 权重）"]
        VID["ARCTIC 14 / HOT3D 18 / H2O 10 / OakInk2 14 / Re:InterHand 16<br/>81 帧窗口，完整 MANO 真值，出画帧全监督"]
        FH["FreiHAND 14：5 帧静态片段，右手，完整 MANO"]
        RHD["RHD 14：5 帧静态片段，21 个 3D 关节，没有 MANO"]
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

    VID -->|"全部项"| Loss
    FH -->|"全部项（静态片段上 L_tmp 按构造接近 0）"| Loss
    RHD -->|"只有 L_joint、L_img、L_pres；L_rot 保持为 0"| Loss
```

优化器（附录 B.2）：AdamW，权重衰减 1e-2，梯度裁剪 1.0，余弦，预热 200 步，共 20k 步；学习率：解码器+头 2e-4，LoRA 1e-4，patch embed 2e-5；每张 GPU 4 个片段；标准配置 16 张 A100（有效 batch 64），K-free 为 8 张 A100（有效 batch 32）。

---

## 图 6：评测协议（附录 A.1）

```mermaid
flowchart TB
    SEG["与基线共用的固定 81 帧测试片段<br/>ARCTIC 291，HOT3D 437，H2O 355，OakInk2 202，HOI4D 498"]
    RUN["每段一次前向（21 或 22 个潜变量帧）"]
    DET["检测：存在性大于 0.5<br/>用投影网格包围盒的 IoU 大于 0 来匹配（真值框膨胀 10%），同一侧"]
    GATE["对真值做画面内门控：某个关节投影在图像内，且 z 大于 1 cm"]
    PEN["带 -p 的惩罚：漏检按规范 MANO 计（单位旋转 R，theta 为零，平均 beta，tau = 0）；<br/>EPE2D 的漏检按图像对角线计"]
    IV["画面内指标：MPJPE-p，PA-p，EPE2D-p，GO-p，CT-p，FAcc，Recall，F1，Jitter"]
    OOS["出画分层：MPJPE^OOS，腕对齐，由真值门控，不惩罚"]
    MIX["MPJPE^+OOS = (n_IV e_IV + n_OOS e_OOS) / (n_IV + n_OOS)（式 oosmix）"]

    SEG --> RUN --> DET
    RUN --> GATE
    DET --> PEN --> IV
    GATE -->|"可见的手-帧"| IV
    GATE -->|"出画的手-帧"| OOS
    IV --> MIX
    OOS --> MIX
```

未发布的：片段 ID（HOT3D 自定义的 126/72 录像划分、OakInk2 的 202 子集、HOI4D 的 498 段）以及评分程序本身（GitHub issue #5、#6）。
