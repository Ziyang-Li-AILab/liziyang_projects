# 清单：ACE-Ego-Hand 官方代码里有什么，复现还缺什么

**仓库**：https://github.com/ggxxii/ACE-Ego-Hand ，MIT。`code/` 是 `main` @ `9757868`（2026-09-10）的压缩包快照（`git clone` 超时，改用 `codeload.github.com` 的 tar 包，所以 `code/` 里没有 `.git`）。仓库创建于 2026-08-21；105 stars、5 forks、6 个开放 / 0 个已关闭 issue（2026-09-21）。论文 v1 时仓库名叫 `dreamhand`。
**权重**：https://huggingface.co/acerobotics2025/ACE-Ego-Hand 上的 `ace_ego_hand_k.pt`、`ace_ego_hand_kfree.pt`（CC BY-NC 4.0；当时从这台网络取不到文件大小）。骨干 `Wan2.2-Fun-5B-Control` 来自 VideoX-Fun（另下，约 10 GB 的 bf16 DiT + VAE，外加约 11 GB 的 umT5-xxl 文本编码器，做字幕缓存时用一次）。
**README 的说法**：「本仓库目前只提供推理；训练代码尚未包含。」待办里评测代码、鱼眼支持、训练代码和数据准备脚本都没勾。
**当时的本地环境**：没装 `torch`，下面任何一条都没实际跑过。以下全是静态阅读。

## 1. 组件表

| 组件 | 有没有 | 路径 | 说明 |
|---|---|---|---|
| 模型定义：骨干包装 | **有** | `code/ace_ego_hand/archs/wan_backbone.py`（298 行） | 包装从 **VideoX-Fun** 导入的 `Wan2_2Transformer3DModel`（`third_party/`，不是内嵌副本）。加载**低噪声**专家（`transformer_low_noise_model_subpath`）。LoRA 用 `peft.inject_adapter_in_model` 加在每块 10 个线性层上。`forward_features` 重写了原版前向，带 block 抽取和可选提前退出（`run_head=False`）。里面还有一个多头「渲染」扩展（`mh_expand_dit`，输入 148→244 通道，输出 48→144 通道），发布配置关掉了（`mh_expand: false`）。 |
| 模型定义：组合体 | **有** | `code/ace_ego_hand/archs/geodit_arch.py`（494） | `GeoDiT`：为 `feature_mode in {emode, noised_video}` 拼输入，射线头（`raymap: true`），固定字幕缓冲，`forward_emode` → 投影器。论文实验 = `noised_video` 且 `nv_sigma=0.0`（文档字符串：「论文实验用的体制」）。还有一个 flow-matching 采样器（`_sample_flow_noise`），给生成式的 “dmode” / `gen_overlay` 损失用，论文没描述这条路径。 |
| 模型定义：解码器 | **有** | `code/ace_ego_hand/archs/projector/memory_projector.py`（1449）、`memory_encoder.py`（638）、`spatial_readout.py`（109） | `temporal_arch: alternating` 时，`MemorySegmentBetasTransformer` 就是论文的双向时空解码器。`MemoryAlternatingEncoder` 自称移植了 VGGT-Omega 的 register-token 交替注意力。消融开关很多（`pooled` 架构、`camera_tokens`、`alt_mem_tokens`、`mano_readout`、`joints2d_readout`、`mixed_pnp_detach_mano`、`shape_ema_logit`……）。对发布配置来说，这个模块大约 40% 是死的（论文 B.1：解码器注册了 20.347M 参数，实际用到 12.287M）。 |
| Mixed-PnP 平移求解 | **有** | `memory_projector.py::_decode_cam_trans_mixed_pnp`（L1259–1421） | 就是式 5，门控为 `_PNP_Z_NEAR=0.05`、`_PNP_INTERIOR_MARGIN=0.02`、`_PNP_MIN_INTERIOR=6`、`_PNP_RESID_FRAC=0.25`、`_PNP_RESID_MIN_PX=15`（= 附录 B.1）。可微；用 `torch.where` 选分支。 |
| K-free 相机拟合（论文 §3.3，附录 B.1） | **部分 / 不一致** | `code/ace_ego_hand/inference.py::_fit_pred_K`（只在事后） | 论文里拟合出的针孔 `(f_hat, c_hat)` 要送进视线方向和腕部回退，**不在模型解码里**。`self_ray_decode: true` 在 2D 锚点处双线性采样射线场（`grid_sample`，`align_corners=True`），这是论文的*回退*路径。`_fit_pred_K` 在前向之后，用逐轴 `lstsq` 拟合 `u = cx + fx*(dx/dz)` 得到 `(fx,fy,cx,cy)`，只把它写进 pickle 的 `pred_intrinsics`。没有任何方差下限或焦距括号保护。 |
| 射线头 | **有** | `geodit_arch.py`（`raymap: true`，约 L223） | 在抽取特征上做 1×1 卷积 → 单位射线；在投影器里做时间平均。论文说零初始化，第 4 阶段冒烟时再核。 |
| MANO 层与转换 | **有** | `code/ace_ego_hand/mano_utils.py`（146）、`code/scripts/convert_mano_pkls.py`（127） | smplx MANO，`flat_hand_mean=False`，16 个关节 + 5 个指尖顶点 → OpenPose-21 重映射（`mano_forward_batch_full`）。转换脚本从官方 pickle 里去掉 chumpy。 |
| 视频读写 + VAE | **有** | `code/ace_ego_hand/video_vae.py`（83）、`code/infer_video.py`（220） | OpenCV 解码 → 缩到 `--encode_w 832` 并对齐到 32 的倍数 → `AutoencoderKLWan3_8`（Wan2.2 VAE，48 通道，空间 ×16，时间因果 4×+1）。输出潜变量是 CPU 上的 fp16。 |
| 推理驱动 | **有** | `code/infer_video.py`、`code/ace_ego_hand/inference.py`（112） | `tiled`（按每个 81 帧片段锚定的 22 个潜变量帧窗口；「基准数字用的就是这个」）和 `full` 两种模式。相机：`--camera cam.json` / `--intrinsics`；K-free 用一个 60° 占位，模型「不读」。每帧写出 `global_orient, hand_pose, betas, cam_trans, direct_joints2d, direct_joints_cam, direct_wrist_cam, exists_2d, exists_3d`。 |
| 模型包装 / 加载 checkpoint | **有** | `code/ace_ego_hand/models/geodit_model.py`（111） | 构建 `GeoDiT`，把 MANO 模型挂到投影器上，加载 `.pt`。 |
| 字幕缓存 | **有** | `code/scripts/precompute_caption.py`（61） | 固定字幕 `"Three geometry renders of two hands on black background: color-coded depth, joint skeleton, surface normals."` → umT5-xxl（512 token，补齐）→ `cache/caption_embed.pt`。被放弃的渲染生成设计的残留，但仍是模型输入。 |
| 可视化 | **有** | `code/scripts/viz_preds.py`（334） | 2D 叠加，CPU 网格光栅化。显示门控：`exists_2d > 0.5`（2D），以及 `exists_3d > 0.5` 且平均 z > 5 cm（3D）。 |
| 配置 | **有** | `code/options/ace_ego_hand_k.yml`、`ace_ego_hand_kfree.yml` | 除了 `self_ray_decode: true` 之外相同。关键值见下。 |
| 数据加载 | **无** | — | 完全没有数据集代码。需要：ARCTIC、HOT3D、H2O、OakInk2、Re:InterHand、FreiHAND、RHD 的读取器，统一成 MANO 格式；以及 HOI4D 测试读取器。 |
| 增强管线 | **无**（只有钩子） | `geodit_arch.py::extract_features(span_mask=...)`、`wan_backbone.py::build_control(visible_mask_lat)` | 有 span-dropout 钩子（潜变量帧置零，并把掩码通道 `[48:52]` 设为 1）。论文实验是否用了它**未知**；论文没提任何增强。 |
| 损失函数 | **无** | — | 损失代码为零。只有暗示：文档字符串提到 `_mano_geometry_losses`、「只对关节的 cam_trans 损失」、「渲染损失」、`gen_overlay`；有 `mixed_pnp_detach_mano` 开关；直接头（`direct_joints_cam`、`direct_wrist_cam`）是当损失目标存在的。式 6 的 8 项（外加 L_fit）都得自己写。 |
| 训练循环 | **无** | — | 文档字符串提到一个在 DDP 下按 `emode/dmode` 分流的训练器（`forward(mode=...)`）、`t2v_flag`、wandb。没有发布。 |
| 优化器与日程 | **无** | — | 论文 B.2 给了 AdamW / 权重衰减 / 裁剪 / 余弦 / 预热 / 3 组学习率；betas、eps、最小学习率没写。 |
| 评测脚本 | **无** | — | 指标定义只在论文附录 A.1。HOT3D（自定义 126/72 录像划分，437 段）、OakInk2（202 子集）、HOI4D（166 段录像 / 498 段）的片段列表未发布（issue #5）。EPE2D-p 的惩罚细节在 issue #6 里被问过。 |
| 预训练权重 | **有** | HF `acerobotics2025/ACE-Ego-Hand` | 两套配置都有。只有针孔。checkpoint 结构覆盖全部消融变体（「一个 checkpoint 结构覆盖每一次消融」，附录 B.1）。 |
| 复现说明 | **部分** | `code/README.md` | 只有推理环境（克隆 VideoX-Fun、转换 MANO、下载骨干、字幕缓存）。`requirements.txt` 钉死：torch 2.5.1、diffusers 0.36.0、transformers 4.57.3、peft 0.18.1、smplx 0.1.28、numpy 1.24.4、CUDA 12.1、Python 3.10。 |

### 发布配置里的取值（`options/ace_ego_hand_k.yml`）

```yaml
seed: 42
backbone:
  tap_layers: [15]            # block 下标，取 block 15 之后的输出（跑了 16 个 block）
  mh_expand: false            # 不做渲染头扩展：保持输入 148 / 输出 48 通道
  train_backbone: true        # 梯度流入 LoRA + patch_embedding
  gradient_checkpointing: true
  feature_mode: noised_video  # x = (1-sigma) z + sigma eps，control = 0
  nv_sigma: 0.0               # => 干净潜变量，t = 0
  lora: {lora_rank: 64, lora_alpha: 64, lora_dropout: 0.0}
  raymap: true                # 打开射线头
  self_ray_pe: true           # 射线位置编码来自预测射线（不是真值 K）
  # 仅 kfree：self_ray_decode: true
projector:
  hidden_dim: 384, num_temporal_layers: 4, num_heads: 8, ffn_mult: 4, dropout: 0.0
  max_spatial_tokens: 4096, max_video_frames: 256
  cam_trans_decode: mixed_pnp
  spatial_pe_mode: interp2d   # 可学习 16x16 网格，双线性缩放
  betas_scope: per_hand, per_hand_betas_source: slot_mean
  spatial_pool_mode: slot, num_direct_joints: 21, joints2d_readout: pooled
  temporal_arch: alternating, alt_num_register: 4, alt_num_layers: 4
  alt_query_mode: joint, alt_temporal_rope: true, alt_mem_tokens: 0
  use_ray_pe: true, ray_pe_n_freqs: 8
```

## 2. 对照：代码架构 vs 论文架构

| 论文说法 | 代码 | 判定 |
|---|---|---|
| 特征网格是 `42×30` 个 `16×16` 像素的格子，`D=3072`（§3.1） | `tap_grid` 默认 `patch`：对 ×16 的 VAE 潜变量再做 `(1,2,2)` patch，得到 **32 像素**的 `21×15` 格子，`D=3072`。`pixelshuffle` 模式（`42×30` 上 `D/4=768`）存在但没开。附录 C.4 的耗时（1.28 s 里 28% 是 DiT+解码器）只和 32 像素网格对得上（见笔记 §3.5）。 | **不一致。论文正文错了，或者它描述的是 VAE 网格。** 复现用 32 像素 token。 |
| 在 30 个 block 的第 15 个截断，跑 16 个 block（式 1） | `tap_layers: [15]`，`run_head=False` 在 block 15 之后返回 | 一致 |
| σ=0 的干净潜变量，「保持发布的输入/输出通道数」 | `nv_sigma=0.0`，`x=z`，`y=zeros(100)`，`t=0`，148 通道的 patch embed | 一致；论文没写控制通道是零、也没写 `t=0` |
| 文本条件 | 每个交叉注意力里都有固定字幕嵌入 | **论文没写** |
| LoRA r=64、α=64、无 dropout，每块 10 个线性层，全部 30 个 block | `LORA_TARGETS` = 自注意力 q,k,v,o + 交叉注意力 q,k,v,o + ffn.0、ffn.2；注入每一个 block | 一致（每块 5.37M） |
| patch embedding 全量可训练；扩散头被注册但前向到不了 | `enable_training(train_patch_embed=True, train_head=True)`；两者都转成 fp32 | 一致 |
| 48 个 query：2 手 + 42 关节 + 4 register | `MemoryAlternatingEncoder` 里的 `hand_token(2)`、`joint_token(42)`、`register_token(4)` | 一致 |
| 4 层交替，时间维 RoPE，无因果掩码 | `alt_num_layers: 4`，`alt_temporal_rope: true`，`_RoPETemporalLayer` | 一致 |
| soft-argmax 用「归一化的网格中心坐标」 | `linspace(0,1,W)` 含端点（`memory_encoder.py` L633），而射线位置编码的像素中心是 `(j+0.5)/W`（`memory_projector.py` L137） | 小的约定不一致（模型会学过去） |
| 相机头只预测对数深度 | `head_cam_trans` 输出 3 个数：`(u_norm, v_norm, log z)`；`u,v` 只给腕部射线回退用 | 论文写少了 |
| 回退：「腕部放在它自己的反投影射线上，深度为 t_z」 | 反投影的是相机头自己的 `(u,v)`，不是 soft-argmax 的腕部锚点 | 回答了开放问题 Q16 |
| K-free 的视线方向来自拟合针孔 | 视线方向是在锚点处从射线场采样的；解码里没有拟合 | **不一致。发布的 K-free 解码 = 论文的无拟合回退** |
| 存在性 + 可见性分数 | `exists_3d`（= 「presence」，存在性）和 `exists_2d`（可见性）；零权重头，偏置 −1 | 一致；命名对照已记下 |
| 片段级 β：对手部 token 做时间池化 | `betas_scope: per_hand`，`per_hand_betas_source: slot_mean`；另外还有 `shape_ema_logit` 携带参数 | 一致（多一个 EMA 旋钮） |

### 代码里有、论文没提的开关
`feature_mode: emode`（渲染生成体制，`t_emode=1000`）、`mh_expand` 多模态头（深度/骨架/法线渲染）、`gen_overlay` / `dmode` 的 flow-matching 损失、`span_mask` 的 span-dropout、`geo_control_mods`（用真值几何渲染当控制，「DETACHED oracle」）、`mixed_pnp_detach_mano`、`shape_ema_logit`、`camera_tokens`、`alt_mem_tokens`（跨窗口记忆）、`tap_grid: pixelshuffle`、`joints2d_readout: spatial`（`SpatialJointReadout`）、`cam_trans_decode: hamer`。这些是 “DreamHand” 设计空间的化石；发布配置只选了其中一点。

## 3. 开放问题分流（编号 = `paper.md` 的「开放问题」）

| # | 问题 | 状态 | 结论 / 位置 |
|---|---|---|---|
| 1 | DiT 的文本条件 | **代码已解决** | 固定字幕（`scripts/precompute_caption.py`），umT5-xxl，512 token，送进全部交叉注意力（`geodit_arch.py` L319） |
| 2 | σ=0 时的控制通道 | **代码已解决** | `y = zeros(100 通道)`；span-dropout 在被丢掉的帧上把 `y[:,48:52]` 设为 1（`geodit_arch.py` L298–313） |
| 3 | σ=0 时的时间步 | **代码已解决** | `t = nv_sigma * 1000 = 0`（L314）；adaLN 经 `time_embedding(sinusoidal(t))`，fp32 |
| 4 | patch embedding 上有没有 LoRA | **代码已解决** | 没有；patch embedding 以 fp32 全量训练（`wan_backbone.py` L155） |
| 5 | 计算 dtype | **代码已解决** | 冻结底座 bf16；LoRA / patch embed / 头的参数 fp32；骨干外包 `autocast(bf16)`；时间嵌入是 fp32 孤岛；投影器收到 `.float()` 特征 |
| 6 | 梯度检查点 | **代码已解决** | `gradient_checkpointing: true`，按 block 的 `torch.utils.checkpoint`（non-reentrant） |
| 7 | 用 Wan2.2 的哪一个子模型 | **代码已解决** | VideoX `wan_civitai_5b.yaml` 里的 `transformer_low_noise_model_subpath` |
| 8 | 解码器头数 / FFN / dropout / 激活 | **代码已解决** | 8 头，FFN ×4，dropout 0，GELU，pre-LayerNorm，`nn.MultiheadAttention` |
| 9 | token / 头的初始化 | **代码已解决** | token `N(0, 0.02)`；全局朝向 / 手部姿态 / cam_trans / betas / exists / 直接腕部头的权重为零；存在性偏置 −1；直接腕部偏置 `(0,0,0.5)`；相机头偏置 `(0,0,-0.693)` → 初始 `t_z = exp(-0.693) = 0.5 m`，在图像中心（`memory_projector.py` L580–581）；交叉注意力读出的 `out_proj` 为零（「第 0 步是空操作」） |
| 10 | Soft-argmax 的坐标约定 | **代码已解决** | `linspace(0,1,W)` 含端点（`memory_encoder.py` L633–634） |
| 11 | 第 0 步的射线位置编码 | **代码已解决** | `RayPositionalEncoding.ray_proj[2]` 零初始化（`memory_projector.py` L116）；射线头按论文零初始化 → 初始化时射线位置编码是空操作 |
| 12 | 直接 3D 的腕部从哪来 | **代码已解决** | 单独的 `head_direct_wrist_cam`（相机系腕部，偏置 z=0.5）+ 相对根的 `direct_joints_cam`；MANO 路径给出另一只腕 |
| 13 | 存在性 vs 可见性 | **代码已解决** | `exists_3d` = 存在 / presence（检测门控），`exists_2d` = 可见性；可视化阈值 0.5 |
| 14 | K-free 相机拟合（回归目标、焦距括号） | **继续挂起** | 只有事后的 `_fit_pred_K`（把 `u` 对 `dx/dz` 回归，像素中心）；模型内拟合、方差下限 `1e-4`、焦距括号都**没有** |
| 15 | L_fit 的精确形式 | **继续挂起** | 没有损失代码 |
| 16 | K-free 模式下的回退 | **代码已解决** | 相机头的 `(u,v)` → 采样射线场（`_decode_cam_trans` 的 `pred_rays` 分支）→ `t_x = r_x/r_z · t_z` |
| 17 | 鱼眼 / 无拟合时的视线采样 | **代码已解决** | 双线性 `grid_sample`，`align_corners=True`，按关节锚点 |
| 18 | L_img 的范数和单位 | **继续挂起** | 线索：锚点在 `[0,1]`，`direct_joints2d` 是 sigmoid 输出 |
| 19 | RHD 上 L_rot 怎么路由 | **继续挂起** | — |
| 20 | 出画帧 / FreiHAND 左手槽的存在性目标 | **继续挂起** | 线索：两个头（2D 可见、3D 存在）→ 出画 =（存在 1，可见 0）是自然读法 |
| 21 | L_tmp 作用在什么上、按什么帧率 | **继续挂起** | 线索：手部 token *特征*在进 MANO/相机头之前从 21 线性插值到 T（`_interp_feat_t`，L1103），2D/3D 关节*坐标*直接插值（`_interp_coords_t`，L1134–1135），所以所有逐帧输出都在 81 帧率上，L_tmp 可以在那里算 |
| 22 | 出画帧上的 L_cam | **继续挂起** | 线索：`torch.where(accept, pnp, inv)` 让每一行都有可微的 τ，所以损失*可以*加在出画行上 |
| 23 | 测地线 / MSE 对 15 个关节怎么归约 | **继续挂起** | — |
| 24 | 5 帧静态片段的损失掩码 | **继续挂起** | — |
| 25 | AdamW 的 betas / eps | **继续挂起** | — |
| 26 | 余弦下限 / 预热形状 | **继续挂起** | — |
| 27 | EMA | **继续挂起** | `shape_ema_logit` 是形状携带旋钮，不是权重 EMA |
| 28 | 混合精度 | **代码已解决**（见 5） | bf16 autocast，可训练参数 fp32 |
| 29 | 随机种子 | **部分解决** | 推理配置 `seed: 42`；训练种子未知 |
| 30 | 训练中的验证协议 | **继续挂起** | — |
| 31 | 数据增强 | **继续挂起** | 有 span-dropout 钩子；是否使用未知；没有翻转/颜色/裁剪代码 |
| 32 | 窗口采样 | **继续挂起** | 推理分块是 22 个潜变量帧 = 85 个像素帧；训练窗口按论文是「21 个潜变量帧 = 81 帧」 |
| 33 | 图像数据集的分辨率 / 画布 | **继续挂起** | — |
| 34 | HOI4D/H2O 的伪真值估计器 | **继续挂起** | — |
| 35 | 统一的 MANO 约定 | **代码已解决** | `flat_hand_mean=False`，smplx MANO，用 5 个指尖顶点重映射到 OpenPose-21（`mano_utils.py`） |
| 36 | 各 GPU 上每个 batch 怎么抽数据集 | **继续挂起** | — |
| 37 | HOT3D 自定义划分 / 437 段 | **继续挂起，无法恢复** | issue #5 没人回；计划：官方 HOT3D 划分 + 自己的片段，分开报告 |
| 38 | OakInk2 的 202 子集 / HOI4D 的 498 段 | **继续挂起，无法恢复** | 继承自 ViDiHand（没有发布） |
| 39 | 评测代码 | **继续挂起** | 按附录 A.1 自己写；issue #6 表明连作者的 EPE2D-p 都需要澄清 |
| 40 | 分块解码还是整段解码 | **代码已解决** | 基准 = `tiled`，`tile_w=22`，窗口锚在 `(81 i)//4`，末端锚定补洞；`full` = 一次前向 |
| 41 | 推理分辨率 | **代码已解决** | `--encode_w 832`，保持宽高比，两轴对齐到 32 |

**代码新暴露、已写进 `gaps_filled.md` 的条目**：N1 固定字幕原文；N2 特征网格 32 像素；N3 发布的 K-free 解码没有相机拟合；N4 射线模式下接受门控用 `f_ref = img_w` 做伪像素缩放；N5 `mixed_pnp_detach_mano` 默认关闭（关闭 → 梯度经 PnP 到达 MANO 头，与论文「梯度流过 PnP 求解器」一致）；N6 span-dropout 是否使用；N7 `exists` 头偏置 −1 的初始化。

## 4. 缺的部分打算放哪（第 4 阶段计划）

```
repro/2608.20308/
├── code/                          # 官方推理代码，在数字落地之前不动
└── src/                           # 第 4 阶段创建
    ├── data/
    │   ├── mano_format.py         # 统一 MANO 记录（Q35 约定）+ 各数据集转换器
    │   ├── arctic.py hot3d.py h2o.py oakink2.py reinterhand.py freihand.py rhd.py hoi4d.py
    │   ├── windows.py             # 随机 21 潜变量帧（81 帧）窗口（Q32），5 帧静态片段
    │   └── mixture.py             # 每个 batch 单来源采样，权重按 Table 2（Q36）
    ├── losses/
    │   ├── rot.py joint.py img.py cam.py pres.py tmp.py ray.py fit.py   # 式 6 各项，附录 B.3 权重
    │   └── routing.py             # RHD / FreiHAND / 出画掩码（Q19，Q20，Q24）
    ├── camera/
    │   └── kfree_fit.py           # 可微的逐轴针孔拟合 + 保护，视线来源（Q14，N3）
    ├── train.py                   # DDP 循环，3 组学习率，AdamW，余弦+预热，梯度检查点（Q25–Q30）
    ├── eval/
    │   ├── metrics.py             # 附录 A.1（MPJPE-p、PA-p、EPE2D-p、GO-p、CT-p、Jitter、出画分层）
    │   └── segments.py            # 我们自己的 81 帧片段列表（Q37–Q39），标明不是官方的
    └── configs/                   # 在 options/ace_ego_hand_*.yml 上扩展的训练 yml
```

模型代码本身从 `code/` 复用（import `ace_ego_hand.*`）；模型侧唯一要加的，是 K-free 的拟合相机视线路径（N3），作为新的 `cam_trans_decode` 选项，而不是去改发布路径。

上面的目录是当时的计划，和现在的 `src/ace_repro/` 不完全一样。转换器实际在 `src/ace_repro/data/converters/`：FreiHAND 有脚本但缺 RGB；HOT3D 的脚本、几何检查和没写完的 `.pt` 见 `HOT3D_STATUS.md`。
