# 已填上的缺口：把 ACE-Ego-Hand 的一次训练跑起来所需要的全部设定

出处标签：`[paper §X]` 表示论文写明了（`paper.md` / `source/main.tex`）；`[code: path:line]` 来自已发布的推理仓库（`code/`）；`[framework default]` 是框架默认；`[venue convention]` 是会议惯例；`[guess: 理由]` 是猜测。`[code-vs-paper]` 标记论文和代码不一致的取值，并说明复现时选哪一个。开放问题编号 Q1–Q41 指向 `paper.md` 的「开放问题」；N1–N7 是 `inventory.md` §3 里由代码暴露出来的条目。

复现两套配置：**标准**（已知内参 K）和 **K-free**（测试时不给内参）。除非某一行另有说明，取值两套共用。

---

## A. 架构：确定性干净潜变量编码器

| 旋钮 | 取值 | 出处 |
|---|---|---|
| 骨干发行版 | `Wan2.2-Fun-5B-Control`（VideoX-Fun），低噪声 DiT 专家 | `[paper App. B.1]`，`[code: ace_ego_hand/archs/wan_backbone.py:108-111]`（Q7） |
| DiT 几何 | 30 个 block，宽度 3072，FFN 14336，输入 148 通道（48 潜变量 + 100 控制），输出 48 通道，patch (1,2,2) | `[paper App. B.1]`；patch 尺寸 `[code: wan_backbone.py:223]` |
| VAE | Wan 2.2 VAE（`AutoencoderKLWan3_8`），48 通道，空间 ×16，时间因果 ×4+1，冻结 | `[paper §3.1, App. B.1]`，`[code: ace_ego_hand/video_vae.py:70]` |
| σ=0 时的输入拼装 | `x = z`（48 通道干净潜变量），`y = zeros(100 通道)`，拼接成 148 通道 | σ=0 `[paper §3.1 Eq. 1]`；148 输入通道 `[paper App. B.1]`；剩余 148−48 = 100 个控制通道全为 0 `[code: geodit_arch.py:310; wan_backbone.py:123]`（Q2） |
| 时间步条件 | `t = 0`（σ·1000） | `[code: geodit_arch.py:314]`（Q3）；与 GenCeption 的 t=0 一致 |
| 文本条件 | 固定字幕 “Three geometry renders of two hands on black background: color-coded depth, joint skeleton, surface normals.” → umT5-xxl，补齐到 512 个 token，缓存后送进每个 block 的交叉注意力 | `[code: scripts/precompute_caption.py:26-27; geodit_arch.py:319]`（Q1，N1）。论文没写。 |
| 抽取点 | 取 block 下标 15 的输出（实际跑了 16 个 block），其余跳过 | `[paper §3.1, App. D]`，`[code: options/*.yml tap_layers: [15]; wan_backbone.py:257-260]` |
| 特征网格 | `21 × (H/32) × (W/32)` 个 token，D=3072（ARCTIC 672×480 → 21×15） | `[code: wan_backbone.py:223, 271-280; geodit_arch.py:103 tap_grid='patch']` — **`[code-vs-paper]`**：论文 §3.1 写的是 42×30、每格 16 像素；代码和附录 C.4 的耗时支持 32 像素（笔记 §3.5）。复现用 32 像素。（N2） |
| LoRA | r=64，α=64，dropout 0，加在全部 30 个 block 的 `self_attn.{q,k,v,o}`、`cross_attn.{q,k,v,o}`、`ffn.0`、`ffn.2` 上（每块 5.37M，共 161.219M）；`B` 零初始化 | `[paper App. B.1, B.2]`，`[code: wan_backbone.py:134-148]` |
| 骨干里非 LoRA 的可训练参数 | patch embedding（1.822M）从发布权重起全量训练；扩散输出头被注册但前向到不了（0.596M） | `[paper App. B.1]`，`[code: wan_backbone.py:155-158]`（Q4） |
| 精度 | 冻结底座 bf16；LoRA / patch-embed / 输出头的参数是 fp32；骨干外包 `autocast(bf16)`；时间嵌入 fp32；进解码器前特征转成 fp32 | `[code: wan_backbone.py:100,121,141-143,232; geodit_arch.py:320,396]`（Q5，Q28） |
| 梯度检查点 | 开启，按 block，non-reentrant | `[code: options/*.yml gradient_checkpointing: true; wan_backbone.py:249-252]`（Q6） |
| 多头渲染扩展 | 关闭（`mh_expand: false`） | `[code: options/*.yml]` |

## B. 架构：双向时空解码器

| 旋钮 | 取值 | 出处 |
|---|---|---|
| 宽度 / 头数 / FFN / dropout / 激活 / 归一化 | d=384，8 头，FFN ×4，dropout 0.0，GELU，pre-LayerNorm，`nn.MultiheadAttention` | 宽度 `[paper §3.2]`；其余 `[code: options/*.yml; memory_encoder.py:77-89]`（Q8） |
| 每个潜变量帧的 query | 2 个手 + 42 个关节 + 4 个 register = 48 | `[paper §3.2]`，`[code: memory_encoder.py:409-420]` |
| 层数 | 4 轮交替：先空间交叉注意力（query → 该帧 token），再时间自注意力（RoPE），无因果掩码 | `[paper §3.2]`，`[code: alt_num_layers: 4, alt_temporal_rope: true]` |
| Token 投影 | `LN(W_F F)`，3072→384 | `[paper §3.2 Eq. 2]` |
| 空间位置编码 | 可学习的 16×16 网格，双线性缩放到 token 网格（`spatial_pe_mode: interp2d`） | `[paper App. B.1]`，`[code: options/*.yml; memory_projector.py:400]` |
| 射线位置编码 | 对方位角、俯仰角做傅里叶，8 个倍频（sin+cos）→ 零初始化 MLP → 解码器宽度 384；来源是预测射线（`self_ray_pe: true`） | 编码方式 `[paper App. B.1]`；宽度 `[paper §3.2]`；`[code: ray_pe_n_freqs: 8; memory_projector.py:107-117]`（Q11） |
| Soft-argmax 网格 | 在 32 像素 token 网格上用 `linspace(0, 1, W)` / `linspace(0, 1, H)`，端点包含在内；输出在 [0,1]²，再乘图像尺寸得到像素 | `[code: memory_encoder.py:633-634]`（Q10）。论文说的是「网格中心」；射线位置编码用的是 `(j+0.5)/W` `[code: memory_projector.py:137]`。这处不一致可以容忍，按代码保留。 |
| 腕部相对 3D MLP | 21 个相对根关节的关节（`num_direct_joints: 21`），另有一个相机坐标系的直接腕部头 | MLP `[paper §3.2]`；21 个关节 `[paper §3 intro]`，`[code: options/*.yml num_direct_joints: 21; memory_projector.py:590-591]`（Q12） |
| 姿态头 | 6D → Gram–Schmidt → 全局朝向（1）+ 关节旋转（15）的旋转矩阵 | `[paper §3.2]`，`[code: memory_projector.py:20]` |
| 相机头 | 3 个输出 `(u_norm, v_norm, log z)`；`t_z = exp(log z)`；`(u,v)` 只给腕部射线回退用 | `[paper §3.2]`（对数深度），`[code: memory_projector.py:1170,1200-1201]` |
| 形状头 | 每只手、每个片段一个，对手部 token 做时间平均（`betas_scope: per_hand`，`per_hand_betas_source: slot_mean`） | `[paper §3.2]`，`[code: options/*.yml]` |
| 存在性头 | `exists_3d` = 是否存在，`exists_2d` = 是否可见；sigmoid；评测阈值 0.5 | `[paper §3.2, App. A.1]`，`[code: memory_projector.py:802-803,957; scripts/viz_preds.py:208-210]`（Q13） |
| 时间上采样 | 手部 token 特征在进入 MANO/相机头之前，从 T'=21 线性插值到 T=81；2D/3D 关节坐标直接插值 | 插值 `[paper §3.2]`；21/81 `[paper §3.1]`；`[code: memory_projector.py:1103-1135]` |
| 初始化 | token 为 `N(0, 0.02)`；全局朝向 / 手部姿态 / betas / 存在性 / 相机 / 直接腕部的输出层权重为零；存在性偏置 −1；相机偏置 `(0, 0, −0.693)`（初始 t_z = 0.5 m，图像中心）；直接腕部偏置 `(0, 0, 0.5)`；读出交叉注意力的 `out_proj` 为零 | `[code: memory_projector.py:65-81,396-438,554-591; memory_encoder.py:293-294]`（Q9） |
| 槽位约定 | 槽 0 = 左手，槽 1 = 右手（固定；不做匈牙利匹配） | `[paper §3.2]`，`[code: memory_projector.py:1229-1230]` |

## C. 架构：基于射线的相机求解器

| 旋钮 | 取值 | 出处 |
|---|---|---|
| 射线头 | 在 F 上做零初始化的 1×1 卷积 → 3 通道 → L2 归一化 → 对 21 个潜变量帧做时间平均 | `[paper §3.3]`，`[code: geodit_arch.py raymap: true]` |
| 射线监督 | `L_ray = mean_u (1 − ⟨r̂(u), r_K(u)⟩)`，在 token 网格上；r_K 是标定相机（针孔或鱼眼）从格子中心反投影的单位射线 | `[paper §3.3 Eq. 4]` |
| 视线方向，标准配置 | 在像素坐标的 soft-argmax 锚点处取 `((u_j − c_x)/f_x, (v_j − c_y)/f_y)` | `[paper §3.3]`，`[code: memory_projector.py:1379-1382]` |
| 视线方向，K-free（论文） | 来自拟合出的等效针孔 `(f̂, ĉ)`：`b_j = (p̂_j − ĉ)/f̂`，腕部回退也用它 | `[paper §3.3, App. B.1]` — **代码里没有**（N3） |
| 视线方向，K-free（已发布代码） | 在锚点处对射线场做双线性 `grid_sample`，`b = (r_x/r_z, r_y/r_z)` | `[code: memory_projector.py:1341-1355]` — 这是论文的*回退*路径；只把它当回退保留 |
| K-free 相机拟合：待实现 | 逐轴，在全部格子上把归一化像素坐标对 `tan = r_x/r_z`（或 `r_y/r_z`）做闭式最小二乘：`f̂ = cov(u, tan)/var(tan)`，`ĉ = mean(u) − f̂·mean(tan)`；可微 | 形式 `[paper App. B.1]`（「在 token 网格上做闭式、可微的逐轴线性回归」）；回归方向（像素对 tan）`[code: inference.py:39-42]`（事后的 `_fit_pred_K` 用同一方向）（Q14） |
| 拟合保护 | `var(tan) ≥ 1e-4`，否则回退；焦距括号 `f̂_norm ∈ [0.35, 3.0]`（以图像宽度为单位 ⇒ 水平视场约 19°–110°），否则回退 | 方差下限 `[paper App. B.1]`；括号 **`[guess: 论文只说「一个括号」没给数字；取得能包住训练时约 80° 视场族并留余量]`** |
| Mixed-PnP | 式 5 对 `(t_x, t_y)` 做加权最小二乘，`t_z` 来自相机头；关节投票 `m_j`：`z_j ≥ 0.05 m` 且锚点在画面内、边距 ≥ 2% | `[paper §3.3 Eq. 5, App. B.1]`，`[code: memory_projector.py:1253-1257,1364-1368]` |
| 是否接受这一行 | 投票 ≥ 6，且重拟合 RMS `≤ max(15 px, 0.25 × 2D 框对角线)`；否则回退 | `[paper App. B.1]`，`[code: memory_projector.py:1411-1415]` |
| 回退 | 把腕部放在相机头自己的 `(u,v)` 反投影射线上，深度为 `t_z`（标准配置经 K；K-free 按论文经拟合相机 / 按代码经射线场） | `[paper §3.3]`，`[code: memory_projector.py:1173-1204]`（Q16） |
| 射线模式下的门控缩放 | 用名义焦距 `f_ref = 图像宽度` 把残差换成伪像素，再套 RMS 阈值 | `[code: memory_projector.py:1391-1398]`（N4）。论文没写。 |
| 穿过 PnP 的梯度 | `torch.where(accept, pnp, fallback)`；`mixed_pnp_detach_mano` 关闭 ⇒ L_cam 的梯度能到 MANO 头、log-z 和锚点 | `[paper §3.4]`（「梯度流过 PnP 求解器」），`[code: memory_projector.py:1325-1328,1421]`（N5） |

## D. 优化器

| 旋钮 | 取值 | 出处 |
|---|---|---|
| 优化器 | AdamW | `[paper App. B.2]` |
| 峰值学习率：解码器 + 读出头 + 射线头 + 扩散头 | 2e-4 | `[paper App. B.2]` |
| 峰值学习率：LoRA | 1e-4 | `[paper App. B.2]` |
| 峰值学习率：patch embedding | 2e-5 | `[paper App. B.2]` |
| 权重衰减 | 1e-2（三组都用） | `[paper App. B.2]`；均匀施加 **`[guess: 论文只给一个数；没说要排除 norm/bias]`** |
| Betas | (0.9, 0.999) | `[framework default]`（Q25） |
| Epsilon | 1e-8 | `[framework default]`（Q25） |
| 梯度裁剪 | 全局范数 1.0 | `[paper App. B.2]` |
| 注册进优化器的参数 | 183.99M：LoRA 161.219M，解码器+头 20.347M，patch embed 1.822M，扩散头 0.596M，射线头 9,219 | `[paper App. B.1]` |
| 逐层学习率衰减 | 无 | `[paper App. B.2]`（三个平坦组） |

## E. Batch

| 旋钮 | 取值 | 出处 |
|---|---|---|
| 每张 GPU 的片段数 | 4 | `[paper App. B.2]` |
| GPU 数 | 标准配置 16 张 A100 / K-free 8 张 A100；Table 3 的 pooled-query 和绝对位置编码两行减半 | `[paper App. B.2]` |
| 有效 batch | 标准 64 个片段 / K-free 32 个 | `[paper App. B.2]` |
| 梯度累积 | 1 | **`[guess: 有效 batch 正好等于 GPU 数 × 4]`** |
| 片段长度 | 视频源 81 帧（21 个潜变量帧）；FreiHAND / RHD 为 5 帧（2 个潜变量帧） | `[paper App. B.2, A.2]` |
| Batch 是否同质 | 每个 batch 只来自一个数据集；**同一步里所有 rank 用同一个数据集** | 单数据集 `[paper App. A.2]`；跨 rank 同步 **`[guess: RHD 的 batch 会跳过 MANO 头，DDP 的归约要求各 rank 参与的参数一致]`**（Q36） |
| 数据源采样权重 | ARCTIC 14，HOT3D 18，H2O 10，OakInk2 14，Re:InterHand 16，FreiHAND 14，RHD 14（%） | `[paper Table 2]` |

## F. 日程

| 旋钮 | 取值 | 出处 |
|---|---|---|
| 总步数 | 20,000 | `[paper App. B.2]` |
| 学习率日程 | 余弦衰减 | `[paper App. B.2]` |
| 预热 | 200 步 | `[paper App. B.2]` |
| 预热形状 / 余弦下限 | 线性预热；余弦收到 0 | **`[guess: diffusers 的 get_cosine_schedule_with_warmup 惯例；仓库钉了 diffusers 并 import 了 diffusers.training_utils]`**（Q26） |
| 验证间隔 | 从 5k 到 20k 每 500 步一次（31 个 checkpoint） | 由 `[paper App. D]` 推算（「在第 5k 步和第 20k 步之间记录了 31 个验证 checkpoint」）（Q30） |
| 验证集 | ARCTIC 测试片段（291） | **`[guess: 附录 D 用腕对齐的 ARCTIC 误差报告这 31 个 checkpoint；论文从没单独命名一个验证划分]`** |
| 保存 checkpoint | 每 500 步，保留最终 | **`[guess: 与验证间隔一致]`** |
| 论文报告的 checkpoint | 第 20k 步 | `[paper App. B.1]`（「20k 步之后」） |
| L_fit 预热（仅 K-free） | 前 500 步线性 | `[paper App. B.3]` |
| 消融配方 | `tab:vdm`：只用 ARCTIC，10k 步。`tab:tap`：ARCTIC+HOT3D，有效 batch 16，20k，反投影解码。Table 3：K-free 配方但不加 L_fit，视线方向直接从射线场读取 | `[paper §4.3, App. D, Table 3 caption]` |

## G. 数据管线与增强

| 旋钮 | 取值 | 出处 |
|---|---|---|
| 视频源 / 划分 | ARCTIC（测试被试 s05），HOT3D（自定义录像划分 126/72），H2O（测试被试 4），OakInk2（序列级，评测子集 202 段），HOI4D（留出，零样本，166 段录像） | `[paper App. A.2, Table 2]`；HOT3D/OakInk2/HOI4D 的片段 ID **无法恢复**（Q37，Q38）。复现时用官方 HOT3D 划分加自己的片段，分开报告 |
| 帧率 | 30 fps，不抽帧 | `[paper App. A.2]`。注意：按 ViDiHand 附录，HOI4D 原生是 15 fps；ACE 论文没处理这一点。 |
| 输入分辨率 | ARCTIC 672×480；HOT3D 480×480；H2O、OakInk2 缩到宽 832，内参按比例缩放 | `[paper App. A.2]` |
| H2O / OakInk2 的高 | 把高补齐或缩放到 32 的倍数（H2O 1280×720 → 832×468 → **480**） | **`[guess: VAE 需要 32 的倍数（代码会对齐到 32）；论文只说「偶数潜变量网格」]`** `[code: infer_video.py:45-53]`（Q33） |
| Re:InterHand | 自我中心鱼眼渲染，10 fps，21 个潜变量帧的窗口，完整 MANO；分辨率未写 → 按原生尺寸对齐到 32 再编码 | `[paper App. A.2]`；分辨率 **`[guess]`**（Q33） |
| FreiHAND / RHD | 静态图像复制成 5 帧片段；按原生 224×224 / 320×320 编码（都已是 32 的倍数）；使用数据集自带的 K | 复制 `[paper App. A.2]`；分辨率 **`[guess: 原生尺寸已经满足 VAE 网格；论文没写缩放]`**（Q33） |
| VAE 潜变量 | 每段完整录像预先编码一次；窗口在潜变量空间里抽 | **`[guess: 附录 A.2「随机 21 个潜变量帧的窗口」、附录 D「21 或 22 个潜变量帧，取决于片段从哪开始」都暗示潜变量是按整段录像算的]`** |
| 窗口采样 | 每段录像、每一步均匀随机抽 21 个潜变量帧的窗口；短于 21 帧的录像填充或跳过 | 均匀 **`[guess]`**；短片段处理 **`[guess: 跳过]`**（Q32） |
| 统一的 MANO 格式 | smplx MANO，`flat_hand_mean=False`，16 个关节 + 5 个指尖顶点 → OpenPose-21；OakInk2（`flat_hand_mean=True`）通过从关节角里减去 `hands_mean` 来转换 | `[code: ace_ego_hand/mano_utils.py; memory_projector.py:1211,1228]`（Q35）；OakInk2 转换 **`[guess: 标准 smplx 恒等关系]`** |
| HOI4D 以及部分 H2O 的伪真值 MANO | 逐帧估计器，未点名 | `[paper App. A.2]`；估计器 **未知**（Q34）。若要重新生成标签，只用 HaMeR 风格的拟合，并写清楚 |
| 出画帧 | 用真值 MANO 全监督（不掩掉） | `[paper §3.4]` |
| 数据增强 | 无（不翻转 / 不变颜色 / 不裁剪）；代码里有可选的 span-dropout 钩子（把潜变量帧置零，并把掩码通道 [48:52] 设为 1），**默认关闭** | `[paper: 未提及]`；钩子 `[code: geodit_arch.py:287-313]`；论文实验是否用了 **未知 → 关闭** **`[guess]`**（Q31，N6） |
| 水平翻转 | 无（会交换左右槽，并改变 MANO 的左右手） | **`[guess: 与固定槽位约定冲突]`** |

## H. 损失形式（式 6；权重见附录 B.3）

```yaml
loss:                                  # L = 各组之和，每组内部再按权重
  rot:
    - {name: geodesic, target: [global_orient, articulation], weight: 1.0}      # [paper App. B.3]
    - {name: rotmat_mse, target: [global_orient, articulation], weight: 1.0}    # [paper App. B.3]
    - {name: l1, target: betas, weight: 0.1}                                    # [paper App. B.3]
    reduction: 对手、帧和 16 个旋转取平均                                          # [guess: 论文没写；取平均让权重与尺度无关]
  joint:
    - {name: l1, target: root_relative_3d, weight: 10.0,                        # [paper App. B.3] 主导项
       applied_to: [direct_mlp_joints, mano_joints]}                            # [guess: §3.2 的 MLP「预测腕部相对 3D 关节」，§3.4 的 L_joint 是「相对根」；两路都监督]
    - {name: l1, target: camera_frame_3d, weight: 5.0,                          # [paper App. B.3]
       applied_to: mano_joints + tau}                                           # [guess]
    - {name: l1, target: wrist_3d, weight: 2.0,                                 # [paper App. B.3]
       applied_to: [direct_wrist_cam, mano_wrist + tau]}                        # [guess: 代码里有直接腕部头]
  img:
    - {name: l1, target: softargmax_2d_anchors, weight: 1.0, units: normalized [0,1]}   # 权重 [paper App. B.3]；范数和单位 [guess: 代码里锚点在 [0,1]]（Q18）
    - {name: l1, target: reprojected_mano_joints_under_training_K, weight: 1.0}         # [paper App. B.3]
    - {name: l1, target: wrist_2d, weight: 0.5}                                          # [paper App. B.3]
    visibility_mask: 只对真值投影落在图像内的关节                                     # [guess: 与 EPE2D「画面内关节」一致]
  cam:
    - {name: l1, target: tau_assembled, weight: 1.0, grad_through_pnp: true}    # [paper App. B.3, §3.4]
      apply_on_oos_frames: true                                                 # [guess: torch.where 让每一行的 tau 都可微；出画帧「全监督」]
  pres:
    - {name: bce, target: existence (exists_3d), weight: 0.5}                   # [paper App. B.3]
    - {name: bce, target: visibility (exists_2d), weight: 0.25}                 # [paper App. B.3]
    targets:
      existence: 该帧这只手有真值则为 1（含出画），否则 0                            # [guess: 出画的手必须算「存在」，才能落实 §3.4「全监督出画的手」]
      visibility: 附录 A.1 的画面内门控（某个关节投影在图像内且 z > 1 cm）            # [guess: 复用评测门控]
      freihand_left_slot: 存在性 0，可见性 0                                     # [guess: FreiHAND 只有右手]
  tmp:
    - {name: l1_acceleration, target: predicted_camera_frame_mano_joints, weight: 0.5}  # [paper App. B.3] ||Jhat_{t+1}-2 Jhat_t+Jhat_{t-1}||_1，只惩罚预测加速度，不减真值
      mask: 无。论文公式里没有真值项，也没有可见性掩码                                    # 作用在相机系 MANO 关节 J_can+tau 上 [guess]（Q21）
  ray:
    - {name: cosine, target: unit_rays_from_calibration_at_cell_centers, weight: 1.0}   # [paper §3.3 Eq. 4, App. B.3]
  fit:                                   # 仅 K-free
    - {name: l1_bearing_error, target: calibrated_bearings, weight: 5.0, warmup_linear_steps: 500,   # [paper §3.4, App. B.3]
       evaluated_at: all token-grid cells, gradient only via (f_x, f_y, c_x, c_y)}       # 点集和范数 [guess: 网格就是拟合的自然定义域]（Q15）
routing:
  RHD: {rot: 0, betas: 0, joint: on, img: on, pres: on, cam: off, tmp: on}      # [paper §3.4, App. A.2]「旋转和形状项保持为零」；相机项关闭 [guess: 没有 MANO ⇒ PnP 没有规范关节]
  FreiHAND: 全部项，只走右手槽                                              # [paper App. A.2]
  Re:InterHand（鱼眼）: 关闭 fit（针孔拟合失败 → 回退）                         # [paper App. B.1]
  出画帧: rot、joint（相对根）、pres、tmp、cam 开；img 关                      # [paper §3.4] 全监督；img 关 [guess: 画面内没有 2D 目标]
```

## I. 技巧

| 技巧 | 取值 | 出处 |
|---|---|---|
| LoRA 的 `B` 零初始化；射线头零初始化；射线位置编码 MLP 零初始化 ⇒ 第 0 步的网络等于未改动的骨干 | 是 | `[paper App. B.1, B.2]`，`[code: memory_projector.py:116-117]` |
| 解码器输出头零初始化；存在性偏置 −1；深度偏置 log(0.5) | 是 | `[code: memory_projector.py:554-591]` |
| 混合精度 | 骨干 bf16 autocast，可训练参数和解码器 fp32 | `[code]`（见 A） |
| 梯度检查点 | 开 | `[code: options/*.yml]` |
| 权重 EMA | 无 | **`[guess: 论文没有，代码也没有；shape_ema_logit 是片段内的形状携带，不是权重 EMA]`**（Q27） |
| Dropout / drop-path | 0 / 无 | `[code: options/*.yml dropout: 0.0, lora_dropout: 0.0]` |
| 标签平滑 / mixup | 无 | `[paper: 未提及]` |
| 随机种子 | 42 | 推理 `[code: options/*.yml seed: 42]`；训练种子 **`[guess: 沿用 42]`**（Q29）；论文：每种设定只跑一次 `[paper App. C, D]` |
| 权重衰减的排除项 | 无 | **`[guess]`** |
| 片段级形状携带（`shape_ema_logit`，初始 0.5） | 发布 checkpoint 的结构里有；保留默认 | `[code: memory_projector.py:367]` |

## J. 评测协议（待实现；附录 A.1）

| 项目 | 取值 | 出处 |
|---|---|---|
| 片段 | 固定的 81 帧测试片段：ARCTIC 291，HOT3D 437，H2O 355，OakInk2 202，HOI4D 498 | `[paper Table 2]`；除 ARCTIC/H2O 能按被试确定外，ID 无法恢复（Q37–Q39） |
| 解码 | `tiled`，22 个潜变量帧的窗口，锚在 `(81·i)//4`，末端锚定补洞 | `[code: ace_ego_hand/inference.py:78-107]`（Q40）；论文：21 或 22 个潜变量帧 `[paper App. D]` |
| 检测 | 存在性 > 0.5；用投影网格包围盒的严格正 IoU 匹配（真值框膨胀 10%），同一侧 | `[paper App. A.1]` |
| 画面内门控 | 任一真值关节投影落在 `[0,W)×[0,H)` 且 `z > 1 cm` | `[paper App. A.1]` |
| 惩罚（-p） | 漏检按规范 MANO 计（单位旋转 R、θ 为零、平均 β、τ=0）；EPE2D 的漏检按图像对角线计（ARCTIC 826 px，HOT3D 679 px） | `[paper App. A.1]` |
| 指标 | MPJPE / PA（腕对齐，21 关节，mm）；EPE2D 只对可见关节（px）；GO（度）；CT（m）；Jitter（mm/frame²，作用在 `J̄ + τ`）；FAcc；Recall；F1 | `[paper App. A.1, Table tab:metrics]` |
| 出画分层 | MPJPE^OOS 腕对齐、由真值门控、不惩罚；MPJPE^+OOS 按式 oosmix | `[paper App. A.1]` |
| 效率计时 | 一段 81 帧的 ARCTIC，一张 A100，预热后至少 3 次的中位数 | `[paper App. C.4]` |

## K. 推理默认值

| 旋钮 | 取值 | 出处 |
|---|---|---|
| 编码宽度 | 832，保持宽高比，两轴都对齐到 32 的倍数 | `[code: infer_video.py:45-53,77]`（Q41） |
| 分块宽度 | 22 个潜变量帧 | `[code: infer_video.py:76]` |
| K-free 的占位内参 | 名义 60° 水平视场的针孔，模型不读它 | `[code: infer_video.py:149-156]` |
| 输出 | 每帧 `global_orient, hand_pose, betas, cam_trans, direct_joints2d, direct_joints_cam, direct_wrist_cam, exists_2d, exists_3d`，外加 `pred_intrinsics`（事后拟合） | `[code: inference.py:20-21; infer_video.py:194-198]` |

---

## L. 标签统计与往返核对

对 A–K 里 129 条带标签的行做了脚本统计（`roundtrip_check.txt`）：引用 `[paper]` 的 87 行，引用 `[code]` 的 51 行（其中 30 行两者都有），`[framework default]` 2 行，含 `[guess]` 的行 **30 行（23%）**，猜测标签共 35 个。低于 30% 的门槛，但猜测集中在损失路由（H）和数据管线（G），这正是从零写训练器时会悄悄偏掉的地方。两处 `[code-vs-paper]` 冲突（特征网格分辨率 N2、K-free 视线来源 N3）的处理是：架构跟代码，K-free 拟合跟论文（作为新选项实现）。

与论文的往返核对（2026-09-21 做的；日志在 `roundtrip_check.txt`）：把标了 `[paper §X]` 的行里的 150 个数字记号，拿到 `source/main.tex` 的对应章节里搜（不是在全文里随便搜）。第一遍标出 24 个；其中 4 个是真正的归属错误，已改（148 个输入通道 → 附录 B.1；解码器宽度 384 → §3.2；21 个关节 → §3 开头 + 代码；21/81 帧 → §3.1）。其余是 LaTeX 归一化假象、单位换算（5 cm ↔ 0.05 m）、推算值（验证间隔 500 步），或与论文引用同行的 `[guess]` 取值；每一条都在日志里有处置。`gaps_filled.md` 和论文之间不再有矛盾。阶段指南里提到的 `paper-verification` 技能这里没装，这段脚本核对代替它。

### 第 4 阶段风险最高的未知项（按严重程度）
1. **出画帧和 RHD 上的损失路由**（H）。论文给的是原则，不是掩码；掩码错了，头条的出画结果就会变。
2. **K-free 的拟合相机视线路径 + L_fit**（C，H）。代码里没有；已发布的 K-free checkpoint 的解码和论文描述不一致，所以即使用发布的解码，发布的数字也可能复现不出来。
3. **片段列表**（G，J）。HOT3D/OakInk2/HOI4D 的协议恢复不了；相对 Table 1 会有协议偏移，自己的片段要按自己的片段来报。
4. **HOI4D/H2O 的伪真值来源**（G）。估计器不同，H2O 训练集上的标签分布就不同。
5. **特征网格分辨率**（A）。如果论文对、代码默认错，soft-argmax 的粒度和计算量会差 4 倍；用发布 checkpoint 做一次冒烟来定（抽取点的 token 数必须等于 `21×(H/32)×(W/32)`）。
