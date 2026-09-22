# 复现结果

**论文声称**（Table 1，ARCTIC，已知 K 的 ACE-Ego-Hand）：F1 1.000，MPJPE-p 15.256 mm，PA-p 7.474 mm，MPJPE+OOS 16.783 mm，EPE2D-p 9.180 px，GO-p 11.807 deg，CT-p 0.021 m，Jitter 2.700。

那一行是在 Table 2 的混合数据上、用真实 MANO 训练 20,000 步得到的。这里**没有**重跑。见 `dataset_substitution.md`。

## 实际跑了什么

主机：一张 RTX 5090（32 GB），conda 环境 `ace-ego-hand`（`torch 2.8.0+cu128`）。加载骨干时在 `code/ace_ego_hand/archs/wan_backbone.py` 里使用 `low_cpu_mem_usage=True`。默认的 fp32 构造加上 9.4 GB 的 state dict，内存峰值接近 30 GB；当时交换分区已经满了，上一次第 1 级尝试挂了一夜，一次前向都没记下来。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 第 0 级（VAE、相机拟合、在真值上的损失和指标） | 通过 | `smoke_logs/tier0.log` |
| 第 1 级（训练 + 评测前向，梯度出现在解码器 / LoRA / patch-embed） | 通过，峰值 12.3 GB | `smoke_logs/tier1.log`，损失 7.2576 |
| 第 2 级（一步 AdamW，同一 batch） | 通过，7.25764 → 6.34232 | `smoke_logs/tier2.log` |
| 第 3 级（20 次迭代，5 步滚动均值） | 通过，7.5070 → 4.2977 | `smoke_logs/tier3.log` |
| `train.py --debug-mode`（10 步，batch 2，合成数据） | 0.1 分钟内退出码 0 | `runs/repro-main-k-debug/`，`smoke_logs/debug_train.log` |

第 1 步的梯度范数约 3.3e9（解码器尚未训练）。`grad_clip: 1.0` 把它压到第 2 步的 137、第 10 步的 8.0。合成混合上的损失在这 10 步里从 6.874 降到 5.198。

第 10 步的 debug 评测（`hand_model=fake`，8 段合成验证片段）：FAcc 0.030，Recall 0.000，F1 0.000，MPJPE-p 189.799，PA-p 13.625，EPE2D-p 678.823，GO-p 129.865，CT-p 0.537，Jitter 为 NaN（没有长度 ≥3 帧的匹配轨迹），MPJPE+OOS 175.048。

## 结论

对 Table 1 的每一项指标，判定都是：**[缺口，假设：这次评测不是论文的实验]**。

189.8 mm 的 MPJPE-p，是用 FakeMANO 训练 10 步后，在骨架片段上对漏检手的惩罚（Recall 为 0）。它不是 ARCTIC 上的测量，也不能用来说明论文的 15.256 mm 是错的。要得到可比较的数字，需要 Table 2 的录像、FreiHAND/RHD 图像、MANO pkl，以及 20,000 步的日程。

debug 运行的 checkpoint：`runs/repro-main-k-debug/ckpt_last.pt` 和 `ckpt_step000010.pt`（各 1.5 GB）。它们不是论文权重。

缺口表和往返核对仍是 `gaps_filled.md` 与 `roundtrip_check.txt`。冒烟测试按复现流程的第 6 阶段做完。没有启动论文规模的训练，因此没有和 Table 1 的同一步数对照。
