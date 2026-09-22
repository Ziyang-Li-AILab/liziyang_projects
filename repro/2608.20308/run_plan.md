# 运行计划

论文日程（附录 B.2）：20,000 步，AdamW，每张 GPU 上 4 个片段（16 张 A100，全局 batch 64），窗口 81 帧。主数字是按这个日程训完后，在 ARCTIC / HOT3D / HOI4D 上的 Table 1。

| 运行 | 目标 | 配置 | 状态 |
| --- | --- | --- | --- |
| `repro-main-k` | Table 1，已知内参 K | `src/configs/repro-default.yaml` | 未启动 |
| `repro-main-k-debug` | 第 4 阶段门槛：10 次迭代后退出 | 同一配置，`--debug-mode --allow-fake-mano` | 2026-09-22 完成，见 `results.md` |

`repro-main-k` 需要 Table 2 的视频、FreiHAND/RHD 图像，以及有许可证的 MANO。这台机器上这些都没有，剩余磁盘也放不下 latent 缓存。在这里跑 20k 步，也只是单张 RTX 5090、batch 2，不是 16 张 A100、全局 batch 64。

冒烟测试第 0–3 级（第 6 阶段）已在合成替身上跑完，然后才启动 debug。日志：`smoke_logs/tier{0,1,2,3}.log`。
