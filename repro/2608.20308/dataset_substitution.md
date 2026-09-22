# 数据集替代

论文的训练混合（附录 B.2 / Table 2）是 ARCTIC、HOT3D、H2O、OakInk2、ReInterHand、FreiHAND 和 RHD，外加留出的 HOI4D 做零样本评测。mixed-PnP 解码以及旋转 / 形状损失都需要 MANO（`MANO_RIGHT.pkl` / `MANO_LEFT.pkl`）。

## 磁盘上有什么

2026-09-22 这台 Windows 上的状态。早先在另一台 Linux 上做的 debug 仍然是合成数据。

| 来源 | 状态 |
| --- | --- |
| Wan2.2 VAE | 在 `code/ckpt/Wan2.2-Fun-5B-Control/Wan2.2_VAE.pth`（2,818,839,170 字节）。不进 Git |
| Wan2.2 DiT | 没有 `diffusion_pytorch_model.safetensors`，正式 `train.py` 起不来 |
| HOT3D-Clips | `data/hot3d/raw/` 有 8 段官方训练 tar、2 段 `test_ht_pose` tar。训练 `.pt` 写完 5 段，停在 `clip-001854`。见 `HOT3D_STATUS.md` |
| FreiHAND 标注 | 只有 `data/freihand/raw/training_{K,mano,scale,xyz}.json` |
| FreiHAND RGB | 没有（`training/rgb/%08d.jpg`） |
| ARCTIC、H2O、OakInk2、ReInterHand、RHD、HOI4D | 没有 |
| MANO | 本机有许可证 pkl，转成了 `code/data/mano/`。两处都不进 Git |

## 冒烟和 debug 实际在什么上训练

`src/ace_repro/data/synthetic.py` 用运动学替身（`src/ace_repro/hand_model.py` 里的 `FakeMANO`）渲染按颜色编码的骨架片段，再用真正的 Wan VAE 编码。这些运行的每一个数字都标了 `hand_model=fake`。

这不是论文的划分，也不能和 Table 1 比。FreiHAND 转换器（`src/ace_repro/data/converters/freihand.py`）已经写了，但没有跑，因为 RGB 还缺。HOT3D 转换器已经能写出 `ClipStore` 读得动的 `.pt`，目前只有 5 段，而且没有拿来训练。
