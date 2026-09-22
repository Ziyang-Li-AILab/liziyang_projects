# 闭环（paper2code，最小可运行）

可运行的闭环是 `src/closed_loop.py`。它对已发布的 GeoDiT 训练若干步（式 6，附录 B.2 的学习率），写出一份推理用 checkpoint，再把这份 checkpoint 载入一个新模块，做一次评测前向。这就是训练/推理闭环。它不是论文 Table 1 那种 20,000 步的完整实验。

## MANO：用哪个文件

[mano.is.tue.mpg.de/download.php](https://mano.is.tue.mpg.de/download.php) 会跳转到登录页。没有账号就下不下来，许可证也禁止再分发。注册、接受研究许可后，只下载：

**Models & Code → `mano_v1_2.zip`**

压缩包里，本项目实际读取的只有：

- `mano_v1_2/models/MANO_LEFT.pkl`
- `mano_v1_2/models/MANO_RIGHT.pkl`

示例扫描、配准样本，以及该网站链出去的其他数据集（FreiHAND、HO-3D 等）都不用下。那些不是手部模型。

转换一次（去掉过时的 `chumpy` 依赖）：

```bash
cd repro/2608.20308/code
python scripts/convert_mano_pkls.py --src /path/to/mano_v1_2/models --dst data/mano/mano
```

`smplx` 会找 `{ACE_EGO_HAND_MANO_DIR}/mano/MANO_RIGHT.pkl`。默认根目录是 `code/data/mano`。然后：

```bash
cd repro/2608.20308/src
python closed_loop.py --steps 2 --mano-dir ../code/data/mano
```

## 没有这些 pkl 时实际跑的是什么

`--allow-fake-mano` 使用 `src/ace_repro/hand_model.py` 里的运动学替身。损失和 mixed-PnP 仍会执行。数字不能和论文比。这是明确的替代，不是在声称论文训练时不用 MANO。

## 闭环里用到、但论文没写明的选择

与 `gaps_filled.md` 相同。影响本脚本的几项：AdamW 的 betas 取 `(0.9, 0.999)`（PyTorch 默认；论文没写），图像损失用 L1（论文没指定范数），用合成片段代替 Table 2 的数据混合。
