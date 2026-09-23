# liziyang_projects

[ACE-Ego-Hand](https://arxiv.org/abs/2608.20308)（arXiv 2608.20308）的复现。官方 [ACE-Ego-Hand](https://github.com/ggxxii/ACE-Ego-Hand) 只发布推理。`repro/2608.20308/src/` 补了训练、损失、数据转换、评测和可视化。

## 目录

| 路径 | 内容 |
| --- | --- |
| `repro/2608.20308/code/` | 官方推理代码（MIT）。VideoX-Fun 在 `code/third_party/`，用它自己的许可证 |
| `repro/2608.20308/src/` | 训练（`train.py`）、损失、评测、可视化 |
| `repro/2608.20308/src/ace_repro/data/converters/` | FreiHAND、HOT3D 转成统一的 `Clip` |
| `repro/2608.20308/src/tools/` | ARCTIC 片段构建、HOT3D 针孔 mp4、FreiHAND zip 分段下载 |
| `repro/2608.20308/src/viz_pair.py` | 左预测、右真值。K-free 用拟合相机画预测，真值用片段里的 K |
| `repro/2608.20308/src/viz_kfree_832.py` | 把 K-free 按 `infer_video.py` 的默认宽度 832 再画一遍 |
| `repro/2608.20308/code/results/` | 上面两个脚本写出的对照视频 |
| `DATA.md` | 数据集怎么下、放到哪、用哪条命令转换。`data/` 不进 Git |
| `WEIGHTS.md` | 超过 GitHub 体积上限的权重 |

阅读笔记留在 `repro/2608.20308/`：`architecture.md`、`TRAINING.md`、`gaps_filled.md`、`HOT3D_STATUS.md`、`paper.md`。论文 PDF 和 LaTeX 源只留在本机。

## 环境

```bash
conda create -n ace-ego-hand python=3.10
conda activate ace-ego-hand
pip install -r repro/2608.20308/code/requirements.txt
```

在 `repro/2608.20308/src` 下跑脚本时：

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="code:src"
```

MANO 与 Wan / ACE 权重的下载见 [DATA.md](DATA.md) 和 [WEIGHTS.md](WEIGHTS.md)。

## 可视化

片段 `.pt` 和对应 mp4 准备好之后（命令在 [DATA.md](DATA.md)）：

```bash
cd repro/2608.20308/src
# K-given：求解和画图都用片段里的真值 K
python viz_pair.py --config configs/repro-freihand.yaml \
  --clip ../data/arctic/clips/espressomachine_use_01_ego.pt \
  --video ../data/arctic/clips/espressomachine_use_01_ego.mp4 \
  --ckpt kgiven ../code/checkpoints/ace_ego_hand_k.pt \
  --out ../code/results

# K-free：不把真值 K 送进网络；预测网格用拟合相机
python viz_pair.py --config configs/repro-kfree.yaml \
  --clip ../data/hot3d/train/clip-001849.pt \
  --video ../data/hot3d/pinhole/clip-001849.mp4 \
  --ckpt kfree ../code/checkpoints/ace_ego_hand_kfree.pt \
  --out ../code/results

# 832 宽上的 K-free（射线解码 + 事后拟合的相机）
python viz_kfree_832.py
```

ARCTIC 用头盔相机（图像文件夹 `0/`），编码成 672×480。第三人称裁剪是 `tools/build_arctic_allocentric.py`，不是论文里的那个视角。

## 训练和转换

```bash
cd repro/2608.20308/src
python train.py --config configs/repro-freihand.yaml
python -m ace_repro.data.converters.hot3d --raw ../data/hot3d/raw --out ../data/hot3d --video-dir ../data/hot3d/pinhole
python -m ace_repro.data.converters.freihand --raw ../data/freihand/raw --out ../data/freihand
python tools/build_arctic_ego.py --seq s01/espressomachine_use_01
```

日程和损失对照在 `repro/2608.20308/TRAINING.md`。官方推理入口仍是 `code/infer_video.py`。
