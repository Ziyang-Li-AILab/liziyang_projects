# 数据不进 Git

`repro/2608.20308/data/` 留在本机。标注、图像和编码后的 `.pt` 都有各自的许可证，体积也超过普通 git 能放的范围。下面是本仓库脚本实际读取的布局和下载入口。权重见 [WEIGHTS.md](WEIGHTS.md)。MANO 的 pkl 同样不进仓库。

路径都相对 `repro/2608.20308/`。转换和导出在 `src/` 下跑，并设置：

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="code:src"
```

## MANO

[mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/download.php) 注册并接受研究许可后，下载 **Models & Code → `mano_v1_2.zip`**。本仓库只用其中的：

- `mano_v1_2/models/MANO_LEFT.pkl`
- `mano_v1_2/models/MANO_RIGHT.pkl`

官方 pickle 里嵌了已经不能在新 numpy 上导入的 `chumpy`。转一次：

```bash
cd repro/2608.20308/code
python scripts/convert_mano_pkls.py --src /path/to/mano_v1_2/models --dst data/mano/mano
```

`smplx` 找 `{ACE_EGO_HAND_MANO_DIR}/mano/MANO_RIGHT.pkl`。默认根目录是 `code/data/mano`。

## FreiHAND

评测和 `src/ace_repro/data/converters/freihand.py` 读的是 [FreiHAND](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html) 公开包，解压后：

```text
data/freihand/raw/training_K.json
data/freihand/raw/training_mano.json
data/freihand/raw/training_xyz.json
data/freihand/raw/training/rgb/00000000.jpg
```

整包大约 4 GB。只要标注 json 时，可以用区间请求从官方 zip 里抽成员：

```bash
cd repro/2608.20308/src
python tools/fetch_zip_members.py <FreiHAND zip URL> ../data/freihand/raw training_K.json training_mano.json training_xyz.json
```

转成训练用的 `.pt`（写到 `data/freihand/train` 和 `data/freihand/test`，不进 Git）：

```bash
python -m ace_repro.data.converters.freihand \
  --raw ../data/freihand/raw --out ../data/freihand
```

## HOT3D

公开子集是 Hugging Face 上的 [bop-benchmark/hot3d](https://huggingface.co/datasets/bop-benchmark/hot3d)，说明在 [hot3d clips README](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md)。这不是要在 projectaria.com 登记才能下的完整 VRS。

```text
data/hot3d/raw/train_aria/clip-001849.tar
data/hot3d/raw/test_aria/clip-003365.tar
```

`test_ht_pose` 的 tar 没有 `hands.json`，转换器会跳过，不能当监督。

转换器把 Aria 鱼眼流 `214-1` 去畸变成针孔，再顺时针转 90°，连同 K 和 MANO 一起转。发布权重吃的是这个正立视角。左手形状基按 HOT3D 的做法做了镜像（`build_hand_models`）。

```bash
cd repro/2608.20308/src
python -m ace_repro.data.converters.hot3d \
  --raw ../data/hot3d/raw --out ../data/hot3d \
  --video-dir ../data/hot3d/pinhole
```

只要 mp4、不编码潜变量时：

```bash
python tools/export_hot3d_pinhole.py \
  ../data/hot3d/raw/train_aria/clip-001849.tar \
  ../data/hot3d/pinhole/clip-001849.mp4
```

`clip-001849` 是训练集里的一条，不是论文 Table 1 的测试集。

## ARCTIC

到 [ARCTIC](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md) 注册后下载。一条序列的最小集合是：

| 压缩包 | 放到 |
| --- | --- |
| `raw_seqs.zip` | `data/arctic/raw_seqs.zip` |
| `meta.zip` | `data/arctic/meta.zip` |
| `cropped_images/<subject>/<sequence>.zip` | `data/arctic/cropped_images/<subject>/<sequence>.zip` |

图像文件夹 `0/` 是头盔第一人称，`1/`–`8/` 是第三人称针孔，而且是跟着物体走的裁剪。论文里的 ARCTIC 视角是第一人称，分辨率 672×480。

ARCTIC 的左手是用 smplx 原始形状空间拟合的（[smplx issue 48](https://github.com/vchoutas/smplx/issues/48)，作者没有为此重新拟合）。两个 ARCTIC 脚本都走 `setup_mano_models`，不镜像左手。HOT3D 转换仍走镜像后的 `build_hand_models`。

```bash
cd repro/2608.20308/src
python tools/build_arctic_ego.py --seq s01/espressomachine_use_01
python tools/build_arctic_allocentric.py --seq s01/espressomachine_use_01 --camera 1
```

输出在 `data/arctic/clips/`：`*_ego.pt` / `*_ego.mp4`，以及 `*_cam1.pt` / `*_cam1.mp4`。`viz_pair.py` 看到 `dataset == arctic` 时会把左手形状基翻回这个原始空间再画。
