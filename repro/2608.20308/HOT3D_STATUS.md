# HOT3D 转换：做到哪了

2026-09-22 停在编码中途。转换器能用，8 段训练 clip 里写完了 5 段。没有接着把剩下 3 段编码完，也没有跑 `train.py`。

## 已经落地的

`train.py` 只读 `data/hot3d/train/**/*.pt`（`ClipStore`，共享 `schema.Clip`）。原始 tar 它不认。

转换器是 `src/ace_repro/data/converters/hot3d.py`。它读 [HOT3D-Clips](https://huggingface.co/datasets/bop-benchmark/hot3d)（公开子集，不是要在 projectaria.com 登记才能下的完整 VRS）。上游说明在 [hot3d clips README](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md)。

和论文、工具包对齐的部分：

- 论文附录 A.2：HOT3D 用 480×480、30 fps、81 帧 RGB（因果 VAE 的 4k+1，对应 21 个潜变量帧）。对角线 679 px 就是 \(\sqrt{480^2+480^2}\)。
- 输入是 Aria RGB 流 `214-1`，1408×1408，`FISHEYE624`。按 `hand_tracking_toolkit` 的 `convert_to_pinhole_camera(focal_scale=1)` 去畸变。论文没写这个比例，用的是工具包默认值 1。`clip-001849` 上 162 只手的手腕有 150 只落在针孔画面里，其余只是略微出画。
- 去畸变之后再顺时针转 90°，连同 K、全局朝向和相机系平移一起转。Aria 传感器图的 x 轴在头戴视角里朝上，发布权重训练的是正立自我视角（手从画面下方进入，和 ARCTIC 同一种构图）。只转像素、不转相机，标注会离开手；只去畸变、不转正，几何自洽但推理对不齐。平移不是把 `t` 乘上旋转：smplx 绕静止手腕 `J0` 转，存的是减去 `J0` 之后的残差，所以 `t' = R t + R J0 - J0`。`meta["upright"]` 必须是 `cw90`，否则转换器会重写这份 `.pt`。`clip-001849.pt` 已按这个朝向重编码。`clip-001850` … `clip-001853` 的 tar 不在本机，还是传感器朝向，tar 回来再跑会重写。
- MANO `thetas` 是 15 维 PCA。这台环境的 smplx 0.1.28 默认 `flat_hand_mean=False`，和仓库里 Hot3D 的约定一致。轴角残差是 `thetas @ hands_components[:15]`，模型自己加 `hands_mean`。
- `wrist_xform` 是世界系轴角加平移。写进 clip 的相机系平移是 `tau = R_wc.T @ (J0 + t_world - t_cam) - J0`。smplx 的根平移列是形状化之后的静止手腕 `J0`，漏掉 `J0` 会留下大约 115 mm 的常数偏差。
- 左手 `shapedirs` 的 x 分量按 smplx issue 48 / HOT3D `MANOHandModel` 做了镜像，加在 `build_hand_models` 里，训练和转换用的是同一个模型。

8 段训练 tar（`clip-001849` … `clip-001856`）都做过几何检查：可见手腕落在鱼眼的 amodal 框里；针孔投影在画面内时，去畸变表和鱼眼投影相差不超过 3 px。两段官方 `test_ht_pose`（`clip-003365`、`clip-003366`）没有 `hands.json`，转换器会跳过，不能当监督。

`clip-001849.pt` 用 `ClipStore` 读过，形状是训练要的：

| 字段 | 值 |
| --- | --- |
| `latent` | `(48, 21, 30, 30)` float16 |
| `n_video_frames` | 81 |
| `K` | `[207.90, 207.90, 240.23, 241.19, 480, 480]`（正立之后；主点相对传感器朝向对调过） |
| `go` / `hp` | `(81, 2, 3, 3)` / `(81, 2, 15, 3, 3)`，行列式为 1 |
| `exists` / `has_mano` | 162 / 162（双手、每一帧） |
| `visible` | 162（至少有一个关节在画面内） |
| 手腕在画面内 | 150 / 162 |
| 手腕深度 | 0.20–0.46 m |

同一条命令还写完了 `clip-001850` … `clip-001853`，五个文件都是 1,986,247 字节。`video_vae.load_vae` 能载入 `Wan2.2_VAE.pth`，missing / unexpected keys 都是 0。本机没有 CUDA，一段 81 帧大约 2.5 分钟。

`code/ace_ego_hand/video_vae.py` 改了一处导入：这份 checkout 没有 `videox_fun/models/__init__.py`，官方的 `from videox_fun.models import AutoencoderKLWan3_8` 会失败，改成从 `videox_fun.models.wan_vae3_8` 直接导入。

## 停在这里的

编码进程在写 `clip-001854` 之前被停掉。下面三份还没有 `.pt`：

- `data/hot3d/train/clip-001854.pt`
- `data/hot3d/train/clip-001855.pt`
- `data/hot3d/train/clip-001856.pt`

`meta["upright"]` 不是 `cw90` 的 `.pt` 再跑会被重写。在 `src/` 下：

```
python -m ace_repro.data.converters.hot3d --raw ../data/hot3d/raw --out ../data/hot3d
```

需要本机已有 `code/ckpt/Wan2.2-Fun-5B-Control/Wan2.2_VAE.pth`，以及转好的 MANO（`code/data/mano/mano/MANO_{LEFT,RIGHT}.pkl`，由 `code/scripts/convert_mano_pkls.py` 从有许可证的官方 pkl 生成）。只复查投影、不编码：加上 `--check-only`。

还没做、也不要当成已经对齐论文的：

- 没有用这 5 段跑过 `train.py`。DiT 权重 `diffusion_pytorch_model.safetensors` 这台机器上没有，正式训练起不来。有 HOT3D 的 `.pt` 时，`build_sources` 会把它算进混合；其它数据源仍然空着，验证仍然只看 `data/arctic/test/`。
- `infer_video.py` 不读 tar，也不读 `.pt`。给它的 mp4 必须是同一套正立针孔。转换器 `--video-dir` 写出的就是这份图；单独拿去畸变表再 `remap`、不调用 `rotate_to_upright`，会回到传感器朝向，发布权重对不齐。
- 每段公开 clip 有 150 帧，转换只用前 81 帧（论文的一个窗口，也是 VAE 的一整块）。`ClipStore` 看到潜变量时间长度 ≤ 21 就整段返回，不会再切窗。
- 这 8 段都是同一条录像 `P0001_23fa0ee8` 的连续窗口，只够冒烟，不够当多样本的 HOT3D。
- 论文的 126/72 录像划分和 437 个评测段编号没有公开。这里用的是官方 HOT3D-Clips 划分，不能拿来对 Table 1。
- 测试集两段没有公开的 MANO，写不出监督 `.pt`。

## 没有放进 Git 的

| 本地文件 | 原因 |
| --- | --- |
| `code/ckpt/Wan2.2-Fun-5B-Control/Wan2.2_VAE.pth`（2,818,839,170 字节） | 超过 GitHub 单文件上限。地址见 `WEIGHTS.md` |
| `data/hot3d/raw/**/*.tar` | `clip-001854.tar`、`clip-003366.tar` 超过 100 MB。清单在 `data/hot3d/raw/manifest.json` |
| `mano_v1_2/` 和 `code/data/mano/` | MANO 是研究许可，不随仓库分发 |
