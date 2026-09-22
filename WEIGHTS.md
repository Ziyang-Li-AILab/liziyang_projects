# 权重没有放进 Git

GitHub 普通 git 拒绝超过 100 MB 的文件，Git LFS 也拒绝超过 2 GB 的单个文件。下面这些文件留在本机，不在仓库里。

| 本地路径 | 大小 | 重新获取 |
| --- | --- | --- |
| `repro/2608.20308/code/ckpt/Wan2.2-Fun-5B-Control/diffusion_pytorch_model.safetensors` | 9.4 GB | [Wan2.2-Fun-5B-Control](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control)（或 ModelScope `PAI/Wan2.2-Fun-5B-Control`） |
| `repro/2608.20308/code/ckpt/Wan2.2-Fun-5B-Control/Wan2.2_VAE.pth` | 2.7 GB | 同上 |
| `repro/2608.20308/code/ckpt/Wan2.2-Fun-5B-Control/models_t5_umt5-xxl-enc-bf16.pth` | 11 GB，本机已删 | 只在需要重做字幕向量时下载。现成向量在 `code/cache/caption_embed.pt`，已经提交 |
| `repro/2608.20308/code/checkpoints/ace_ego_hand_k.pt` | 702 MB | [ace_ego_hand_k.pt](https://huggingface.co/acerobotics2025/ACE-Ego-Hand/resolve/main/ace_ego_hand_k.pt) |
| `repro/2608.20308/code/checkpoints/ace_ego_hand_kfree.pt` | 702 MB | [ace_ego_hand_kfree.pt](https://huggingface.co/acerobotics2025/ACE-Ego-Hand/resolve/main/ace_ego_hand_kfree.pt) |
| `repro/2608.20308/runs/repro-main-k-debug/ckpt_*.pt` | 各 1.5 GB | 本机 debug 训练产物，不是论文权重。日志 `train_log.jsonl` / `eval_log.jsonl` 已提交 |

DiT 的 `config.json`、tokenizer（`google/umt5-xxl/`）和 `README_T5_REMOVED.txt` 已提交。
