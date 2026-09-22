# Dataset substitution

The paper's training mixture (Appendix B.2 / Table 2) is ARCTIC, HOT3D, H2O, OakInk2, ReInterHand, FreiHAND, and RHD, plus held-out HOI4D for zero-shot eval. MANO (`MANO_RIGHT.pkl` / `MANO_LEFT.pkl`) is required for the mixed-PnP decode and for rotation / shape losses.

## What is on disk

| Source | Status |
| --- | --- |
| Wan2.2-Fun-5B-Control DiT + VAE | present under `code/ckpt/` (T5 encoder weights deleted after `code/cache/caption_embed.pt` was cached) |
| FreiHAND annotations | `data/freihand/raw/training_{K,mano,scale,xyz}.json` only |
| FreiHAND RGB | absent (`training/rgb/%08d.jpg`) |
| ARCTIC, HOT3D, H2O, OakInk2, ReInterHand, RHD, HOI4D | absent |
| MANO pkls | absent anywhere under `/home/remote` |

Disk free at the time of this note: about 16 GB on `/` after the debug checkpoints. A latent cache of the paper mixture does not fit.

## What the smoke and debug runs actually trained on

`src/ace_repro/data/synthetic.py` renders colour-coded skeleton clips from a kinematic stand-in (`FakeMANO` in `src/ace_repro/hand_model.py`) and encodes them with the real Wan VAE. Every number from those runs is labelled `hand_model=fake`.

This is not the paper's split, not metric MANO, and not comparable to Table 1. The FreiHAND converter (`src/ace_repro/data/converters/freihand.py`) is written but was not run, because the RGB files and the MANO pkls are both missing.
