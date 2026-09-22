# Run plan

Paper schedule (Appendix B.2): 20,000 steps, AdamW, batch 4 clips/GPU on 16 A100s (global batch 64), 81-frame windows. Main number is Table 1 on ARCTIC / HOT3D / HOI4D after that schedule.

| Run | Goal | Config | Status |
| --- | --- | --- | --- |
| `repro-main-k` | Table 1, K-given | `src/configs/repro-default.yaml` | not launched |
| `repro-main-k-debug` | Stage-4 gate: 10 iterations, then exit | same config, `--debug-mode --allow-fake-mano` | finished 2026-09-22, see `results.md` |

`repro-main-k` needs the Table 2 videos, FreiHAND/RHD images, and licensed MANO. None of those are on this machine, and the free disk cannot hold the latent cache. A 20k-step run here would also be a single RTX 5090 at batch 2, not 16 A100s at global batch 64.

Smoke tiers 0–3 (stage 6) completed on the synthetic stand-in before the debug launch. Logs: `smoke_logs/tier{0,1,2,3}.log`.
