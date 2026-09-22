# Reproduction results

**Paper claim** (Table 1, ARCTIC, K-given ACE-Ego-Hand): F1 1.000, MPJPE-p 15.256 mm, PA-p 7.474 mm, MPJPE+OOS 16.783 mm, EPE2D-p 9.180 px, GO-p 11.807 deg, CT-p 0.021 m, Jitter 2.700.

That row was trained for 20,000 steps on the Table 2 mixture with real MANO. It was **not** rerun here. See `dataset_substitution.md`.

## What did run

Host: one RTX 5090 (32 GB), conda env `ace-ego-hand` (`torch 2.8.0+cu128`). Backbone load uses `low_cpu_mem_usage=True` in `code/ace_ego_hand/archs/wan_backbone.py`. The default fp32 construct plus the 9.4 GB state dict peaks near 30 GB RAM; with swap already full, the previous tier-1 attempt sat overnight and never logged a forward.

| Check | Result | Evidence |
| --- | --- | --- |
| Tier 0 (VAE, camera fit, loss and metrics on GT) | pass | `smoke_logs/tier0.log` |
| Tier 1 (train + eval forward, grads in decoder / LoRA / patch-embed) | pass, 12.3 GB peak | `smoke_logs/tier1.log`, loss 7.2576 |
| Tier 2 (one AdamW step, same batch) | pass, 7.25764 → 6.34232 | `smoke_logs/tier2.log` |
| Tier 3 (20 iters, 5-step rolling mean) | pass, 7.5070 → 4.2977 | `smoke_logs/tier3.log` |
| `train.py --debug-mode` (10 steps, batch 2, synthetic) | exit 0 in 0.1 min | `runs/repro-main-k-debug/`, `smoke_logs/debug_train.log` |

Step 1 gradient norm is ~3.3e9 (untrained decoder). `grad_clip: 1.0` brings it to 137 at step 2 and 8.0 at step 10. Loss on the synthetic mixture goes 6.874 → 5.198 over those 10 steps.

Debug eval at step 10 (`hand_model=fake`, 8 synthetic val clips): FAcc 0.030, Recall 0.000, F1 0.000, MPJPE-p 189.799, PA-p 13.625, EPE2D-p 678.823, GO-p 129.865, CT-p 0.537, Jitter NaN (no matched run of ≥3 frames), MPJPE+OOS 175.048.

## Verdict

**[gap, hypothesis: this eval is not the paper's experiment]** for every Table 1 metric.

The 189.8 mm MPJPE-p is the missed-hand penalty on skeleton clips after 10 steps with FakeMANO (Recall 0). It is not an ARCTIC measurement and is not evidence that the paper's 15.256 mm is wrong. A comparable number needs the Table 2 recordings, FreiHAND/RHD images, MANO pkls, and the 20,000-step schedule.

Checkpoints from the debug run: `runs/repro-main-k-debug/ckpt_last.pt` and `ckpt_step000010.pt` (1.5 GB each). They are not paper weights.

Gap table and the round-trip check remain `gaps_filled.md` and `roundtrip_check.txt`. Smoke followed reproduce stage 6. No paper-scale launch was started, so there is no same-step comparison against Table 1.
