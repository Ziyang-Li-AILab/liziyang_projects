# Closed loop (paper2code, minimal)

The runnable loop is `src/closed_loop.py`. It trains the released GeoDiT for a few steps (Eq. 6, Appendix B.2 learning rates), writes an inference checkpoint, loads that checkpoint into a new module, and runs one eval forward. That is the training/inference loop. It is not the paper's 20,000-step Table 1 run.

## MANO — which file

[mano.is.tue.mpg.de/download.php](https://mano.is.tue.mpg.de/download.php) redirects to a login. The files cannot be fetched without an account, and the license forbids redistributing them. Register, accept the research license, and download only:

**Models & Code → `mano_v1_2.zip`**

Inside it, the only files this project reads are:

- `mano_v1_2/models/MANO_LEFT.pkl`
- `mano_v1_2/models/MANO_RIGHT.pkl`

Skip example scans, sample registrations, and any other dataset linked from that site (FreiHAND, HO-3D, and the rest). Those are not the hand model.

Convert once (strips the obsolete `chumpy` dependency):

```bash
cd repro/2608.20308/code
python scripts/convert_mano_pkls.py --src /path/to/mano_v1_2/models --dst data/mano/mano
```

`smplx` looks for `{ACE_EGO_HAND_MANO_DIR}/mano/MANO_RIGHT.pkl`. The default root is `code/data/mano`. Then:

```bash
cd repro/2608.20308/src
python closed_loop.py --steps 2 --mano-dir ../code/data/mano
```

## What runs when those pkls are absent

`--allow-fake-mano` uses the kinematic stand-in in `src/ace_repro/hand_model.py`. Losses and mixed-PnP still execute. Numbers are not comparable to the paper. This is an explicit substitute, not a claim that the paper trained without MANO.

## Unspecified choices used by the loop

Same as `gaps_filled.md`. The ones that affect this script: AdamW betas `(0.9, 0.999)` (PyTorch default; paper silent), image losses as L1 (paper does not name the norm), synthetic clips instead of the Table 2 mixture.
