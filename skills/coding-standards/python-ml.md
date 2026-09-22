# Python / PyTorch conventions

Use with `coding-standards`. If the file you are editing disagrees, follow that file.

## Imports

```python
from __future__ import annotations  # only if the project already does

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
```

- stdlib → third party → local. One blank line between groups.
- No `from torch.nn import *`. No unused imports.

## Types and tensors

- Annotate public function signatures. Internal helpers may infer.
- Shapes in a one-line docstring or comment on the first use: `x: (B, T, C)`.
- `dtype` and `device` flow from inputs (`x.new_zeros(...)`, `torch.empty_like`). Do not hard-code `cuda:0` unless the project does.
- Prefer `einops` only if the repo already depends on it.

## Modules

```python
class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {num_heads}")
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        # ...

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C), ctx: (B, M, C)
        ...
```

- `forward` stays tensor in → tensor out. Side effects (logging, wandb) live in the training loop.
- Loss functions are modules or pure functions, not buried in `train.py` unless the file is already that style.
- Don't wrap every line in `.cuda()`. Move a batch once.

## Training scripts

- Config (yaml/argparse) is the source of truth. Construct optimizer/schedule from config, not scattered literals.
- Seed, device, and output dir are explicit.
- A `--debug` / dry-run path that runs a few steps is required for new training entrypoints.
- Checkpoints save `state_dict`, step, and config — not the whole module pickle unless the project already does.

## Tests

- Test behavior (shapes, invariants, a known numeric case), not mock call counts.
- One obvious fixture per test module. No 200-line setup.
- For paper code: a CPU forward-pass smoke test with tiny tensors beats an unrunnable "full training" test.

## Anti-patterns

| Don't | Do |
| --- | --- |
| `list.append` in a hot Python loop over batch | batched tensor ops |
| `for i in range(len(x))` | iterate values / `enumerate` / vectorize |
| `except: pass` | catch specific errors or let it fail |
| `print(x.shape)` left in | delete, or a debug logger behind a flag |
| `ModelV2CopyFinal` | one name, git history for the rest |
| silent `clip_grad` / `eval()` mismatches | set train/eval explicitly at loop boundaries |
