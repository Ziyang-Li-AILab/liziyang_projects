"""Per-batch single-source sampling with Table 2 weights (gaps_filled E/G).

Each step draws ONE source with probability proportional to its weight, then
``batch_size`` clips from that source. The source choice is seeded by the
global step so all DDP ranks pick the same dataset (RHD batches skip the MANO
heads, so parameter participation must match across ranks).
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ace_repro.data.schema import Clip, collate


class MixtureLoader:
    def __init__(self, sources: dict[str, tuple[Dataset, float]], batch_size: int, seed: int = 0):
        self.names = [n for n, (ds, w) in sources.items() if len(ds) > 0 and w > 0]
        self.datasets = {n: sources[n][0] for n in self.names}
        w = torch.tensor([sources[n][1] for n in self.names], dtype=torch.float64)
        self.probs = w / w.sum()
        self.batch_size = batch_size
        self.seed = seed
        self.step = 0

    def source_for_step(self, step: int) -> str:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + step)   # rank-independent
        return self.names[int(torch.multinomial(self.probs, 1, generator=g))]

    def next_batch(self, rank: int = 0, world_size: int = 1) -> dict:
        name = self.source_for_step(self.step)
        ds = self.datasets[name]
        g = torch.Generator().manual_seed(self.seed * 7_919 + self.step * world_size + rank)
        idx = torch.randint(0, len(ds), (self.batch_size,), generator=g).tolist()
        clips: list[Clip] = [ds[i] for i in idx]
        self.step += 1
        return collate(clips)

    def __iter__(self):
        while True:
            yield self.next_batch()
