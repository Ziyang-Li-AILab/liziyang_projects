# Design It Twice

When exploring alternative interfaces for a module, use this pattern. Based on "Design It Twice" (Ousterhout): the first idea is unlikely to be the best.

Uses the vocabulary in [SKILL.md](SKILL.md): **module**, **interface**, **seam**, **adapter**, **leverage**.

## Process

### 1. Frame the problem space

Write the constraints any new interface must satisfy, the dependencies it would rely on (see [DEEPENING.md](DEEPENING.md)), and a rough sketch that makes the constraints concrete — not a proposal.

Show this to the user, then immediately proceed to Step 2.

### 2. Produce 3 radically different interfaces

Give each a different constraint:

1. Minimize the interface: 1–3 entry points. Maximise leverage per entry point.
2. Maximise flexibility: many use cases and extension.
3. Optimise for the most common caller: make the default case trivial.

Each design includes: interface (types, methods, invariants, errors), a usage example, what the implementation hides, adapter strategy, and trade-offs.

### 3. Present and compare

Compare on **depth**, **locality**, and **seam placement**. Recommend one design (or a hybrid). Be opinionated.
