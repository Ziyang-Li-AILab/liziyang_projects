# {Paper Title}

## 0. One-Sentence Summary

{What it does + key trick + outcome, in 1 sentence.}

---

## 1. Problem

- **Task:** ... We want to... *(Source: ...)*
- **Setting:** ... *(Source: ...)*
- **Difficulty / bottleneck:** ... *(Source: ...)*
- **Objective:** ... (keep LaTeX if any) *(Source: ...)*
- **Impl. environment:** ... *(Source: ...)*

---

## 2. Why Prior Work Fails

- **Main limitation:** ... *(Source: ...)*
- **Root cause:** ... *(Source: ...)*
- **What this paper changes:** ... *(Source: ...)*

---

## 3. Core Mechanism

### High-Level Idea

{Intuitive explanation — make the reader think "ah, I get it" before any equations. Keep it short without sacrificing intuition.} *(Source: ...)*

### Mechanism Details

Include ONLY the relevant blocks from Phase 1 classification:

**Learning-oriented blocks:**
- **Model / Representation:** ... *(Source: ...)*
- **Objective / Loss:** ... (key equations allowed; use $...$) *(Source: ...)*
- **Optimization / Training:** ... *(Source: ...)*
- **Inference / Simulation:** ... *(Source: ...)*
- **Data:** ... *(Source: ...)*

**Physics-oriented blocks:**
- **State representation:** ... *(Source: ...)*
- **Governing equations / Energy:** ... *(Source: ...)*
- **Constraints:** ... *(Source: ...)*
- **Solver type:** ... *(Source: ...)*
- **Time integration:** ... *(Source: ...)*

**Hybrid:** include relevant blocks from both groups, plus:
- **Coupling:** ... (how physics and learning modules interact) *(Source: ...)*

**General (neither Learning nor Physics):** Do NOT force-fit the blocks above. Instead, create **4–6 domain-appropriate blocks** that mirror the same format (`- **Label:** concise explanation *(Source: ...)*`).

---

## 4. How It Works

(5 steps max, less is preferred)
> 1. ... *(Source: ...)*
> 2. ... *(Source: ...)*
> 3. ... *(Source: ...)*
> 4. ... *(Source: ...)*
> 5. ... *(Source: ...)*

---

## 5. Results

- **Best metric:** ... *(Source: ...)*
- **Improvement over baseline:** ... *(Source: ...)*
- **Benchmark / dataset:** ... *(Source: ...)*
- **Main qualitative effect:** ... *(Source: ...)*

---

## 6. Works / Fails

- **Works well when:** ... *(Source: ...)*
- **Weak when:** ... *(Source: ...)*

---

## 7. Pseudocode or Algorithm

Choose whichever communicates the method more clearly. Pick one. Do not include both.

### Option A: Python pseudocode

```python
def core_algorithm(inputs, params):
    """
    Inputs: ...
    Outputs: ...
    """
    state = initialize(inputs, params)
    for t in range(params.max_iter):
        state = update(state, inputs, params)
    return finalize(state)
```

### Option B: Algorithm block

> **Input:** ...
> **Output:** ...
>
> 1. ...
> 2. **for** ... **do**
>    1. ...
> 3. **return** ...
