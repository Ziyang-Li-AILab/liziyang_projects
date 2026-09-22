---
name: coding-standards
description: Enforces clean, consistent, maintainable code when writing, editing, implementing, refactoring, reviewing, or generating any code (Python, PyTorch, JavaScript, TypeScript, C++, CUDA). Use whenever the user asks to write code, implement a feature, fix a bug, add a file, reproduce a paper in code, or when the agent is about to create messy, over-engineered, or inconsistent code.
---

# Coding Standards

Apply this skill on **every** code change — paper reproduction, product features, scripts, and refactors. Messy agent code is a process failure, not a style preference.

Read [python-ml.md](python-ml.md) when the language is Python / PyTorch.

After substantial edits, load `code-review` (or `requesting-code-review`). Before claiming done, load `verification-before-completion`. For new behavior (not cloned paper code), prefer `tdd` / `test-driven-development`. When shaping modules, use `codebase-design`.

## Match the repo first

1. Read neighboring files. Copy **their** naming, import style, formatter, and layout.
2. If `ruff` / `black` / `prettier` / `eslint` / `clang-format` exist, obey them. Do not invent a second style.
3. Change only what the task needs. No drive-by cleanup, no "while I'm here" rewrites, no unrelated renaming.

## Structure

- One job per module. New file only when an existing file would mix two reasons to change.
- Public surface stays small. Hide helpers. Prefer a deep module (lots of behavior, few parameters) over a bag of pass-through functions.
- Functions default to < 40 lines; files default to < 400 lines. Split when either is exceeded unless the project already uses long generated/model files.
- No god objects, no `utils.py` dumping ground, no `temp_final_v3`.
- Keep training / data / model / eval in separate modules when building ML code from scratch.

## Naming

- Names describe role, not type soup: `pred_joints`, not `arr2` / `data_new` / `tmp`.
- Booleans are predicates: `is_training`, `has_gt`.
- Paper symbols may appear in comments or a short alias (`attn  # A in Eq. 3`); runtime names stay readable English.
- Files: `snake_case.py`, `PascalCase` types in Python, match existing JS/TS convention.

## Comments and dead weight

- Comment **why** and **invariants**, never narrate the next line.
- Delete unused imports, dead branches, commented-out code, and unused flags.
- Do not leave `TODO` unless it names the missing paper detail and a default.

## Correctness over cleverness

- No speculative abstraction ("we might need a plugin system"). YAGNI.
- Don't swallow exceptions. Fail with the actual error; log once at the boundary.
- Don't copy-paste with a one-token diff — extract or parameterize.
- Magic numbers belong in named constants or a config. In reproductions, every hyperparameter cites a paper section or `[unspecified — assumed X]`.
- Mutable default arguments, global side effects, and hidden `chdir` are forbidden.

## Diff discipline

- Prefer editing an existing function to adding a parallel `_new` copy.
- Do not reformat whole files. Do not add type-hint/docstring campaigns onto cloned third-party code.
- Generated or vendored files: skip style rewrites.

## Before you stop

- [ ] Diff is scoped to the request
- [ ] Names and layout match neighbors
- [ ] No unused code or debug prints
- [ ] Ran the project's check (tests, lint, or a smoke import). If none exist, run the smallest command that would catch a syntax/import error
- [ ] Did not claim success without that output (`verification-before-completion`)

## Exceptions

- **Cloned official paper repos:** fill gaps; do not restyle or "improve" their code until a number lands (`reproduce` stage 4).
- **Throwaway probes:** a single script is fine; still no unused junk, still delete it or mark it clearly.
