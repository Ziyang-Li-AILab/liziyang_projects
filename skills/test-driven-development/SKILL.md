---
name: test-driven-development
description: Use when implementing any feature or bugfix, before writing implementation code
---

# Test-Driven Development (TDD)

## Overview

Write the test first. Watch it fail. Write minimal code to pass.

**Core principle:** If you didn't watch the test fail, you don't know if it tests the right thing.

**Violating the letter of the rules is violating the spirit of the rules.**

## When to Use

**Always:**
- New features
- Bug fixes
- Refactoring
- Behavior changes

**Exceptions (ask your human partner):**
- Throwaway prototypes
- Generated code
- Configuration files
- Filling gaps in a cloned official paper repo until a reproduction number lands

Thinking "skip TDD just this once"? Stop. That's rationalization — unless one of the exceptions above applies.

## The Iron Law

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

Write code before the test? Delete it. Start over.

## Red-Green-Refactor

1. **RED** — one failing test for one behavior. Clear name. Real code, not mocks unless unavoidable.
2. **Verify RED** — run the test. It must fail because the feature is missing, not because of a typo.
3. **GREEN** — smallest code that passes. No extra options, no "while I'm here".
4. **Verify GREEN** — that test and the project's full suite (`pytest`, `npm test`, `cargo test`, …).
5. **REFACTOR** — names, duplication, helpers. Stay green. No new behavior.

## Good Tests

| Quality | Good | Bad |
| --- | --- | --- |
| **Minimal** | One thing. "and" in the name? Split it. | `test('validates email and domain')` |
| **Clear** | Name describes behavior | `test('test1')` |
| **Shows intent** | Demonstrates desired API | Asserts mock call counts |

Read [writing-good-tests.md](writing-good-tests.md) when writing or changing tests.

## Rationalizations that are not allowed

- "Too simple to test" / "I'll test after" / "already manually tested"
- "Keep as reference, write tests first" — that's testing after. Delete the code.
- "Need to explore first" — explore, then throw it away and start with TDD.
- "Existing code has no tests" — add tests for the behavior you are changing.

## Bug fix pattern

Write a failing test that reproduces the bug. Watch it fail. Fix the code. Watch it pass. That test is the regression lock.

## Final rule

```
Production code → test exists and failed first
Otherwise → not TDD
```

No exceptions without the user's permission (or the paper-repo exception above).
