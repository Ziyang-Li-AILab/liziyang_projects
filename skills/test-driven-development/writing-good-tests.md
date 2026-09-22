# Writing Good Tests

**Load this reference when:** writing or changing tests, adding mocks, or adding cleanup/helper methods for tests.

## Overview

A test exists to catch a specific break. Two principles govern everything here:

```
1. Every test names the break it catches
2. Every test exercises the real thing
```

## Principle 1: Name the Break

Before writing the test body, answer: **what production change should make this test fail — and is that change a bug or a decision?**

**Derive expectations independently.** Use literals and hand-checked fixtures. An expectation computed by the code under test always passes:

```typescript
// BAD — mirror assertion
const expected = buildSearchQuery({ tag: 'urgent' });
expect(buildSearchQuery({ tag: 'urgent' })).toBe(expected);

// GOOD — hand-derived literal
expect(buildSearchQuery({ tag: 'urgent' })).toBe('tag:"urgent"');
```

**No change detectors.** Don't assert `MAX_RETRIES === 5`. Assert "a failing call is retried 5 times and the 6th never happens."

**Behavior, not text.** Don't grep source for a string. Run the artifact and assert outputs, side effects, or exit codes.

## Principle 2: Exercise the Real Thing

**The mock earns no assertions.** If you are checking the mock, unmock it or delete the assertion. Assert the real component.

**Mock at the right level.** Mock the slow/external operation; keep the side effects the test depends on real.

**Mirror real data completely.** Partial mocks fail silently when downstream reads an omitted field.

**Production classes carry production methods only.** Test-only `destroy()` belongs in test utilities.

## The Mutation Check

Before finishing, mentally mutate the production code; at least one test should fail for each realistic mutation:

- Wrong constant or argument
- Wrong branch
- Missing state change
- Empty / default return
- Missing validation for empty, nil, or malformed input

## Warning signs

- Setup and assertion share the same object
- Expected values come from the code under test
- Assertions on `*-mock` test IDs
- Mock setup is more than half the test
- Test would still pass if only the framework remained
