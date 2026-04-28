# Test Engineer Review (Cycle 25) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-24 fixes

## Previous Findings Status

C24-01 (TYPE_FRACTIONAL_RE space+inch test) and C24-02 (bare 1-inch test) implemented and passing.

## New Findings

### TE25-01: No test for SIZE_RE / PITCH_RE inconsistency with shared patterns

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

There is no test verifying that `parse_sensor_field` handles Unicode × or Greek mu μ. The existing test `test_parse_sensor_field` only tests with ASCII `x` and micro sign `µ`. If `SIZE_RE` is upgraded to match `SIZE_MM_RE` behavior, tests should verify:

```python
# SIZE_RE handles × and spaces
result1 = pp.parse_sensor_field('CMOS 36.0×24.0mm')
expect("SIZE_RE handles ×", result1["size"], (36.0, 24.0), tol=0.01)

result2 = pp.parse_sensor_field('CMOS 36.0 x 24.0 mm')
expect("SIZE_RE handles spaces", result2["size"], (36.0, 24.0), tol=0.01)

# PITCH_RE handles Greek mu
result3 = pp.parse_sensor_field('CMOS 5.12μm')
expect("PITCH_RE handles Greek mu", result3["pitch"], 5.12, tol=0.01)
```

These tests would FAIL with current code, correctly documenting the gap.

### TE25-02: No test for ValueError guard in parse_sensor_field

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** MEDIUM

No test exists for the case where `float()` is called on a regex match that produces an unparseable string. If a ValueError guard is added, a test should verify it:

```python
# Malformed dimension string — should not crash
result = pp.parse_sensor_field('CMOS 36.0.1x24.0mm')
expect("malformed size returns None for size", result["size"], None)
```

This test would CRASH with current code (ValueError), correctly documenting the bug.

---

## Summary

- TE25-01 (MEDIUM): No test for SIZE_RE/PITCH_RE Unicode/space handling gap
- TE25-02 (MEDIUM): No test for ValueError guard in parse_sensor_field
