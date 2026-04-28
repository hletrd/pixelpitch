# Test Engineer Review (Cycle 34) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-33 fixes, focusing on NEW issues

## Previous Findings Status

TE33-01 (derive_spec 0.0 test) implemented in C33. TE33-02 (sorted_by 0.0 test) implemented in C33. TE33-03 (template 0.0 rendering test) implemented in C33.

## New Findings

### TE34-01: No test for match_sensors with megapixels=0.0 (ZeroDivisionError)

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

There is no test verifying that `match_sensors` handles megapixels=0.0 without crashing. The existing `test_match_sensors` tests with None mpix and positive mpix, but not 0.0. The CR34-03/DBG34-01 bug (ZeroDivisionError) would not be caught by CI.

**Fix:** Add a test case:
```python
matches_zero = pp.match_sensors(36.0, 24.0, 0.0, sensors_db)
expect("match with zero mpix: no crash", True, True)
expect("match with zero mpix: empty or size-only match",
       isinstance(matches_zero, list), True)
```

---

### TE34-02: No test for match_sensors with width=0.0 or height=0.0

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

There is no test verifying behavior of `match_sensors` with width=0.0 or height=0.0. Currently the truthy guard `not width or not height` returns [] for 0.0, which is the same result as for None but for the wrong reason (truthy vs None conflation).

**Fix:** Add a test case verifying match_sensors with 0.0 dimensions.

---

## Summary

- TE34-01 (MEDIUM): No test for match_sensors with megapixels=0.0 (ZeroDivisionError)
- TE34-02 (LOW): No test for match_sensors with width=0.0 or height=0.0
