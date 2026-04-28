# Test Engineer Review (Cycle 27) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-26 fixes

## Previous Findings Status

TE26-01 (MPIX_RE "MP"/"Mega pixels" tests) and TE26-02 (ValueError guard tests in source modules) — both addressed by the C26 implementation which added `test_mpix_re_format_handling()` and ValueError guards with existing fixture tests still passing.

## New Findings

### TE27-01: No test for PITCH_UM_RE "um" (lowercase ASCII) variant

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** HIGH

The `test_parse_sensor_field()` function tests PITCH_UM_RE with `µm`, `μm`, and `microns` but does NOT test the lowercase ASCII `um` variant. If `um` is added to the shared pattern (as suggested by CR27-01), a test should be added to verify:

```python
# PITCH_UM_RE handles lowercase ASCII "um"
result = pp.parse_sensor_field('CMOS 5.12um')
expect("PITCH handles um", result["pitch"], 5.12, tol=0.01)
```

Currently, `um` is NOT matched by the shared pattern, so this test would fail. Adding the test documents the expected behavior when the fix is applied.

---

### TE27-02: No test for parse_existing_csv year validation edge cases

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

The `test_parse_existing_csv()` function does not test edge cases for the year column:
- year=0 (should be rejected or treated as None)
- year=-1 (should be rejected or treated as None)
- year=99999 (should be validated against a reasonable range)

Currently, these values are accepted verbatim and would display on the website. If validation is added (as suggested by CR27-02), tests should verify the rejection behavior.

---

## Summary

- TE27-01 (LOW): No test for PITCH_UM_RE "um" variant
- TE27-02 (LOW): No test for parse_existing_csv year validation edge cases
