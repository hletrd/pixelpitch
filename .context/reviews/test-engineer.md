# Test Engineer Review (Cycle 28) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-27 fixes

## Previous Findings Status

TE27-01 (PITCH_UM_RE "um" test) addressed in C27. TE27-02 (year validation tests) addressed in C27. All existing tests passing.

## New Findings

### TE28-01: No test for imaging_resource.py pitch ValueError guard

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

The `test_imaging_resource()` function tests `IR_PITCH_RE` matching but does not test what happens when the matched value is a malformed float (e.g., "5.1.2 microns"). If the ValueError guard is added (as suggested by CR28-01), a test should verify:

```python
# fetch_one handles malformed pitch value gracefully
fields = {"Approximate Pixel Pitch": "5.1.2 microns", ...}
spec = imaging_resource.fetch_one(...)
expect("malformed pitch returns None", spec.pitch, None)  # or spec is None
```

Currently, no test exercises this code path. The test for `parse_sensor_field` tests "5.1.2µm" but that's the shared pattern, not the IR-specific code.

### TE28-02: No test for CineD year range validation

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

The CineD source module is browser-dependent and not tested offline. However, if year range validation is added to `cined._parse_camera_page()` (as suggested by V28-03), a unit test for the year validation logic would be valuable. Currently, there are no offline tests for the CineD source at all — only `test_cined_format_coverage()` which tests the FORMAT_TO_MM lookup table.

---

## Summary

- TE28-01 (MEDIUM): No test for imaging_resource.py pitch ValueError guard
- TE28-02 (LOW): No test for CineD year range validation
