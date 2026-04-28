# Test Engineer Review (Cycle 40)

**Reviewer:** test-engineer
**Date:** 2026-04-28

## Previous Findings Status

TE39-01 implemented. Tests for negative/NaN pitch/mpix template rendering pass. `test_parse_existing_csv_negative_values` passes.

## New Findings

### TE40-01: No test for `derive_spec` producing pitch=0.0 from computed path (mpix=0.0)

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** HIGH

There is no test verifying that `derive_spec` with `spec.pitch=None` and `spec.mpix=0.0` produces `derived.pitch=None` (or correctly handles the 0.0 sentinel from `pixel_pitch()`). The existing `test_derive_spec_zero_pitch` only tests `spec.pitch=0.0` (direct), not the computed path.

After CR40-01 is fixed (convert 0.0 to None in derive_spec), a test should verify:
- `derive_spec` with `pitch=None, mpix=0.0` → `derived.pitch is None`
- `derive_spec` with `pitch=None, mpix=-1.0` → `derived.pitch is None`

### TE40-02: No test for `write_csv` non-finite float output

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

No test verifies that `write_csv` does not write inf/nan strings to the CSV file. After CR40-02 is fixed, a test should verify that non-finite values produce empty strings in the CSV.

---

## Summary

- TE40-01 (LOW): No test for `derive_spec` computed pitch=0.0 path (mpix=0.0)
- TE40-02 (LOW): No test for `write_csv` non-finite float output
