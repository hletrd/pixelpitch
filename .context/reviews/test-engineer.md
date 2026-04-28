# Test Engineer Review (Cycle 41)

**Reviewer:** test-engineer
**Date:** 2026-04-28

## Previous Findings Status

TE40-01 implemented. Tests for derive_spec computed 0.0 pitch pass. TE40-02 implemented. Tests for write_csv non-finite guards pass.

## New Findings

### TE41-01: No test for `derive_spec` with direct `spec.pitch=0.0` producing wrong pitch value

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** HIGH

The existing `test_derive_spec_zero_pitch` tests that `spec.pitch=0.0` is *preserved* as `derived.pitch=0.0`. This test was written before the understanding that 0.0 pitch is physically meaningless and should be treated like None. After CR41-01 is fixed, this test needs to be updated to expect `derived.pitch=None` instead of `derived.pitch=0.0`.

The test at line 1345-1346:
```python
expect("derive_spec: spec.pitch=0.0 preserved over computed",
       d_zero.pitch, 0.0, tol=0.01)
```

After the fix, this should assert `d_zero.pitch is None`.

**Fix:** Update `test_derive_spec_zero_pitch` to expect None for spec.pitch=0.0 direct. Add tests for spec.pitch=-1.0 and spec.pitch=nan direct.

---

### TE41-02: No test for `write_csv` with 0.0 or negative mpix/pitch

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

The existing `test_write_csv_nonfinite_guards` tests inf and nan but not 0.0 or negative values. After CR41-02 is fixed (positivity checks in write_csv), tests should verify that:
- `write_csv` with `spec.mpix=0.0` produces empty mpix cell (not "0.0")
- `write_csv` with `spec.mpix=-5.0` produces empty mpix cell (not "-5.0")
- `write_csv` with `derived.pitch=0.0` produces empty pitch cell (not "0.00")
- `write_csv` with `derived.pitch=-1.0` produces empty pitch cell (not "-1.00")

**Fix:** Add test cases for 0.0 and negative values in write_csv.

---

## Summary

- TE41-01 (LOW): No test for derive_spec direct spec.pitch=0.0 → None (test currently expects 0.0)
- TE41-02 (LOW): No test for write_csv with 0.0/negative mpix/pitch
