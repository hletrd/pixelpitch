# Test Engineer Review (Cycle 37) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-36 fixes

## Previous Findings Status

TE36-01 through TE36-04 all implemented and passing. NaN/inf tests confirmed working.

## New Findings

### TE37-01: No test for `derive_spec` area being `nan` when size has NaN dimension

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** HIGH

The existing `test_derive_spec_negative_size` test checks that `size=(nan, 24.0)` produces `pitch=0.0`, but does not check what `area` becomes. When one dimension is NaN:
```python
spec = Spec(name='NaN Size', category='fixed', type=None,
             size=(float('nan'), 24.0), pitch=None, mpix=10.0, year=2020)
d = pp.derive_spec(spec)
# d.area = nan (should ideally be None)
# d.pitch = 0.0 (correctly guarded)
```

After the fix for CR37-02, `area` should be `None` for NaN dimensions. A test should verify this.

**Fix:** Add area assertion to `test_derive_spec_negative_size`:
```python
expect("derive_spec NaN size: area is None", d_nan.area, None)
```

---

### TE37-02: No test for CSV write-read round-trip of NaN/inf `area` field

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

If `derived.area` is `nan` (possible from code-constructed Spec with NaN size), `write_csv` writes `f"{nan:.2f}"` = `"nan"`, and `_safe_float("nan")` reads it back as `None`. This asymmetry should have a test.

**Fix:** Add test after the CR37-02 fix to verify that `area=nan` round-trips as `area=None`.

---

## Summary

- TE37-01 (LOW): No test for `derive_spec` area being `nan` when size has NaN dimension
- TE37-02 (LOW): No test for CSV write-read round-trip of NaN area field
