# Verifier Review (Cycle 16) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository verification after cycles 1-15 fixes

## Previously Fixed (Cycles 1-15) — Confirmed Resolved
All previous fixes verified as correctly applied and tested.

## Verification of C15 Fixes

### C15-01: Canon EOS xxxD DSLR regex — VERIFIED
Test `test_openmvg_csv_parser` includes `expect("EOS 250D category", ..., "dslr")` which passes.

### C15-02: Samsung NX mirrorless — VERIFIED
Test includes `expect("Samsung NX300 category", ..., "mirrorless")` which passes.

### C15-03: Rangefinder dedup — VERIFIED
`test_category_dedup` verifies the dedup logic. Passing.

### C15-04: openMVG BOM defense — VERIFIED
`test_openmvg_bom` verifies BOM handling. Passing.

### C15-05: Sigma SD regex — VERIFIED
Test includes `expect("Sigma SD14 category", ..., "dslr")`. Passing.

### C15-06: Docstring update — VERIFIED
OpenMVG docstring now warns about heuristic limitations.

## New Findings

### V16-01: `sensor_size_from_type` crash on invalid input — REPRODUCED
**File:** `pixelpitch.py`, lines 152-165
**Severity:** MEDIUM | **Confidence:** HIGH

Reproduced with test:
```python
spec = Spec(name='Test', category='fixed', type='1/0', size=None, pitch=None, mpix=10.0, year=2020)
derive_spec(spec)  # ZeroDivisionError
```
Also crashes with `type='1/'` (ValueError) and `type='1/0.0'` (ZeroDivisionError).
A single bad spec in `derive_specs()` crashes the entire batch.

**Fix:** Add try/except in `sensor_size_from_type` returning None on error.

---

### V16-02: `merge_camera_data` duplicate entries confirmed — REPRODUCED
**File:** `pixelpitch.py`, lines 349-407
**Severity:** MEDIUM | **Confidence:** HIGH

Reproduced with test:
```python
new = [derive('Canon EOS 250D', 'dslr', (22.3, 14.9), 24.1, 2019),
       derive('Canon EOS 250D', 'dslr', (22.3, 14.9), 24.1, 2019)]
merged = merge_camera_data(new, [])
# Result: 2 entries for Canon EOS 250D instead of 1
```

**Fix:** Dedup among new_specs before/within the merge loop.

---

### V16-03: Pentax DSLR regex misses 10+ models — VERIFIED
**File:** `sources/openmvg.py`, line 47
**Severity:** LOW | **Confidence:** HIGH

Verified with regex testing: Pentax K3, K5, K7, K1, KP, KF, K-r, K-x, K-30, K-50, K-70, K30, K50, K70, K100D, K200D, K10D, K20D all fail to match. Only Pentax K-1, K-3, K-5, K-7 (with hyphen) match.

**Fix:** Broaden regex pattern.

---

## Summary
- NEW findings: 3 (2 MEDIUM, 1 LOW)
- V16-01: sensor_size_from_type crash — MEDIUM (reproduced)
- V16-02: merge_camera_data duplicate — MEDIUM (reproduced)
- V16-03: Pentax regex misses models — LOW (verified)
