# Verifier Review (Cycle 17) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository verification after cycles 1-16 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-16) — Verified

All 98 gate tests pass. C16-01 through C16-05 all verified as correctly applied and tested.

## Verification of C16 Fixes

### C16-01: sensor_size_from_type crash guard — VERIFIED
Tests for 1/0, 1/0.0, 1/, 1/-1 all pass, returning None instead of crashing.

### C16-02: merge_camera_data new_specs dedup — VERIFIED
Test `test_merge_camera_data` includes duplicate key tests. Both "dedup new_specs same key" and "dedup new_specs with existing" pass.

### C16-03: Pentax DSLR regex broadened — PARTIALLY VERIFIED
Pentax K3 and 645Z now correctly classified. However, KP, KF, K-r, K-x are still misclassified. See V17-01.

### C16-04: digicamdb removed from SOURCE_REGISTRY — VERIFIED
`"digicamdb" in SOURCE_REGISTRY` returns False.

### C16-05: OSError added to http_get — VERIFIED
`OSError` present in the except clause.

## New Findings

### V17-01: Pentax KP, KF, K-r, K-x still misclassified — C16-03 fix incomplete
**File:** `sources/openmvg.py`, line 47
**Severity:** MEDIUM | **Confidence:** HIGH

Reproduced with live regex test:
```python
from sources.openmvg import _DSLR_NAME_RE
_DSLR_NAME_RE.search("Pentax KP")     # None (should match)
_DSLR_NAME_RE.search("Pentax KF")     # None (should match)
_DSLR_NAME_RE.search("Pentax K-r")    # None (should match)
_DSLR_NAME_RE.search("Pentax K-x")    # None (should match)
```

The regex `Pentax\s+K[-\s]?\d+[A-Za-z]*` requires at least one digit after K[-\s]?. These four models have no digit — they go directly from K (or K-) to a letter.

**Fix:** Change to `Pentax\s+K[-\s]?[\dA-Za-z]+[A-Za-z]*`.

---

### V17-02: Nikon Df missed — not classified as DSLR
**File:** `sources/openmvg.py`, line 46
**Severity:** LOW | **Confidence:** HIGH

Reproduced:
```python
_DSLR_NAME_RE.search("Nikon Df")  # None (should match DSLR)
```

The Nikon Df is a known DSLR with no digit after D. The regex `Nikon\s+D\d{1,4}` requires at least one digit.

**Fix:** Add `|Nikon\s+Df` to the regex alternation.

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- V17-01: Pentax KP/KF/K-r/K-x still misclassified — MEDIUM (reproduced)
- V17-02: Nikon Df missed — LOW (reproduced)
