# Verifier Review (Cycle 15) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository verification after cycles 1-14 fixes

## Previously Fixed (Cycles 1-14) — Verification Status
All previous fixes verified still working. Gate tests pass cleanly. C14-01 through C14-05 all verified.

## New Findings

### V15-01: Canon EOS xxxD DSLRs misclassified as mirrorless by openMVG regex — verified by regex test
**File:** `sources/openmvg.py`, lines 36-48
**Severity:** MEDIUM | **Confidence:** HIGH

Verified by regex test:
```python
_DSLR_NAME_RE.search("Canon EOS 250D")   # None (WRONG - is DSLR)
_DSLR_NAME_RE.search("Canon EOS 800D")   # None (WRONG - is DSLR)
_DSLR_NAME_RE.search("Canon EOS 850D")   # None (WRONG - is DSLR)
_DSLR_NAME_RE.search("Canon EOS 1200D")  # None (WRONG - is DSLR)
_DSLR_NAME_RE.search("Canon EOS 2000D")  # None (WRONG - is DSLR)
_DSLR_NAME_RE.search("Canon EOS 4000D")  # None (WRONG - is DSLR)
_DSLR_NAME_RE.search("Canon EOS 90D")    # None (WRONG - is DSLR)
_DSLR_NAME_RE.search("Canon EOS 70D")    # None (WRONG - is DSLR)
_DSLR_NAME_RE.search("Canon EOS 80D")    # None (WRONG - is DSLR)
```
All return None, but these are all DSLRs. The pattern `\dD` only matches a single digit before D.

**Fix:** Change `Canon\s+EOS[-\s]+\dD` to `Canon\s+EOS[-\s]+\d+D`.

---

### V15-02: Samsung NX cameras misclassified as DSLR by openMVG regex — verified by regex test
**File:** `sources/openmvg.py`, line 44
**Severity:** MEDIUM | **Confidence:** HIGH

Verified by regex test:
```python
_DSLR_NAME_RE.search("Samsung NX100")   # Match (WRONG - is mirrorless)
_DSLR_NAME_RE.search("Samsung NX200")   # Match (WRONG - is mirrorless)
_DSLR_NAME_RE.search("Samsung NX300")   # Match (WRONG - is mirrorless)
_DSLR_NAME_RE.search("Samsung NX500")   # Match (WRONG - is mirrorless)
```
All match the DSLR regex, but all Samsung NX cameras are mirrorless.

**Fix:** Remove `Samsung\s+NX\d{3}` from the DSLR regex.

---

### V15-03: 43 cameras have triple-category duplicates in dist data — verified by data analysis
**File:** `pixelpitch.py`, line 339 (`create_camera_key`); dist data
**Severity:** MEDIUM | **Confidence:** HIGH

Verified by analyzing the dist/camera-data.csv:
- 43 cameras appear under 3 categories (dslr + mirrorless + rangefinder)
- All 43 are misclassified as "rangefinder" by Geizhals
- 10 cameras are actual rangefinders (Leica M series)
- None of the 43 misclassified cameras are actual rangefinders

The triple-duplicate is caused by `create_camera_key(name, category)` producing 3 different keys for the same camera.

**Fix:** Normalize Geizhals rangefinder data against dslr/mirrorless data before merge.

---

### V15-04: Sigma SD2-digit models missed by openMVG regex — verified by regex test
**File:** `sources/openmvg.py`, line 43
**Severity:** LOW | **Confidence:** HIGH

Verified:
```python
_DSLR_NAME_RE.search("Sigma SD10")   # None (WRONG - is DSLR)
_DSLR_NAME_RE.search("Sigma SD14")   # None (WRONG - is DSLR)
_DSLR_NAME_RE.search("Sigma SD15")   # None (WRONG - is DSLR)
_DSLR_NAME_RE.search("Sigma SD1")    # Match (correct)
_DSLR_NAME_RE.search("Sigma SD9")    # Match (correct)
```

**Fix:** Change `Sigma\s+SD\d?` to `Sigma\s+SD\d+`.

---

## Summary
- NEW findings: 4 (3 MEDIUM, 1 LOW)
- V15-01: Canon EOS xxxD regex false negatives — MEDIUM
- V15-02: Samsung NX regex false positives — MEDIUM
- V15-03: 43 triple-category duplicates in data — MEDIUM
- V15-04: Sigma SD regex misses 2-digit models — LOW
