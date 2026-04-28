# Designer Review (Cycle 17) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28
**Scope:** Full repository UI/UX re-review after cycles 1-16 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-16) — Confirmed Resolved

- UX16-01 (duplicate entries): Fixed — merge dedup working.
- UX16-02 (Pentax misclassification): Partially fixed — K3 etc. now DSLR, but KP/KF/K-r/K-x still mirrorless.

## New Findings

### UX17-01: Pentax KP/KF/K-r/K-x still appear under wrong category
**File:** `sources/openmvg.py`, line 47
**Severity:** LOW | **Confidence:** HIGH

Same as C17-01. Four Pentax DSLR models (KP, KF, K-r, K-x) still appear on the Mirrorless page instead of DSLR. Users browsing the DSLR page won't find these cameras.

**Fix:** Fix the regex to match letter-suffix Pentax K-mount models.

---

### UX17-02: Nikon Df appears under wrong category
**File:** `sources/openmvg.py`, line 46
**Severity:** LOW | **Confidence:** HIGH

Nikon Df is a well-known retro DSLR that would appear on the Mirrorless page. Photography enthusiasts would notice this misclassification.

**Fix:** Add Nikon Df to the DSLR regex.

---

## Summary
- NEW findings: 2 (both LOW)
- UX17-01: Pentax KP/KF/K-r/K-x misclassification — LOW
- UX17-02: Nikon Df misclassification — LOW
