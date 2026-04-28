# Document Specialist Review (Cycle 40) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28

## Previous Findings Status

DOC39-01 fixed. `_safe_float` docstring mentions negative value handling.

## New Findings

### DOC40-01: `pixel_pitch` docstring says "Returns 0.0 when mpix <= 0" but `derive_spec` doesn't document how it handles this sentinel

**File:** `pixelpitch.py`, lines 178-187 (pixel_pitch docstring), lines 722-769 (derive_spec docstring)
**Severity:** LOW | **Confidence:** HIGH

`pixel_pitch` documents that it "Returns 0.0 when mpix <= 0, area <= 0, or either argument is NaN/inf". However, `derive_spec`'s docstring does not mention that computed pitch from `pixel_pitch()` can be 0.0 (a sentinel value), or how this value is handled downstream. The docstring says "pixel pitch is computed as `1000 * sqrt(area / (mpix * 10**6))`" but does not mention that this computation can fail and produce 0.0.

After CR40-01 is fixed (derive_spec converts 0.0 to None), the docstring should be updated to note that computed pitch is None when `pixel_pitch` returns 0.0.

**Fix:** Update `derive_spec` docstring to mention: "When pixel_pitch returns 0.0 (invalid input sentinel), the computed pitch is set to None."

---

## Summary

- DOC40-01 (LOW): `derive_spec` docstring doesn't document 0.0 sentinel handling from `pixel_pitch`
