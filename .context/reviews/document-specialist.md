# Document Specialist Review (Cycle 37) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-36 fixes

## Previous Findings Status

DOC36-01 (`pixel_pitch` docstring should mention NaN/inf handling) — fixed. The docstring now says: "Returns 0.0 when mpix <= 0, area <= 0, or either argument is NaN / inf".

## New Findings

### DOC37-01: `derive_spec` docstring does not mention NaN/inf handling for size dimensions

**File:** `pixelpitch.py`, lines 706-724
**Severity:** LOW | **Confidence:** HIGH

The `derive_spec` docstring describes the computation logic:
> Area: computed as width * height when both are known.

After the C37 fix (adding `isfinite` validation in `derive_spec`), the docstring should mention that NaN/inf size dimensions are treated as unknown (size=None, area=None).

**Fix:** Update docstring after the `derive_spec` fix is implemented.

---

## Summary

- DOC37-01 (LOW): `derive_spec` docstring should mention NaN/inf handling for size dimensions
