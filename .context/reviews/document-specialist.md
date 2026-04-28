# Document Specialist Review (Cycle 41) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28

## Previous Findings Status

DOC40-01 fixed. `derive_spec` docstring documents 0.0 sentinel handling for computed path.

## New Findings

### DOC41-01: `derive_spec` docstring documents computed-path sentinel but not direct-path validation

**File:** `pixelpitch.py`, lines 722-745 (derive_spec docstring)
**Severity:** LOW | **Confidence:** HIGH

The `derive_spec` docstring now says: "When pixel_pitch returns 0.0 (sentinel for invalid inputs such as non-positive mpix/area or NaN/inf), the computed pitch is set to None instead of propagating the sentinel." However, it does NOT document that direct `spec.pitch` values (0.0, negative, NaN) are currently propagated without validation. After CR41-01 is fixed (adding validation to the direct path), the docstring should be updated to document that both paths produce the same contract: `derived.pitch` is either None or a positive finite value.

**Fix:** Update docstring to document that direct `spec.pitch` values that are non-finite or non-positive are also converted to None, establishing a uniform output contract.

---

## Summary

- DOC41-01 (LOW): `derive_spec` docstring doesn't document direct-path validation (or lack thereof)
