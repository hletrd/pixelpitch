# Document Specialist Review (Cycle 39) — Doc/Code Mismatches

**Reviewer:** document-specialist
**Date:** 2026-04-28

## Previous Findings Status

DOC37-01 fixed. `derive_spec` docstring mentions NaN/inf handling.

## New Findings

### DOC39-01: `_safe_float` docstring says "returning None for NaN/inf/empty" but allows negative values

**File:** `pixelpitch.py`, lines 268-269
**Severity:** LOW | **Confidence:** HIGH

```python
def _safe_float(s: str) -> Optional[float]:
    """Parse a float string, returning None for NaN/inf/empty."""
```

The docstring says "returning None for NaN/inf/empty" but the function also returns negative values without filtering them. For fields like pitch, mpix, area, and size dimensions, negative values are physically meaningless. The docstring should either document this behavior explicitly or the function should be updated to reject negative values for relevant fields.

**Fix:** Either:
1. Update docstring to mention that negative values are allowed through (documenting current behavior)
2. Add a `positive_only` parameter to `_safe_float` for use with physical-quantity fields

---

## Summary

- DOC39-01 (LOW): `_safe_float` docstring incomplete — does not mention negative value handling
