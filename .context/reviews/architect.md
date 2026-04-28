# Architect Review (Cycle 41) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28

## Previous Findings Status

ARCH40-01 fixed. `derive_spec` converts computed 0.0 sentinel to None.

## New Findings

### ARCH41-01: `derive_spec` has inconsistent validation between direct and computed pitch paths — contract gap

**File:** `pixelpitch.py`, lines 759-769
**Severity:** MEDIUM | **Confidence:** HIGH

`derive_spec` now validates the *computed* pitch path (converting `pixel_pitch()`'s 0.0 sentinel to None), but the *direct* pitch path (`spec.pitch is not None`) performs no validation at all. This creates an inconsistency in the output contract:

- Computed path: `derived.pitch` is guaranteed to be None or a positive finite value
- Direct path: `derived.pitch` can be 0.0, negative, or NaN

This contract gap means downstream consumers cannot rely on a uniform contract for `derived.pitch`. The fix should be a single validation step applied to both paths, not just one.

**Fix:** Add a unified `_validate_positive_float` helper used by both paths:

```python
def _validate_positive_float(val: Optional[float]) -> Optional[float]:
    """Return val if it is a positive finite number, else None."""
    if val is not None and isfinite(val) and val > 0:
        return val
    return None
```

Then in `derive_spec`:
```python
pitch = _validate_positive_float(spec.pitch) if spec.pitch is not None else None
if pitch is None and spec.mpix is not None and area is not None:
    pitch = _validate_positive_float(pixel_pitch(area, spec.mpix))
```

This establishes a uniform contract: `derived.pitch` is always None or a positive finite value, regardless of input path.

---

## Summary

- ARCH41-01 (MEDIUM): `derive_spec` has inconsistent validation between direct and computed pitch paths — contract gap
