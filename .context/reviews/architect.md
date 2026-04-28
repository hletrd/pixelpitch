# Architect Review (Cycle 34) — Architectural/Design Risks

**Reviewer:** architect
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-33 fixes, focusing on NEW issues

## Previous Findings Status

ARCH33-01 (systemic truthy-vs-None) fixed in C33. All previous architectural concerns remain deferred.

## New Findings

### ARCH34-01: match_sensors percentage comparison lacks defensive boundary check

**File:** `pixelpitch.py`, lines 230-238
**Severity:** MEDIUM | **Confidence:** HIGH

The `match_sensors` function computes percentage differences for width, height, and megapixel matching:

```python
width_match = abs(width - sensor_width) / width * 100 <= size_tolerance
height_match = abs(height - sensor_height) / height * 100 <= size_tolerance
...
abs(megapixels - mp) / megapixels * 100 <= megapixel_tolerance
```

All three divisions can produce ZeroDivisionError when the denominator is 0.0:
- `width=0.0` → lines 230-231 crash
- `height=0.0` → line 231 crash
- `megapixels=0.0` → line 238 crash

The width/height cases are guarded by `if not width or not height: return []` on line 217, but this guard uses a truthy check (0.0 is treated as None). The megapixels case has NO guard against 0.0.

**Architectural recommendation:** All percentage-based comparisons should guard their denominators explicitly. Consider a helper function:

```python
def _pct_diff(value: float, reference: float) -> Optional[float]:
    """Return percentage difference, or None if reference is <= 0."""
    if reference <= 0:
        return None
    return abs(value - reference) / reference * 100
```

---

## Summary

- ARCH34-01 (MEDIUM): match_sensors percentage comparisons lack ZeroDivisionError guards
