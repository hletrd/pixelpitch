# Critic Review (Cycle 36) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-35 fixes, focusing on NEW issues

## Previous Findings Status

All C35 findings fixed and verified. No regressions.

## New Findings

### CRIT36-01: The C35-01 fix (negative area guard) is incomplete — NaN and inf bypass the guard

**Files:** `pixelpitch.py` line 184, `sources/openmvg.py` line 96
**Severity:** MEDIUM | **Confidence:** HIGH

The C35-01 fix added `if mpix <= 0 or area <= 0: return 0.0` to `pixel_pitch`. This correctly handles negative and zero values but is incomplete because:

1. `float('nan') <= 0` is `False` — NaN bypasses the guard
2. `float('inf') <= 0` is `False` — inf bypasses the guard
3. `float('nan') <= 0` is `False` — NaN mpix also bypasses the guard

The C35 reviews specifically identified that negative area crashes via `sqrt()` and that negative mpix returns 0.0. But NaN is worse than negative — it doesn't even crash. It silently propagates through every subsequent computation, producing NaN output that renders as "nan µm" on the page.

This is the same class of issue as C35-01 (data integrity) but with a different input domain. The fix pattern should be `math.isfinite()` rather than just `> 0` checks.

**Fix:** Replace the `<= 0` guard with a `math.isfinite` check:
```python
def pixel_pitch(area: float, mpix: float) -> float:
    if not math.isfinite(area) or not math.isfinite(mpix) or mpix <= 0 or area <= 0:
        return 0.0
    return 1000 * sqrt(area / (mpix * 10**6))
```

And add `math.isfinite` validation in `parse_existing_csv` for all float fields, and in `openmvg.fetch` for sensor dimensions.

---

## Summary

- CRIT36-01 (MEDIUM): C35-01 fix incomplete — NaN and inf bypass the `<= 0` guard
