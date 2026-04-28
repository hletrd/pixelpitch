# Debugger Review (Cycle 34) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-33 fixes, focusing on NEW issues

## Previous Findings Status

DBG33-01 (derive_spec 0.0 override) and DBG33-02 (template 0.0 rendering) fixed in C33. All gate tests pass.

## New Findings

### DBG34-01: match_sensors ZeroDivisionError crash with megapixels=0.0

**File:** `pixelpitch.py`, line 238
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:** If `match_sensors` is called with `megapixels=0.0`, the percentage comparison `abs(megapixels - mp) / megapixels * 100` divides by zero. The guard `if megapixels is not None and sensor_megapixels:` does not protect against 0.0 because `0.0 is not None` is True.

**Trigger scenario:**
1. Source parser produces `Spec(mpix=0.0)` from malformed data or a computation
2. `merge_camera_data` calls `match_sensors` with that camera's mpix
3. `0.0 is not None` → True, enters the division
4. `ZeroDivisionError` raised, not caught by any try/except
5. The entire merge/render pipeline crashes

This is the most significant finding because it's an unhandled exception that would crash the production build.

**Fix:** Guard against `megapixels <= 0` before the division:
```python
if megapixels is not None and megapixels > 0 and sensor_megapixels:
```

---

### DBG34-02: `list` command truthy check skips cameras with pitch=0.0

**File:** `pixelpitch.py`, line 1170
**Severity:** LOW | **Confidence:** HIGH

**Failure mode:** The `list` command uses `if spec.pitch:` to filter before prettyprinting. If `spec.pitch=0.0`, the camera is silently omitted from the listing. This is the same class of truthy-vs-None bug fixed in C33-01 across 4 other locations.

**Fix:** Replace with `if spec.pitch is not None:`

---

## Summary

- DBG34-01 (MEDIUM): match_sensors ZeroDivisionError crash with megapixels=0.0
- DBG34-02 (LOW): `list` command truthy check skips cameras with pitch=0.0
