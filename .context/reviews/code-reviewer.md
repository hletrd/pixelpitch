# Code Review (Cycle 37) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-36 fixes, focusing on NEW issues

## Previous Findings Status

All C36 findings (CR36-01 through CR36-03) confirmed fixed. `_safe_float` correctly rejects NaN/inf. `pixel_pitch` uses `isfinite` guard. `openmvg.fetch` uses `math.isfinite`. All gate tests pass.

## New Findings

### CR37-01: `write_csv` writes `0.0` pitch/mpix as `0.00`/`0.0` — round-trip asymmetry

**File:** `pixelpitch.py`, lines 838-839
**Severity:** LOW | **Confidence:** HIGH

When `spec.mpix` is `0.0`, `write_csv` outputs `f"{0.0:.1f}"` = `"0.0"`. When `derived.pitch` is `0.0`, it outputs `f"{0.0:.2f}"` = `"0.00"`. These are valid float strings that `_safe_float` will parse back to `0.0`, so the round-trip test passes. However, `0.0` for mpix and `0.0` for pitch represent invalid/malformed data (a camera with zero megapixels or zero pixel pitch is physically impossible). These values survive the CSV round-trip but represent a semantic issue — the guard in `pixel_pitch` returns `0.0` for invalid inputs, and that `0.0` propagates as if it were a legitimate measurement.

This is a design concern, not a bug. The `0.0` sentinel value for "invalid" is indistinguishable from a legitimate `0.0` value. A cleaner approach would be to store `None` for invalid-derived values instead of `0.0`, but this is a larger refactor.

**Fix:** No immediate fix required. Consider converting `pixel_pitch` to return `None` instead of `0.0` for invalid inputs in a future refactor, and update the template to display "unknown" for `None` pitch values.

---

### CR37-02: `derive_spec` computes `area = nan * 24.0 = nan` when one dimension is NaN but the other is valid

**File:** `pixelpitch.py`, lines 731-733
**Severity:** MEDIUM | **Confidence:** HIGH

When `spec.size = (float('nan'), 24.0)` (NaN width, valid height), `derive_spec` computes:
```python
area = size[0] * size[1]  # = nan * 24.0 = nan
```

The `area` becomes `nan`, which then flows to `pixel_pitch(nan, mpix)` → returns `0.0`. The final `pitch` is `0.0` rather than `None`, which is incorrect — it's not that the camera has a zero pitch, it's that we have invalid data.

The `parse_existing_csv` path now rejects NaN via `_safe_float`, so NaN can't enter through CSV. But a Spec object constructed in code with `size=(nan, 24.0)` still reaches `derive_spec` with NaN in one dimension.

**Fix:** Add a `math.isfinite` guard in `derive_spec` for the size dimensions:
```python
if size is not None and spec.mpix is not None:
    if isfinite(size[0]) and isfinite(size[1]) and size[0] > 0 and size[1] > 0:
        area = size[0] * size[1]
    else:
        size = None
        area = None
```

---

### CR37-03: `match_sensors` division by zero when `width` or `height` is very small

**File:** `pixelpitch.py`, line 236
**Severity:** LOW | **Confidence:** MEDIUM

```python
width_match = abs(width - sensor_width) / width * 100 <= size_tolerance
```

If `width` is extremely small (e.g., `1e-15`), the division `abs(width - sensor_width) / width` can produce an enormous number, causing false negatives. While `width <= 0` is already guarded, extremely small positive values are not. This is a theoretical edge case — real sensor widths are always >= 2mm.

**Fix:** No immediate fix required. The guard `width <= 0` handles the practical cases. Add a minimum width check only if sensor data includes anomalous entries.

---

## Summary

- CR37-01 (LOW): `pixel_pitch` returns `0.0` sentinel for invalid inputs — design concern, not a bug
- CR37-02 (MEDIUM): `derive_spec` propagates NaN area from partially-NaN size tuple
- CR37-03 (LOW): `match_sensors` division by near-zero width — theoretical edge case
