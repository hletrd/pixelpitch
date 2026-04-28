# Code Review (Cycle 36) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-35 fixes, focusing on NEW issues

## Previous Findings Status

All C35 findings (CR35-01 through CR35-04) confirmed fixed. `_BOM` uses escape sequence. `pixel_pitch` guards `area <= 0`. matched_sensors filters empty strings. openmvg uses `pw > 0 and ph > 0`.

## New Findings

### CR36-01: `pixel_pitch` does not guard against NaN or inf inputs

**File:** `pixelpitch.py`, line 184
**Severity:** MEDIUM | **Confidence:** HIGH

The guard `if mpix <= 0 or area <= 0: return 0.0` does not reject NaN or inf:

- `pixel_pitch(float('nan'), 10.0)` returns `nan` (not 0.0)
- `pixel_pitch(float('inf'), 10.0)` returns `inf` (not 0.0)
- `pixel_pitch(864.0, float('inf'))` returns 0.0 (only this one works because inf <= 0 is False but the sqrt produces 0)

NaN propagates through `derive_spec` when size contains NaN: `area = nan * 24.0 = nan`, then `pixel_pitch(nan, mpix) = nan`.

**Concrete scenario:**
1. Corrupted CSV contains `nan` or `inf` for sensor_width_mm
2. `parse_existing_csv` calls `float("nan")` which succeeds
3. `derive_spec` computes `area = nan * 24.0 = nan`
4. `pixel_pitch(nan, mpix)` returns `nan`
5. `write_csv` writes `nan` to CSV
6. Template renders "nan µm" in the visible cell and `data-pitch="nan"` in the HTML

**Fix:** Add `math.isfinite` guard in `pixel_pitch`:
```python
def pixel_pitch(area: float, mpix: float) -> float:
    if mpix <= 0 or area <= 0 or not math.isfinite(area) or not math.isfinite(mpix):
        return 0.0
    return 1000 * sqrt(area / (mpix * 10**6))
```

---

### CR36-02: `parse_existing_csv` accepts NaN and inf values from CSV strings

**File:** `pixelpitch.py`, lines 328-339
**Severity:** MEDIUM | **Confidence:** HIGH

Python's `float()` accepts `"nan"`, `"inf"`, `"-inf"`, and `"NaN"` as valid inputs. The CSV parser uses bare `float()` calls for width, height, area, mpix, and pitch without checking for finite values. This allows NaN and inf to enter the data pipeline from corrupted or manually edited CSV files.

**Verified:** `parse_existing_csv` with `"nan"` and `"inf"` in CSV cells produces `float('nan')` and `float('inf')` in the resulting SpecDerived objects.

**Fix:** Add `math.isfinite` validation after each `float()` call in `parse_existing_csv`:
```python
width = float(width_str)
height = float(height_str)
if not (math.isfinite(width) and math.isfinite(height)):
    size = None
```
Same pattern for area, mpix, and pitch.

---

### CR36-03: `openmvg.fetch` accepts inf sensor dimensions

**File:** `sources/openmvg.py`, line 96
**Severity:** LOW | **Confidence:** HIGH

The size guard `sw > 0 and sh > 0` passes for `inf` because `inf > 0` is True. While NaN is rejected (because `nan > 0` is False), inf dimensions produce `(inf, inf)` size which propagates through the pipeline.

**Concrete scenario:**
1. CSV row has `SensorWidth(mm)=inf`
2. `float("inf")` succeeds
3. `sw > 0 and sh > 0` is True
4. `size = (inf, inf)` is accepted
5. `derive_spec` computes `area = inf * inf = inf`
6. `pixel_pitch(inf, mpix)` returns `inf`

**Fix:** Replace with `math.isfinite` check:
```python
size = (sw, sh) if sw and sh and math.isfinite(sw) and math.isfinite(sh) and sw > 0 and sh > 0 else None
```

---

## Summary

- CR36-01 (MEDIUM): `pixel_pitch` does not guard against NaN or inf inputs
- CR36-02 (MEDIUM): `parse_existing_csv` accepts NaN and inf from CSV strings
- CR36-03 (LOW): `openmvg.fetch` accepts inf sensor dimensions
