# Debugger Review (Cycle 36) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-35 fixes, focusing on NEW issues

## Previous Findings Status

All C35 findings fixed. No regressions detected. Gate tests pass.

## New Findings

### DBG36-01: NaN and inf values propagate silently through the entire data pipeline

**Files:** `pixelpitch.py` (pixel_pitch, derive_spec, parse_existing_csv, write_csv), `sources/openmvg.py`
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:** Python's `float()` function accepts `"nan"`, `"inf"`, `"-inf"` as valid inputs. The CSV parser uses bare `float()` for all numeric fields. Once NaN or inf enters the pipeline, it propagates through every computation without raising an exception:

1. `parse_existing_csv` accepts `"nan"` and `"inf"` CSV values
2. `derive_spec` propagates NaN through `area = nan * 24.0 = nan`
3. `pixel_pitch(nan, 10.0)` returns `nan` (guard only checks `<= 0`)
4. `write_csv` writes `nan` and `inf` to CSV as strings
5. Template renders "nan µm" and `data-pitch="nan"`
6. JS `isInvalidData` does not catch NaN (parseFloat("nan") || 0 = 0)

**Trigger scenario:**
1. Manual CSV edit or data corruption introduces `nan` or `inf` in a cell
2. `parse_existing_csv` happily parses it
3. NaN propagates through derive_spec, write_csv, and template rendering
4. The build succeeds but produces "nan µm" in visible output

**Fix:** Add `math.isfinite()` guards in `pixel_pitch` and `parse_existing_csv`.

---

### DBG36-02: `pixel_pitch` returns inf for inf inputs instead of 0.0

**File:** `pixelpitch.py`, line 184
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:** `pixel_pitch(float('inf'), 10.0)` returns `inf` because:
- `inf <= 0` is False (guard does not catch it)
- `sqrt(inf / 10e6)` = `inf`

The C35-01 fix added `area <= 0` guard but did not account for inf. While negative area correctly returns 0.0, infinite area returns `inf`, which then renders as "inf µm" in the template.

**Fix:** Add `math.isfinite` guard in `pixel_pitch`.

---

## Summary

- DBG36-01 (MEDIUM): NaN and inf propagate silently through the entire pipeline
- DBG36-02 (MEDIUM): `pixel_pitch` returns inf for inf inputs instead of 0.0
