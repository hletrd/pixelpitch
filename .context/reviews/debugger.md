# Debugger Review (Cycle 37) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-36 fixes

## Previous Findings Status

All C36 findings fixed. No regressions detected. Gate tests pass.

## New Findings

### DBG37-01: `derive_spec` computes `area = nan` for partially-NaN size, writes "nan" to CSV

**File:** `pixelpitch.py`, lines 731-733
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:** When `spec.size = (nan, 24.0)`, `derive_spec` computes `area = nan * 24.0 = nan`. This `nan` area then flows to:
1. `pixel_pitch(nan, mpix)` → returns `0.0` (guarded, OK)
2. `SpecDerived(area=nan, ...)` — the `area` field is `nan`
3. `write_csv` formats `f"{nan:.2f}"` = `"nan"` — writes literal "nan" to CSV
4. `_safe_float("nan")` on re-read returns `None` — area changes from nan to None

The failure mode is: on the first render after a camera with NaN size enters the system, the CSV contains "nan" strings. On the next render, those "nan" strings are parsed as `None`, so the data "self-heals" but with data loss (area changes from nan to None, when it should have been None from the start).

**Trigger scenario:**
1. Source parser or test code constructs `Spec(size=(nan, 24.0))`
2. `derive_spec` produces `SpecDerived(area=nan, pitch=0.0)`
3. `write_csv` writes "nan" to the area column
4. Next build reads "nan" → `_safe_float` → `None`
5. Area becomes None — data "healed" but via a corrupt intermediate CSV

**Fix:** Add `isfinite` guard in `derive_spec` for size dimensions. Set `size=None` and `area=None` when dimensions are not finite.

---

## Summary

- DBG37-01 (MEDIUM): `derive_spec` area=nan writes "nan" to CSV, self-heals to None on next read
