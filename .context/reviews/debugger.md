# Debugger Review (Cycle 32) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-31 fixes, focusing on NEW issues

## Previous Findings Status

DBG31-01 (merge pitch inconsistency) fixed in C31. DBG31-02 (BOM literal) fixed in C31.

## New Findings

### DBG32-01: write_csv falsy check silently drops 0.0 float values

**File:** `pixelpitch.py`, lines 824-827
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

**Failure mode:** If a camera's `spec.mpix` or `derived.pitch` or `derived.area` is ever `0.0`, `write_csv` writes an empty string for that field. On next build, `parse_existing_csv` reads the empty string as `None`. The value silently changes from `0.0` to `None`.

**Root cause:** Python truthiness check `if x` treats `0.0` as falsy, identical to `None`. The correct check is `if x is not None`.

**Trigger:** `pixel_pitch(area, mpix)` returns `0.0` when `mpix <= 0`. If a source parser produces a camera with `mpix=0.0` (unlikely but possible through computation), the derived pitch would be `0.0`, which write_csv would drop.

**Fix:** Replace `if x` with `if x is not None` for float and integer fields in `write_csv`.

---

## Summary

- DBG32-01 (LOW-MEDIUM): write_csv falsy checks lose 0.0 float values on CSV round-trip
