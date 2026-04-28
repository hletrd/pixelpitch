# Debugger Review (Cycle 40) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28

## Previous Findings Status

DBG39-01 fixed. Template renders "unknown" for negative/NaN pitch/mpix.

## New Findings

### DBG40-01: `derive_spec` produces pitch=0.0 from computed path — CSV round-trip data loss

**File:** `pixelpitch.py`, lines 757-762; lines 839-872 (write_csv)
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:**
1. Camera has `spec.pitch=None, spec.mpix=0.0, spec.size=(5.0, 3.7)`
2. `derive_spec` computes: `pixel_pitch(18.5, 0.0) = 0.0`, stores as `derived.pitch=0.0`
3. `write_csv` writes "0.00" for pitch column
4. `parse_existing_csv` reads "0.00", `_safe_float` returns 0.0, positivity check rejects it (0.0 <= 0), sets pitch=None
5. Data loss: camera that had `derived.pitch=0.0` now has `derived.pitch=None`

This is a silent data corruption on CSV round-trip. The `pitch=0.0` value is lost and becomes `pitch=None`. While the 0.0 value itself is meaningless (it's a sentinel), the data model inconsistency could cause issues if the round-trip is used for incremental updates — `merge_camera_data` would treat the None pitch differently from 0.0 pitch.

**Fix:** In `derive_spec`, convert `pixel_pitch()`'s 0.0 return to None. This ensures `write_csv` writes an empty string for pitch, and `parse_existing_csv` reads it back as None — consistent round-trip.

---

## Summary

- DBG40-01 (MEDIUM): `derive_spec` pitch=0.0 causes CSV round-trip data loss (0.0 -> None)
