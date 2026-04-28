# Debugger Review (Cycle 41) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28

## Previous Findings Status

DBG40-01 fixed. `derive_spec` converts computed 0.0 sentinel to None.

## New Findings

### DBG41-01: `derive_spec` direct pitch path lacks validation — 0.0, negative, NaN propagate undetected

**File:** `pixelpitch.py`, line 759-760
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode (0.0 direct pitch):**
1. `Spec(pitch=0.0)` (e.g., from legacy CSV data before positivity checks)
2. `derive_spec`: `pitch = spec.pitch = 0.0` — no validation
3. `write_csv` writes "0.00" — `isfinite(0.0)` is True, passes through
4. `parse_existing_csv` rejects 0.0 — data loss on round-trip
5. Template: camera in wrong table section (selectattr includes 0.0)

**Failure mode (negative direct pitch):**
1. `Spec(pitch=-1.0)` (e.g., from corrupted data)
2. `derive_spec`: `pitch = spec.pitch = -1.0` — no validation
3. `write_csv` writes "-1.00" — `isfinite(-1.0)` is True
4. `parse_existing_csv` rejects -1.0 — data loss on round-trip

**Failure mode (NaN direct pitch):**
1. `Spec(pitch=nan)` (e.g., from float arithmetic error)
2. `derive_spec`: `pitch = spec.pitch = nan` — no validation
3. `write_csv`: `isfinite(nan)` is False — writes empty string (safe)
4. Template: `nan > 0` is False — renders "unknown" (safe for display)
5. But `selectattr('pitch', 'ne', None)` with NaN is unpredictable in Jinja2

The C40 fix only addressed the computed path. These are the same class of bug but on the direct path.

**Fix:** Validate `spec.pitch` in `derive_spec`: reject non-finite or non-positive values.

---

### DBG41-02: `write_csv` isfinite guard insufficient — 0.0 and negative values pass through

**File:** `pixelpitch.py`, lines 866-868
**Severity:** LOW | **Confidence:** HIGH

The C40 fix added `isfinite()` checks for mpix, pitch, and area in `write_csv`. But `isfinite(0.0)` is True and `isfinite(-1.0)` is True. For physically meaningful quantities (mpix, pitch, area), these values are invalid and should not be written to the CSV, consistent with `parse_existing_csv`'s rejection of <=0 values.

**Fix:** Replace `isfinite` with positivity checks (`> 0`) for mpix, pitch, and area in `write_csv`.

---

## Summary

- DBG41-01 (MEDIUM): `derive_spec` direct pitch path lacks validation — 0.0, negative, NaN propagate undetected
- DBG41-02 (LOW): `write_csv` isfinite guard insufficient — 0.0 and negative values pass through
