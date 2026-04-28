# Critic Review (Cycle 41) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28

## Previous Findings Status

CRIT40-01 fixed. `derive_spec` converts computed 0.0 sentinel to None. But the fix was narrow — only the computed path.

## New Findings

### CRIT41-01: C40 fix was incomplete — `derive_spec` still propagates invalid direct `spec.pitch` values

**File:** `pixelpitch.py`, lines 759-760; `templates/pixelpitch.html`, line 183
**Severity:** MEDIUM | **Confidence:** HIGH

The C40 fix addressed the computed path: when `pixel_pitch()` returns 0.0, `derive_spec` converts it to None. But the direct path (`spec.pitch is not None`) was left unguarded. This means:

1. `Spec(pitch=0.0)` → `derived.pitch = 0.0` — still passes through selectattr
2. `Spec(pitch=-1.0)` → `derived.pitch = -1.0` — negative pitch in data model
3. `Spec(pitch=nan)` → `derived.pitch = nan` — NaN in data model

The template `> 0` guard renders these as "unknown", but the camera is still in the wrong section. The `write_csv` `isfinite` guard catches NaN but not 0.0 or negative values.

This is the same class of bug as CRIT40-01, but on the direct path. The correct fix is a unified validation in `derive_spec` that applies to both paths, establishing the contract that `derived.pitch` is always either None or a positive finite value.

**Fix:** In `derive_spec`, validate `spec.pitch` the same as computed pitch: reject non-finite or non-positive values by converting to None.

---

### CRIT41-02: `write_csv` positivity gap — `isfinite` insufficient for physical quantities

**File:** `pixelpitch.py`, lines 866-868
**Severity:** LOW | **Confidence:** HIGH

The C40 fix added `isfinite()` checks for mpix, pitch, and area in `write_csv`. But `isfinite(0.0)` and `isfinite(-1.0)` both return True. For physical quantities, these values are just as invalid as inf/nan. The `parse_existing_csv` function correctly rejects them (positivity check), creating a data loss on round-trip.

The fix is simple: replace `isfinite(x)` with `x > 0` for all physical quantity fields. This is consistent with `parse_existing_csv` and ensures lossless round-trips.

**Fix:** Replace `isfinite` checks with `> 0` checks in `write_csv` for mpix, pitch, and area.

---

## Summary

- CRIT41-01 (MEDIUM): C40 fix incomplete — derive_spec still propagates invalid direct spec.pitch values
- CRIT41-02 (LOW): write_csv isfinite guard insufficient for physical quantities — 0.0 and negative values pass
