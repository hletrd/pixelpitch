# Tracer Review (Cycle 39) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28

## Previous Findings Status

TR38-01 fixed. Template now renders "unknown" for 0.0 pitch/mpix.

## New Findings

### TR39-01: Negative value data flow — `_safe_float` → `parse_existing_csv` → template renders as numeric

**Files:** `pixelpitch.py` (_safe_float, parse_existing_csv), `templates/pixelpitch.html`
**Severity:** MEDIUM | **Confidence:** HIGH

**Causal trace for negative pitch entering the system:**

1. CSV file contains `pixel_pitch_um` = `-1.00` (corrupted or manually edited)
2. `parse_existing_csv` reads the CSV, calls `_safe_float("-1.00")`
3. `_safe_float` returns `-1.0` (passes `isfinite` check — -1.0 IS finite)
4. `spec.pitch = -1.0` is stored in the `Spec` object
5. `derive_spec` with `spec.pitch=-1.0` → `derived.pitch = -1.0` (preserved as-is, precedence rule)
6. `write_csv` writes `-1.00` for pitch (round-trip preserved)
7. Template: `spec.pitch is not none and spec.pitch != 0.0` → both True → renders "-1.0 µm"
8. JS `isInvalidData`: `pitch < 0` → returns true → row hidden by default

**Comparison with C38-01 trace (zero pitch):**

The same three-layer disagreement exists for negative values that existed for zero values before C38-01 was fixed:
- **Python logic**: `_safe_float` allows -1.0 through (only rejects NaN/inf)
- **Template**: treats -1.0 as a valid number, renders "-1.0 µm"
- **JS**: treats -1.0 as invalid, hides the row

The C38-01 fix addressed `0.0` specifically but did not generalize to all invalid numeric values.

**Fix:** Change template guards from `!= 0.0` to `> 0`, which covers zero, negative, and NaN in a single condition.

---

## Summary

- TR39-01 (MEDIUM): Negative value data flow — `_safe_float` allows negatives through CSV pipeline, template renders them as numeric
