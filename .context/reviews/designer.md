# Designer Review (Cycle 39) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28

## Previous Findings Status

DES38-01 fixed. Template renders "unknown" for 0.0 pitch/mpix. DES38-02 verified — scatter plot still correct.

## New Findings

### DES39-01: `!= 0.0` guard incomplete — negative/NaN pitch renders as numeric, confusing UX

**File:** `templates/pixelpitch.html`, lines 84-88
**Severity:** MEDIUM | **Confidence:** HIGH

The C38-01 fix changed the template guard to `!= 0.0`. This correctly handles zero values but leaves a UX inconsistency for negative and NaN values:

- A camera with `pitch=-1.0` renders "-1.0 µm" in the table — a physically impossible value displayed as if legitimate
- A camera with `pitch=NaN` renders "nan µm" — a malformed display string
- The JS filter correctly hides these rows by default, but unchecking "Hide possibly invalid data" reveals the confusing values

The `> 0` guard would solve both: negative values and NaN both fail the `> 0` check, rendering "unknown" instead.

**Fix:** Change `!= 0.0` to `> 0` in template pitch and mpix guards.

---

## Summary

- DES39-01 (MEDIUM): `!= 0.0` guard incomplete — negative/NaN pitch renders as numeric, inconsistent UX
