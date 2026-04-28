# Designer Review (Cycle 35) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-34 fixes, focusing on NEW issues

## Previous Findings Status

DES33-01 (template truthy checks for 0.0) fixed in C33. All previous UI/UX findings (F35-F39) remain deferred.

## New Findings

### DES35-01: Negative pitch/mpix values render as "-2.0 µm" and "-10.0 MP" in the table

**File:** `templates/pixelpitch.html`, lines 69-89
**Severity:** LOW | **Confidence:** HIGH

The template renders negative pitch and mpix values as negative numbers (e.g., "-2.0 µm", "-10.0 MP"). While physically meaningless, these values pass through the template rendering without any visual indicator that they're invalid.

The `isInvalidData` JS function (line 263) checks for `pitch > 10` but does NOT check for negative values. So a camera with pitch=-2.0 would be visible in the table (if the "Hide possibly invalid data" checkbox is on, it would pass the check).

**Fix options:**
1. Add a negative value check to `isInvalidData`: `if (pitch < 0) return true;`
2. Or fix the data pipeline to reject negative values at the source (recommended)

---

### DES35-02: NaN pitch value renders as `data-pitch="nan"` and displays "nan µm"

**File:** `templates/pixelpitch.html`, lines 50, 84-85
**Severity:** LOW | **Confidence:** MEDIUM

If a NaN pitch value reaches the template, it renders as `data-pitch="nan"` in the HTML attributes and "nan µm" in the visible cell. While NaN is unlikely to reach the template in practice (the `pixel_pitch` function crashes before producing NaN), this is a defense-in-depth gap.

**Fix:** Add a NaN/finite check in the template or in `write_csv`.

---

## Summary

- DES35-01 (LOW): Negative pitch/mpix renders as negative numbers in the table
- DES35-02 (LOW): NaN pitch renders as "nan µm" in the template
