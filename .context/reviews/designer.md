# Designer Review (Cycle 36) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-35 fixes, focusing on NEW issues

## Previous Findings Status

DES35-01 (negative pitch renders) and DES35-02 (NaN pitch renders) partially addressed. Negative pitch now guarded by `pixel_pitch` returning 0.0. JS `isInvalidData` now checks `pitch < 0`. However, NaN rendering was NOT fully addressed.

## New Findings

### DES36-01: NaN pitch renders as "nan µm" in visible table cell

**File:** `templates/pixelpitch.html`, lines 84-85
**Severity:** MEDIUM | **Confidence:** HIGH

When `spec.pitch` is `float('nan')`, the Jinja2 template renders `{{ spec.pitch|round(1) }}` as `nan` (Jinja2's `round` filter propagates NaN). The visible cell shows "nan µm". The `data-pitch` attribute becomes `data-pitch="nan"`.

The JS `isInvalidData` function uses `parseFloat(row.attr('data-pitch')) || 0`. In JS, `parseFloat("nan")` returns `NaN`, and `NaN || 0` evaluates to `0`. So the NaN value gets a computed `pitch` of 0, which passes both `pitch > 10` and `pitch < 0` checks. The row is NOT hidden by the "Hide possibly invalid data" filter.

**Verified:** A `spec.pitch=NaN` renders as "nan µm" in the visible cell and is NOT filtered by isInvalidData.

**Fix options (defense-in-depth):**
1. Fix the data pipeline to reject NaN at the source (pixel_pitch guard) -- primary fix
2. Add NaN check to `isInvalidData` JS function: `if (isNaN(parseFloat(row.attr('data-pitch')))) return true;`
3. Add a Jinja2 filter or check in the template

---

## Summary

- DES36-01 (MEDIUM): NaN pitch renders as "nan µm" and is not caught by isInvalidData
