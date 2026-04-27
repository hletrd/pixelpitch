# Debugger Review (Cycle 11) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-10 fixes

## Previously Fixed (Cycles 1-10) — Confirmed Resolved
All previous fixes verified. No regressions detected.

## New Findings

### D11-01: `create_camera_key` year component causes duplicates when sources disagree on year
**File:** `pixelpitch.py`, lines 313-315, 318-367
**Severity:** MEDIUM | **Confidence:** HIGH

The year in `create_camera_key` creates different keys for the same camera when different sources provide different year values. The failure mode:

1. Camera "Canon EOS R5" from Geizhals: key = `"canon eos r5-mirrorless-2020"`
2. Same camera from openMVG (year=None): key = `"canon eos r5-mirrorless-unknown"`
3. `merge_camera_data` sees different keys → treats as different cameras
4. Both cameras appear in the merged output → duplicate on website

The `year=None` → `"unknown"` mapping means this affects EVERY camera that appears in both openMVG (or digicamdb, which wraps openMVG) and any other source.

**Failure scenario:** Run `pixelpitch.py` with openMVG + Geizhals data. Observe duplicate cameras on the mirrorless and DSLR pages.

**Fix:** Remove year from `create_camera_key`.

---

### D11-02: `parse_existing_csv` category field not stripped — same bug pattern as fixed C10-01
**File:** `pixelpitch.py`, lines 262, 275
**Severity:** LOW | **Confidence:** HIGH

The category field is parsed without `.strip()`. If a manually edited CSV has whitespace in the category column (e.g., `" mirrorless"`), the camera would have `category=" mirrorless"`, which would not match the category filter in `render_html` (line 793: `if s.spec.category == cat`). The camera would be excluded from the mirrorless page and only appear on the "All Cameras" page.

Same root cause as C10-01 which was fixed for the type field. The fix was applied to `type_str` but not to `category`.

**Fix:** Add `.strip()` to category field parsing.

---

### D11-03: `merge_camera_data` year preservation only works when existing has year and new doesn't
**File:** `pixelpitch.py`, lines 342-344
**Severity:** LOW | **Confidence:** MEDIUM

Line 343-344: `if new_spec.spec.year is None and existing_spec.spec.year is not None: new_spec.spec.year = existing_spec.spec.year`. This only preserves the year from existing data when the NEW data has no year. It does NOT preserve the year when the existing data has a year and the new data has a DIFFERENT year (e.g., existing year=2020, new year=2021 from a revised source). The new year silently overwrites the old year.

This is not necessarily a bug (updated data should take precedence), but it's worth noting that if a source incorrectly provides a year, it will overwrite the correct year from existing data without any warning.

**Fix:** Consider logging when years differ, or preferring the existing year when the new source is known to have less reliable year data.

---

## Summary
- NEW findings: 3 (1 MEDIUM, 2 LOW)
- D11-01: create_camera_key year mismatch — duplicates across sources — MEDIUM
- D11-02: Category field whitespace not stripped — LOW (same as C10-01 pattern)
- D11-03: Merge year overwrite without warning — LOW
