# Code Review (Cycle 11) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-10 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-10) — Confirmed Resolved
All previous fixes remain intact. C10-01 through C10-10 all verified as fixed.

## New Findings

### C11-01: `create_camera_key` produces different keys for same camera when year differs across sources — causes duplicates
**File:** `pixelpitch.py`, lines 313-315
**Severity:** MEDIUM | **Confidence:** HIGH

`create_camera_key` includes the year in the key: `f"{spec.name.lower().strip()}-{spec.category}-{year}"`. When `year is None`, the key uses the string `"unknown"`. When the same camera is provided by two different sources (e.g., Geizhals with year=2021, openMVG with year=None), the keys differ: `"sony a7 iv-mirrorless-2021"` vs `"sony a7 iv-mirrorless-unknown"`. This causes `merge_camera_data` to treat them as separate cameras, producing duplicate entries.

openMVG always provides `year=None` (its CSV has no year column). Any camera that appears in both openMVG and Geizhals or Imaging Resource will be duplicated.

**Failure scenario:** Camera "Sony A7 IV" from Geizhals (year=2021) and from openMVG (year=None) produces two entries in the merged output. Both appear on the mirrorless page and the All Cameras page.

**Fix:** Remove the year from `create_camera_key` or normalize None years. Camera names are already unique within categories, so year is not needed for deduplication. The merge code already handles year preservation on line 343-344.

---

### C11-02: `parse_existing_csv` doesn't strip whitespace from category field — same pattern as fixed C10-01
**File:** `pixelpitch.py`, lines 262-263, 275-276
**Severity:** LOW | **Confidence:** HIGH

C10-01 fixed the type field by adding `.strip()`. The same pattern applies to the category field: `category = values[2]` (no-id) or `category = values[1]` (has-id). A category like `" mirrorless"` would not match the category filters in `render_html` (lines 785-802), causing the camera to appear only on the "All Cameras" page but not on its category page.

While `write_csv` never introduces whitespace in the category field, manually edited CSVs could. This is the exact same vulnerability that C10-01 fixed for the type field.

**Fix:** Add `.strip()` to category field parsing, same as was done for the type field.

---

### C11-03: `deduplicate_specs` name unification loses the year from non-first variants
**File:** `pixelpitch.py`, lines 523-543
**Severity:** LOW | **Confidence:** MEDIUM

When color variants are unified (same specs, different EXTRAS suffix), the code takes `year = min(years) if years else None` where `years = [s.year for s in grouped_specs if s.year]`. This correctly picks the earliest year. However, if the first variant has a year and a later variant doesn't, the unified result always uses the min year from variants that have years. This is correct behavior, but it's worth noting for completeness — a variant with `year=None` is silently ignored in the min calculation.

---

## Summary
- NEW findings: 3 (1 MEDIUM, 2 LOW)
- C11-01: `create_camera_key` year mismatch causes duplicates across sources — MEDIUM
- C11-02: Category field whitespace not stripped — LOW (same pattern as C10-01)
- C11-03: deduplicate_specs year handling for variants — LOW (informational)
- All cycle 1-10 fixes remain intact
