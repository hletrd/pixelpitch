# Code Review — Cycle 46

**Date:** 2026-04-28
**Reviewer:** code-reviewer

## Review Scope

All Python source files (`pixelpitch.py`, `models.py`, `sources/*.py`), templates, and tests.

## Previous Cycle Findings (1-45) Status

All previous fixes confirmed still in place. C45-01 GSMArena decimal MP regex fix working correctly. C44-01 CineD dead code cleanup complete.

## New Findings

### CR46-01: matched_sensors not preserved in merge_camera_data — data loss

**File:** `pixelpitch.py`, lines 439-521 (merge_camera_data)
**Severity:** MEDIUM | **Confidence:** HIGH

The merge_camera_data function preserves `type`, `size`, `pitch`, `mpix`, `year`, `area` fields from existing data when new data has `None`. However, `matched_sensors` is never checked for preservation. When `derive_spec` is called without `sensors_db` (or with an empty `sensors_db`), it returns `matched_sensors=[]`. The merge code treats `[]` as "we have data" (not `None`), so it overwrites existing sensor matches from the previous CSV with an empty list.

**Failure scenario:**
1. Previous CSV has `matched_sensors=['IMX309', 'IMX366', 'IMX609']` for Canon R5
2. `sensors.json` is temporarily unavailable (missing/corrupt)
3. `derive_specs` -> `derive_spec(spec, {})` -> `matched_sensors=[]`
4. `merge_camera_data` overwrites existing `['IMX309', 'IMX366', 'IMX609']` with `[]`
5. CSV download loses all sensor match data

**Root cause:** `matched_sensors=[]` and `matched_sensors=None` are semantically different (empty after checking vs. not checked), but the merge code doesn't distinguish them.

**Fix:** Change `derive_spec` to return `matched_sensors=None` when `sensors_db` is not provided (or is empty), and add a `matched_sensors` preservation check in `merge_camera_data` similar to other fields.

---

### CR46-02: LENS_RE dead code in gsmarena.py

**File:** `sources/gsmarena.py`, lines 45-50
**Severity:** LOW | **Confidence:** HIGH

The `LENS_RE` regex is defined at module level but never referenced anywhere in the codebase — not in `gsmarena.py` itself nor in any other module. This is dead code similar to the C44-01 `FORMAT_TO_MM` removal in `cined.py`.

**Fix:** Remove the `LENS_RE` definition entirely.
