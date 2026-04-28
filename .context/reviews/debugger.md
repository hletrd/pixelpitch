# Debugger Review — Cycle 46

**Date:** 2026-04-28
**Reviewer:** debugger

## Previous Findings Status

DBG45-01 (GSMArena decimal MP regex) — COMPLETED. Fix applied and tested.

## New Findings

### DBG46-01: matched_sensors silently lost in merge_camera_data when sensors_db unavailable

**File:** `pixelpitch.py`, merge_camera_data (lines 439-521)
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:** When `sensors.json` is missing or corrupt (e.g., during a git merge conflict), `load_sensors_database()` returns `{}`. `derive_spec` then sets `matched_sensors=[]` for all cameras. During merge, the `[]` overwrites existing `matched_sensors` data from the previous CSV because the merge code only preserves `None` values.

**Concrete failure scenario:**
1. Build succeeds with `sensors.json` available -> Canon R5 gets `matched_sensors=['IMX309', 'IMX366', 'IMX609']`
2. CSV is written to dist/ with sensor data
3. Next build: `sensors.json` is missing (git conflict, accidental deletion)
4. `derive_spec(spec, {})` -> `matched_sensors=[]`
5. `merge_camera_data` overwrites `['IMX309', 'IMX366', 'IMX609']` with `[]`
6. CSV loses sensor match data
7. On subsequent builds (even with sensors.json restored), the sensor data is not in the existing CSV, so it must be re-derived

**Why it was missed for 45 cycles:** No test covers `matched_sensors` preservation in merge. The template doesn't display `matched_sensors` (it's in a TODO block), so the bug doesn't produce visible UI changes. It only affects the CSV download.

**Fix:** Return `matched_sensors=None` from `derive_spec` when `sensors_db` is falsy. Add `matched_sensors` preservation in `merge_camera_data`.

---

## Summary

- DBG46-01 (MEDIUM): matched_sensors silently lost in merge_camera_data when sensors_db unavailable
