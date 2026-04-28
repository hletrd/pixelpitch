# Performance Review (Cycle 17)

**Reviewer:** perf-reviewer
**Date:** 2026-04-28
**Scope:** Full repository performance re-review after cycles 1-16 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-16) — Confirmed Resolved

- P16-01 (merge dedup overhead): Fixed by C16-02's `seen_new_keys` set — O(1) dedup lookup confirmed working.

## New Findings

### P17-01: merge_camera_data loads sensors_db even when no existing specs need sensor matching
**File:** `pixelpitch.py`, line 372 (`sensors_db = load_sensors_database()`)
**Severity:** LOW | **Confidence:** HIGH

`merge_camera_data` always calls `load_sensors_database()` at the start, reading and parsing `sensors.json` from disk. However, `sensors_db` is only used in the "preserve existing-only cameras" loop (lines 413-421). If all existing cameras are also in the new data (common case for a fresh CI run), the sensors_db is loaded but never used.

**Concrete impact:** The sensors.json file is small (~5KB) and the parse is fast, so this is a minor inefficiency. In CI, this runs once per deploy. Not worth optimizing.

**Fix (if desired):** Lazy-load sensors_db — only call `load_sensors_database()` when we actually have existing-only cameras to match.

---

## Summary
- NEW findings: 1 (LOW)
- No significant performance issues found
