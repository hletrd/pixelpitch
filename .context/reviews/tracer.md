# Tracer Review — Cycle 46

**Date:** 2026-04-28
**Reviewer:** tracer

## Previous Findings Status

TR45-01 (GSMArena decimal MP regex) — COMPLETED. Fix verified.

## New Findings

### TR46-01: matched_sensors data loss in merge_camera_data — causal trace

**File:** `pixelpitch.py`, merge_camera_data (lines 439-521)
**Severity:** MEDIUM | **Confidence:** HIGH

Causal trace:
1. `render_html` calls `get_category` -> `derive_specs` -> `derive_spec(spec, sensors_db)`
2. `derive_spec` checks `if sensors_db and size:` — when `sensors_db` is empty (sensors.json missing), the condition is False
3. `matched_sensors` is set to `[]` (the default initialization at line 811)
4. `merge_camera_data` processes new specs against existing specs from `parse_existing_csv`
5. For overlapping cameras (same key), the merge preserves fields when `new.X is None and existing.X is not None`
6. `matched_sensors` is never checked because `[]` is not `None`
7. The existing `matched_sensors=['IMX309', 'IMX366', 'IMX609']` from CSV is overwritten by `[]`
8. `write_csv` writes `[]` as empty string in the matched_sensors column
9. On the next build cycle, the sensor data is permanently lost from the CSV

Competing hypotheses:
- H1: sensors_db is always available so this never happens -> REJECTED. `load_sensors_database` returns `{}` on FileNotFoundError/JSONDecodeError/OSError
- H2: derive_specs always passes sensors_db -> CONFIRMED but irrelevant. The issue is when sensors_db is empty (not when it's missing)
- H3: The empty-list vs None ambiguity causes the data loss -> CONFIRMED. `matched_sensors=[]` bypasses the `is None` preservation check

**Fix:** Change `derive_spec` to return `matched_sensors=None` when `sensors_db` is `None` or falsy, indicating "not checked." Then add preservation logic in `merge_camera_data`.

---

## Summary

- TR46-01 (MEDIUM): matched_sensors data loss in merge_camera_data — causal trace confirmed
