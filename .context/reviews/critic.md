# Critic Review — Cycle 46

**Date:** 2026-04-28
**Reviewer:** critic

## Previous Findings Status

CRIT45-01 (GSMArena decimal MP regex) — COMPLETED. Fix applied and tested.

## New Findings

### CRIT46-01: matched_sensors data loss in merge_camera_data — semantic ambiguity between [] and None

**File:** `pixelpitch.py`, merge_camera_data (lines 439-521)
**Severity:** MEDIUM | **Confidence:** HIGH

The merge_camera_data function has a systematic gap in its field preservation logic: `matched_sensors` is not included in the set of fields checked for preservation from existing data. This is because the field preservation code was written to check `if new_spec.X is None and existing_spec.X is not None`, which works for fields that default to `None` but fails for `matched_sensors` which defaults to `[]` (empty list) rather than `None`.

The semantic ambiguity is the core problem: `matched_sensors=[]` can mean either "we checked the sensors database and found no matches" (authoritative) or "we did not check the sensors database" (non-authoritative). The current code cannot distinguish these cases.

When `derive_spec` is called with `sensors_db=None` (the default for `derive_specs` which loads the DB), it calls `match_sensors` only when `sensors_db` is truthy. When `sensors_db` is `None` or empty, `matched_sensors` is set to `[]` — the same value as "checked and found nothing."

**Concrete failure scenario:**
1. Previous build cycle: Canon R5 gets `matched_sensors=['IMX309', 'IMX366', 'IMX609']` from sensors_db
2. This data is written to `camera-data.csv` in dist/
3. Next build cycle: `sensors.json` is missing/corrupt (e.g., git merge conflict)
4. `derive_specs` -> `derive_spec(spec, {})` -> `matched_sensors=[]`
5. `merge_camera_data` overwrites `['IMX309', 'IMX366', 'IMX609']` with `[]`
6. CSV download loses sensor match information

**Fix:** Make `derive_spec` return `matched_sensors=None` when `sensors_db` is `None` or empty (meaning "not checked"), and `matched_sensors=[]` only when the database was actually consulted. Then add `matched_sensors` preservation in `merge_camera_data` following the same pattern as other fields: preserve from existing when new is `None`.

---

## Summary

- CRIT46-01 (MEDIUM): matched_sensors data loss in merge_camera_data — semantic ambiguity between [] and None
