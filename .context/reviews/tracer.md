# Tracer Review (Cycle 56)

**Reviewer:** tracer
**Date:** 2026-04-29
**HEAD:** `e8d5414`

## Causal traces

### Flow 1: per-source CSV → merge → render → write-back (post-C55-01)

1. `fetch_source` writes per-source CSV with current matched_sensors.
2. `_load_per_source_csvs` parses; if sensors_db loads, refresh;
   if sensors_db is empty, **preserve the parsed cache**; if size
   is None, force matched_sensors = None.
3. `merge_camera_data` honors tri-valued sentinel.
4. `write_csv` joins with `;` and guards `;`-in-name.
5. Next cycle: `parse_existing_csv` splits on `;` (strip+dedup).

End-to-end the contract holds. Cache fallback is now consistent
with merge existing-only branch on the empty-db case.

### Flow 2: BOM → parse_existing_csv → write_csv (verified at C55-01)

1. Excel saves CSV with leading `﻿`.
2. `parse_existing_csv` `strip_bom` runs first.
3. `header[0] == "id"` resolves correctly post-strip.
4. Re-write produces no BOM.

Confirmed correct via new BOM regression test.

### Flow 3: sensors.json missing during render_html (post-C55-01)

1. `merge_camera_data` lazy-loads only for existing-only cameras.
   When file missing, returns `{}`. Existing-only cameras get
   matched_sensors UNCHANGED in this branch.
2. `_load_per_source_csvs` lazy-loads when row has size; on `{}`
   PRESERVES the parsed cache (was: dropped to None).

Inconsistency from cycle 55 is now resolved on the empty-db path.

## Findings

### F56-T-01 (resolved): empty-db cache-discard inconsistency — RESOLVED at e8d5414.

### F56-T-02 (informational): size-less branch in _load_per_source_csvs still drops cache, by design

- See F56-D-04 / F56-A-01. Intentional; documented; tests pin
  refresh and empty-db paths but not the size-less path.

## Competing hypotheses

- H1 (chosen): C55-01 fix is correct and complete for the empty-db
  case. Remaining size-less-drop behavior is intentional.
- H2 (rejected): force size-less branch to also preserve cache.
  Rejected because `derive_spec` documents matched_sensors = None
  when size is unknown ("not checked"); preserving stale cache
  would lie about staleness.
