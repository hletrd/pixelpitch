# Tracer Review (Cycle 55)

**Reviewer:** tracer
**Date:** 2026-04-29
**HEAD:** `f08c3c4`

## Causal traces

### Flow 1: per-source CSV → merge → render → write-back

1. `fetch_source` writes per-source CSV with current matched_sensors.
2. `_load_per_source_csvs` parses and (post-C54-01) recomputes
   matched_sensors against current sensors.json.
3. `merge_camera_data` honors tri-valued sentinel.
4. `write_csv` joins with `;` and guards `;`-in-name.
5. Next cycle: `parse_existing_csv` splits on `;` (strip+dedup).

End-to-end the contract holds.

### Flow 2: BOM → parse_existing_csv → write_csv

1. Excel saves CSV with leading `﻿`.
2. `parse_existing_csv` `strip_bom` runs.
3. `header[0] == "id"` resolves correctly post-strip.
4. Re-write produces no BOM.

Confirmed correct.

### Flow 3: sensors.json missing during render_html

1. `merge_camera_data` lazy-loads only for existing-only cameras.
   When file missing, returns `{}`. Existing-only cameras get
   matched_sensors UNCHANGED in this branch.
2. `_load_per_source_csvs` always overwrites with lookup; on `{}`
   sets matched_sensors to `None` (drops cache).

Inconsistency confirmed → F55-T-01.

## Findings

### F55-T-01 (consensus with F55-CRIT-01): inconsistent sensors_db-failure behavior between merge and per-source-load paths — LOW

- See F55-CRIT-01.
- **Severity:** LOW. **Confidence:** MEDIUM.

## Competing hypotheses

- H1 (chosen): C54-01 happy-path correct; failure path too aggressive.
  Fix the failure path only.
- H2 (rejected): Revert C54-01.
