# Performance Review (Cycle 56)

**Reviewer:** perf-reviewer
**Date:** 2026-04-29
**HEAD:** `e8d5414`

## Hot-path inventory

`merge_camera_data`, `_load_per_source_csvs`, `derive_specs`,
`match_sensors`, `parse_existing_csv`, `write_csv`, `render_html`,
source `fetch()`s, `extract_specs`.

## Findings

### F56-PR-01: `_load_per_source_csvs` per-row `match_sensors` (carry-over) — INFORMATIONAL

- Same as F55-PR-01 / deferred F49-04. Sub-second on commodity HW.

### F56-PR-02: `match_sensors` `sorted()` per call (carry-over) — INFORMATIONAL

- Same as F55-PR-02. Trivial allocation.

### F56-PR-03: `parse_existing_csv` 6× `_safe_float` per row (carry-over) — INFORMATIONAL

- Same as F55-PR-03. <100 ms on full load.

### F56-PR-04: C55-01 cache-preservation does NOT change call count of `match_sensors` — INFORMATIONAL

- When sensors_db is empty, `match_sensors` is never called. Net
  perf effect: tiny improvement (one fewer per-row call) on the
  empty-db path. Not actionable.

## No new actionable performance issues this cycle.
