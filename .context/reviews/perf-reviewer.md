# Performance Review (Cycle 55)

**Reviewer:** perf-reviewer
**Date:** 2026-04-29
**HEAD:** `f08c3c4`

## Hot-path inventory

`merge_camera_data`, `_load_per_source_csvs`, `derive_specs`,
`match_sensors`, `parse_existing_csv`, `write_csv`, `render_html`,
source `fetch()`s, `extract_specs`.

## Findings

### F55-PR-01: `_load_per_source_csvs` per-row `match_sensors` — INFORMATIONAL

- **File:** `pixelpitch.py:1080-1082`
- **Detail:** ~1000 rows × 5 sources × ~200 sensors per call ≈ 1M
  comparisons. Sub-second on commodity hardware. Same complexity
  class as the deferred F49-04 entry. No new action.

### F55-PR-02: `match_sensors` `sorted()` per call — INFORMATIONAL

- **File:** `pixelpitch.py:253`
- **Detail:** Per-call list-sort allocates trivially. Not a hotspot.

### F55-PR-03: `parse_existing_csv` 6× `_safe_float` per row — INFORMATIONAL

- **File:** `pixelpitch.py:414-432`
- **Detail:** ~36k float-parse attempts on full load; <100 ms.

## No new actionable performance issues this cycle.
