# Performance Review (Cycle 57)

**Reviewer:** perf-reviewer
**Date:** 2026-04-29
**HEAD:** `01c31d8`

## Inventory

- `pixelpitch.py` hot paths: `merge_camera_data`,
  `_load_per_source_csvs`, `match_sensors`, `parse_existing_csv`,
  `derive_specs`.
- Sources: scrapers — I/O bound.
- Templates: rendered once per category.

All hot paths examined.

## New findings (cycle 57)

### F57-PR-01: `match_sensors` is O(n_sensors) per camera, called O(n_cameras) times — INFORMATIONAL

- **File:** `pixelpitch.py:215-253`, called from
  `pixelpitch.py:899-900`, `pixelpitch.py:622-628`,
  `pixelpitch.py:1078-1080`.
- **Detail:** Every camera linearly scans the entire sensors_db.
  Current sensors.json has ~150 entries; cameras ~3000 — total
  ~450k comparisons, each cheap. Total wall time well under 1s.
  No action.
- **Severity:** INFORMATIONAL. **Confidence:** HIGH.

### F57-PR-02: `_load_per_source_csvs` re-parses every per-source CSV on every build — LOW (informational)

- **File:** `pixelpitch.py:1059-1091`
- **Detail:** Called once per `render_html` call. The 5 source CSVs
  total ~3000 rows, parsed via `csv.reader` — well under 100ms total.
  Negligible.
- **Severity:** INFORMATIONAL. **Confidence:** HIGH.

### F57-PR-03: `merge_camera_data` `existing_by_key` dict O(n) build then O(n) iteration — OK

- Standard hash-table merge; dominated by I/O elsewhere. No action.

## Carry-over deferred (from cycles ≤ 56)

- F49-04 perf, F55-PR-01..03 perf, F56-PR-04 informational. Re-defer.

## Sweep for hot-path issues

- No nested loops > O(n*m) where m is small constant.
- No string concat in loop without `join`.
- No accidental O(n²) `in list` checks.
- I/O is always batched into dict/list construction.

## Confidence summary

- 0 new actionable findings.
- All performance carry-overs remain informational.
