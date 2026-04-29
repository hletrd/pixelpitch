# Perf Reviewer — Cycle 60 (Orchestrator Cycle 13)

**Date:** 2026-04-29
**HEAD:** `a0cd982`

## Inventory

`pixelpitch.py` (render_html, merge_camera_data, match_sensors,
_load_per_source_csvs); `sources/*.py` (fetch loops, http_get).

## Status

No new perf regressions observed. All previously-deferred perf items
(F49-04 sensor-DB linear scan, F55-PR-01..03, F56-PR-04, F57-PR-01..03,
F59-PR-01) remain valid as deferred — none have crossed re-open
thresholds.

## Cycle 60 New Findings

### F60-PR-01 (deferred, informational): `match_sensors` recomputed
twice for source-CSV cameras during full render

- **File:** `pixelpitch.py:1139` (`_load_per_source_csvs`) and
  `pixelpitch.py:644` (`merge_camera_data` existing-only branch).
- **Detail:** A camera that exists only in a per-source CSV but not
  in the merged Geizhals+source set may have `match_sensors`
  computed twice during one `render_html` invocation: once in
  `_load_per_source_csvs` (refresh against current sensors_db) and
  again in `merge_camera_data` (existing-only re-match). The
  computation is idempotent and cheap (~200 sensor comparisons), so
  this is a wasteful-but-correct double-call, not a bug.
- **Severity:** LOW. **Confidence:** MEDIUM.
- **Disposition:** Defer (informational, ≤2x of an already-cheap
  operation; same class as F49-04). Re-open if render time
  exceeds 30s.

## Carry-over deferred

F49-04, F55-PR-01..03, F56-PR-04, F57-PR-01..03, F59-PR-01 — all
informational, no thresholds crossed.

## Summary

No actionable perf findings for cycle 60.
