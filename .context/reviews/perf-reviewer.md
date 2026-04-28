# perf-reviewer Review (Cycle 51)

**Date:** 2026-04-29
**HEAD:** 3b35dcc

## Inventory

- Hot paths: `merge_camera_data` (O(N×S) sensor re-match for existing-only cameras),
  `match_sensors` (linear scan over sensor DB), `write_csv` (O(N) over specs),
  `parse_existing_csv` (O(N) row parse), template rendering (Jinja2 single pass).

## Findings

### Carry-forward (deferred; still applicable)

- F49-04: `merge_camera_data` re-runs `match_sensors` per existing-only camera. Linear
  sensor-DB scan ~1000 cameras × ~200 sensors. Acceptable at current scale.
- F40: `openmvg.fetch` re-fetches CSV each run; ~500KB; CI cache is not a concern.

### F51-P-01: `write_csv` matched_sensors filter is per-row Python loop — VERY LOW
- **File:** `pixelpitch.py:925-934`
- **Detail:** The new defensive filter iterates each element of `derived.matched_sensors`
  in pure Python. Even at N=4000 cameras × ~5 sensors each = ~20k iterations, this is sub-ms.
- **Severity:** none (informational)
- **Action:** No fix required.

## No new actionable performance findings this cycle.
