# Test Engineer Review (Cycle 56)

**Reviewer:** test-engineer
**Date:** 2026-04-29
**HEAD:** `e8d5414`

## Coverage map

`tests/test_parsers_offline.py` (2456 lines) exercises:
- pitch/area, TYPE_SIZE, year/id/matched_sensors parse tolerance
- merge preservation, write_csv ↔ parse round-trip (incl. `;`)
- `_load_per_source_csvs` refresh (C54-01) and cache fallback (C55-01)
- BOM has_id detection regression (C55-01 / F55-03)

## Gaps

### F56-TE-01: no test for `_load_per_source_csvs` size-less rows — LOW

- **File:** tests/test_parsers_offline.py
- **Detail:** When a per-source CSV row has no size (width=height=None
  parsed from empty cells), `_load_per_source_csvs` forces
  `matched_sensors = None`. No test pins this. Could regress
  silently if the size-less branch logic is refactored.
- **Severity:** LOW. **Confidence:** HIGH.

### F56-TE-02: no boundary test for `match_sensors` (carry-over) — LOW

- Carried over from F55-TE-02. Boundary at exactly 2% size / 5%
  mpix deviation untested. Already deferred (see deferred.md F55-02).

### F56-TE-03: no test for `_load_per_source_csvs` row with empty cache string and no sensors_db — LOW

- **File:** tests/test_parsers_offline.py
- **Detail:** A per-source row whose matched_sensors column is
  empty (`""`) and sensors.json is missing would today have
  `matched_sensors = []` (empty list from `parse_existing_csv`'s
  semicolon split with strip+dedup). The cache-preservation branch
  preserves `[]`. This is correct but untested. Adding the test
  pins the contract.
- **Severity:** LOW. **Confidence:** MEDIUM.
