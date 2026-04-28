# Test Engineer Review (Cycle 55)

**Reviewer:** test-engineer
**Date:** 2026-04-29
**HEAD:** `f08c3c4`

## Coverage map

`tests/test_parsers_offline.py` exercises pitch/area, TYPE_SIZE,
year/id/matched_sensors parse tolerance, merge preservation,
write_csv ↔ parse round-trip (incl. `;`), `_load_per_source_csvs`
refresh.

## Gaps

### F55-TE-01: no test for `_load_per_source_csvs` when sensors.json fails — LOW

- **File:** tests/test_parsers_offline.py
- **Detail:** No assertion of fallback behavior when
  `load_sensors_database` returns `{}`. F55-CRIT-01's chosen
  contract should be locked in by a test.
- **Severity:** LOW. **Confidence:** HIGH.

### F55-TE-02: no boundary tolerance test for `match_sensors` — LOW

- **File:** tests/test_parsers_offline.py
- **Detail:** Boundary at exactly 2% size / 5% mpix deviation
  untested.
- **Severity:** LOW. **Confidence:** HIGH.

### F55-TE-03: no direct BOM test for `parse_existing_csv` — LOW

- **File:** tests/test_parsers_offline.py
- **Detail:** Add a regression test that prepends `﻿` and asserts
  has_id detection still works.
- **Severity:** LOW. **Confidence:** HIGH.
