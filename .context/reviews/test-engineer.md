# Test Engineer Review (Cycle 57)

**Reviewer:** test-engineer
**Date:** 2026-04-29
**HEAD:** `01c31d8`

## Coverage map

- `parse_existing_csv`: many sections cover BOM, header detection,
  matched_sensors round-trip, year/id parse tolerance.
- `merge_camera_data`: matched_sensors preservation (C46), tri-valued
  sentinel (F50-04).
- `_load_per_source_csvs`: refresh (C54), cache fallback (C55),
  size-less drop (C56). All branches now pinned.
- `derive_spec`: pixel_pitch sentinel (C40), TYPE_SIZE phone formats
  (C8).
- `match_sensors`: indirectly tested via the round-trip and refresh
  tests.

## New findings (cycle 57)

### F57-TE-01: no test pins `parse_existing_csv` `area` column round-trip with width/height — LOW

- **File:** `tests/test_parsers_offline.py`
- **Detail:** No section asserts that, when width and height are
  both present, the parsed `area` matches `width * height`. Adding
  this test will pin F57-CR-01's fix and prevent future regressions.
- **Severity:** LOW. **Confidence:** HIGH.

### F57-TE-02: `match_sensors` direct unit tests absent — LOW (gap)

- **File:** `tests/test_parsers_offline.py`
- **Detail:** The function is exercised indirectly through
  `_load_per_source_csvs` and `merge_camera_data` tests, but no
  direct unit test pins:
  - megapixel disagreement → empty match
  - megapixel agreement → match
  - tolerance edges (2% size, 5% mpix)
- **Severity:** LOW. **Confidence:** MEDIUM.
- **Disposition:** Defer — indirect coverage suffices. Schedule
  only if a new tolerance is added.

### F57-TE-03: F57-TE-01 fix can co-locate with the CSV round-trip section — INFO

- A 5-line addition to the existing `parse_existing_csv` round-trip
  test section is sufficient.

## Carry-over

- F32 monolith (test file size) — re-defer.

## Confidence summary

- 1 actionable LOW (F57-TE-01 area consistency test).
- 1 LOW deferred (F57-TE-02 direct match_sensors unit tests).
