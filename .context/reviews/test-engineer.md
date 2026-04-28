# test-engineer Review (Cycle 52)

**Date:** 2026-04-29
**HEAD:** 331c6f5

## Coverage map

The offline gate (`tests/test_parsers_offline.py`, ~2206 LOC) covers:

- pixel_pitch sentinel handling
- TYPE_SIZE phone formats
- write_csv finite/positive guards
- write_csv → parse_existing_csv round-trip for matched_sensors
- write_csv guard against `;` in matched_sensors
- parse_existing_csv whitespace + dedup tolerance for matched_sensors
- merge_camera_data matched_sensors preservation

## Gap identified

### F52-04: No parse-tolerance test for `year_str = "2023.0"` — LOW

- **File:** `tests/test_parsers_offline.py` (gap)
- **Detail:** All current parse-tolerance tests exercise the
  `matched_sensors` column. The F52-01 fix needs an accompanying test
  asserting that `year_str ∈ {"2023", " 2023 ", "2023.0", "2023.5",
  "abc", ""}` parses to `{2023, 2023, 2023, None, None, None}`
  (`2023.5` is rejected because it cannot represent a calendar year
  cleanly; or accept `2023` if rounding is acceptable — the
  implementation will pick the conservative path).
- **Severity:** LOW
- **Confidence:** HIGH (companion to F52-01)

## No flaky tests detected.

The offline gate is fully deterministic (fixture-based).
