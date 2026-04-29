# Test-Engineer Review (Cycle 59, orchestrator cycle 12)

**Date:** 2026-04-29
**HEAD:** `fa0ae66`

## Coverage assessment

`tests/test_parsers_offline.py` is now 2659 lines (up from 2456
in cycle 56). Section count is high; gate runtime remains under
10s.

Existing write_csv guard tests:

- `test_write_csv_nonfinite_guards` (line 1952): pins
  `inf`/`nan` in mpix and pitch.
- `test_write_csv_zero_negative_guards` (line 1992): pins zero
  and negative mpix and pitch (and computed-pitch with both
  zero inputs).

Both tests focus on `mpix` and `pitch` columns. **Width/height
columns are not pinned for non-finite or non-positive values.**
This is the test-side mirror of the F59-CR-01 defensive-parity
gap.

## New findings

### F59-TE-01 (test gap, LOW): no test pins write_csv width/height non-finite/non-positive guards

- **File:** `tests/test_parsers_offline.py` (gap, paired with
  F59-CR-01 fix in `pixelpitch.py:1018-1019`)
- **Severity:** LOW. **Confidence:** HIGH.
- **Detail:** When the F59-CR-01 fix lands, a regression test
  must pin the new behavior. Recommended sub-tests in a new
  section `write_csv width/height non-finite/non-positive
  guards`:
  - `derived.size = (inf, 24.0)` -> CSV row's width and height
    cells empty.
  - `derived.size = (35.9, nan)` -> both empty.
  - `derived.size = (0.0, 0.0)` -> both empty.
  - `derived.size = (-1.0, -1.0)` -> both empty.
  - Sanity: `derived.size = (35.9, 23.9)` -> both populated as
    `"35.90"` and `"23.90"`.
  - Sanity: `derived.size = None` -> both empty.
- **Disposition:** Schedule alongside F59-CR-01.

## Carry-over

- F58-06 (boundary tests at exact range edges) - still deferred
  per F55-02 pattern.
- F56-CRIT-02 (test monolith) - still deferred (architectural).
