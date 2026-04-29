# Code Reviewer — Cycle 61 (Orchestrator Cycle 14)

**Date:** 2026-04-29
**HEAD:** `a781933`
**Scope:** Full repository review for code quality, logic, SOLID,
maintainability.

## Inventory

- `pixelpitch.py` (1488 lines)
- `models.py` (27 lines)
- `sources/__init__.py`, `openmvg.py`, `apotelyt.py`,
  `imaging_resource.py`, `gsmarena.py`, `cined.py`, `digicamdb.py`
- `tests/test_parsers_offline.py` (2748 lines)
- `tests/test_sources.py` (111 lines)
- `templates/*.html`

## Status at HEAD

All cycle 1-60 fixes confirmed in place. Both gates pass:

- `flake8 .` -> 0 errors.
- `python3 -m tests.test_parsers_offline` -> all sections green.

Re-verified key invariants:

- `derive_spec` (line 900) filters non-finite/non-positive size.
- `parse_existing_csv` (line 430-433) rejects non-positive width/height.
- `write_csv` (line 1052-1062) hardens all five numeric columns
  including width/height per F59-01.
- `merge_camera_data` matched_sensors tri-valued contract intact
  (line 616-617).
- `_safe_year`, `_safe_int_id` reject out-of-range values.

## Cycle 61 New Findings

### F61-CR-01 (LOW, by-design): CSV `matched_sensors` column cannot
distinguish None vs [] — round-trip lossy

- **File:** `pixelpitch.py:462-466` (parse_existing_csv) and
  `pixelpitch.py:1069-1081` (write_csv).
- **Detail:** `derive_spec` documents a tri-valued sentinel for
  `matched_sensors` (None = "not checked", [] = "checked, found
  nothing", non-empty = matches). However the CSV format conflates
  the first two: write_csv emits `""` for both None and [], and
  parse_existing_csv reads `""` back as `[]`. After round-trip,
  the "not checked" sentinel is lost. Practical impact is nil:
  downstream consumers (template, write_csv) treat None and [] the
  same when displaying or writing back, and the test pins `[]` as
  the canonical post-parse value (see test_parsers_offline.py:691,
  701). Same class as F60-D-01 (Spec/SpecDerived size asymmetry).
- **Severity:** LOW. **Confidence:** HIGH (round-trip behavior
  is deterministic and tested).
- **Disposition:** Defer (no observable bug; documented behavior
  pinned by existing tests). Re-open if a future consumer needs to
  distinguish "never checked" from "checked, empty" after CSV
  round-trip.

## Carry-over from cycles 1-60

All previously-actionable findings either fixed or deferred per
`deferred.md`. No regressions observed at HEAD.

## Confidence

- HIGH: cycles 48-60 fixes still in place.
- HIGH: gates green.
- LOW: any new actionable defect this cycle.

## Summary

No new actionable findings for cycle 61. Repository at steady-state
post-C59-01.
