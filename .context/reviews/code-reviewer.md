# Code Reviewer — Cycle 60 (Orchestrator Cycle 13)

**Date:** 2026-04-29
**HEAD:** `a0cd982`
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

All cycle 1-59 fixes confirmed in place. Both gates pass:

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

## Cycle 60 New Findings

### F60-CR-01 (LOW, defensive parity gap): `_load_per_source_csvs`
has no per-source try/except wrapping `parse_existing_csv`

- **File:** `pixelpitch.py:1132`
- **Detail:** `parse_existing_csv(content)` is wrapped only by a
  per-row `try/except Exception` *inside* itself. If
  `csv.reader(io.StringIO(...))` ever raises a top-level exception
  (e.g. on a corrupt cache file), the surrounding
  `_load_per_source_csvs` has no per-source try/except wrapping the
  call. The exception propagates up through `render_html`, killing
  the build. Today `csv.reader` is permissive enough that this is
  largely theoretical, but the docstring promises "Missing files are
  silently skipped — failure of one source must not block the
  build" — which is broken if `parse_existing_csv` itself raises.
- **Severity:** LOW. **Confidence:** LOW (theoretical at present).
- **Disposition:** Defer (no observed failure mode; defensive parity
  with the docstring contract).

### F60-CR-02 (informational): `cined.fetch` ImportError handling
correctly returns []

- **File:** `sources/cined.py:114-121`
- **Detail:** Confirmed working. Non-finding.

## Carry-over from cycles 1-59

All previously-actionable findings either fixed or deferred per
`deferred.md`. No regressions observed at HEAD.

## Confidence

- HIGH: cycles 48-59 fixes still in place.
- HIGH: gates green.
- LOW: any new actionable defect this cycle.

## Summary

No new actionable findings for cycle 60. Repository at steady-state
post-C59-01.
