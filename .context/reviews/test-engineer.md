# Test Engineer — Cycle 50

**Date:** 2026-04-29
**HEAD:** `ed45eed`

## Inventory

- `tests/test_parsers_offline.py` (2096 LOC) — single-file offline suite
- `tests/test_sources.py` (111 LOC) — source-module sanity
- `tests/fixtures/` — captured HTML samples

## Findings

### F50-04 — No round-trip test for `write_csv` → `parse_existing_csv` matched_sensors (LOW / HIGH)
- File: `tests/test_parsers_offline.py`
- `write_csv` joins `matched_sensors` with `;` (`pixelpitch.py:920-922`). `parse_existing_csv` splits on `;` (`pixelpitch.py:373`). Each side is tested independently, but no test asserts the round-trip preserves the list verbatim, including ordering and multi-element cases.
- Add a small round-trip test that:
  1. Builds a `SpecDerived` with `matched_sensors=["IMX455", "IMX571", "IMX989"]`.
  2. Calls `write_csv` to a `tempfile.NamedTemporaryFile`.
  3. Reads the file back with `parse_existing_csv`.
  4. Asserts `parsed[0].matched_sensors == ["IMX455", "IMX571", "IMX989"]`.

## Confirmations
- write_csv non-finite float guards (cycle 40 fixes) covered by existing tests.
- matched_sensors merge-preservation (cycle 46) covered by existing tests.
- Sensor-size-from-type phone formats (cycle 8) covered.
- CSV parser BOM handling (cycle 14-15) covered.

## Summary

Single new finding (F50-04): a small round-trip regression net for matched_sensors. Test surface otherwise solid; the 2096-line offline suite exercises every parser and the merge pipeline thoroughly.
