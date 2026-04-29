# Verifier Review (Cycle 57)

**Reviewer:** verifier
**Date:** 2026-04-29
**HEAD:** `01c31d8`

## Evidence-based correctness check

### Gate evidence

- `python3 -m flake8 .` exit 0 (no diagnostics).
- `python3 -m tests.test_parsers_offline` — all sections green,
  including:
  - `_load_per_source_csvs refresh against sensors.json` (C54-01)
  - `_load_per_source_csvs cache fallback when sensors.json missing` (C55-01)
  - `_load_per_source_csvs size-less row drops cache (sensors_db non-empty)` (C56-01, NEW)
  - `parse_existing_csv BOM has_id detection` (C55-01)
  - all matched_sensors preservation tests (C46)

### Behavior verification

- `derive_spec` "size unknown means matched_sensors is None"
  contract: VERIFIED (lines 899-906).
- `_load_per_source_csvs` matches contract: VERIFIED (lines
  1072-1088). Test C56-01 pins it.
- `merge_camera_data` matched_sensors preservation: VERIFIED (lines
  595-596). Test C46 pins it.
- `parse_existing_csv` matched_sensors round-trip: VERIFIED (lines
  441-445). Test C50, C51 pin it.

### F57-V-01: write_csv area-emission predicate matches parse-back acceptance — VERIFIED

- **Files:** `pixelpitch.py:998` (write), `pixelpitch.py:425-426` (parse).
- **Detail:** write_csv emits area only when `derived.area is not
  None and isfinite(derived.area) and derived.area > 0`. parse
  rejects `area <= 0`. Both reject NaN/inf via `_safe_float`.
  Round-trip is consistent for valid inputs.
- **Verdict:** VERIFIED. No action.

### F57-V-02: F57-CR-01 (area consistency) reproducible — VERIFIED

- **Repro steps:**
  1. Construct CSV row: `1,Foo,dslr,,23.6,15.6,999.0,24.0,3.85,2020,`
  2. Call `parse_existing_csv`.
  3. Inspect `result[0].area`. Expected: 23.6 * 15.6 = 368.16.
     Actual: 999.0.
- **Verdict:** Confirmed reproducible. F57-CR-01 is a real bug
  surface (not a flaw in tests).

## Confidence summary

- All gate-level invariants verified.
- F57-CR-01 reproduction confirms it as real LOW finding.
- 0 critical/high findings unverified.
