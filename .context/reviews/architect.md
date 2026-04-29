# Architect Review (Cycle 59, orchestrator cycle 12)

**Date:** 2026-04-29
**HEAD:** `fa0ae66`

## Architectural posture

Repo continues to be a single-file CLI (`pixelpitch.py`,
1448 lines) plus modular sources (`sources/*.py`). The
write-csv contract is the durable boundary between consecutive
CI runs.

## New findings

### F59-A-01 (defensive-parity, LOW): write-csv contract enforcement is split between derive/parse and write

- **File:** `pixelpitch.py:1000-1047` (write_csv) vs.
  `pixelpitch.py:866-930` (derive_spec) / `pixelpitch.py:352-480`
  (parse_existing_csv)
- **Severity:** LOW. **Confidence:** HIGH (architectural
  hygiene, not a bug).
- **Detail:** The CSV artifact's float-cell contract - "no
  inf, no nan, no zero, no negative" - is enforced for area,
  mpix, and pitch directly at the write boundary (lines
  1020-1022) but for width and height only via upstream
  producers (derive_spec line 900, parse_existing_csv line
  430-433). This split-enforcement model means a contract
  violation in width/height could slip through if either
  upstream path regresses or a new producer is added that
  doesn't go through them.
- **Recommended fix:** co-locate the contract enforcement at
  `write_csv`, mirroring the area/mpix/pitch guards. This is
  the same pattern as F40 (write_csv finite-guard for pitch /
  mpix). Same scope as F40 fix; LOW risk.

## Carry-over (still deferred)

- F32 monolith - repo policy.
- F58-A-02 argparse drift - repo policy.
- F58-A-02 / F57-A-02 / F56-A-02 category list duplication -
  repo policy.

No new HIGH/CRITICAL architectural risks.
