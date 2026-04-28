# Critic Review (Cycle 56)

**Reviewer:** critic
**Date:** 2026-04-29
**HEAD:** `e8d5414`

## Overview

55 cycles of fixes. C55-01 just landed (preserve matched_sensors
cache on sensors.json failure, add BOM regression test, README
update for smartphone/cinema pages). Both gates green. This cycle's
job is again to surface real issues without manufacturing churn.

## Findings

### F56-CRIT-01: `_load_per_source_csvs` and `merge_camera_data` still have separate refresh implementations — LOW (cleanup, gated)

- **File:** `pixelpitch.py:613-628` and `pixelpitch.py:1071-1087`
- **Detail:** Same root cause as F55-A-01 / F55-06: both branches
  now agree on the empty-db fallback (preserve cache). They disagree
  on the size-less branch (merge skips, per-source-load drops to
  None). The latter intentionally honors `derive_spec` sentinel.
  An extracted `_refresh_matched_sensors(d, sensors_db)` helper
  would still need a flag for the size-less behavior, so the
  reduction is small. Re-deferred per F55-06 rationale.
- **Severity:** LOW. **Confidence:** HIGH.

### F56-CRIT-02: `tests/test_parsers_offline.py` is now 2456 lines — LOW

- Same class as deferred F32 (`pixelpitch.py` monolith) and
  F55-CRIT-03. Test file growth (+120 lines from C55-01 BOM and
  cache-fallback tests) is expected. Not a bug.
- **Severity:** LOW. **Confidence:** HIGH.
- **Disposition:** Defer per F55-CRIT-03 / F32 rationale.

### F56-CRIT-03: cycle-counter inflation (orchestrator vs project cycle) is doc-confusing — INFORMATIONAL

- The orchestrator runs 100 cycles; the project has accumulated
  55+ named C-cycles. Plan IDs (`C55-01`, `C56-01`) are not the
  same as orchestrator cycle 9. Plan filenames remain the
  authoritative identifier. No fix needed; informational.

## No HIGH/CRITICAL findings.
