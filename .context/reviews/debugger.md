# Debugger Review (Cycle 58, orchestrator cycle 11)

**Date:** 2026-04-29
**HEAD:** `aef726b`

## Latent bug surface

### F58-D-01: `--limit -1` silent no-op (HIGH-confidence latent bug)

- Same as F58-CR-01 / F58-CRIT-01 / F58-T-01.
- Failure mode: the user runs the CLI, sees a zero-exit
  status, finds an empty CSV in `dist/`, and assumes the
  scrape failed silently. Wastes investigation time.
- Fix: input validation. One-line guard.

### F58-D-02: `--limit` value not enforced as positive integer

- `int("0")` → `0`, returns empty list (silent no-op).
- `int("-100")` → `-100`, slices `urls[:-100]` (drops last
  100 items).
- Fix: `if limit <= 0: print(...); sys.exit(1)`.

### F58-D-03 (theoretical, deferred): `--out` path validation

- **File:** `pixelpitch.py:1400-1401`
- **Detail:** `--out --limit` (typo) sets `out_dir =
  Path("--limit")`. The script will then try to create a
  directory named `--limit/`, which works on POSIX but is
  user-hostile. Same root cause as F58-A-02 (manual argparse
  drift).
- **Severity:** LOW. **Confidence:** MEDIUM.
- **Disposition:** defer (covered by F58-A-02 architectural
  refactor when accepted).

### F58-D-04: F57-D-06 semicolon-in-sensor-name still deferred

- Carry-over. No action.

## No other regressions

All 25+ test sections pass. The C57-01 fix has not introduced
any new latent failure mode.

## Summary

Two actionable latent findings (F58-D-01, F58-D-02 — both
collapsed into the F58-CR-01 fix). One deferred (F58-D-03).
