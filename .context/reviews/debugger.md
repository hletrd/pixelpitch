# debugger Review (Cycle 51)

**Date:** 2026-04-29
**HEAD:** 3b35dcc

## Latent bug surface scan

### F51-D-01 (paired with code-reviewer F51-01): Whitespace tokens in `matched_sensors` — LOW / MEDIUM
- **File:** `pixelpitch.py:373`
- **Failure mode:** Hand-edited CSV with `IMX455; IMX571` produces a stray ` IMX571` token
  that round-trips. No crash, no data loss, but visible in any future template that surfaces
  matched_sensors text.
- **Severity:** LOW
- **Confidence:** MEDIUM

### F51-D-02: `parse_existing_csv` does not deduplicate `matched_sensors` — LOW
- **File:** `pixelpitch.py:373`
- **Detail:** If a CSV row contained `IMX455;IMX455`, the parser produces two identical
  entries. `match_sensors` returns sorted unique entries (line 253), so duplicates only
  arise via external CSV editing. Currently latent.
- **Fix:** Either dedup in `parse_existing_csv` (`list(dict.fromkeys(...))`) or accept
  current behavior since `match_sensors` produces unique values.
- **Confidence:** MEDIUM
- **Severity:** LOW (no current trigger)

## No high-severity latent bugs identified this cycle.
