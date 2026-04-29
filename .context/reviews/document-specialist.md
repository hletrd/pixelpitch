# Document-Specialist Review (Cycle 58, orchestrator cycle 11)

**Date:** 2026-04-29
**HEAD:** `aef726b`

## Doc/code consistency

- `README.md`: enumerates generated HTML pages — matches
  `render_html` output (8 categories + index + about + CSV).
- `pixelpitch.py --help`: matches CLI parsing for `html`,
  `source`, `list`.
- `parse_existing_csv` docstring (cycle 57): accurately
  describes the F57-01 area trust contract.

## New findings

### F58-DOC-01: `--help` does not document `--limit` constraints — LOW

- **File:** `pixelpitch.py:1422-1426`
- **Detail:** The `--help` text says
  `source <name> [--limit N] [--out DIR]` without stating
  that N must be a positive integer. After the F58-CR-01 fix
  the help text should reflect the validated input range.
- **Severity:** LOW. **Confidence:** HIGH.
- **Fix:** one-line update to the help string.

### F58-DOC-02 (deferred carry-over): `.context/plans/deferred.md` is growing past 25 entries

- Same as F56-DOC-03 / F57-DOC-03. Periodic sweep is hygiene,
  not a correctness concern.
- **Disposition:** keep deferred.

## Summary

One new documentation finding (F58-DOC-01). One carry-over
deferred.
