# Architect Review (Cycle 58, orchestrator cycle 11)

**Date:** 2026-04-29
**HEAD:** `aef726b`

## Layering check

- `models.py` → dataclasses, no I/O.
- `sources/*` → HTTP + parse, return `list[Spec]`.
- `pixelpitch.py` → orchestration, merge, render, CLI.
- `tests/*` → pure functions tested offline.

Layering remains clean. No new coupling introduced this cycle.

## Architectural risks

### F58-A-01 (carry-over of F32): `pixelpitch.py` 1437 LOC

- Same class as deferred F32. No correctness concern.
- **Disposition:** keep deferred.

### F58-A-02: CLI argument parsing reinvented manually — LOW (deferred)

- **File:** `pixelpitch.py:1368-1431`
- **Detail:** `main()` rolls a hand-coded argv parser for
  `html`, `source`, `list`. Each branch handles flag-value
  pairing differently (the `html` branch uses a
  while-with-counter; the `source` branch uses
  `for i, a in enumerate(args)` and consumes `args[i+1]` but
  does not advance `i`). The patterns drift, opening the
  door to F58-CR-01 (negative `--limit`) and F58-CRIT-02
  (`--out --limit` typo).
- **Severity:** LOW. **Confidence:** HIGH (architectural).
- **Fix:** migrate to `argparse`. Standard library, no new
  dependency. Adds ~30 LOC, removes ~50 LOC of hand-coded
  parsing, and produces argparse-standard `--help` output.
- **Disposition:** defer per repo policy (refactor risk
  in a file already flagged monolithic; same class as F32).
  Note in deferred.md.

## Carry-over deferred (no action this cycle)

- F32 monolith.
- F55-A-02 / F56-A-02 / F57-A-02: render_html category list
  duplication.

## Summary

One new architectural finding (F58-A-02), deferred. No new
required refactors this cycle.
