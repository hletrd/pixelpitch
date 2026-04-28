# Aggregate Review (Cycle 52) — Deduplicated, Merged Findings

**Date:** 2026-04-29
**HEAD:** 331c6f5
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier,
test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-51 Status

All previous fixes confirmed still working. Both gates pass at HEAD = `331c6f5`:

- `flake8 .` — 0 errors (also enforced in CI by `.github/workflows/github-pages.yml`)
- `python3 -m tests.test_parsers_offline` — PASS

No regressions detected. Cycle 51 findings F51-01 / F51-02 are fully resolved
(commits `a0ac8bc`, `d1b0ca1`).

## Cycle 52 New Findings

### F52-01 (consensus): `parse_existing_csv` rejects `2023.0`-style year strings — LOW / MEDIUM

- **Flagged by:** code-reviewer, verifier, tracer, debugger, test-engineer
- **File:** `pixelpitch.py:366-372`
- **Detail:** `int(year_str)` raises ValueError on `"2023.0"`. The except branch
  silently drops the year. Same class as F51-01: defense-in-depth against Excel
  hand-edit of `dist/camera-data.csv`. `write_csv` emits clean integer years
  today, so no current internal trigger.
- **Failure scenario:** Maintainer opens `dist/camera-data.csv` in Excel → makes
  a small edit → saves → CI re-renders → year column blanks for every edited
  row.
- **Fix:** Try `int(year_str)` first; on `ValueError`, fall back to
  `int(float(year_str))`; keep the 1900-2100 range guard. Add a parse-tolerance
  test in `tests/test_parsers_offline.py`.
- **Confidence:** MEDIUM
- **Severity:** LOW (no current trigger; defense-in-depth alongside cycle-51
  parse-tolerance change)

### F52-02 (related): `record_id` parsing has the same int-vs-float vulnerability — LOW

- **Flagged by:** debugger
- **File:** `pixelpitch.py:319`
- **Detail:** `int(values[0])` raises on `"5.0"`. The broad except at line 390
  catches it and SKIPS the row entirely. Recoverable because
  `merge_camera_data` regenerates ids (line 559-560), but more disruptive
  than the year case — the entire row is lost on parse, and any data not
  also produced by the new fetch is gone.
- **Fix:** Pair with F52-01 in the same comprehension: `int(values[0]) if
  values[0] else None` becomes `_safe_int(values[0])` where `_safe_int`
  tolerates `"5"`, `"5.0"`, `" 5 "` and rejects everything else.
- **Confidence:** MEDIUM
- **Severity:** LOW (recoverable)

### F52-03: per-agent reviews and aggregate uncommitted at cycle start — LOW (process)

- **Flagged by:** code-reviewer, critic
- **File:** `.context/reviews/*.md`
- **Detail:** All 12 review files were modified-but-uncommitted in the working
  tree at cycle start. The cycle's docs commit must include the refreshed
  snapshots.
- **Fix:** Process-only. Commit refreshed reviews + aggregate + plan in the
  cycle's docs commit, matching cycle-51's `331c6f5` pattern.
- **Severity:** LOW (process)
- **Confidence:** HIGH

### F52-04: no parse-tolerance test for `year_str = "2023.0"` — LOW

- **Flagged by:** test-engineer
- **File:** `tests/test_parsers_offline.py` (gap)
- **Detail:** All current parse-tolerance tests exercise `matched_sensors`. The
  F52-01 fix needs an accompanying test for the year column.
- **Fix:** Add a synthetic-CSV test that asserts `year_str ∈ {"2023",
  " 2023 ", "2023.0"}` all parse to `2023`, while `"abc"` and `""` parse
  to None.
- **Severity:** LOW
- **Confidence:** HIGH (companion to F52-01)

### F52-DS-01: docstring/comment update for parse_existing_csv year tolerance — LOW

- **Flagged by:** document-specialist
- **File:** `pixelpitch.py:285-291` and inline comment near line 366
- **Detail:** Cosmetic; gate-bound to F52-01 implementation.
- **Severity:** LOW (cosmetic)
- **Confidence:** HIGH

## Cross-Agent Agreement Matrix

| Finding | Flagged By                                                      | Highest Severity |
|---------|-----------------------------------------------------------------|------------------|
| F52-01  | code-reviewer, verifier, tracer, debugger, test-engineer        | LOW / MEDIUM     |
| F52-02  | debugger                                                        | LOW              |
| F52-03  | code-reviewer, critic                                           | LOW (process)    |
| F52-04  | test-engineer                                                   | LOW              |
| F52-DS-01 | document-specialist                                           | LOW (cosmetic)   |

## AGENT FAILURES

No agents failed.

## Summary statistics

- Total distinct new findings: 5 (3 actionable code/test/docs, 1 process,
  1 paired latent)
- Cross-agent consensus findings (3+ agents): 1 (F52-01)
- Highest severity: LOW (one MEDIUM-confidence)
- Actionable findings: 4 (F52-01 + companion test F52-04 + companion docs
  F52-DS-01 + paired record_id F52-02)
