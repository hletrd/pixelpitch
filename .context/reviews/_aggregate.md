# Aggregate Review (Cycle 58, orchestrator cycle 11) — Deduplicated, Merged Findings

**Date:** 2026-04-29
**HEAD:** `aef726b` (after C57-01 plan-completed marker)
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic,
verifier, test-engineer, tracer, architect, debugger,
document-specialist, designer.

## Cycle 1–57 Status

All previous fixes confirmed still working at HEAD `aef726b`.
Both gates pass:

- `flake8 .` → 0 errors (also enforced in CI).
- `python3 -m tests.test_parsers_offline` → all sections green
  (including the C57-01 `parse_existing_csv area recomputed
  from width*height` section, with 9 new assertions).

No regressions. Cycle 57's findings (F57-01 area recomputation
+ F57-02 mpix-disagree comment) are fixed and verified.

## Cycle 58 New Findings

### F58-01 (BUG): `source` CLI accepts negative or zero `--limit` silently — LOW

- **Flagged by:** code-reviewer (F58-CR-01), critic
  (F58-CRIT-01), verifier (reproduced), tracer (F58-T-01),
  debugger (F58-D-01, F58-D-02).
- **File:** `pixelpitch.py:1393-1399`
- **Detail:** `int(args[i + 1])` accepts `-1`, `0`, and any
  negative integer without validation. Downstream consumers
  silently truncate / empty the result list:
  - `apotelyt.py:162` / `cined.py:127` / `gsmarena.py:243`:
    `urls[:limit]` slicing with negative limit drops trailing
    items; with `limit=0` returns empty.
  - `openmvg.py:73`: `if i >= limit: break` short-circuits
    immediately when `limit <= 0`, returning empty.
- **Repro (verifier):** `python pixelpitch.py source openmvg
  --limit -1` writes an empty CSV silently.
- **Fix:** add `if limit <= 0: print(...); sys.exit(1)` to
  the `--limit` branch in `main()`. Also update the `--help`
  string to document the positive-integer constraint
  (F58-DOC-01).
- **Severity:** LOW. **Confidence:** HIGH.
- **Cross-agent agreement:** 5 reviewers (HIGH signal).

### F58-02 (DOC): `--help` does not document `--limit` constraints — LOW

- **Flagged by:** document-specialist (F58-DOC-01).
- **File:** `pixelpitch.py:1422-1426`
- **Detail:** Help text says `[--limit N]` without stating
  N must be positive. Pair with F58-01 fix.
- **Severity:** LOW. **Confidence:** HIGH.

### F58-03 (test gap): no test for `--limit` validation — LOW

- **Flagged by:** test-engineer (F58-TE-01).
- **File:** `tests/test_parsers_offline.py` (gap)
- **Detail:** F58-01 fix needs a regression test pinning the
  validation behavior. Use a sub-test that calls `main()`
  with patched `sys.argv` and captures `SystemExit`.
- **Severity:** LOW. **Confidence:** HIGH.
- **Disposition:** Schedule alongside F58-01.

### F58-04 (deferred): `--out`/`--limit` consume value-arg without skipping loop counter — LOW

- **Flagged by:** critic (F58-CRIT-02), debugger
  (F58-D-03).
- **File:** `pixelpitch.py:1393-1401`
- **Detail:** typo `--out --limit` would set
  `out_dir = Path("--limit")`. Same root cause as the
  manual argparse drift in F58-A-02 (architectural).
- **Disposition:** Defer; covered by F58-A-02 architectural
  refactor when accepted.

### F58-05 (deferred, architectural): hand-coded argv parser drift — LOW

- **Flagged by:** architect (F58-A-02).
- **File:** `pixelpitch.py:1368-1431`
- **Detail:** `html` branch uses while+counter; `source`
  branch uses for+enumerate without counter advance. Drift
  between the two is the root of F58-01 / F58-04. Migrating
  to `argparse` (stdlib) would consolidate. Same class as
  F32 monolith.
- **Disposition:** Defer per repo policy.

### F58-06 (deferred): boundary tests at exactly 1900/2100/0/1_000_000 — LOW

- **Flagged by:** code-reviewer (F58-CR-02).
- **Disposition:** Defer (analogous to F55-02 boundary
  tolerance defer; indirect coverage suffices).

## Carry-over deferred (no action this cycle)

- F32 monolith, F55-CRIT-03 / F56-CRIT-02 / F57-CRIT-01
  test monolith.
- F49-04 perf, F55-PR-01..03 perf, F56-PR-04 / F57-PR-01..03
  informational.
- C10-07 redirects, C10-08 debug port (F58-SR-old).
- F35..F40 UI carry-overs (re-confirmed by designer).
- F55-A-02 / F56-A-02 / F57-A-02 / F58-A-02 category list
  duplication / argparse drift.
- F55-04 (existing_specs in-place mutation), F55-05
  (hand-edited blank-leading-cell defeats has_id).
- F56-DOC-03 / F57-DOC-03 / F58-DOC-02 (`deferred.md` size).
- F57-CR-03 (cosmetic comment redundancy in
  `_load_per_source_csvs`).
- F57-D-06 (semicolon-in-sensor-name parse-back).
- F58-CR-03 (`_load_per_source_csvs` ignores parsed
  record_id — informational, no fix).

## Cross-Agent Agreement Matrix

| Finding | Flagged By | Severity |
|---------|------------|----------|
| F58-01  | code-reviewer, critic, verifier, tracer, debugger | LOW (high signal — 5 agents) |
| F58-02  | document-specialist | LOW |
| F58-03  | test-engineer | LOW |
| F58-04  | critic, debugger | LOW (defer) |
| F58-05  | architect | LOW (defer) |
| F58-06  | code-reviewer | LOW (defer) |

## AGENT FAILURES

No agents failed.

## Summary statistics

- 11 reviewer perspectives executed.
- 6 findings produced this cycle:
  - 1 actionable bug (F58-01) flagged by 5/11 reviewers.
  - 1 actionable doc (F58-02), pairs with F58-01.
  - 1 actionable test (F58-03), pairs with F58-01.
  - 3 deferred per repo policy (F58-04 typo tolerance,
    F58-05 argparse refactor, F58-06 boundary tests).
- 0 new HIGH/CRITICAL findings.
- 0 regressions vs cycle 57.
