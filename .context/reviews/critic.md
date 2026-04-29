# Critic Review (Cycle 57)

**Reviewer:** critic
**Date:** 2026-04-29
**HEAD:** `01c31d8`

## Multi-perspective critique

After 56 cycles of fan-out review and incremental hardening, the
codebase has converged. Recent cycles added micro-tests for
edge cases (size-less branch, BOM has_id detection, year/id
parse tolerance). The marginal value of additional micro-fixes
is decreasing.

### F57-CRIT-01: parser-tests file is 2536 lines, monolith carried over from F55-CRIT-03 — LOW (deferred)

- **File:** `tests/test_parsers_offline.py`
- **Detail:** Test file continues to grow. Each cycle adds 10–30
  lines. At ~2536 lines a developer onboarding to the project must
  scroll-search to find the relevant section. Splitting by feature
  area (CSV parse, merge, _load_per_source_csvs, sensors) would
  improve maintainability.
- **Severity:** LOW. **Confidence:** MEDIUM.
- **Disposition:** Re-defer; mechanical refactor with test-rerun
  risk.

### F57-CRIT-02: `area` round-trip via CSV is independent of `width*height` round-trip — LOW

- **Files:** `pixelpitch.py:413-426`, `pixelpitch.py:996`
- **Detail:** Same finding as F57-CR-01 from a different angle:
  the CSV format encodes redundant information (width, height,
  area). The redundancy is useful for human eyeballing but creates
  a consistency hazard. Deserves a small fix in `parse_existing_csv`
  to recompute area when width and height are present.
- **Severity:** LOW. **Confidence:** HIGH.
- **Disposition:** Schedule as actionable plan in cycle 57.

### F57-CRIT-03: `match_sensors` partial-match disagreement is implicit — LOW

- Same as F57-CR-02. Comment-only fix worth one line.

### F57-CRIT-04: scope drift toward over-testing — INFO

- **Detail:** Recent cycles added increasingly narrow tests
  (size-less + sensors_db non-empty, BOM has_id, year tolerance,
  id tolerance, range guard). This is rigorous but each new
  test is now a 5-line micro-check. Consider declaring
  diminishing returns on parse-tolerance tests after cycle 56.
- **Disposition:** Note but no action required.

## Confidence summary

- 1 actionable LOW (F57-CRIT-02 area consistency, overlaps F57-CR-01).
- 2 LOW deferred (F57-CRIT-01 monolith, F57-CRIT-03 comment).
- 1 INFO (F57-CRIT-04 over-testing scope).
