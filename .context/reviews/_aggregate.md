# Aggregate Review (Cycle 50) — Deduplicated, Merged Findings

**Date:** 2026-04-29
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-49 Status

All previous fixes confirmed still working. Both gates pass at HEAD = `ed45eed`:
- `flake8` — 0 errors (now also enforced in CI by `.github/workflows/github-pages.yml`)
- `python3 -m tests.test_parsers_offline` — PASS

No regressions detected. Cycle 49's headline finding F49-01 (CI flake8 enforcement) is fully resolved (commit `4a10b4d`).

## Cycle 50 New Findings

### F50-01 (consensus): `git pull --rebase || true` still swallows rebase failures — LOW / HIGH

- **Flagged by:** code-reviewer (carry-forward of F49-02), critic, verifier
- **File:** `.github/workflows/github-pages.yml`, line 108
- **Detail:** The defensive `|| true` after `git pull --rebase` masks a true rebase conflict. Although the subsequent `git push` will fail noisily on a non-fast-forward, the audit trail in workflow logs is muddled: the rebase-failed step shows green, then the push step shows red, making it harder for an operator to identify the root cause.
- **Failure scenario:** Two simultaneous workflow runs (e.g. manual dispatch coinciding with the monthly cron, or a manual master commit during the workflow window) produce non-fast-forward divergence. The rebase step appears to succeed, then push fails with a misleading error, costing operator-debug time.
- **Fix:** Replace `git pull --rebase || true` with one of:
  - `git pull --rebase` (let conflicts fail the step explicitly), or
  - `git pull --rebase || { echo "::error::rebase failed"; exit 1; }`
- **Confidence:** HIGH (logic is unambiguous)
- **Severity:** LOW (idempotent monthly workflow, low frequency of concurrent commits)

### F50-02: Per-agent review files were not refreshed in cycle 49 commit

- **Flagged by:** verifier
- **File:** `.context/reviews/*.md` (all 11 per-agent files)
- **Detail:** The cycle-49 aggregate references findings F49-01 through F49-11, but per-agent files were only lightly updated. This is a process-hygiene issue, not a bug. Re-running the full fan-out each cycle ensures the per-agent files stay in lockstep with the aggregate.
- **Fix:** This cycle (50) refreshes all per-agent review files alongside the aggregate.
- **Confidence:** HIGH
- **Severity:** N/A (process)

### F50-03: `parse_existing_csv` matched_sensors splits on `";"` without escape — LOW / MEDIUM

- **Flagged by:** code-reviewer, debugger
- **File:** `pixelpitch.py:373` and `pixelpitch.py:920-922`
- **Detail:** `write_csv` joins `matched_sensors` with `;` and `parse_existing_csv` splits on the same delimiter. There is no escaping rule — if a sensor name ever contains a literal `;` (e.g. a future "IMX455;v2" datapoint, or imported third-party data), the round-trip would silently corrupt the matched-sensors list. Currently all sensor names in `sensors.json` are alphanumeric so this is theoretical.
- **Fix:** Either (a) document the contract (sensor names MUST NOT contain `;`) and add an assertion in `write_csv`, or (b) switch to a delimiter unlikely to appear in sensor names (e.g. `|`) with a one-shot migration.
- **Confidence:** MEDIUM (no current trigger, but no defense)
- **Severity:** LOW (cosmetic data-loss; matched_sensors is a hint, not authoritative)

### F50-04: No automated test of write_csv → parse_existing_csv round-trip for matched_sensors — LOW / HIGH

- **Flagged by:** test-engineer
- **File:** `tests/test_parsers_offline.py`
- **Detail:** Existing tests cover write_csv outputs and parse_existing_csv inputs separately, and there is a recent matched_sensors merge-preservation test. There is no single test that exercises a write_csv → parse_existing_csv → write_csv round-trip on a row carrying multiple matched_sensors entries to confirm the join/split symmetry.
- **Fix:** Add a small round-trip test in `tests/test_parsers_offline.py` that constructs a `SpecDerived` with `matched_sensors=["IMX455", "IMX571"]`, writes it via `write_csv`, parses it back, and asserts the matched_sensors list is preserved verbatim (and ordered).
- **Confidence:** HIGH
- **Severity:** LOW (no current bug, but useful regression net for F50-03 and any future delimiter changes)

## Cross-Agent Agreement Matrix

| Finding | Flagged By                       | Highest Severity |
|---------|-----------------------------------|------------------|
| F50-01  | code-reviewer, critic, verifier  | LOW              |
| F50-02  | verifier                         | N/A (process)    |
| F50-03  | code-reviewer, debugger          | LOW              |
| F50-04  | test-engineer                    | LOW              |

## AGENT FAILURES

No agents failed.

## Summary Statistics

- Total distinct new findings: 4 (3 actionable, 1 process-hygiene)
- Cross-agent consensus findings (3+ agents): 1 (F50-01)
- Highest severity: LOW
- Actionable findings: 3
