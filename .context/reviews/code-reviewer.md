# Code Reviewer — Cycle 49

**Date:** 2026-04-29
**Scope:** Full repository

## Inventory

- `pixelpitch.py` (1291 LOC)
- `models.py`, `sources/__init__.py`, `sources/{openmvg,digicamdb,imaging_resource,apotelyt,gsmarena,cined}.py`
- `tests/test_parsers_offline.py` (2096 LOC), `tests/test_sources.py` (111 LOC)
- `templates/{index,pixelpitch,about}.html`, `setup.cfg`, `.github/workflows/github-pages.yml`

## Findings

### F49-01: CI does not enforce flake8 — gate cleanup will silently regress (MEDIUM / HIGH)
- **File:** `.github/workflows/github-pages.yml`
- **Detail:** Cycle 48 fixed 33 flake8 errors and the orchestrator now lists flake8 as a gate. However, the CI workflow only runs `python -m tests.test_parsers_offline`. flake8 is not in the CI pipeline, so any new lint regression introduced via PR or scheduled run goes undetected until someone runs it locally.
- **Failure scenario:** A future PR or auto-update introduces F401/E303/etc. and merges to master. The next deploy succeeds. The lint regression accumulates until the next manual review-plan-fix cycle re-runs flake8.
- **Fix:** Add a `flake8` step after the offline tests step in `github-pages.yml`. Treat lint failures as workflow-blocking.
- **Confidence:** HIGH

### F49-02: `git pull --rebase || true` masks rebase failures in CI (LOW / HIGH)
- **File:** `.github/workflows/github-pages.yml`, line 100
- **Detail:** The CI commit step uses `git pull --rebase || true` followed by `git push`. If the rebase fails (e.g., conflicting `[skip ci]` updates), the failure is swallowed and the push attempts a non-fast-forward.
- **Fix:** Remove `|| true`, or replace with explicit error reporting. Low priority — workflow runs monthly, idempotent.
- **Confidence:** HIGH (logic), LOW (impact)

## Summary

After 48 prior review cycles, the codebase is in solid shape:
- All gates pass (flake8 0 errors, tests.test_parsers_offline PASS).
- No new logic bugs found by direct inspection.
- The remaining findings are infrastructure (CI gate enforcement) rather than code bugs.
