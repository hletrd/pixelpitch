# Security Review (Cycle 14) — OWASP Top 10, Secrets, Unsafe Patterns

**Reviewer:** security-reviewer
**Date:** 2026-04-28
**Scope:** Full repository security re-review after cycles 1-13 fixes

## Previously Fixed (Cycles 1-13) — Confirmed Resolved
- All SRI hashes present on all 7 CDN resources
- `data-name` attribute `|e` filter — FIXED
- All `target="_blank"` links have `rel="noopener noreferrer"` — FIXED
- `UnicodeDecodeError` now caught in both `load_csv` and `_load_per_source_csvs` — FIXED

## Deferred Items Still Valid
- C10-07: HTTP redirect chain SSRF risk — DEFERRED
- C10-08: Remote debugging port — DEFERRED
- F34: `importlib.import_module` with user-controllable input — DEFERRED

## New Findings

### S14-01: UTF-8 BOM in CSV enables denial-of-service against site generation
**File:** `pixelpitch.py`, lines 250-330
**Severity:** LOW (availability impact only, not data breach) | **Confidence:** HIGH

A malformed `camera-data.csv` with a UTF-8 BOM causes `parse_existing_csv` to produce 0 rows, effectively wiping the site of all camera data on the next build. This is a denial-of-service against the site generation pipeline.

While the threat model is limited (an attacker would need write access to the dist/ directory or the repository), the fix is trivial and the risk is real for developer workflow (Excel saves with BOM by default).

**Fix:** Strip BOM at the entry point of `parse_existing_csv`.

---

## Summary
- NEW findings: 1 (1 LOW)
- S14-01: BOM enables site-generation DoS — LOW
- No security regressions
- Deferred items remain appropriate
