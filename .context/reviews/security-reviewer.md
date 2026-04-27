# Security Review (Cycle 15) — OWASP Top 10, Secrets, Unsafe Patterns

**Reviewer:** security-reviewer
**Date:** 2026-04-28
**Scope:** Full repository security re-review after cycles 1-14 fixes

## Previously Fixed (Cycles 1-14) — Confirmed Resolved
- All SRI hashes present on all 7 CDN resources
- `data-name` attribute `|e` filter — FIXED
- All `target="_blank"` links have `rel="noopener noreferrer"` — FIXED
- `UnicodeDecodeError` now caught in both `load_csv` and `_load_per_source_csvs` — FIXED
- BOM defense in `parse_existing_csv` — FIXED (C14-02)

## Deferred Items Still Valid
- C10-07: HTTP redirect chain SSRF risk — DEFERRED
- C10-08: Remote debugging port — DEFERRED
- F34: `importlib.import_module` with user-controllable input — DEFERRED

## New Findings

### S15-01: openMVG CSV fetch has no BOM defense — remote data tampering could cause 0-record parse
**File:** `sources/openmvg.py`, lines 52-56
**Severity:** LOW | **Confidence:** HIGH

Same issue as C15-05 from code-reviewer perspective. If the openMVG GitHub repository CSV is modified to include a BOM (either by maintainer or through repository compromise), the `csv.DictReader` would produce mangled field names, causing `KeyError` on every row and returning 0 records silently. The existing BOM defense only covers `parse_existing_csv`, not the `DictReader` path used by openMVG.

**Fix:** Strip BOM from the CSV body in `openmvg.fetch()` before passing to `DictReader`.

---

## Summary
- NEW findings: 1 (1 LOW)
- S15-01: openMVG CSV fetch lacks BOM defense — LOW
- No security regressions
- Deferred items remain appropriate
