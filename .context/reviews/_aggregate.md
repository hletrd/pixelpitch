# Aggregate Review (Cycle 45) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-44 Status

All previous fixes confirmed still working. No regressions in core logic. Gate tests pass. C44 fixes (FORMAT_TO_MM dead code removal, fmt extraction removal) verified working correctly.

## Cross-Agent Agreement Matrix (Cycle 45 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| GSMArena _select_main_lens regex split breaks on decimal MP | CR45-01, CRIT45-01, V45-01, TR45-01, ARCH45-01, DBG45-01 | HIGH |
| No test for decimal MP in _select_main_lens | TE45-01 | MEDIUM |
| No test for decimal MP in _phone_to_spec | TE45-02 | MEDIUM |

## Deduplicated New Findings (Ordered by Severity)

### C45-01: GSMArena _select_main_lens regex split breaks on decimal MP values — data corruption

**Sources:** CR45-01, CRIT45-01, V45-01, TR45-01, ARCH45-01, DBG45-01
**Severity:** HIGH | **Confidence:** HIGH (6-agent consensus, verified by live execution)

The `re.split(r'(?=\b\d+(?:\.\d+)?\s*MP\b)', raw)` in `_select_main_lens` (gsmarena.py line 82) uses `\b` (word boundary) before `\d+`. The word boundary fires between a digit and a decimal point, causing "12.2 MP" to be split into "12." and "2 MP, ...". The function then selects the corrupted "2 MP" fragment as the main lens, extracting mpix=2.0 instead of 12.2 and losing the sensor type designation.

This affects all phones with decimal-megapixel main cameras (Google Pixel 1-6 at 12.2 MP, various Samsung/Apple/OnePlus phones). The bug causes three-fold data corruption:

1. **mpix wrong**: 12.2 becomes 2.0
2. **sensor type lost**: TYPE_FRACTIONAL_RE may fail to match the format designation in the corrupted fragment
3. **derived.size wrong**: Without spec.type, derive_spec cannot compute sensor dimensions

**Fix:** Remove `\b` from the start of the split regex: `r'(?=\d+(?:\.\d+)?\s*MP\b)'`

---

### C45-02: No test coverage for decimal MP values in GSMArena _select_main_lens and _phone_to_spec

**Sources:** TE45-01, TE45-02
**Severity:** MEDIUM | **Confidence:** HIGH

The existing GSMArena tests only exercise integer MP values (200, 10, 50, 50 from the S25 Ultra fixture). No test covers decimal MP values like 12.2 MP, which is the exact input that triggers C45-01. Adding test coverage would have caught this bug earlier and will prevent regression.

**Fix:** Add test cases for `_select_main_lens` and `_phone_to_spec` with decimal MP camera values.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 2 (C45-01, C45-02)
- Cross-agent consensus findings (3+ agents): 1 (C45-01 with 6 agents)
- Highest severity: HIGH (C45-01)
- Actionable findings: 2
- Verified safe / deferred: 0
