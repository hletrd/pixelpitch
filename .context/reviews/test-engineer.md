# Test Engineer Review — Cycle 48

**Date:** 2026-04-29
**Reviewer:** test-engineer

## Inventory

- `tests/test_parsers_offline.py` (2098 lines) — main offline test suite
- `tests/test_sources.py` (111 lines) — registry & dispatch tests

## New Findings (Cycle 48)

### F48-TEST-01: `merged2` unused in `tests/test_parsers_offline.py:1271`
- **Severity:** LOW | **Confidence:** MEDIUM
- **Why it's a problem:** F841. The variable's assigned but never asserted. Either the test omitted an assertion or the call is dead code.
- **Fix:** Inspect the surrounding test; either add an assertion comparing `merged2` to expectations, or drop the assignment.

### F48-TEST-02: Repeated unused `models.SpecDerived` / `models.Spec` imports
- **Severity:** LOW | **Confidence:** HIGH
- **Why it's a problem:** F401. Test scaffolding boilerplate that was copy-pasted with imports never used in that test function. Compiler/runtime cost is zero, but lint gate flags them.
- **Fix:** Remove unused imports.

### F48-TEST-03: Continuation-line indentation noise in `tests/test_parsers_offline.py`
- **Severity:** LOW | **Confidence:** HIGH
- **Why it's a problem:** Nine E127 errors indicate inconsistent indentation in multi-line literal calls. Doesn't affect behavior but is gate noise.
- **Fix:** Reformat to satisfy E127 (align continuation lines with the opening parenthesis or use hanging indent).

## Confirmation

- Test gate: PASS.
- All cycle 1–47 regression tests still pass.

## Confidence Summary

| Finding     | Severity | Confidence |
|-------------|----------|------------|
| F48-TEST-01 | LOW      | MEDIUM     |
| F48-TEST-02 | LOW      | HIGH       |
| F48-TEST-03 | LOW      | HIGH       |
