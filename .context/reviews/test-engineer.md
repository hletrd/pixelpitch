# Test Engineer Review (Cycle 45) — Test Coverage Gaps, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28

## Previous Findings Status

TE44-01, TE44-02 — COMPLETED. test_cined_format_coverage removed after FORMAT_TO_MM removal.

## New Findings

### TE45-01: No test for GSMArena _select_main_lens with decimal MP values

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

The existing `test_gsmarena_select_main_lens` test only uses integer MP values (12 MP, 50 MP, 48 MP, 10 MP, 8 MP). There is no test case with decimal MP values like "12.2 MP" or "10.7 MP", which is exactly the input that triggers the regex split bug (CR45-01). A test with decimal MP would have caught this bug earlier.

**Fix:** Add test cases for `_select_main_lens` with decimal MP values:
- `'12.2 MP, f/1.9, (wide), 1/2.55", 1.25µm'` should select the full "12.2 MP" entry
- `'10.7 MP, f/4.3, 240mm (periscope)'` should select the full "10.7 MP" entry
- Multi-lens with decimal MP: `'50 MP, f/1.7, (wide)\n12.2 MP, f/2.2, (ultrawide)'` should pick the 50 MP wide lens

### TE45-02: No test for GSMArena _phone_to_spec with decimal MP

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

The existing GSMArena test uses the Galaxy S25 Ultra fixture which has integer MP values (200, 10, 50, 50). There is no test verifying that `_phone_to_spec` correctly extracts decimal MP values, sensor type, and pitch when the camera value contains decimal MP. This is a direct test gap for the CR45-01 bug.

**Fix:** Add a unit test for `_phone_to_spec` with synthetic fields containing decimal MP camera values. Verify mpix, type, and pitch are correctly extracted.

---

## Summary

- TE45-01 (MEDIUM): No test for GSMArena _select_main_lens with decimal MP values
- TE45-02 (MEDIUM): No test for GSMArena _phone_to_spec with decimal MP
