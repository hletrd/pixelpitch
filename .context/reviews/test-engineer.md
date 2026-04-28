# Test Engineer Review (Cycle 38) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-37 fixes

## Previous Findings Status

TE37-01 and TE37-02 both implemented. NaN area tests confirmed working.

## New Findings

### TE38-01: `test_template_zero_pitch_rendering` tests the wrong behavior — should test that 0.0 pitch renders as "unknown"

**File:** `tests/test_parsers_offline.py`, lines 496-523
**Severity:** LOW | **Confidence:** HIGH

The test `test_template_zero_pitch_rendering` asserts that 0.0 pitch renders as "0.0 µm" and 0.0 mpix renders as "0.0 MP". However, a 0.0 µm pixel pitch is physically impossible — the correct rendering should be "unknown" (same as None). The test is asserting the wrong expected behavior.

After fixing the template to render "unknown" for 0.0 pitch, this test needs to be updated to expect "unknown" instead of "0.0 µm". The 0.0 mpix case is less clear — 0.0 MP is also physically impossible, but the mpix field is less critical. I recommend fixing both to "unknown" for consistency.

**Fix:** Update `test_template_zero_pitch_rendering` to verify:
- 0.0 pitch renders as "unknown" (not "0.0 µm")
- 0.0 mpix renders as "unknown" (not "0.0 MP")

---

## Summary

- TE38-01 (LOW): `test_template_zero_pitch_rendering` tests wrong behavior — should expect "unknown" for 0.0 values
