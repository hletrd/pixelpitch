# Test Engineer Review (Cycle 31) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-30 fixes, focusing on NEW issues

## Previous Findings Status

TE30-01 (no test for GSMArena fetch() per-phone resilience) remains deferred — requires network access. All existing tests passing.

## New Findings

### TE31-01: No test for merge_camera_data spec.pitch/derived.pitch inconsistency

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

The existing `test_merge_field_preservation` test checks that `spec.pitch` and `derived.pitch` are both preserved when new has None for both. However, it does NOT test the scenario where new has `spec.pitch=None` but `derived.pitch` is computed (not None) because `spec.mpix` is set. This is the gap that allows the spec/derived pitch inconsistency (CR31-01, CRIT31-01) to persist.

**Fix:** Add a test case to `test_merge_field_preservation`:
- Existing: spec.pitch=2.0, derived.pitch=2.0
- New: spec with size and mpix but pitch=None, so derive_spec() computes derived.pitch from area+mpix
- Assert: after merge, derived.pitch should equal spec.pitch (the preserved authoritative value)

---

### TE31-02: No test for BOM handling with escape sequence vs literal

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

The existing BOM tests (`test_parse_existing_csv` BOM section and `test_openmvg_bom`) use a literal BOM character in the test string. This means the test passes regardless of whether the implementation uses a literal or escape sequence. However, the point of CR31-02 is that the literal in the *implementation* is fragile. The test cannot detect this fragility since the literal BOM in the test data itself is also fragile in the same way. No direct test fix is needed — the fix is in the implementation code.

---

## Summary

- TE31-01 (MEDIUM): No test for merge pitch inconsistency when derived.pitch is computed but spec.pitch is None
- TE31-02 (LOW): BOM literal fragility is in implementation, not testable from outside
