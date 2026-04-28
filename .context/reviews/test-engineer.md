# Test Engineer Review — Cycle 46

**Date:** 2026-04-28
**Reviewer:** test-engineer

## Previous Findings Status

TE45-01, TE45-02 — COMPLETED. Decimal MP tests added for _select_main_lens and _phone_to_spec.

## New Findings

### TE46-01: No test for matched_sensors preservation in merge_camera_data

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

The existing `test_merge_field_preservation` test verifies preservation of `type`, `size`, `pitch`, `mpix` from existing data when new has `None`. But there is no test for `matched_sensors` preservation. This gap allowed the matched_sensors merge bug (CR46-01) to go undetected through 45 cycles.

The test should verify:
1. When new has `matched_sensors=None` and existing has `matched_sensors=['IMX455']`, the merged result preserves `['IMX455']`
2. When new has `matched_sensors=[]` (from derive_spec with empty sensors_db) and existing has `matched_sensors=['IMX455']`, the merged result should still preserve `['IMX455']` (after the fix makes derive_spec return None for unchecked sensors)

**Fix:** Add a test case in `test_merge_field_preservation` that verifies `matched_sensors` is preserved from existing data.

---

## Summary

- TE46-01 (MEDIUM): No test for matched_sensors preservation in merge_camera_data
