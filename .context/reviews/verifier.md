# Verifier Review (Cycle 20) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28

## V20-01: `pixel_pitch(area, 0.0)` crashes — verified
**Severity:** MEDIUM | **Confidence:** HIGH (reproduced)

Tested: `pixel_pitch(864.0, 0.0)` raises `ZeroDivisionError`. Tested: `pixel_pitch(864.0, -1.0)` raises `ValueError` (sqrt of negative). The `derive_spec` function calls `pixel_pitch` without a guard, so this crashes the pipeline.

**Evidence:** Direct Python execution confirms both crash paths.

---

## V20-02: Sony FX naming — verified misnaming
**Severity:** MEDIUM | **Confidence:** HIGH (reproduced)

Tested: `_parse_camera_name({'Model Name': 'Sony FX3'}, '...sony-fx3-review/specifications/')` returns `'Sony Fx3'` instead of `'Sony FX3'`. The `.title()` method capitalizes the 'x' in 'fx', producing 'Fx'.

**Evidence:** Direct Python execution confirms the misnaming for FX3, FX6, FX30.

---

## V20-03: Merge field preservation — verified year-only behavior
**Severity:** LOW | **Confidence:** HIGH (reproduced)

Tested: Merging new spec with `type=None` against existing spec with `type='1/2.3'` loses the existing type. Only `year` has preservation logic. Verified by constructing specs and running `merge_camera_data`.

---

## Summary

- V20-01 (MEDIUM): pixel_pitch crash confirmed
- V20-02 (MEDIUM): Sony FX misnaming confirmed
- V20-03 (LOW): Merge only preserves year, not other fields
