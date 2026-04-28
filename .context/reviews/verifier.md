# Verifier Review (Cycle 31) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-30 fixes, focusing on NEW issues

## V31-01: Gate tests pass — all checks verified

**Evidence:** Ran `python3 -m tests.test_parsers_offline` — all checks passed. C30-01 and C30-02 fixes verified working. No regressions.

## V31-02: merge_camera_data spec/derived pitch inconsistency — verified

**File:** `pixelpitch.py`, lines 413-432
**Severity:** MEDIUM | **Confidence:** HIGH

**Evidence:** Traced the merge logic step by step:

1. Line 417-418: `if new_spec.spec.pitch is None and existing_spec.spec.pitch is not None: new_spec.spec.pitch = existing_spec.spec.pitch`
2. Line 431-432: `if new_spec.pitch is None and existing_spec.pitch is not None: new_spec.pitch = existing_spec.pitch`

These are independent conditions. The gap: when new data arrives with `spec.pitch=None` but a computed `derived.pitch` (from area+mpix), condition 1 fires (preserving spec.pitch from existing), but condition 2 does NOT fire (because new's derived.pitch is not None). The template reads `derived.pitch`, so the computed value wins over the preserved measurement.

**Reproduction path:**
```python
spec = Spec("Cam", "fixed", None, (5.0, 3.7), None, 10.0, 2020)
derived = derive_spec(spec)  # derived.pitch ~= 1.36
# existing: spec.pitch=2.0, derived.pitch=2.0
# After merge: spec.pitch=2.0, derived.pitch=1.36 — INCONSISTENT
```

## V31-03: BOM literal character in two files — verified

**File:** `pixelpitch.py` line 276; `sources/openmvg.py` line 67
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

**Evidence:** Both files contain the literal UTF-8 BOM character (U+FEFF) inside a Python string literal. If the .py file is re-encoded by a tool that normalizes Unicode or strips BOMs from string literals, the comparison silently breaks.

---

## Summary

- V31-01: All gate tests pass
- V31-02 (MEDIUM): merge_camera_data spec/derived pitch inconsistency — verified
- V31-03 (LOW-MEDIUM): BOM literal character fragility — verified
