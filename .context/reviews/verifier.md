# Verifier Review (Cycle 33) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-32 fixes, focusing on NEW issues

## V33-01: Gate tests pass — all checks verified

**Evidence:** Ran `python3 -m tests.test_parsers_offline` — all checks passed. C32-01 fix verified working. No regressions.

## V33-02: derive_spec spec.pitch=0.0 bug — verified

**File:** `pixelpitch.py`, line 722
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

**Evidence:** Traced derive_spec execution with spec.pitch=0.0:

```python
spec = Spec(name="Test", category="fixed", type=None,
            size=(35.9, 23.9), pitch=0.0, mpix=33.0, year=2021)
# derive_spec:
# if spec.pitch: → False (bool(0.0) is False)
# elif spec.mpix is not None and area is not None: → True
# pitch = pixel_pitch(858.61, 33.0) → 5.12
# Result: derived.pitch=5.12, NOT 0.0
```

The docstring says "spec.pitch (direct measurement) always takes precedence" but for 0.0 it does not. Confirmed bug.

## V33-03: Template truthy check for spec.pitch — verified

**File:** `templates/pixelpitch.html`, lines 84-89
**Severity:** LOW | **Confidence:** HIGH

**Evidence:** Jinja2's `{% if spec.pitch %}` evaluates `bool(0.0)` as False. If spec.pitch=0.0, the template renders `<span class="text-muted">unknown</span>` instead of "0.0 µm". The Jinja2 test `{% if spec.pitch is not none %}` correctly distinguishes 0.0 from None.

## V33-04: sorted_by 0.0 sorting — verified

**File:** `pixelpitch.py`, lines 752-756
**Severity:** LOW | **Confidence:** HIGH

**Evidence:** Traced `sorted_by` with a camera having pitch=0.0:
```python
key_functions["pitch"](camera_with_pitch_0)  # 0.0 if 0.0 else -1 → -1
```
0.0 would be sorted as -1, placing it below cameras with small positive pitch values.

---

## Summary

- V33-01: All gate tests pass
- V33-02 (LOW-MEDIUM): derive_spec spec.pitch=0.0 bug verified
- V33-03 (LOW): Template truthy check for 0.0 verified
- V33-04 (LOW): sorted_by 0.0 sorting verified
