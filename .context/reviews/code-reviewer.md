# Code Review (Cycle 33) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-32 fixes, focusing on NEW issues

## Previous Findings Status

C32-01 (write_csv falsy checks) fixed and verified. C32-02 (IR_MPIX_RE) deferred. All previous fixes stable.

## New Findings

### CR33-01: derive_spec truthy check for spec.pitch violates docstring contract for 0.0

**File:** `pixelpitch.py`, line 722
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

The `derive_spec` docstring states: "spec.pitch (direct measurement) always takes precedence." But the code uses a truthy check:

```python
if spec.pitch:
    pitch = spec.pitch
elif spec.mpix is not None and area is not None:
    pitch = pixel_pitch(area, spec.mpix)
```

If `spec.pitch=0.0`, the truthy check is False, and pitch is computed from area+mpix instead. The docstring's "always takes precedence" guarantee is violated for 0.0. This is the same class of issue as C32-01 (write_csv falsy checks), but in the computation layer.

**Concrete scenario:**
1. Source parser produces `Spec(pitch=0.0, mpix=33.0, size=(35.9, 23.9))`
2. `derive_spec`: `if spec.pitch:` → False (0.0 is falsy)
3. Falls to elif: computes `pixel_pitch(858.61, 33.0)` = 5.12
4. Result: `derived.pitch=5.12` instead of `0.0`
5. The "direct measurement" was silently discarded

**Fix:** Replace with explicit None check:
```python
if spec.pitch is not None:
    pitch = spec.pitch
```

---

### CR33-02: sorted_by truthy checks sort 0.0 values as -1

**File:** `pixelpitch.py`, lines 752-756
**Severity:** LOW | **Confidence:** HIGH

The `sorted_by` function uses truthy checks for sort keys:

```python
"pitch": lambda c: c.pitch if c.pitch else -1,
"area": lambda c: c.area if c.area else -1,
"mpix": lambda c: c.spec.mpix if c.spec.mpix else -1,
```

If any of these values is 0.0, it would be sorted as -1 instead of 0.0. This is inconsistent with the C32-01 fix that treats 0.0 as a valid value distinct from None.

**Fix:** Use explicit None checks:
```python
"pitch": lambda c: c.pitch if c.pitch is not None else -1,
"area": lambda c: c.area if c.area is not None else -1,
"mpix": lambda c: c.spec.mpix if c.spec.mpix is not None else -1,
```

---

### CR33-03: prettyprint truthy checks display "unknown" for 0.0 values

**File:** `pixelpitch.py`, lines 772-778
**Severity:** LOW | **Confidence:** HIGH

The `prettyprint` function uses truthy checks:

```python
if spec.mpix:
    print(f", {spec.mpix:.1f} MP", end="")
else:
    print(", unknown resolution", end="")

if derived.pitch:
    print(f", {derived.pitch:.1f}µm pixel pitch", end="")
```

If mpix or pitch is 0.0, it would display "unknown" instead of "0.0 MP" / "0.0 µm". Inconsistent with C32-01 fix.

**Fix:** Use explicit None checks:
```python
if spec.mpix is not None:
    print(f", {spec.mpix:.1f} MP", end="")
else:
    print(", unknown resolution", end="")

if derived.pitch is not None:
    print(f", {derived.pitch:.1f}µm pixel pitch", end="")
```

---

### CR33-04: Missing test for derive_spec with spec.pitch=0.0

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** HIGH

The `test_pixel_pitch` function tests `pixel_pitch(864.0, 0.0) == 0.0`, and `test_csv_round_trip` tests 0.0 CSV preservation. But there is no test verifying that `derive_spec` with `spec.pitch=0.0` correctly preserves the 0.0 value instead of computing pitch from area+mpix. Without this test, the CR33-01 bug in `derive_spec` would not be caught by CI.

**Fix:** Add a test case:
```python
spec_zero_pitch = Spec(name="Zero Pitch Cam", category="fixed", type=None,
                       size=(35.9, 23.9), pitch=0.0, mpix=33.0, year=2021)
d_zp = pp.derive_spec(spec_zero_pitch)
expect("derive_spec: spec.pitch=0.0 preserved", d_zp.pitch, 0.0, tol=0.01)
```

---

## Summary

- CR33-01 (LOW-MEDIUM): derive_spec truthy check violates docstring contract for spec.pitch=0.0
- CR33-02 (LOW): sorted_by truthy checks sort 0.0 values as -1
- CR33-03 (LOW): prettyprint truthy checks display "unknown" for 0.0 values
- CR33-04 (LOW): Missing test for derive_spec with spec.pitch=0.0
