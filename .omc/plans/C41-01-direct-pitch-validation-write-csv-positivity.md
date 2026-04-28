# Plan: Cycle 41 Findings — Direct Pitch Validation & write_csv Positivity

**Created:** 2026-04-28
**Status:** PENDING
**Source Reviews:** CR41-01, CRIT41-01, V41-02, TR41-01, ARCH41-01, DBG41-01, DES41-01, CR41-02, CRIT41-02, V41-03, DBG41-02, CR41-03, V41-04, TE41-01, TE41-02, DOC41-01

---

## Task 1: Validate direct `spec.pitch` in `derive_spec` — C41-01 (core)

**Finding:** C41-01 (7-agent consensus)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `pixelpitch.py`, lines 759-760

### Problem

The C40 fix added 0.0-to-None conversion for the computed pitch path. The direct pitch path (`spec.pitch is not None`) still propagates invalid values (0.0, negative, NaN) without validation. This causes:
- `selectattr` misclassification (camera in wrong table section)
- `write_csv` writes invalid values to CSV
- CSV round-trip data loss

### Implementation

1. In `pixelpitch.py`, `derive_spec` function, change the direct pitch path from:

```python
if spec.pitch is not None:
    pitch = spec.pitch
```

To:

```python
if spec.pitch is not None and isfinite(spec.pitch) and spec.pitch > 0:
    pitch = spec.pitch
else:
    pitch = None
```

This establishes the uniform contract: `derived.pitch` is either None or a positive finite value.

Note: The `else` clause now also covers the case where `spec.pitch is not None` but invalid. The subsequent `elif spec.mpix is not None and area is not None` block computes pitch from area+mpix when direct pitch is invalid. This is the correct behavior — if we have a direct but invalid pitch and can compute one, we should compute it. If both fail, pitch stays None.

Wait — this needs more thought. If `spec.pitch` is set to 0.0 (meaning "I explicitly set this to an invalid value"), should we try to compute it from area+mpix instead? Let me reconsider.

The current logic is:
```python
if spec.pitch is not None:
    pitch = spec.pitch   # takes precedence over computed
elif spec.mpix is not None and area is not None:
    pitch = pixel_pitch(area, spec.mpix)  # fallback
```

The intent is: direct measurement > computed value. If the direct measurement is invalid (0.0, negative, NaN), we should treat it as "no valid direct measurement" and fall through to the computed path.

So the correct implementation is:

```python
if spec.pitch is not None and isfinite(spec.pitch) and spec.pitch > 0:
    pitch = spec.pitch
elif spec.mpix is not None and area is not None:
    pitch = pixel_pitch(area, spec.mpix)
    if pitch == 0.0:
        pitch = None
else:
    pitch = None
```

This way:
- `spec.pitch=5.12` → uses 5.12 (direct, valid)
- `spec.pitch=0.0` with `mpix=33.0, area=858.0` → computes ~5.12 (fallback)
- `spec.pitch=0.0` with no mpix/area → pitch=None (no data available)
- `spec.pitch=-1.0` → falls through to computed or None
- `spec.pitch=nan` → falls through to computed or None

---

## Task 2: Replace `isfinite` with positivity checks in `write_csv` — C41-02

**Finding:** C41-02 (4-agent consensus)
**Severity:** LOW | **Confidence:** HIGH
**Files:** `pixelpitch.py`, lines 866-868

### Problem

`write_csv` uses `isfinite()` to guard mpix, pitch, and area. But `isfinite(0.0)` and `isfinite(-1.0)` both return True, so these physically invalid values are written to the CSV. `parse_existing_csv` rejects them on re-read, causing data loss on round-trip.

### Implementation

Change lines 866-868 in `pixelpitch.py` from:

```python
area_str = f"{derived.area:.2f}" if derived.area is not None and isfinite(derived.area) else ""
mpix_str = f"{spec.mpix:.1f}" if spec.mpix is not None and isfinite(spec.mpix) else ""
pitch_str = f"{derived.pitch:.2f}" if derived.pitch is not None and isfinite(derived.pitch) else ""
```

To:

```python
area_str = f"{derived.area:.2f}" if derived.area is not None and derived.area > 0 else ""
mpix_str = f"{spec.mpix:.1f}" if spec.mpix is not None and spec.mpix > 0 else ""
pitch_str = f"{derived.pitch:.2f}" if derived.pitch is not None and derived.pitch > 0 else ""
```

Note: `x > 0` already implies `isfinite(x)` because NaN > 0 is False and inf > 0 is True. For inf, the positivity check alone would pass it through. However, `derive_spec` and `parse_existing_csv` already reject inf values (via isfinite guards), so inf should never reach `write_csv` in practice. If we want defense in depth, we can use `isfinite(x) and x > 0`, but `x > 0` alone is sufficient given the upstream guards.

Actually, let me reconsider: if somehow `derived.area=inf` reaches write_csv, `inf > 0` is True and we'd write "inf" or a very large number. Let's keep both checks for safety:

```python
area_str = f"{derived.area:.2f}" if derived.area is not None and isfinite(derived.area) and derived.area > 0 else ""
mpix_str = f"{spec.mpix:.1f}" if spec.mpix is not None and isfinite(spec.mpix) and spec.mpix > 0 else ""
pitch_str = f"{derived.pitch:.2f}" if derived.pitch is not None and isfinite(derived.pitch) and derived.pitch > 0 else ""
```

---

## Task 3: Update `derive_spec` docstring — C41-06

**Finding:** DOC41-01
**Severity:** LOW | **Confidence:** HIGH
**Files:** `pixelpitch.py`, derive_spec docstring

### Implementation

Update the docstring to document that both direct and computed paths produce the same contract. Change the relevant paragraph from:

"When ``pixel_pitch`` returns 0.0 (sentinel for invalid inputs such as non-positive mpix/area or NaN/inf), the computed pitch is set to None instead of propagating the sentinel."

To:

"Direct ``spec.pitch`` values that are non-finite or non-positive (0.0, negative, NaN, inf) are treated as invalid and converted to None, allowing the computed path to serve as a fallback. When ``pixel_pitch`` returns 0.0 (sentinel for invalid inputs), the computed pitch is also set to None. The output contract is uniform: ``derived.pitch`` is either None (unknown) or a positive finite value (valid measurement or approximation)."

---

## Task 4: Update tests for direct pitch validation — C41-04

**Finding:** TE41-01
**Severity:** LOW | **Confidence:** HIGH
**Files:** `tests/test_parsers_offline.py`

### Implementation

1. Update `test_derive_spec_zero_pitch` — change the expectation for `spec.pitch=0.0` direct from `0.0` to `None`:

```python
# Before: expect("derive_spec: spec.pitch=0.0 preserved over computed", d_zero.pitch, 0.0, tol=0.01)
# After:
expect("derive_spec: spec.pitch=0.0 direct is None (invalid)",
       d_zero.pitch, None)
```

2. Add test cases to `test_derive_spec_zero_pitch` or a new test function:

```python
# spec.pitch=-1.0 direct → derived.pitch=None (falls through to computed or None)
spec_neg = Spec(name="Neg Direct Cam", category="fixed", type=None,
                size=(35.9, 23.9), pitch=-1.0, mpix=33.0, year=2021)
d_neg = pp.derive_spec(spec_neg)
expect("derive_spec: spec.pitch=-1.0 direct is None",
       d_neg.pitch, None)

# spec.pitch=nan direct → derived.pitch=None
spec_nan = Spec(name="NaN Direct Cam", category="fixed", type=None,
                size=(35.9, 23.9), pitch=float('nan'), mpix=33.0, year=2021)
d_nan = pp.derive_spec(spec_nan)
expect("derive_spec: spec.pitch=nan direct is None",
       d_nan.pitch, None)

# spec.pitch=0.0 with mpix=33.0, area=858 → computed ~5.12 (fallback)
spec_zero_fb = Spec(name="Zero Fallback Cam", category="fixed", type=None,
                     size=(35.9, 23.9), pitch=0.0, mpix=33.0, year=2021)
d_zero_fb = pp.derive_spec(spec_zero_fb)
expect("derive_spec: spec.pitch=0.0 falls back to computed",
       d_zero_fb.pitch is not None, True)
expect("derive_spec: fallback pitch ≈ 5.12",
       d_zero_fb.pitch, 5.12, tol=0.05)
```

3. Update `test_sorted_by_zero_values` — the zero-pitch camera now has `derived.pitch=None` instead of 0.0, so it sorts at -1 (same as None). Change the test to reflect this.

---

## Task 5: Add tests for `write_csv` 0.0/negative mpix/pitch — C41-05

**Finding:** TE41-02
**Severity:** LOW | **Confidence:** MEDIUM
**Files:** `tests/test_parsers_offline.py`

### Implementation

Add test function:

```python
def test_write_csv_zero_negative_guards():
    """Verify write_csv does not output 0.0 or negative mpix/pitch."""
    section("write_csv zero/negative value guards")
    import tempfile
    import pixelpitch as pp
    from models import Spec

    # spec.mpix=0.0 → CSV should have empty mpix cell
    spec_zero = Spec(name="Zero MP Cam", category="fixed", type=None,
                     size=(35.9, 23.9), pitch=5.12, mpix=0.0, year=2020)
    d_zero = pp.derive_spec(spec_zero)
    d_zero.id = 0
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        out_path = Path(f.name)
    try:
        pp.write_csv([d_zero], out_path)
        csv_text = out_path.read_text(encoding="utf-8")
    finally:
        out_path.unlink(missing_ok=True)
    row = csv_text.splitlines()[1]
    expect("write_csv zero mpix: no '0.0' in CSV row for mpix",
           ",0.0," not in row, True)

    # spec.mpix=-5.0 → CSV should have empty mpix cell
    spec_neg = Spec(name="Neg MP Cam", category="fixed", type=None,
                    size=(35.9, 23.9), pitch=5.12, mpix=-5.0, year=2020)
    d_neg = pp.derive_spec(spec_neg)
    d_neg.id = 0
    with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as f:
        out_path2 = Path(f.name)
    try:
        pp.write_csv([d_neg], out_path2)
        csv_text2 = out_path2.read_text(encoding="utf-8")
    finally:
        out_path2.unlink(missing_ok=True)
    row2 = csv_text2.splitlines()[1]
    expect("write_csv neg mpix: no '-5.0' in CSV row",
           "-5.0" not in row2, True)
```

---

## Task 6: Validate preserved pitch in `merge_camera_data` — C41-03

**Finding:** C41-03
**Severity:** LOW | **Confidence:** MEDIUM
**Files:** `pixelpitch.py`, lines 449, 471-473

### Problem

`merge_camera_data` preserves `spec.pitch=0.0` from existing data, re-introducing the sentinel.

### Implementation

After the field preservation block (around line 465), add validation:

```python
# Validate preserved pitch: non-positive or non-finite values are invalid
if new_spec.spec.pitch is not None and (not isfinite(new_spec.spec.pitch) or new_spec.spec.pitch <= 0):
    new_spec.spec.pitch = None
```

And update the consistency check to also validate:

```python
if (new_spec.spec.pitch is not None and new_spec.spec.pitch > 0
        and new_spec.pitch != new_spec.spec.pitch):
    new_spec.pitch = new_spec.spec.pitch
```

---

## Verification

- Gate tests (`python3 -m tests.test_parsers_offline`) — all checks must pass
- `derive_spec` with `spec.pitch=0.0` produces None (not 0.0)
- `derive_spec` with `spec.pitch=0.0` and valid mpix/area computes pitch as fallback
- `derive_spec` with `spec.pitch=-1.0` produces None (not -1.0)
- `derive_spec` with `spec.pitch=nan` produces None (not nan)
- `write_csv` with mpix=0.0 writes empty string (not "0.0")
- `write_csv` with pitch=-1.0 writes empty string (not "-1.00")
- CSV round-trip is lossless for all valid values
- Updated tests pass

---

## Deferred Findings

None. All findings are scheduled for implementation.
