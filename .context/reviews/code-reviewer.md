# Code Review (Cycle 38) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-37 fixes, focusing on NEW issues

## Previous Findings Status

All C37 findings confirmed fixed. `derive_spec` isfinite guard working. `isInvalidData` zero-pitch check added. Gate tests pass.

## New Findings

### CR38-01: `isInvalidData` pitch===0 check hides legitimate 0.0 pitch rows — UX contradiction with template rendering

**File:** `templates/pixelpitch.html`, lines 277-279
**Severity:** MEDIUM | **Confidence:** HIGH

The C37-02 fix added `if (pitch === 0) { return true; }` to `isInvalidData`. This causes rows with `pitch=0.0` to be hidden when "Hide possibly invalid data" is checked (which is the default state — `checked` on line 157).

However, `test_template_zero_pitch_rendering` explicitly verifies that 0.0 pitch renders as "0.0 µm" in the template — NOT as "unknown". The intent of the test is to confirm that 0.0 is a valid numeric display value. But the JS filter then hides those rows by default, making the template rendering moot for users.

The contradiction: the code renders "0.0 µm" in the table, but then hides the row. If 0.0 is truly invalid data that should be hidden, the template should render "unknown" instead. If 0.0 is a valid value, it shouldn't be hidden by default.

Furthermore, `pixel_pitch` returns `0.0` for `mpix=0.0` cameras. If a camera genuinely has zero megapixels listed (which can happen from source data), its computed pitch would be `0.0`, and that row would be hidden.

**Fix:** Two options:
1. (Preferred) Change `pixel_pitch` to return `None` for invalid inputs instead of `0.0`, update template to show "unknown" for `None` pitch, and keep the JS `pitch === 0` check as defense-in-depth. This eliminates the `0.0` sentinel entirely.
2. Remove the `pitch === 0` check from `isInvalidData` and accept that 0.0 pitch rows are visible (matching the template behavior).

Option 1 is cleaner but is a larger refactor. Option 2 is simpler but leaves the semantic ambiguity.

---

### CR38-02: `match_sensors` ZeroDivisionError when `megapixels=0.0` and `sensor_megapixels` is non-empty

**File:** `pixelpitch.py`, line 243
**Severity:** MEDIUM | **Confidence:** HIGH

```python
megapixel_match = any(
    abs(megapixels - mp) / megapixels * 100 <= megapixel_tolerance
    for mp in sensor_megapixels
)
```

When `megapixels=0.0` (which is `> 0` is False, but `megapixels is not None and megapixels > 0` on line 242 evaluates to `0.0 > 0 = False`), the code skips to the elif branch. So in practice this is currently safe because the guard `megapixels > 0` rejects `0.0`. However, if that guard were ever changed to `megapixels is not None and megapixels >= 0`, a ZeroDivisionError would occur.

The existing test `test_match_sensors` with `megapixels=0.0` uses the size-only match path (the elif branch), which works. So this is not a current bug, but the division by `megapixels` without a zero guard in the `any()` expression is a latent risk.

**Fix:** Add an explicit `megapixels > 0` guard inside the `any()` expression as defense-in-depth:
```python
if megapixels is not None and megapixels > 0 and sensor_megapixels:
    megapixel_match = any(
        megapixels > 0 and abs(megapixels - mp) / megapixels * 100 <= megapixel_tolerance
        for mp in sensor_megapixels
    )
```
Or more simply, since the outer guard already ensures `megapixels > 0`, no fix is strictly needed. Deferring.

---

## Summary

- CR38-01 (MEDIUM): `isInvalidData` pitch===0 hides rows that template renders as "0.0 µm" — UX contradiction
- CR38-02 (LOW): Latent ZeroDivisionError risk in `match_sensors` megapixel matching — currently guarded
