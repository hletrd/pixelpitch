# Debugger Review (Cycle 35) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-34 fixes, focusing on NEW issues

## Previous Findings Status

DBG34-01 (match_sensors ZeroDivisionError) fixed in C34. DBG34-02 (list truthy check) fixed in C34. Verified.

## New Findings

### DBG35-01: `derive_spec` crashes with ValueError when area is negative

**File:** `pixelpitch.py`, line 725 (calls `pixel_pitch`)
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:** `pixel_pitch(area, mpix)` calls `sqrt(area / (mpix * 10**6))`. When `area < 0`, `sqrt()` raises `ValueError: expected a nonnegative input`. This exception is not caught anywhere.

**Trigger scenario:**
1. `parse_existing_csv` reads a CSV with negative sensor dimensions (e.g., corrupted data)
2. `derive_spec` is called on the parsed spec
3. `area = size[0] * size[1]` produces a negative value
4. Since `spec.pitch is None`, `pixel_pitch(area, mpix)` is called
5. `sqrt(negative)` raises `ValueError`
6. Unhandled exception crashes the build

**Fix:** Add a guard in `pixel_pitch` for `area <= 0`:

```python
def pixel_pitch(area: float, mpix: float) -> float:
    if mpix <= 0 or area <= 0:
        return 0.0
    return 1000 * sqrt(area / (mpix * 10**6))
```

---

### DBG35-02: `_BOM` literal character — silent failure if editor strips it

**File:** `sources/__init__.py`, line 90
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:** The literal BOM character U+FEFF is used instead of the documented escape sequence. If an editor or CI pipeline strips invisible characters during normalization:
1. `_BOM` becomes `''` (empty string)
2. `strip_bom` always returns the input unchanged
3. BOM-prefixed CSVs produce mangled headers in DictReader
4. `KeyError` on every row — 0 records, no error message

This is a silent data loss bug — the build succeeds but produces empty/incomplete output.

**Fix:** Replace the literal with escape sequence `﻿`.

---

### DBG35-03: Empty strings in matched_sensors from semicolon splitting

**File:** `pixelpitch.py`, line 343
**Severity:** LOW | **Confidence:** HIGH

**Failure mode:** If the matched_sensors CSV field contains leading/trailing semicolons (e.g., `;IMX455;`), the `split(";")` produces `['', 'IMX455', '']`. These empty strings are written back to CSV on the next `write_csv` call, perpetuating corruption.

**Fix:** Filter empty strings after split.

---

## Summary

- DBG35-01 (MEDIUM): `derive_spec` crashes with ValueError on negative area
- DBG35-02 (MEDIUM): `_BOM` literal — silent failure if editor strips invisible character
- DBG35-03 (LOW): Empty strings in matched_sensors from semicolon splitting
