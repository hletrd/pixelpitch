# Code Review (Cycle 35) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-34 fixes, focusing on NEW issues

## Previous Findings Status

C34-01 (list command truthy check) fixed. C34-02 (match_sensors width/height truthy) fixed. C34-03 (match_sensors ZeroDivisionError) fixed. All verified by gate tests.

## New Findings

### CR35-01: `_BOM` uses literal character despite comment saying it uses escape sequence

**File:** `sources/__init__.py`, line 90
**Severity:** MEDIUM | **Confidence:** HIGH

The comment on lines 87-89 says:

> Using the escape sequence rather than the literal character guards against editors or CI pipelines that silently strip or normalise the invisible BOM glyph when re-encoding source files.

But the actual code on line 90 uses the literal BOM character:

```python
_BOM = '﻿'  # Actually a LITERAL character, not an escape sequence
```

When examined as raw bytes, the file contains `ef bb bf` (the UTF-8 encoding of U+FEFF) directly in the source, confirming the literal is present. An escape sequence would appear as `﻿` in the raw bytes instead.

**Concrete scenario:**
1. An editor or CI pipeline normalizes UTF-8 files and strips invisible characters
2. The BOM literal is silently removed, making `_BOM = ''`
3. `strip_bom` stops working — BOM-prefixed CSVs produce mangled headers
4. DictReader fails with KeyError on every row, producing 0 records

**Fix:** Replace the literal with the actual escape sequence in the source file:

```python
_BOM = '﻿'
```

The source file must be edited so the raw bytes contain the ASCII characters `\`, `u`, `f`, `e`, `f`, `f` rather than the UTF-8 BOM bytes.

---

### CR35-02: `parse_existing_csv` produces empty strings in `matched_sensors` from semicolons

**File:** `pixelpitch.py`, line 343
**Severity:** LOW | **Confidence:** HIGH

The matched_sensors field is split by semicolons:

```python
matched_sensors = sensors_str.split(";") if sensors_str else []
```

If the CSV contains leading/trailing semicolons (e.g., `;IMX455;`), the split produces `['', 'IMX455', '']`. These empty strings are not valid sensor names and will be written back to the CSV on the next `write_csv` call, perpetuating the corruption.

**Concrete scenario:**
1. A manual CSV edit adds `;IMX455;` as the sensors column
2. `parse_existing_csv` produces `matched_sensors = ['', 'IMX455', '']`
3. `write_csv` writes `;IMX455;` again (from `;".join(...)`)
4. Empty strings appear as phantom sensor matches

**Fix:** Filter out empty strings after split:

```python
matched_sensors = [s for s in sensors_str.split(";") if s] if sensors_str else []
```

---

### CR35-03: `derive_spec` crashes with ValueError when area is negative

**File:** `pixelpitch.py`, line 725 (calls `pixel_pitch`)
**Severity:** MEDIUM | **Confidence:** HIGH

`pixel_pitch` calls `sqrt(area / (mpix * 10**6))`, which raises `ValueError` when the argument is negative. `derive_spec` calls `pixel_pitch(area, spec.mpix)` when `spec.pitch is None` and both `area` and `spec.mpix` are known. If `area` is negative (from negative sensor dimensions), this crashes.

**Concrete scenario:**
1. Source parser produces `Spec(size=(-5.0, 3.7), pitch=None, mpix=10.0)`
2. `derive_spec` computes `area = -5.0 * 3.7 = -18.5`
3. Since `spec.pitch is None`, calls `pixel_pitch(-18.5, 10.0)`
4. `sqrt(-18.5 / 10_000_000)` raises `ValueError`
5. Unhandled exception crashes the build pipeline

While negative sensor dimensions are physically meaningless, the data model allows them, and the error is unhandled.

**Fix:** Add a guard in `pixel_pitch`:

```python
def pixel_pitch(area: float, mpix: float) -> float:
    if mpix <= 0 or area <= 0:
        return 0.0
    return 1000 * sqrt(area / (mpix * 10**6))
```

Or guard in `derive_spec` before calling `pixel_pitch`.

---

### CR35-04: `openmvg.fetch` produces positive mpix from negative pixel dimensions

**File:** `sources/openmvg.py`, lines 87-89
**Severity:** LOW | **Confidence:** HIGH

The mpix calculation doesn't guard against negative pixel dimensions:

```python
pw = int(float(row["SensorWidth(pixels)"]))
ph = int(float(row["SensorHeight(pixels)"]))
mpix = round(pw * ph / 1_000_000, 1) if pw and ph else None
```

If `pw=-100` and `ph=-200`, their product is `20000`, producing `mpix=20.0`. The truthy check `if pw and ph` passes because non-zero integers are truthy. The mm-dimension guard (`sw > 0 and sh > 0`) correctly rejects negative mm, but the pixel-dimension calculation does not.

**Fix:** Add sign check:

```python
mpix = round(pw * ph / 1_000_000, 1) if pw > 0 and ph > 0 else None
```

---

## Summary

- CR35-01 (MEDIUM): `_BOM` uses literal despite comment promising escape sequence
- CR35-02 (LOW): Empty strings in matched_sensors from semicolon splitting
- CR35-03 (MEDIUM): `derive_spec` crashes on negative area via `pixel_pitch` ValueError
- CR35-04 (LOW): `openmvg.fetch` produces positive mpix from negative pixel dimensions
