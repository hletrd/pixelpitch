# Verifier Review (Cycle 35) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-34 fixes, focusing on NEW issues

## V35-01: Gate tests pass — all checks verified

**Evidence:** Ran `python3 -m tests.test_parsers_offline` — all 153 checks passed. C34 fixes verified working. No regressions.

## V35-02: `_BOM` literal vs escape sequence — verified

**File:** `sources/__init__.py`, line 90
**Severity:** MEDIUM | **Confidence:** HIGH

**Evidence:** Examined raw bytes of `sources/__init__.py`:

```
_BOM = '\xef\xbb\xbf'
```

The file contains the literal UTF-8 BOM bytes (`ef bb bf`), not the ASCII escape sequence characters (`﻿`). The comment on lines 87-89 explicitly states the escape sequence is used "rather than the literal character" — but this is false. The literal is present.

If the BOM literal is stripped by an editor, `_BOM` becomes an empty string, and `strip_bom` stops functioning, causing CSV header mangling.

## V35-03: `derive_spec` crashes with ValueError on negative area — verified

**File:** `pixelpitch.py`, line 725
**Severity:** MEDIUM | **Confidence:** HIGH

**Evidence:** Tested directly:

```python
spec = Spec(name='Neg', category='fixed', type=None,
            size=(-5.0, 3.7), pitch=None, mpix=10.0, year=2020)
derive_spec(spec)
# → ValueError: expected a nonnegative input, got -1.85e-06
```

The crash occurs because `pixel_pitch(-18.5, 10.0)` calls `sqrt(-18.5/10e6)` which raises `ValueError`. No try/except catches this.

## V35-04: Empty strings in matched_sensors from semicolons — verified

**File:** `pixelpitch.py`, line 343
**Severity:** LOW | **Confidence:** HIGH

**Evidence:** Tested:

```python
csv = '...0,Test,mirrorless,,36.00,24.00,864.00,45.0,5.00,2021,;IMX455;\n'
parsed = parse_existing_csv(csv)
# parsed[0].matched_sensors → ['', 'IMX455', '']
```

The `split(";")` produces empty strings from leading/trailing semicolons.

## V35-05: openmvg negative pixel dimensions produce positive mpix — verified

**File:** `sources/openmvg.py`, lines 87-89
**Severity:** LOW | **Confidence:** HIGH

**Evidence:** With `pw=-100, ph=-200`:

```python
mpix = round((-100) * (-200) / 1_000_000, 1)  # → 20.0
# if pw and ph → True (non-zero ints are truthy)
```

The guard `if pw and ph` passes for negative values because `bool(-100)` is True.

---

## Summary

- V35-01: All gate tests pass
- V35-02 (MEDIUM): `_BOM` literal vs escape — comment contradicts code, verified
- V35-03 (MEDIUM): `derive_spec` crashes with ValueError on negative area, verified
- V35-04 (LOW): Empty strings in matched_sensors, verified
- V35-05 (LOW): openmvg negative pixel dims produce positive mpix, verified
