# Critic Review (Cycle 35) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-34 fixes, focusing on NEW issues

## Previous Findings Status

CRIT34-01 (residual truthy-vs-None) fixed in C34 across list command, match_sensors guard, and inner loop. CRIT34-02 (ZeroDivisionError) fixed. All verified.

## New Findings

### CRIT35-01: `_BOM` literal vs escape sequence — comment-doc-code mismatch with real consequences

**File:** `sources/__init__.py`, line 90
**Severity:** MEDIUM | **Confidence:** HIGH

The comment on lines 87-89 explicitly promises an escape sequence is used to guard against editor stripping. But the actual code uses the literal BOM character. This is not just a documentation error — it's a defense mechanism that was designed but never actually implemented. The whole point of the escape sequence was to survive editor normalization, and using the literal defeats that purpose entirely.

This is the same class of issue as "security measure documented but not enforced" — the code claims a protection exists that it doesn't actually provide.

**Fix:** Replace the literal BOM with the escape sequence `﻿` in the source file.

---

### CRIT35-02: Data integrity chain has no negative-value guards

**Files:** `pixelpitch.py` (pixel_pitch, derive_spec), `sources/openmvg.py` (mpix calc)
**Severity:** MEDIUM | **Confidence:** HIGH

The data pipeline has no guards against negative numeric values at any stage:

1. **`pixel_pitch(area, mpix)`** crashes with `ValueError` when `area < 0` (from `sqrt()` of negative number)
2. **`derive_spec`** can produce negative `area` from negative sensor dimensions, then crashes when computing pitch
3. **`openmvg.fetch`** produces positive `mpix` from negative pixel dimensions (product of two negatives)
4. **`parse_existing_csv`** accepts negative values for all numeric fields without validation
5. **`write_csv`** writes negative values to CSV without warning
6. **Template renders** negative pitch/mpix as `-2.0 µm` and `-10.0 MP`

The pipeline has validation for year (1900-2100) but no validation for any other numeric field. While negative values are physically meaningless, they can crash the build (item 1) or produce nonsensical output (items 3-6).

**Fix:** Add a validation guard in `pixel_pitch` for `area <= 0` and in `openmvg.fetch` for negative pixel dimensions. Consider adding a validation step in `derive_spec` or `parse_existing_csv`.

---

### CRIT35-03: `parse_existing_csv` matched_sensors can contain empty strings

**File:** `pixelpitch.py`, line 343
**Severity:** LOW | **Confidence:** HIGH

The split-by-semicolon produces empty strings from leading/trailing/doubled semicolons in the matched_sensors CSV field. This is a data quality issue that perpetuates through the CSV round-trip.

**Fix:** Filter empty strings after split.

---

## Summary

- CRIT35-01 (MEDIUM): `_BOM` literal vs escape — documented protection not actually implemented
- CRIT35-02 (MEDIUM): No negative-value guards in data pipeline — crashes and nonsensical output
- CRIT35-03 (LOW): Empty strings in matched_sensors from semicolon splitting
