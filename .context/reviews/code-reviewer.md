# Code Review (Cycle 31) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-30 fixes, focusing on NEW issues

## Previous Findings Status

C30-01 (GSMArena per-phone try/except) and C30-02 (deduplicate_specs DRY) both implemented and verified. All previous fixes stable.

## New Findings

### CR31-01: merge_camera_data can leave spec.pitch and derived.pitch inconsistent

**File:** `pixelpitch.py`, lines 413-432
**Severity:** MEDIUM | **Confidence:** HIGH

When `merge_camera_data` preserves `spec.pitch` from existing data (because new has None), it does NOT update `derived.pitch` if `derived.pitch` was already computed from area+mpix in the new data. This leaves `spec.pitch` and `derived.pitch` pointing at different values.

**Concrete scenario:**
1. openMVG provides camera "X" with `spec.pitch=None`, `spec.size=(5.0, 3.7)`, `spec.mpix=10.0`
2. `derive_spec()` computes `derived.pitch = pixel_pitch(18.5, 10.0)` = ~1.36um
3. Existing CSV has same camera with `spec.pitch=2.0`, `derived.pitch=2.0` (direct measurement from Geizhals)
4. Merge preserves `spec.pitch=2.0` from existing (new has None)
5. Merge does NOT preserve `derived.pitch` because new's `derived.pitch=1.36` is not None
6. Result: `spec.pitch=2.0`, `derived.pitch=1.36` — inconsistent
7. Template displays `derived.pitch=1.36`, ignoring the more authoritative `spec.pitch=2.0`
8. On next CSV write, `derived.pitch=1.36` is persisted, permanently losing the 2.0 measurement

**Fix:** After all Spec field preservation, if `spec.pitch` was changed (preserved from existing) and `derived.pitch` was computed (not directly from `spec.pitch`), update `derived.pitch` to match `spec.pitch`. Alternatively, re-derive `derived.pitch` from the updated spec fields.

---

### CR31-02: BOM check uses literal character instead of escape sequence

**File:** `pixelpitch.py`, line 276; `sources/openmvg.py`, line 67
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

Both files compare `csv_content[0] == "﻿"` using the literal BOM character (U+FEFF) inside the string literal. If the source file is re-encoded by an editor or tool that strips or mangles the BOM character, the comparison silently fails and BOM-prefixed CSVs produce mangled headers (e.g., `"﻿id"` instead of `"id"`), causing 0-row parses.

**Fix:** Replace `csv_content[0] == "﻿"` with `csv_content.startswith('﻿')` using the explicit escape sequence `﻿`. Same for openmvg.py line 67.

---

### CR31-03: Spec and SpecDerived constructed with positional args in parse_existing_csv and extract_specs

**File:** `pixelpitch.py`, lines 346-347 and 625
**Severity:** LOW | **Confidence:** MEDIUM

`parse_existing_csv` creates `Spec(name, category, type_str, size, pitch, mpix, year)` and `SpecDerived(spec, size, area, pitch, matched_sensors, record_id)` using positional arguments. Similarly, `extract_specs` at line 625 creates `Spec(name, category, typ, size, pitch, mpix, year=None)`. If the dataclass field order changes, these would silently produce wrong objects. The C30-02 fix addressed `deduplicate_specs()` with `dataclasses.replace()`, but these parser code paths still use positional args.

**Fix:** Use keyword arguments: `Spec(name=name, category=category, type=type_str, size=size, pitch=pitch, mpix=mpix, year=year)` and similarly for SpecDerived.

---

## Summary

- CR31-01 (MEDIUM): merge_camera_data spec.pitch/derived.pitch inconsistency after field preservation
- CR31-02 (LOW-MEDIUM): BOM check uses literal U+FEFF instead of '﻿' escape
- CR31-03 (LOW): Spec/SpecDerived positional args in parsers — fragile if field order changes
