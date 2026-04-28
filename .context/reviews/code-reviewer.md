# Code Review (Cycle 26) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-25 fixes, focusing on NEW issues

## Previous Findings Status

C25-01 (SIZE_RE/PITCH_RE centralization) and C25-02 (ValueError guard in parse_sensor_field) both implemented and verified. All previous fixes stable.

## New Findings

### CR26-01: MPIX_RE in pixelpitch.py is still a DRY violation — not centralized like SIZE_MM_RE/PITCH_UM_RE

**File:** `pixelpitch.py`, line 42 vs `sources/__init__.py`, line 67
**Severity:** MEDIUM | **Confidence:** HIGH

The C25-01 aggregate review explicitly stated: "Update the MPIX_RE in pixelpitch.py similarly (it only matches 'Megapixel', not 'MP' or 'Mega pixels')." However, the C25-01 fix plan (Task 1, step 3) noted: "MPIX_RE.search() stays (no shared equivalent for 'Megapixel')" — this was incorrect. There IS a shared `MPIX_RE` in `sources/__init__.py`:

- `pixelpitch.py` line 42: `MPIX_RE = re.compile(r"([\d\.]+)\s*Megapixel")` — only matches "Megapixel"
- `sources/__init__.py` line 67: `MPIX_RE = re.compile(r"([\d.]+)\s*(?:effective\s+)?(?:Mega ?pixels?|MP)", re.IGNORECASE)` — matches "Megapixel", "Mega pixels", "MP", and more

The shared pattern is a superset of the local pattern (it matches everything the local one does, plus more). It is also exported in `__all__` (line 89).

This is the same DRY violation that was fixed for SIZE_RE and PITCH_RE in C25-01, but MPIX_RE was incorrectly left out.

**Impact:** If Geizhals ever changes "Megapixel" to "MP" or "Mega pixels" in their HTML, the megapixel value is silently lost. The camera would show "unknown" resolution.

**Concrete failure scenario:** Geizhals updates their HTML to show "33.0 MP" instead of "33.0 Megapixel". `MPIX_RE.search("33.0 MP")` returns None. Camera shows "unknown" resolution and computed pixel pitch is lost.

**Fix:** Import `MPIX_RE` from `sources` in `pixelpitch.py`, replacing the local definition. Update `extract_specs()` line 596 to use the imported pattern. Add test cases for "MP" and "Mega pixels" formats.

---

### CR26-02: ValueError guard missing in source module float() calls — same pattern as C25-02

**File:** `sources/cined.py` line 98, `sources/apotelyt.py` lines 119-120/123/129, `sources/gsmarena.py` lines 130/133, `sources/imaging_resource.py` line 228
**Severity:** MEDIUM | **Confidence:** MEDIUM

The C25-02 fix added ValueError guards only in `pixelpitch.py`'s `parse_sensor_field()`. But the same vulnerable pattern exists in all source modules:

1. **`sources/cined.py` line 98:** `size = (float(s.group(1)), float(s.group(2)))` — no try/except. SIZE_RE `([\d.]+)` allows multi-dot values.
2. **`sources/apotelyt.py` lines 119-120:** `size = (float(m.group(1)), float(m.group(2)))` — no try/except.
3. **`sources/apotelyt.py` line 123:** `pitch = float(m.group(1))` — no try/except.
4. **`sources/apotelyt.py` line 129:** `mpix = float(m.group(1))` — no try/except.
5. **`sources/gsmarena.py` line 130:** `mpix = float(mp_match.group(1))` — no try/except.
6. **`sources/gsmarena.py` line 133:** `pitch = float(pitch_match.group(1))` — no try/except.
7. **`sources/imaging_resource.py` line 228:** `size = (float(m.group(1)), float(m.group(2)))` — no try/except.

**Impact:** A malformed value from any source (e.g., "36.0.1" in a size field) raises ValueError and the entire camera record is lost for that source. Unlike the Geizhals path (where the entire category could be lost), source modules process one camera at a time, so the blast radius is smaller — only the individual camera is lost. However, the inconsistency with C25-02's defensive pattern is notable.

**Fix:** Wrap float() calls in try/except ValueError, setting the affected field to None (consistent with `parse_sensor_field()` pattern). This allows the remaining fields (e.g., type, pitch, mpix) to still be extracted even if one field is malformed.

---

## Summary

- CR26-01 (MEDIUM): MPIX_RE DRY violation — not centralized despite being in C25-01 aggregate scope
- CR26-02 (MEDIUM): ValueError guard missing in source module float() calls — same pattern as C25-02
