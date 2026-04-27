# Verifier Review (Cycle 12) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository verification after cycles 1-11 fixes

## Previously Fixed (Cycles 1-11) — Verification Status
All previous fixes verified still working. Gate tests pass cleanly. V11-01 (year in key) verified as fixed — `create_camera_key` now uses name+category only.

## New Findings

### V12-01: `parse_existing_csv` name field not stripped — verified via runtime test
**File:** `pixelpitch.py`, lines 277, 291
**Severity:** MEDIUM | **Confidence:** HIGH

Verified by runtime test:
```python
csv = 'id,name,category,...\n0, Sony A7 IV ,mirrorless,...\n'
parsed = pp.parse_existing_csv(csv)
# parsed[0].spec.name == ' Sony A7 IV '  (leading/trailing spaces preserved)
```

The category field (fixed in C11-02) now has `.strip()`, and the type field (fixed in C10-01) also has `.strip()`. The name field is the last remaining field without `.strip()`. This is an inconsistency that could produce visible whitespace in camera names on the website.

**Fix:** Add `.strip()` to name field parsing.

---

### V12-02: `_parse_camera_name` Sony slug extraction verified broken for legacy URLs
**File:** `sources/imaging_resource.py`, line 151
**Severity:** MEDIUM | **Confidence:** HIGH

Verified by runtime test:
```python
# Legacy spec URL (matched by LEGACY_SPEC_URL_RE)
url = 'https://www.imaging-resource.com/cameras/sony-zv-e10-specifications/'
url.rstrip('/').rsplit('/', 2)[-2]  # → 'cameras' (WRONG)

# Modern spec URL (produced by _spec_url)
url = 'https://www.imaging-resource.com/cameras/sony-zv-e10-review/specifications/'
url.rstrip('/').rsplit('/', 2)[-2]  # → 'sony-zv-e10-review' (correct)
```

The Sony-specific name parsing branch at line 151 uses `rsplit('/', 2)[-2]` which only works for URLs with an extra path segment after the slug. Legacy spec URLs don't have this extra segment, so the parent directory name `'cameras'` is extracted instead of the camera slug.

The non-Sony fallback at line 165 uses `rsplit('/', 1)[-1]` which works correctly for both URL formats.

**Fix:** Use `rsplit('/', 1)[-1]` in the Sony branch too, then strip the review/specifications suffix.

---

### V12-03: `_load_per_source_csvs` UnicodeDecodeError not caught — verified via code analysis
**File:** `pixelpitch.py`, line 754
**Severity:** LOW | **Confidence:** HIGH

Verified: The `except OSError` at line 755 does not catch `UnicodeDecodeError` (which is `ValueError`, not `OSError`). The `path.read_text(encoding='utf-8')` call at line 754 can raise `UnicodeDecodeError` for corrupt files. This would crash `render_html`.

**Fix:** Add `UnicodeDecodeError` to the except clause.

---

## Summary
- NEW findings: 3 (2 MEDIUM, 1 LOW)
- V12-01: Name field whitespace not stripped — verified — MEDIUM
- V12-02: Sony slug extraction fails for legacy IR URLs — verified — MEDIUM
- V12-03: UnicodeDecodeError not caught — verified — LOW
