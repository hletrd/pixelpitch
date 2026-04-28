# Code Review (Cycle 25) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-24 fixes, focusing on NEW issues

## Previous Findings Status

All previous findings confirmed addressed. C24-01 through C24-05 implemented or deferred as documented.

## New Findings

### CR25-01: SIZE_RE and PITCH_RE in pixelpitch.py are less robust than their equivalents in sources/__init__.py

**File:** `pixelpitch.py`, lines 42-43
**Severity:** MEDIUM | **Confidence:** HIGH

The Geizhals-specific regex patterns in `pixelpitch.py` are significantly less robust than the shared patterns in `sources/__init__.py`:

- `SIZE_RE = re.compile(r"([\d\.]+)x([\d\.]+)mm")` — only matches lowercase `x` with no spaces. Does not match Unicode multiplication sign `×` or spaces around `x`.
- `SIZE_MM_RE` (sources/__init__.py line 65) — matches `x`, `×`, optional spaces around the separator, case-insensitive.

- `PITCH_RE = re.compile(r"([\d\.]+)µm")` — only matches micro sign (U+00B5) `µ`. Does not match Greek mu (U+03BC) `μ`, "microns", "um", or HTML entities.
- `PITCH_UM_RE` (sources/__init__.py line 66) — matches all the above variants.

These inconsistencies were verified by testing:

```
SIZE_RE.search('36.0×24.0mm')  → NO MATCH (Unicode ×)
SIZE_MM_RE.search('36.0×24.0mm')  → ('36.0', '24.0')
SIZE_RE.search('36.0 x 24.0mm')  → NO MATCH (spaces)
SIZE_MM_RE.search('36.0 x 24.0mm')  → ('36.0', '24.0')

PITCH_RE.search('5.12μm')  → NO MATCH (Greek mu)
PITCH_UM_RE.search('5.12μm')  → ('5.12',)
PITCH_RE.search('5.12 microns')  → NO MATCH
PITCH_UM_RE.search('5.12 microns')  → ('5.12',)
```

**Impact:** If Geizhals sensor text ever uses `×` instead of `x`, or `μ` instead of `µ`, or spaces around the dimension separator, the sensor size or pixel pitch is silently lost. The `parse_sensor_field()` function uses the limited patterns.

**Concrete failure scenario:** Geizhals updates their SPA to render sensor fields with Unicode `×` (e.g., `"36.0×24.0mm"`). Every camera's sensor dimensions become `None`. The entire site shows "unknown" for sensor size.

**Fix:** Replace `SIZE_RE` and `PITCH_RE` in `pixelpitch.py` with the shared patterns from `sources/__init__.py` (imported via `from sources import SIZE_MM_RE, PITCH_UM_RE`), or make them equally robust.

---

### CR25-02: parse_sensor_field has no ValueError guard on float() calls

**File:** `pixelpitch.py`, lines 556, 561
**Severity:** MEDIUM | **Confidence:** MEDIUM

`parse_sensor_field()` calls `float(size_match.group(1))` and `float(pitch_match.group(1))` without try/except. The regex pattern `[\d\.]+` can match strings with multiple dots (e.g., `"36.0.1"`) which `float()` rejects with `ValueError`.

Verified:
```python
SIZE_RE.search('36.0.1x24.0mm')  → matches group(1)="36.0.1"
float("36.0.1")  → ValueError
```

**Impact:** A single malformed sensor field from Geizhals would raise `ValueError` from `parse_sensor_field` → `extract_specs` → `get_category`. The outer `try/except Exception` in `render_html` would catch it, but the **entire category** (all cameras) would be dropped, not just the single malformed row.

**Concrete failure scenario:** A Geizhals HTML row has a corrupted sensor field with text like `"36.0.1x24.0mm"`. The ValueError propagates up and the entire category (e.g., all mirrorless cameras) is skipped, falling back to previous data.

**Fix:** Wrap the float() calls in try/except ValueError, returning None for unparseable values. This is consistent with the defensive pattern used elsewhere (e.g., `sensor_size_from_type`, `parse_existing_csv`).

---

## Summary

- CR25-01 (MEDIUM): SIZE_RE and PITCH_RE less robust than shared patterns — inconsistency
- CR25-02 (MEDIUM): parse_sensor_field float() without ValueError guard — crash risk
