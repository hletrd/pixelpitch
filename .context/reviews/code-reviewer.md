# Code Review (Cycle 27) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-26 fixes, focusing on NEW issues

## Previous Findings Status

C26-01 (MPIX_RE centralization) and C26-02 (ValueError guards in source modules) both implemented and verified. All previous fixes stable. Gate tests pass.

## New Findings

### CR27-01: PITCH_UM_RE does not match lowercase "um" — inconsistent with GSMArena PITCH_RE

**File:** `sources/__init__.py`, line 66 vs `sources/gsmarena.py`, line 50
**Severity:** LOW | **Confidence:** HIGH

The shared `PITCH_UM_RE` pattern matches `µm`, `μm`, `microns`, `&micro;m`, `&#956;m` but NOT plain lowercase `um`. Meanwhile, `sources/gsmarena.py` defines its own `PITCH_RE` at line 50 that explicitly includes `um`:

```python
# sources/__init__.py line 66 (shared):
PITCH_UM_RE = re.compile(r"([\d.]+)\s*(?:µm|microns?|μm|&micro;m|&#0?956;m)", re.IGNORECASE)

# sources/gsmarena.py line 50 (local):
PITCH_RE = re.compile(r"([\d.]+)\s*(?:µm|μm|um)", re.IGNORECASE)
```

The shared pattern is used by `pixelpitch.py`'s `parse_sensor_field()` for Geizhals data, while GSMArena uses its own pattern. Currently, Geizhals consistently uses `µm` or `μm`, so this is not a runtime bug. However, it is a DRY inconsistency — the shared pattern should be the authoritative one, and `um` is a common alternative representation used in some technical documentation.

**Impact:** If any source starts producing "um" (ASCII-only) pitch values, the shared pattern would not match. Currently only GSMArena encounters "um" and has its own pattern, so no data is lost.

**Fix:** Add `um` to the shared `PITCH_UM_RE` alternation: `(?:µm|μm|um|microns?|&micro;m|&#0?956;m)`. This makes the shared pattern a true superset of all local patterns. This is a LOW severity because no current data path uses the shared pattern against "um" text.

---

### CR27-02: parse_existing_csv accepts year=0 and negative years without validation

**File:** `pixelpitch.py`, line 336
**Severity:** LOW | **Confidence:** HIGH

The CSV parser converts the year column with `int(year_str) if year_str else None`, accepting any integer value including 0 and negative numbers. While sources use `parse_year()` which only matches 19xx or 20xx patterns, a manually edited CSV or corrupted data could introduce invalid years.

```python
year = int(year_str) if year_str else None  # line 336
```

The template renders year directly: `{{ spec.spec.year }}`. A year of 0 or -1 would display as "0" or "-1" on the website.

**Impact:** No current data path produces year=0 or negative years. The `parse_year()` function in `sources/__init__.py` only matches `\b(19\d{2}|20\d{2})\b`. This is a defensive hardening issue, not a current bug.

**Fix:** Add validation: `year = int(year_str) if year_str and int(year_str) >= 1900 else None`. Or use a try/except with a range check.

---

## Summary

- CR27-01 (LOW): PITCH_UM_RE missing "um" — DRY inconsistency with GSMArena PITCH_RE
- CR27-02 (LOW): parse_existing_csv accepts year=0 and negative years without validation
