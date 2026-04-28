# Debugger Review (Cycle 25) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-24 fixes

## Previous Findings Status

DBG24-01 (TYPE_FRACTIONAL_RE space+inch) and DBG24-02 (bare 1-inch) fixed. DBG24-03 (rstrip) remains deferred.

## New Findings

### DBG25-01: parse_sensor_field ValueError crash on malformed float

**File:** `pixelpitch.py`, lines 556, 561
**Severity:** MEDIUM | **Confidence:** MEDIUM

**Failure mode:** If `SIZE_RE` matches a string containing multiple dots (e.g., `"36.0.1x24.0mm"` produces group(1)=`"36.0.1"`), `float()` raises `ValueError`. This exception is NOT caught in `parse_sensor_field()` or `extract_specs()`, and propagates to `render_html()` where the outer `try/except Exception` drops the entire Geizhals category.

**Root cause:** The regex `[\d\.]+` allows multiple dots, but `float()` requires at most one. No try/except wraps the conversion.

**Impact:** All cameras in the affected category are lost for that deployment. Previous data is preserved via CSV merge, but current Geizhals data for that category is gone.

**Verified:** `float("36.0.1")` raises `ValueError`.

### DBG25-02: SIZE_RE does not match Unicode × or spaces — silent data loss

**File:** `pixelpitch.py`, line 42
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:** If Geizhals sensor text uses `×` (U+00D7) instead of `x` (U+0078), or includes spaces around `x`, `SIZE_RE` returns None. The sensor dimensions are silently lost.

**Root cause:** `SIZE_RE` pattern is `([\d\.]+)x([\d\.]+)mm` — only ASCII lowercase `x`, no spaces. The shared `SIZE_MM_RE` handles both `×` and spaces.

**Impact:** Camera shows "unknown" sensor size on website.

**Verified:** `SIZE_RE.search('36.0×24.0mm')` returns None.

---

## Summary

- DBG25-01 (MEDIUM): parse_sensor_field ValueError crash on malformed float input
- DBG25-02 (MEDIUM): SIZE_RE does not match Unicode × or spaces
