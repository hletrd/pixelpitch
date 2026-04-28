# Debugger Review (Cycle 26) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-25 fixes

## Previous Findings Status

DBG25-01 (ValueError guard) and DBG25-02 (SIZE_RE Unicode) both fixed in C25-01/C25-02. DBG24-03 (rstrip) remains deferred.

## New Findings

### DBG26-01: MPIX_RE only matches "Megapixel" — will silently lose data if Geizhals uses "MP"

**File:** `pixelpitch.py`, line 42
**Severity:** MEDIUM | **Confidence:** HIGH

**Failure mode:** If Geizhals changes their "Megapixel effektiv" field to show "33.0 MP" (a common abbreviation), `MPIX_RE.search("33.0 MP")` returns None. The megapixel value is silently lost.

**Root cause:** `MPIX_RE = re.compile(r"([\d\.]+)\s*Megapixel")` only matches the literal string "Megapixel" (case-sensitive). The shared `MPIX_RE` in `sources/__init__.py` handles "MP", "Mega pixels", "Megapixel" etc.

**Impact:** Camera shows "unknown" resolution and computed pixel pitch is lost.

**Verified:** `MPIX_RE.search("33.0 MP")` returns None with the local pattern.

### DBG26-02: ValueError in source module float() calls — individual camera crash

**File:** `sources/apotelyt.py` lines 119-129, `sources/cined.py` line 98, `sources/gsmarena.py` lines 130/133, `sources/imaging_resource.py` line 228
**Severity:** LOW | **Confidence:** MEDIUM

**Failure mode:** A malformed value in any source module's HTML (e.g., "35.9.1x23.9 mm") causes `float()` to raise ValueError. The exception is caught by the outer loop in each source's `fetch()`, so only that camera is lost. Other cameras continue processing.

**Root cause:** The C25-02 fix only addressed ValueError guards in `parse_sensor_field()`. Source modules that parse the same data types still lack the same guards.

**Impact:** Individual camera record lost. Not category-wide like the Geizhals path.

---

## Summary

- DBG26-01 (MEDIUM): MPIX_RE only matches "Megapixel" — silent data loss on format change
- DBG26-02 (LOW): ValueError in source module float() calls — individual camera crash
