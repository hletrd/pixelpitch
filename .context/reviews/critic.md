# Critic Review (Cycle 27) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-26 fixes, focusing on NEW issues

## Previous Findings Status

C26-01 (MPIX_RE centralization) and C26-02 (ValueError guards in source modules) both implemented and verified. All previous fixes stable.

## New Findings

### CRIT27-01: PITCH_UM_RE missing "um" — incomplete DRY unification of pitch patterns

**File:** `sources/__init__.py`, line 66 vs `sources/gsmarena.py`, line 50
**Severity:** LOW | **Confidence:** HIGH

The C25-01 fix centralized SIZE_MM_RE and PITCH_UM_RE from pixelpitch.py into sources/__init__.py. The C26-01 fix centralized MPIX_RE. However, the shared PITCH_UM_RE pattern still does not cover all variants that local patterns handle:

- Shared `PITCH_UM_RE` (`sources/__init__.py` line 66): matches `µm`, `μm`, `microns`, `&micro;m`, `&#956;m` — but NOT `um`
- Local `PITCH_RE` (`sources/gsmarena.py` line 50): matches `µm`, `μm`, AND `um`

The `um` variant (plain ASCII "u" + "m") is commonly used in English technical documentation and on GSMArena spec pages. The shared pattern should be a superset of all local patterns to truly centralize pitch matching. Currently GSMArena's local pattern handles this, so no data is lost, but the shared pattern is incomplete as a single source of truth.

**Fix:** Add `um` to the shared `PITCH_UM_RE` alternation in `sources/__init__.py`.

---

### CRIT27-02: parse_existing_csv year validation gap — accepts year=0 and negative years

**File:** `pixelpitch.py`, line 336
**Severity:** LOW | **Confidence:** HIGH

The CSV parser blindly converts the year column to int without any range validation: `year = int(year_str) if year_str else None`. While source modules use `parse_year()` which only matches 19xx/20xx, a corrupted or manually edited CSV could introduce year=0 or negative years. The template renders these verbatim, showing "0" or "-1" on the website.

This is a defensive hardening issue, not a current runtime bug.

**Fix:** Add a range check after int conversion: `year = int(year_str) if year_str and 1900 <= int(year_str) <= 2100 else None`.

---

## Summary

- CRIT27-01 (LOW): PITCH_UM_RE missing "um" — incomplete DRY unification
- CRIT27-02 (LOW): parse_existing_csv year validation gap
