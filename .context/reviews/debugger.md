# Debugger Review (Cycle 27) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-26 fixes

## Previous Findings Status

DBG26-01 (MPIX_RE only matches "Megapixel") and DBG26-02 (ValueError in source modules) both fixed in C26. DBG24-03 (rstrip) remains deferred.

## New Findings

### DBG27-01: PITCH_UM_RE does not match "um" — latent failure if Geizhals uses ASCII-only pitch

**File:** `sources/__init__.py`, line 66
**Severity:** LOW | **Confidence:** HIGH

**Failure mode:** If Geizhals (or any future source parsed via `parse_sensor_field()`) uses "5.12 um" (ASCII-only, no micro sign), `PITCH_UM_RE.search("5.12 um")` returns None. The pitch value is silently lost. If size and mpix are available, `derive_spec()` will compute a pitch from area/mpix, so the camera may still show a pitch value — but it would be computed rather than the authoritative value from the source.

**Root cause:** The shared `PITCH_UM_RE` alternation includes `µm`, `μm`, `microns`, `&micro;m`, `&#956;m` but not plain `um`. The GSMArena local `PITCH_RE` includes `um` but uses its own pattern.

**Impact:** Currently no data path triggers this — Geizhals uses µm/μm. This is a latent bug that would surface only if a data source starts using ASCII "um".

**Verified:** `PITCH_UM_RE.search("5.12 um")` returns None.

### DBG27-02: parse_existing_csv accepts year=0 — displays "0" on website

**File:** `pixelpitch.py`, line 336
**Severity:** LOW | **Confidence:** HIGH

**Failure mode:** A corrupted or manually edited CSV with year=0 passes through validation. The template's Jinja2 `{% if spec.spec.year %}` treats int(0) as truthy (Jinja2 follows Python truthiness for integers), so "0" is displayed on the website.

**Root cause:** `year = int(year_str) if year_str else None` has no range check. An empty string is falsy (correctly produces None), but "0" is truthy and produces year=0.

**Impact:** No current source produces year=0. Requires corrupted CSV input.

**Verified:** CSV with year=0 produces `spec.year == 0`, which displays as "0" in the template.

---

## Summary

- DBG27-01 (LOW): PITCH_UM_RE missing "um" — latent failure if source uses ASCII "um"
- DBG27-02 (LOW): parse_existing_csv year=0 — displays "0" on website
