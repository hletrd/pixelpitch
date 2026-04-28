# Critic Review (Cycle 19) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository critique after cycles 1-18 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-18) — Confirmed Resolved

All C18 fixes verified. No regressions from C18-01 through C18-08 except as noted below.

## New Findings

### CR19-01: Tablesorter column configuration regression — sensor-width parser on wrong column
**File:** `templates/pixelpitch.html`, lines 228-258
**Severity:** MEDIUM | **Confidence:** HIGH

The C18-08 fix added a `sensor-width` custom parser for numeric sorting of the Sensor Size column, but it hardcoded column index 2. On 8 of 9 pages (everything except "All Cameras"), the Category column is absent, shifting all column indices left by 1. Sensor Size is at index 1 on those pages, not 2.

This means the sensor-width parser is applied to the Resolution column on non-"all" pages, and the text parser is applied to Sensor Size. The C18-08 fix improved sorting on exactly 1 page (All Cameras) while breaking it on the other 8.

**Fix:** Conditional column index assignment in the Jinja2 template.

---

### CR19-02: `fetch_source` crashes on malformed GSMARENA_MAX_PAGES_PER_BRAND env var
**File:** `pixelpitch.py`, line 1046
**Severity:** LOW | **Confidence:** HIGH

The C18-04 fix wired the env var but didn't add error handling for `int()` conversion. An empty or non-numeric value causes a crash instead of a graceful fallback.

**Fix:** Wrap in try/except with default value.

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- CR19-01: Tablesorter column regression on non-"all" pages — MEDIUM
- CR19-02: fetch_source crashes on bad env var — LOW
