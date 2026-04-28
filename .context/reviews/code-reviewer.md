# Code Review (Cycle 18) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-17 fixes, focusing on NEW issues missed or introduced by previous fixes

## Previously Fixed (Cycles 1-17) — Confirmed Resolved

All previous fixes remain intact. Gate tests pass (98 checks). C17-01 (Pentax KP/KF/K-r/K-x), C17-02 (Nikon Df), C17-03 (GSMArena Unicode quotes), C17-04 (openMVG docstring), C17-05 (sensors_db lazy load) all verified as correctly applied.

## New Findings

### C18-01: `SENSOR_TYPE_RE` in pixelpitch.py doesn't handle Unicode quotes — inconsistent with C17-03 fix
**File:** `pixelpitch.py`, line 43 (`SENSOR_TYPE_RE = re.compile(r"(1/[\d\.]+)\"")`)
**Severity:** LOW | **Confidence:** MEDIUM

The C17-03 fix updated GSMArena's `SENSOR_FORMAT_RE` to match Unicode curly quotes (`″`). However, `SENSOR_TYPE_RE` in pixelpitch.py itself still only matches ASCII double-quote `"`.

The central `TYPE_FRACTIONAL_RE` in `sources/__init__.py` correctly handles both `"|inch|-inch|-type|″`. Having three separate regexes for the same concept is a maintenance risk — format changes require updating all three.

For pixelpitch.py specifically, the `SENSOR_TYPE_RE` is used in `parse_sensor_field()` (line 510) to parse Geizhals HTML data. Geizhals sensor descriptions typically use ASCII quotes in their `title` attributes, so the practical risk is low. However, the inconsistency means that if any source ever uses Unicode quotes, the Geizhals parser would silently lose the sensor type.

**Fix:** Either (a) reuse `TYPE_FRACTIONAL_RE` from `sources/__init__.py`, or (b) update `SENSOR_TYPE_RE` to match both ASCII and Unicode quotes: `re.compile(r'(1/[\d.]+)(?:\"|″)')`.

---

### C18-02: `SENSOR_TYPE_RE` / `TYPE_FRACTIONAL_RE` / `SENSOR_FORMAT_RE` regex duplication — maintenance risk
**Files:** `pixelpitch.py` line 43, `sources/__init__.py` line 68, `sources/gsmarena.py` line 50
**Severity:** LOW | **Confidence:** HIGH

Three separate regex patterns exist for matching fractional-inch sensor types:
1. `SENSOR_TYPE_RE` in pixelpitch.py: `r"(1/[\d\.]+)\""` — ASCII-only quote
2. `TYPE_FRACTIONAL_RE` in sources/__init__.py: `r"(1/[\d.]+)(?:\"|inch|-inch|-type|\s*type|″)"` — comprehensive
3. `SENSOR_FORMAT_RE` in gsmarena.py: `r'(1/[\d.]+)(?:\"|″)'` — ASCII + Unicode quotes

If the sensor type format convention changes (e.g., a new suffix variant appears), all three must be updated independently. `TYPE_FRACTIONAL_RE` in `sources/__init__.py` is the most complete version and should be the single source of truth.

**Fix:** Replace `SENSOR_TYPE_RE` and `SENSOR_FORMAT_RE` with imports from `sources/__init__.py`, or at minimum ensure they all use the same suffix pattern.

---

### C18-03: `fetch_source()` doesn't pass `max_pages_per_brand` to GSMArena — CI env var is dead code
**File:** `pixelpitch.py`, lines 1021-1043; `.github/workflows/github-pages.yml`, line 74
**Severity:** LOW | **Confidence:** HIGH

The CI workflow sets `GSMARENA_MAX_PAGES_PER_BRAND: ${{ env.GSMARENA_PAGES }}` as an environment variable, but `fetch_source()` in pixelpitch.py only passes `limit` to `module.fetch()`. The `max_pages_per_brand` parameter of `gsmarena.fetch()` defaults to 2 and is never overridden.

The CI configuration suggests the intent was to control this parameter, but the code doesn't read the env var. The `fetch_source` CLI also lacks a `--max-pages` flag.

**Concrete failure:** CI workflow sets `GSMARENA_PAGES: "1"` but GSMArena fetch always uses `max_pages_per_brand=2`, fetching more pages than intended.

**Fix:** Either (a) have `fetch_source` read the env var and pass it as `max_pages_per_brand` to GSMArena, or (b) add a `--max-pages` CLI flag, or (c) remove the dead env var from the CI workflow.

---

## Summary
- NEW findings: 3 (all LOW)
- C18-01: SENSOR_TYPE_RE doesn't handle Unicode quotes — LOW
- C18-02: Three duplicated sensor-type regexes — maintenance risk — LOW
- C18-03: CI GSMARENA_MAX_PAGES_PER_BRAND env var is dead code — LOW
