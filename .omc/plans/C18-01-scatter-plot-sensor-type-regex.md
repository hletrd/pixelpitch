# Plan: Cycle 18 Findings — Scatter Plot Hidden Data, Sensor-Type Regex DRY, CI Dead Code, Test Gaps

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** C18-01 through C18-08 (aggregate)

---

## Task 1: Fix scatter plot to exclude hidden/invalid data rows — C18-01 — DONE
**Finding:** C18-01 (4-agent consensus: designer, critic, tracer, debugger)
**Severity:** MEDIUM | **File:** `templates/pixelpitch.html`, lines 337-346
**Commit:** 0b0e547

### What was done
Added `if (!row.is(':visible')) return;` at the start of the `.each()` callback in `createPlot()`. Hidden rows (filtered by "Hide possibly invalid data") are now excluded from scatter plot data collection.

### Verification
- Gate tests pass (105 checks)
- Visual verification: scatter plot respects visibility filter

---

## Task 2: Fix `SENSOR_TYPE_RE` in pixelpitch.py to handle Unicode quotes — C18-02 — DONE (superseded by Task 3)
**Finding:** C18-02 (3-agent consensus: code-reviewer, verifier, debugger)
**Severity:** LOW | **File:** `pixelpitch.py`, line 43
**Commit:** 81729ab

### What was done
Instead of merely updating `SENSOR_TYPE_RE`, we consolidated it into the canonical `TYPE_FRACTIONAL_RE` from `sources/__init__.py`. This supersedes Task 2 by providing full Unicode + "inch" suffix support.

---

## Task 3: Consolidate sensor-type regexes — C18-03 — DONE
**Finding:** C18-03 (2-agent consensus: code-reviewer, architect)
**Severity:** LOW | **Files:** `pixelpitch.py` line 43, `sources/__init__.py` line 68, `sources/gsmarena.py` line 50
**Commit:** 81729ab

### What was done
- Removed `SENSOR_TYPE_RE` from `pixelpitch.py` and replaced all usages with `TYPE_FRACTIONAL_RE` from `sources/__init__.py`
- Removed `SENSOR_FORMAT_RE` from `sources/gsmarena.py` and replaced with `TYPE_FRACTIONAL_RE` import
- Single source of truth: `TYPE_FRACTIONAL_RE` in `sources/__init__.py` now handles all fractional-inch sensor type matching

### Verification
- Gate tests pass (105 checks)
- GSMArena `_phone_to_spec()` still extracts sensor formats correctly
- pixelpitch.py `parse_sensor_field()` still extracts sensor types correctly

---

## Task 4: Wire CI `GSMARENA_MAX_PAGES_PER_BRAND` env var — C18-04 — DONE
**Finding:** C18-04 (3-agent consensus: code-reviewer, critic, tracer)
**Severity:** LOW | **Files:** `pixelpitch.py` lines 1021-1043, `.github/workflows/github-pages.yml` line 74
**Commit:** e31ee4c

### What was done
Updated `fetch_source()` to read `GSMARENA_MAX_PAGES_PER_BRAND` env var and pass it as `max_pages_per_brand` to `gsmarena.fetch()`. Default is 2 (unchanged behavior when env var is not set). CI workflow's `GSMARENA_PAGES: "1"` setting now takes effect.

### Verification
- Gate tests pass (105 checks)
- Default behavior preserved (env var absent → max_pages_per_brand=2)

---

## Task 5: Add test for GSMArena Unicode curly-quote regex — C18-05 — DONE
**Finding:** C18-05 (test-engineer)
**Severity:** LOW | **File:** `tests/test_parsers_offline.py`
**Commit:** f9171b3

### What was done
Added `test_gsmarena_unicode_quotes()` with 4 checks:
- ASCII double-quote match
- Unicode curly quote (U+2033) match
- "-inch" suffix match
- No suffix → no match

### Verification
- All 4 new checks pass

---

## Task 6: Add Pentax KF, K-r, K-x DSLR classification tests — C18-06 — DONE
**Finding:** C18-06 (test-engineer)
**Severity:** LOW | **File:** `tests/test_parsers_offline.py`
**Commit:** f9171b3

### What was done
Added Pentax KF, K-r, K-x rows to the openMVG test CSV and 3 new expect() assertions verifying they are classified as "dslr".

### Verification
- All 3 new checks pass
- Test CSV row count: 10 → 13

---

## Task 7: Add test for `SENSOR_TYPE_RE` in pixelpitch.py — C18-07 — DONE (superseded by Task 3)
**Finding:** C18-07 (test-engineer)
**Severity:** LOW | **File:** `tests/test_parsers_offline.py`
**Commit:** f9171b3

### What was done
Since Task 3 replaced `SENSOR_TYPE_RE` with `TYPE_FRACTIONAL_RE`, the new `test_gsmarena_unicode_quotes()` test covers `TYPE_FRACTIONAL_RE` functionality including ASCII quotes, Unicode quotes, and "-inch" suffix. This supersedes a standalone SENSOR_TYPE_RE test.

---

## Task 8: Add custom tablesorter parser for Sensor Size column — C18-08 — DONE
**Finding:** C18-08 (designer)
**Severity:** LOW | **File:** `templates/pixelpitch.html`
**Commit:** 0b0e547

### What was done
Added custom tablesorter parser `sensor-width` that reads the `data-sensor-width` attribute for numeric sorting. Updated both `#table_with_pitch` and `#table_without_pitch` header configs to use `sorter: "sensor-width"` for the Sensor Size column.

### Verification
- Gate tests pass (105 checks)

---

## Deferred Findings

### DS18-01: SENSOR_TYPE_RE has no comment about ASCII-only limitation — NEGLIGIBLE — MOOT
**File:** `pixelpitch.py`, line 43
**Original Severity:** NEGLIGIBLE | **Confidence:** HIGH
**Status:** MOOT — `SENSOR_TYPE_RE` has been removed entirely (Task 3). The canonical `TYPE_FRACTIONAL_RE` handles both ASCII and Unicode quotes. No comment needed.

---

## Summary

All 8 tasks completed. 4 commits made:
1. `0b0e547` — fix(ui): scatter plot visibility + sensor-size numeric sort (C18-01, C18-08)
2. `81729ab` — refactor(regex): consolidate sensor-type regex into TYPE_FRACTIONAL_RE (C18-02, C18-03)
3. `e31ee4c` — fix(ci): wire GSMARENA_MAX_PAGES_PER_BRAND env var (C18-04)
4. `f9171b3` — test(sources): add Unicode quotes, Pentax KF/K-r/K-x, TYPE_FRACTIONAL_RE tests (C18-05, C18-06, C18-07)
