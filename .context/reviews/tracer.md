# Tracer Review (Cycle 18) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository causal tracing after cycles 1-17 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-17) — Confirmed Resolved

All C17 fixes confirmed. Pentax, Nikon Df, GSMArena quotes, sensors_db lazy load all working.

## New Findings

### T18-01: Scatter plot data collection ignores row visibility — hidden invalid data leaks into plot
**File:** `templates/pixelpitch.html`, lines 337-346
**Severity:** MEDIUM | **Confidence:** HIGH

Traced data flow:
1. `applyInvalidFilter()` hides rows with `row.hide()` when "Hide possibly invalid data" is checked
2. User clicks "Create Scatter Plot"
3. `createPlot()` iterates ALL `#table_with_pitch tbody tr` elements
4. No visibility check is performed
5. Hidden rows with suspicious pitch > 10 µm appear as data points
6. User sees outlier in scatter plot that isn't visible in the table

**Concrete failure:** A misclassified camera with 12 µm pixel pitch is hidden by the filter. The scatter plot shows this as a data point at 12 µm. The user can't find the corresponding row in the table.

**Fix:** Add `if (!row.is(':visible')) return;` in the scatter plot `.each()` callback.

---

### T18-02: CI GSMARENA_MAX_PAGES_PER_BRAND env var is read by nobody
**File:** `.github/workflows/github-pages.yml`, line 74; `pixelpitch.py`, lines 1031-1033
**Severity:** LOW | **Confidence:** HIGH

Traced configuration path:
1. CI sets `GSMARENA_MAX_PAGES_PER_BRAND: "1"` as env var
2. `python pixelpitch.py source gsmarena --limit 150 --out dist` is executed
3. `fetch_source()` calls `module.fetch(limit=limit)` — only passes `limit`
4. `gsmarena.fetch()` uses `max_pages_per_brand=2` (default) — never reads env var
5. CI fetches 2 pages per brand instead of the intended 1

**Fix:** Wire the env var through `fetch_source()` or add CLI flag.

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- T18-01: Scatter plot includes hidden data — MEDIUM
- T18-02: CI env var dead code — LOW
