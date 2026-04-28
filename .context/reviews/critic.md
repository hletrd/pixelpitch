# Critic Review (Cycle 18) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository critique after cycles 1-17 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-17) — Confirmed Resolved

All C17 fixes verified. Pentax KP/KF/K-r/K-x now DSLR, Nikon Df DSLR, GSMArena Unicode quotes working, docstrings updated, sensors_db lazy-loaded.

## New Findings

### CR18-01: Scatter plot violates user's visibility expectations by including hidden rows
**File:** `templates/pixelpitch.html`, lines 337-346
**Severity:** MEDIUM | **Confidence:** HIGH

The scatter plot reads data from ALL table rows, including those hidden by the "Hide possibly invalid data" toggle. This is a user-trust issue — the user explicitly hid suspect data, but the plot still includes it. The user sees a data point in the scatter plot, clicks the corresponding row, and finds it's not visible in the table. This undermines confidence in the data quality.

**Fix:** Add `if (!row.is(':visible')) return;` in the scatter plot data collection loop.

---

### CR18-02: CI GSMARENA_MAX_PAGES_PER_BRAND env var is dead code — suggests incomplete wiring
**File:** `.github/workflows/github-pages.yml`, line 74; `pixelpitch.py`, lines 1021-1043
**Severity:** LOW | **Confidence:** HIGH

The CI workflow sets an environment variable that the Python code never reads. This is a code/configuration mismatch — it suggests the developer intended to control GSMArena pagination via CI but didn't complete the wiring. The result: CI always fetches 2 pages per brand regardless of the `GSMARENA_PAGES: "1"` setting.

**Fix:** Wire the env var through `fetch_source()` to `gsmarena.fetch(max_pages_per_brand=...)`, or remove the dead env var.

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- CR18-01: Scatter plot includes hidden data — MEDIUM
- CR18-02: CI env var dead code — LOW
