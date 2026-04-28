# Debugger Review (Cycle 19) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository latent bug review after cycles 1-18 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-18) — Confirmed Resolved

All C18 fixes confirmed. Scatter plot hidden data exclusion, TYPE_FRACTIONAL_RE consolidation, CI env var wiring all working.

## New Findings

### D19-01: Tablesorter column mismatch on non-"all" pages — functional regression from C18-08
**File:** `templates/pixelpitch.html`, lines 228-258
**Severity:** MEDIUM | **Confidence:** HIGH

The C18-08 fix introduced a custom `sensor-width` parser and assigned it to column index 2 for both "all" and non-"all" pages. On non-"all" pages, the Category column is absent, so Sensor Size is at index 1, not 2.

**Failure mode:** On the DSLR page, the user clicks the "Resolution" column header to sort by megapixels. Instead, the table sorts by sensor width because the `sensor-width` parser is assigned to column 2 (Resolution) instead of column 1 (Sensor Size). The user sees cameras sorted by sensor width when they expected megapixel ordering.

Similarly, clicking "Sensor Size" sorts alphabetically (text parser) instead of numerically.

**Fix:** Conditional column index assignment based on `{% if page == "all" %}`.

---

### D19-02: `int()` on env var can crash `fetch_source` with unhandled ValueError
**File:** `pixelpitch.py`, line 1046
**Severity:** LOW | **Confidence:** HIGH

`int(os.environ.get("GSMARENA_MAX_PAGES_PER_BRAND", "2"))` has no error handling. An empty string or non-numeric value in the env var causes an unhandled crash.

**Failure mode:** CI or local user sets `GSMARENA_MAX_PAGES_PER_BRAND=""` (accidentally empty). `int("")` raises ValueError. The entire `python pixelpitch.py source gsmarena` command fails.

**Fix:** Add try/except with fallback to default value 2.

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- D19-01: Tablesorter column indices wrong for non-"all" pages — MEDIUM
- D19-02: int() on env var can crash — LOW
