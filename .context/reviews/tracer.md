# Tracer Review (Cycle 19) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28
**Scope:** Full repository causal tracing after cycles 1-18 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-18) — Confirmed Resolved

All C18 fixes confirmed. Scatter plot hidden data, CI env var wiring, TYPE_FRACTIONAL_RE consolidation all working.

## New Findings

### T19-01: Tablesorter column index mismatch on non-"all" pages — regression from C18-08
**File:** `templates/pixelpitch.html`, lines 228-258
**Severity:** MEDIUM | **Confidence:** HIGH

Traced data flow:
1. C18-08 fix added `sensor-width` custom parser to column index 2
2. On "all" page: Name(0), Category(1), Sensor Size(2), ... — column 2 IS Sensor Size ✓
3. On non-"all" page: Name(0), Sensor Size(1), Resolution(2), ... — column 2 IS Resolution ✗
4. User clicks "Sensor Size" header on DSLR page
5. Tablesorter applies column 1's parser: "text" (alphabetical sort)
6. "9.84 x 7.40 mm" sorts after "35.9 x 23.9 mm" alphabetically
7. User sees incorrect sort order

The root cause is that the C18-08 fix did not account for the variable column count caused by the conditional `{% if page == "all" %}` Category column.

**Fix:** Make the header config conditional on `page == "all"`.

---

## Summary
- NEW findings: 1 (MEDIUM)
- T19-01: Tablesorter column index mismatch on non-"all" pages — MEDIUM
