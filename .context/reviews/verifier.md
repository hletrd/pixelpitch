# Verifier Review (Cycle 19) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository verification after cycles 1-18 fixes, focusing on NEW issues

## Previously Fixed (Cycles 1-18) — Verified

All 105 gate tests pass. All C18 fixes verified as correctly applied.

## Verification of C18 Fixes

### C18-01: Scatter plot excludes hidden data — VERIFIED
`if (!row.is(':visible')) return;` is present in the `.each()` callback. Hidden rows excluded from plot data.

### C18-02/C18-03: TYPE_FRACTIONAL_RE consolidation — VERIFIED
`SENSOR_TYPE_RE` removed from pixelpitch.py. `from sources import TYPE_FRACTIONAL_RE` at line 50. GSMArena also imports `TYPE_FRACTIONAL_RE`. Single source of truth confirmed.

### C18-04: CI GSMARENA_MAX_PAGES_PER_BRAND wired — VERIFIED (with caveat)
`fetch_source()` reads env var and passes to gsmarena.fetch(). However, no error handling on `int()` conversion — see V19-02.

### C18-05/C18-06/C18-07: Test additions — VERIFIED
Unicode quote test, Pentax KF/K-r/K-x test, TYPE_FRACTIONAL_RE tests all present and passing.

### C18-08: Sensor-size numeric sort — PARTIAL REGRESSION
Custom `sensor-width` parser added. Works correctly on "all" page. **Broken on non-"all" pages** — see V19-01.

## New Findings

### V19-01: Tablesorter column index mismatch on non-"all" pages — regression
**File:** `templates/pixelpitch.html`, lines 228-258
**Severity:** MEDIUM | **Confidence:** HIGH

Reproduced the column index mismatch by tracing the HTML template:

For `#table_with_pitch` on non-"all" page (e.g., DSLR):
```
<th>Name</th>     → index 0
<th>Sensor Size</th> → index 1
<th>Resolution</th>  → index 2
<th>Pixel Pitch</th> → index 3
<th>Year</th>       → index 4
```

Config assigns: `1: text, 2: sensor-width, 3: digit, 4: digit`
Correct would be: `1: sensor-width, 2: digit, 3: digit, 4: digit`

Sensor Size column uses "text" parser → alphabetical sort, not numeric.

**Fix:** Conditional header config based on `page == "all"`.

---

### V19-02: `int()` on env var without error handling
**File:** `pixelpitch.py`, line 1046
**Severity:** LOW | **Confidence:** HIGH

```python
max_pages = int(os.environ.get("GSMARENA_MAX_PAGES_PER_BRAND", "2"))
```

If env var is empty string: `int("")` → `ValueError: invalid literal for int() with base 10: ''`

**Fix:** try/except with fallback default.

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- V19-01: Tablesorter column indices wrong for non-"all" pages — MEDIUM
- V19-02: int() on env var crashes on bad input — LOW
