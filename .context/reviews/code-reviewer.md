# Code Review (Cycle 19) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-18 fixes, focusing on NEW issues missed or introduced by previous fixes

## Previously Fixed (Cycles 1-18) — Confirmed Resolved

All C18 fixes verified. Scatter plot visibility filter, TYPE_FRACTIONAL_RE consolidation, CI env var wiring, Unicode quote tests, Pentax KF/K-r/K-x tests, sensor-size numeric sort — all intact. Gate tests pass (105 checks).

## New Findings

### C19-01: Tablesorter column indices wrong for non-"all" pages — regression from C18-08
**File:** `templates/pixelpitch.html`, lines 228-258
**Severity:** MEDIUM | **Confidence:** HIGH

The C18-08 fix added a custom `sensor-width` tablesorter parser but assigned it to column index 2 unconditionally. On non-"all" pages (DSLR, mirrorless, fixed, etc.), there is no Category column, so all columns shift left by 1:

**`#table_with_pitch` column mapping:**
- "all" page: Name(0), Category(1), **Sensor Size(2)**, Resolution(3), Pixel Pitch(4), Year(5)
- non-"all" page: Name(0), **Sensor Size(1)**, Resolution(2), Pixel Pitch(3), Year(4)

**Current config for non-"all" `#table_with_pitch`:**
```
0: { sorter: "text" },         # Name — OK
1: { sorter: "text" },         # Sensor Size — WRONG (should be sensor-width)
2: { sorter: "sensor-width" }, # Resolution — WRONG (should be digit)
3: { sorter: "digit" },       # Pixel Pitch — OK (by coincidence)
4: { sorter: "digit" },       # Year — OK (by coincidence)
```

**Same issue in `#table_without_pitch` for non-"all":**
- Current: 1:text, 2:sensor-width, 3:digit
- Correct: 1:sensor-width, 2:digit, 3:digit

**Concrete failure:** On the DSLR page, clicking "Sensor Size" header sorts alphabetically ("9.84 x 7.40 mm" after "35.9 x 23.9 mm") because the text parser is applied instead of sensor-width. Clicking "Resolution" sorts by sensor width instead of MP count.

**Fix:** Use conditional Jinja2 blocks to assign the correct column index based on `page == "all"`:

For `#table_with_pitch` non-"all":
```javascript
{% if page == "all" %}
1: { sorter: "text" },        // Category
2: { sorter: "sensor-width" }, // Sensor Size
3: { sorter: "digit" },       // Resolution
4: { sorter: "digit" },       // Pixel Pitch
5: { sorter: "digit" }        // Year
{% else %}
1: { sorter: "sensor-width" }, // Sensor Size
2: { sorter: "digit" },       // Resolution
3: { sorter: "digit" },       // Pixel Pitch
4: { sorter: "digit" }        // Year
{% endif %}
```

---

### C19-02: `fetch_source` ValueError on non-integer `GSMARENA_MAX_PAGES_PER_BRAND` env var
**File:** `pixelpitch.py`, line 1046
**Severity:** LOW | **Confidence:** MEDIUM

The C18-04 fix added `int(os.environ.get("GSMARENA_MAX_PAGES_PER_BRAND", "2"))` without try/except. If the env var is accidentally set to a non-integer (e.g., empty string, "abc"), the entire `fetch_source` command crashes with an unhandled ValueError instead of falling back to the default of 2.

**Concrete failure:** `GSMARENA_MAX_PAGES_PER_BRAND=""` causes `int("")` → `ValueError: invalid literal for int() with base 10: ''`

**Fix:** Wrap in try/except with fallback:
```python
try:
    max_pages = int(os.environ.get("GSMARENA_MAX_PAGES_PER_BRAND", "2"))
except (ValueError, TypeError):
    max_pages = 2
```

---

## Summary
- NEW findings: 2 (1 MEDIUM, 1 LOW)
- C19-01: Tablesorter column indices wrong for non-"all" pages — MEDIUM
- C19-02: fetch_source ValueError on bad env var — LOW
