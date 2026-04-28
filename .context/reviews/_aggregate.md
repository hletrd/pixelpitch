# Aggregate Review (Cycle 19) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-18 Status

All previous fixes confirmed still working. No regressions. Gate tests pass (105 checks).

## Cross-Agent Agreement Matrix (Cycle 19 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| Tablesorter column indices wrong on non-"all" pages | code-reviewer, debugger, designer, critic, tracer, verifier | MEDIUM |
| `fetch_source` ValueError on bad env var | code-reviewer, critic, verifier, debugger | LOW |
| No JS-side test for tablesorter config | test-engineer | NEGLIGIBLE |

## Deduplicated New Findings (Ordered by Severity)

### C19-01: Tablesorter column indices wrong for non-"all" pages — regression from C18-08
**Sources:** C19-01, D19-01, UX19-01, CR19-01, T19-01, V19-01
**Severity:** MEDIUM | **Confidence:** HIGH (6-agent consensus)

The C18-08 fix added a custom `sensor-width` parser for numeric sorting of the Sensor Size column, but it hardcoded column index 2. On non-"all" pages (DSLR, mirrorless, fixed, rangefinder, camcorder, actioncam, smartphone, cinema), the Category column is absent, shifting all column indices left by 1. Sensor Size is at index 1 on those pages, not 2.

**Impact on `#table_with_pitch` (non-"all"):**
- Column 1 (Sensor Size): uses "text" parser → alphabetical sort instead of numeric
- Column 2 (Resolution): uses "sensor-width" parser → sorts by sensor width instead of MP

**Impact on `#table_without_pitch` (non-"all"):**
- Column 1 (Sensor Size): uses "text" parser → alphabetical sort instead of numeric
- Column 2 (Resolution): uses "sensor-width" parser → sorts by sensor width instead of MP

**Concrete failure scenario:**
1. User navigates to the DSLR page
2. Clicks "Sensor Size" column header
3. Table sorts alphabetically: "9.84 x 7.40 mm" appears after "35.9 x 23.9 mm"
4. User expected numeric sort (9.84 first, 35.9 last in ascending)

**Fix:** Use conditional Jinja2 blocks in the tablesorter header configuration:

```javascript
// Table with pitch column
$('#table_with_pitch').tablesorter($.extend({}, tsBase, {
  headers: {
    0: { sorter: "text" },  // Name
    {% if page == "all" %}
    1: { sorter: "text" },  // Category
    2: { sorter: "sensor-width" },  // Sensor Size
    3: { sorter: "digit" }, // Resolution
    4: { sorter: "digit" }, // Pixel Pitch
    5: { sorter: "digit" }  // Year
    {% else %}
    1: { sorter: "sensor-width" },  // Sensor Size
    2: { sorter: "digit" }, // Resolution
    3: { sorter: "digit" }, // Pixel Pitch
    4: { sorter: "digit" }  // Year
    {% endif %}
  }
}));

// Table without pitch column
$('#table_without_pitch').tablesorter($.extend({}, tsBase, {
  headers: {
    0: { sorter: "text" },  // Name
    {% if page == "all" %}
    1: { sorter: "text" },  // Category
    2: { sorter: "sensor-width" },  // Sensor Size
    3: { sorter: "digit" }, // Resolution
    4: { sorter: "digit" }  // Year
    {% else %}
    1: { sorter: "sensor-width" },  // Sensor Size
    2: { sorter: "digit" }, // Resolution
    3: { sorter: "digit" }  // Year
    {% endif %}
  }
}));
```

---

### C19-02: `fetch_source` ValueError on non-integer `GSMARENA_MAX_PAGES_PER_BRAND` env var
**Sources:** C19-02, CR19-02, V19-02, D19-02
**Severity:** LOW | **Confidence:** HIGH (4-agent consensus)

The C18-04 fix wired `GSMARENA_MAX_PAGES_PER_BRAND` through `fetch_source()` but didn't add error handling for `int()` conversion. If the env var is set to an empty string or non-numeric value, the entire `python pixelpitch.py source gsmarena` command crashes with an unhandled ValueError.

**Concrete failure:** `GSMARENA_MAX_PAGES_PER_BRAND=""` causes `int("")` → `ValueError`.

**Fix:** Add try/except with fallback:
```python
try:
    max_pages = int(os.environ.get("GSMARENA_MAX_PAGES_PER_BRAND", "2"))
except (ValueError, TypeError):
    max_pages = 2
```

---

### C19-03: No JS-side test for tablesorter column config correctness
**Sources:** TE19-01
**Severity:** NEGLIGIBLE | **Confidence:** MEDIUM

The tablesorter configuration is JavaScript that runs in the browser. The offline Python test suite cannot exercise it. The C19-01 column index bug would not be caught by automated tests. This is informational — not actionable in the current test framework.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics
- Total distinct new findings: 2 (1 MEDIUM, 1 LOW) + 1 NEGLIGIBLE (excluded from actionable count)
- Cross-agent consensus findings (3+ agents): 2 (C19-01, C19-02)
- 1 MEDIUM finding: tablesorter column indices wrong on non-"all" pages
- NEGLIGIBLE finding (TE19-01) excluded from actionable count
