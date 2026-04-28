# Aggregate Review (Cycle 18) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-17 Status

All previous fixes confirmed still working. No regressions. Gate tests pass (98 checks).

## Cross-Agent Agreement Matrix (Cycle 18 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| Scatter plot includes hidden/invalid data | designer, critic, tracer, debugger | MEDIUM |
| SENSOR_TYPE_RE in pixelpitch.py doesn't match Unicode quotes | code-reviewer, verifier, debugger | LOW |
| Three divergent sensor-type regexes (DRY violation) | code-reviewer, architect | LOW |
| CI GSMARENA_MAX_PAGES_PER_BRAND env var is dead code | code-reviewer, critic, tracer | LOW |
| No test for GSMArena Unicode quote regex | test-engineer | LOW |
| No test for Pentax KF/K-r/K-x DSLR classification | test-engineer | LOW |
| No test for SENSOR_TYPE_RE in pixelpitch.py | test-engineer | LOW |
| SENSOR_TYPE_RE has no ASCII-only comment | document-specialist | NEGLIGIBLE |
| Sensor Size column sorts as text, not numerically | designer | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C18-01: Scatter plot includes hidden/invalid data points — user-trust violation
**Sources:** UX18-01, CR18-01, T18-01, D18-01
**Severity:** MEDIUM | **Confidence:** HIGH (4-agent consensus)

When the "Hide possibly invalid data" toggle is active (default ON), hidden rows are excluded from the table but still included in the scatter plot. The `createPlot()` function iterates over all `#table_with_pitch tbody tr` elements without checking `row.is(':visible')`.

**Concrete failure scenario:**
1. Default: "Hide possibly invalid data" is checked
2. A camera with pixel pitch > 10 µm is hidden from the table
3. User clicks "Create Scatter Plot"
4. The hidden outlier appears as a data point in the scatter plot
5. User sees data in the plot that is not visible in the table

**Fix:** Add `if (!row.is(':visible')) return;` at the start of the `.each()` callback in `createPlot()`.

---

### C18-02: `SENSOR_TYPE_RE` in pixelpitch.py doesn't match Unicode quotes
**Sources:** C18-01, V18-01, D18-02
**Severity:** LOW | **Confidence:** MEDIUM (3-agent consensus)

The C17-03 fix updated GSMArena's `SENSOR_FORMAT_RE` to match Unicode curly quotes (U+2033), but the `SENSOR_TYPE_RE` in pixelpitch.py (line 43) still only matches ASCII double-quotes. If Geizhals HTML ever uses Unicode quotes, `parse_sensor_field()` would silently lose the sensor type.

Practical impact is LOW since Geizhals HTML title attributes typically use ASCII quotes.

**Fix:** Update `SENSOR_TYPE_RE` to `re.compile(r'(1/[\d.]+)(?:\"|″)')`, or reuse `TYPE_FRACTIONAL_RE`.

---

### C18-03: Three divergent sensor-type regexes — DRY violation
**Sources:** C18-02, A18-01
**Severity:** LOW | **Confidence:** HIGH (2-agent consensus)

The fractional-inch sensor type pattern is defined independently in three places with divergent capabilities:
1. `SENSOR_TYPE_RE` in pixelpitch.py — ASCII-only
2. `TYPE_FRACTIONAL_RE` in sources/__init__.py — comprehensive
3. `SENSOR_FORMAT_RE` in gsmarena.py — ASCII + Unicode quotes

Changes must be synchronized across all three, violating Single Source of Truth.

**Fix:** Import `TYPE_FRACTIONAL_RE` from `sources/__init__.py` in pixelpitch.py and gsmarena.py, or create a shared regex module.

---

### C18-04: CI `GSMARENA_MAX_PAGES_PER_BRAND` env var is dead code
**Sources:** C18-03, CR18-02, T18-02
**Severity:** LOW | **Confidence:** HIGH (3-agent consensus)

The CI workflow sets `GSMARENA_MAX_PAGES_PER_BRAND: "1"` as an environment variable, but `fetch_source()` never reads it. GSMArena always uses `max_pages_per_brand=2`. The env var suggests incomplete wiring.

**Fix:** Wire the env var through `fetch_source()` to `gsmarena.fetch(max_pages_per_brand=...)`, or remove the dead env var from CI.

---

### C18-05: No test for GSMArena Unicode curly-quote regex
**Sources:** TE18-01
**Severity:** LOW | **Confidence:** HIGH

C17-03 fixed GSMArena's `SENSOR_FORMAT_RE` to match Unicode quotes, but no test was added. Regression risk.

**Fix:** Add a test verifying `SENSOR_FORMAT_RE.search('1/1.3″')` matches.

---

### C18-06: No test for Pentax KF, K-r, K-x DSLR classification
**Sources:** TE18-02
**Severity:** LOW | **Confidence:** HIGH

The test CSV includes Pentax KP and Nikon Df but not Pentax KF, K-r, or K-x. Only partial coverage of the C17-01 fix.

**Fix:** Add these models to the test CSV and verify DSLR classification.

---

### C18-07: No test for `SENSOR_TYPE_RE` in pixelpitch.py
**Sources:** TE18-03
**Severity:** LOW | **Confidence:** MEDIUM

No dedicated test for `SENSOR_TYPE_RE` used in Geixhals parsing. If the regex is broken, sensor type extraction silently fails.

**Fix:** Add a basic test verifying `SENSOR_TYPE_RE` matches standard patterns.

---

### C18-08: Sensor Size column sorts as text, not numerically
**Sources:** UX18-02
**Severity:** LOW | **Confidence:** HIGH

Clicking the Sensor Size column header sorts alphabetically ("9.84 x 7.40 mm" after "35.9 x 23.9 mm") instead of numerically by width.

**Fix:** Add a custom tablesorter parser that reads `data-sensor-width` attribute.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics
- Total distinct new findings: 8 (1 MEDIUM, 6 LOW, 1 NEGLIGIBLE excluded from count)
- Cross-agent consensus findings (3+ agents): 3 (C18-01, C18-04, C18-02)
- All cycle 1-17 fixes remain intact
- 1 MEDIUM finding: scatter plot includes hidden data
- NEGLIGIBLE finding (DS18-01: ASCII-only comment) excluded from actionable count
