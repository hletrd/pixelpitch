# Aggregate Review (Cycle 25) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-24 Status

All previous fixes confirmed still working. No regressions. Gate tests pass (222/222 checks). C24-01 (TYPE_FRACTIONAL_RE space+inch) and C24-02 (bare 1-inch type) implemented and verified.

## Cross-Agent Agreement Matrix (Cycle 25 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| SIZE_RE/PITCH_RE less robust than shared patterns | code-reviewer, critic, verifier, tracer, architect, debugger, test-engineer | MEDIUM |
| parse_sensor_field ValueError crash on malformed float | code-reviewer, critic, verifier, tracer, debugger, test-engineer | MEDIUM |
| parse_sensor_field docstring format limitations | document-specialist | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C25-01: SIZE_RE and PITCH_RE in pixelpitch.py are less robust than shared patterns in sources/__init__.py

**Sources:** CR25-01, CRIT25-01, V25-02, V25-03, TR25-02, ARCH25-01, DBG25-02, TE25-01
**Severity:** MEDIUM | **Confidence:** HIGH (8-agent consensus)

The Geizhals-specific regex patterns in `pixelpitch.py` are significantly less robust than the shared patterns in `sources/__init__.py`:

- `SIZE_RE` (pixelpitch.py line 42): only matches ASCII `x` with no spaces. Does not match Unicode `×` or spaces around separator.
- `SIZE_MM_RE` (sources/__init__.py line 65): matches `x`, `×`, optional spaces, case-insensitive.

- `PITCH_RE` (pixelpitch.py line 43): only matches micro sign `µ` (U+00B5). Does not match Greek mu `μ` (U+03BC), "microns", "um", or HTML entities.
- `PITCH_UM_RE` (sources/__init__.py line 66): matches all the above variants.

This is a DRY violation — `TYPE_FRACTIONAL_RE` was already centralized (imported from sources), but `SIZE_RE` and `PITCH_RE` were not.

**Impact:** If Geizhals sensor text uses `×` instead of `x`, or `μ` instead of `µ`, or spaces around dimension separator, sensor data is silently lost.

**Fix:** Import `SIZE_MM_RE` and `PITCH_UM_RE` from `sources` in `pixelpitch.py`, replacing the local `SIZE_RE` and `PITCH_RE`. Follow the same pattern already used for `TYPE_FRACTIONAL_RE`. Update `parse_sensor_field()` and `extract_specs()` to use the imported patterns. Update the `MPIX_RE` in pixelpitch.py similarly (it only matches "Megapixel", not "MP" or "Mega pixels"). Add tests for the expanded format support.

---

### C25-02: parse_sensor_field has no ValueError guard on float() calls

**Sources:** CR25-02, CRIT25-02, V25-04, TR25-01, DBG25-01, TE25-02
**Severity:** MEDIUM | **Confidence:** MEDIUM (6-agent consensus)

`parse_sensor_field()` calls `float(size_match.group(1))` and `float(pitch_match.group(1))` without try/except. The regex `[\d\.]+` allows multiple dots, and `float("36.0.1")` raises `ValueError`. This exception propagates up through `extract_specs` → `get_category` → `render_html`, where the outer try/except drops the entire Geizhals category.

**Impact:** A single malformed sensor field from Geizhals HTML can cause all cameras in that category to be lost for the deployment cycle.

**Fix:** Add try/except ValueError around float() calls in `parse_sensor_field()`, returning None for unparseable values. Add a test for the malformed input case.

---

### C25-03: parse_sensor_field docstring does not mention format limitations

**Sources:** DOC25-01
**Severity:** LOW | **Confidence:** MEDIUM

The docstring examples only show ASCII `x` and micro sign `µ`. If the regex is upgraded, the docstring should be updated.

**Status:** Deferred — will be addressed as part of C25-01 fix (regex upgrade implies docstring update).

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 3 (2 MEDIUM, 1 LOW)
- Cross-agent consensus findings (3+ agents): 2 (C25-01: 8-agent, C25-02: 6-agent)
- 2 MEDIUM findings: SIZE_RE/PITCH_RE inconsistency, ValueError guard missing
- 1 LOW finding: docstring limitations (deferred with C25-01)
