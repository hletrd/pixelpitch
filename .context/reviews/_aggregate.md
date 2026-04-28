# Aggregate Review (Cycle 24) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-23 Status

All previous fixes confirmed still working. No regressions. Gate tests pass (123/123 checks). C23-01 (body-category fallback tests) implemented and verified.

## Cross-Agent Agreement Matrix (Cycle 24 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| TYPE_FRACTIONAL_RE misses "1/x.y inch" (space+inch) | critic, verifier, tracer, architect, debugger, test-engineer | LOW |
| parse_sensor_field misses bare 1-inch sensor type ("1") | critic, verifier, tracer, debugger, test-engineer | LOW |
| _parse_fields rstrip("</") strips chars not string | code-reviewer, debugger | LOW (previously deferred C3-08) |
| TYPE_FRACTIONAL_RE comment imprecise | document-specialist | LOW |
| SpecDerived.size shadows Spec.size | code-reviewer | LOW (maintainability, not correctness) |

## Deduplicated New Findings (Ordered by Severity)

### C24-01: TYPE_FRACTIONAL_RE does not match "1/x.y inch" (space before "inch")

**Sources:** CRIT24-01, V24-02, TR24-01, ARCH24-01, DBG24-01, TE24-01
**Severity:** LOW | **Confidence:** HIGH (6-agent consensus)

The `TYPE_FRACTIONAL_RE` pattern in `sources/__init__.py` line 68 is:
```
(1/[\d.]+)(?:\"|inch|-inch|-type|\s*type|″)
```

It has `inch` (no space) and `-inch` but lacks `\s*inch` (space before "inch"). The pattern has `\s*type` but not the corresponding `\s*inch`.

**Impact:** If a source page uses `1/2.3 inch` format, the sensor type is not extracted. Camera shows "unknown" sensor size.

**Real-world risk:** LOW — all current sources use `1/2.3"` (quote) or `1/2.3-inch` (hyphenated).

**Fix:** Change `inch` to `\s*inch` in the regex pattern, and add a test case.

---

### C24-02: parse_sensor_field does not extract bare 1-inch sensor type ("1"")

**Sources:** CRIT24-02, V24-03, TR24-02, DBG24-02, TE24-02
**Severity:** LOW | **Confidence:** HIGH (5-agent consensus)

`parse_sensor_field()` in `pixelpitch.py` uses `TYPE_FRACTIONAL_RE` which only matches fractional-inch formats (`1/x.y` prefix). The bare 1-inch format (`1"`) is not matched, even though `TYPE_SIZE` has a `"1"` key with value `(13.2, 8.8)`.

**Impact:** Camera with 1-inch sensor and no explicit mm dimensions in Geizhals data shows "unknown" sensor size.

**Real-world risk:** LOW — Geizhals typically includes mm dimensions.

**Fix:** Add a separate check after `TYPE_FRACTIONAL_RE` for bare 1-inch format. Add a test case.

---

### C24-03: _parse_fields rstrip("</") strips individual chars, not the string

**Sources:** CR24-02, DBG24-03, TE24-03
**Severity:** LOW | **Confidence:** HIGH

Previously deferred as C3-08. Still present. `rstrip("</")` strips any trailing `<`, `/`, or `"` characters, not the string `"</"`. A value ending in `"` would have it stripped.

**Status:** Remains deferred (C3-08). Adding to deferred list for continuity.

---

### C24-04: TYPE_FRACTIONAL_RE comment could be more precise

**Sources:** DOC24-01
**Severity:** LOW | **Confidence:** MEDIUM

The comment on pixelpitch.py lines 47-49 mentions "ASCII/Unicode quotes" as primary focus and "inch", "-type" as "etc." The `\s*type` alternative is not mentioned.

**Fix:** Update comment to list all suffix alternatives explicitly.

---

### C24-05: SpecDerived.size shadows Spec.size — maintainability concern

**Sources:** CR24-01
**Severity:** LOW | **Confidence:** MEDIUM

`SpecDerived.size` shadows `Spec.size`, making `spec.size` vs `spec.spec.size` confusing. After merge, both paths are preserved correctly (verified by testing). This is purely a readability/maintainability concern, not a correctness bug.

**Fix:** Consider renaming to clarify the relationship in a future refactor. Not urgent.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 5 (5 LOW)
- Cross-agent consensus findings (3+ agents): 2 (C24-01, C24-02)
- 0 MEDIUM findings
- 0 HIGH findings
- 5 LOW findings: TYPE_FRACTIONAL_RE gap (6-agent), 1-inch type gap (5-agent), rstrip (3-agent, previously deferred), comment precision (1-agent), SpecDerived shadowing (1-agent)
