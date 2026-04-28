# Aggregate Review (Cycle 27) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-26 Status

All previous fixes confirmed still working. No regressions. Gate tests pass (all checks). C26-01 (MPIX_RE centralization) and C26-02 (ValueError guards in source modules) implemented and verified.

## Cross-Agent Agreement Matrix (Cycle 27 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| PITCH_UM_RE missing "um" — doc says it matches but regex doesn't | code-reviewer, critic, verifier, tracer, architect, debugger, document-specialist, test-engineer | LOW |
| parse_existing_csv accepts year=0 and negative years without validation | code-reviewer, critic, verifier, tracer, debugger, test-engineer | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C27-01: PITCH_UM_RE missing "um" — doc/code mismatch (comment claims support but regex does not match)

**Sources:** CR27-01, CRIT27-01, V27-02, TR27-01, ARCH27-01, DBG27-01, DOC27-01, TE27-01
**Severity:** LOW | **Confidence:** HIGH (8-agent consensus)

The comment in `pixelpitch.py` line 44 explicitly states:
```
# PITCH_UM_RE matches µm, μm (Greek mu), "microns", "um", and HTML entities.
```

However, the actual regex in `sources/__init__.py` line 66 does NOT include `um`:
```python
PITCH_UM_RE = re.compile(r"([\d.]+)\s*(?:µm|microns?|μm|&micro;m|&#0?956;m)", re.IGNORECASE)
```

The GSMArena source has its own local `PITCH_RE` (line 50) that includes `um`:
```python
PITCH_RE = re.compile(r"([\d.]+)\s*(?:µm|μm|um)", re.IGNORECASE)
```

**Impact:** The comment is misleading — it claims "um" is supported but the code does not match it. Currently no data path triggers this (Geizhals uses µm/μm, GSMArena has its own pattern), so this is not a runtime bug. However, it is:
1. A doc/code mismatch (the comment lies about what the code does)
2. A DRY incompleteness (the shared pattern should be a superset of all local patterns)
3. A latent bug (if a source parsed via `parse_sensor_field()` ever uses "um", pitch is silently lost)

**Fix:** Add `um` to the shared `PITCH_UM_RE` alternation: `(?:µm|um|microns?|μm|&micro;m|&#0?956;m)`. This makes the regex match the existing documentation comment and makes the shared pattern a true superset of GSMArena's local PITCH_RE. Add a test for "um" matching.

---

### C27-02: parse_existing_csv accepts year=0 and negative years without validation

**Sources:** CR27-02, CRIT27-02, V27-03, TR27-02, DBG27-02, TE27-02
**Severity:** LOW | **Confidence:** MEDIUM (6-agent consensus)

The CSV parser converts the year column with `int(year_str) if year_str else None` (line 336), accepting any integer value including 0 and negative numbers. The template's Jinja2 `{% if spec.spec.year %}` treats int(0) as truthy, so year=0 displays as "0" on the website.

No current source produces year=0 or negative years — `parse_year()` only matches `\b(19\d{2}|20\d{2})\b`. This requires corrupted CSV input to trigger.

**Impact:** A corrupted or manually edited CSV with year=0 would display "0" on the website. Very low probability.

**Fix:** Add a range check: `year = int(year_str) if year_str and 1900 <= int(year_str) <= 2100 else None`. This requires a try/except for non-integer strings. Add test cases for year=0, year=-1, and year=99999.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 2 (both LOW)
- Cross-agent consensus findings (3+ agents): 2 (C27-01: 8-agent, C27-02: 6-agent)
- 2 LOW findings: PITCH_UM_RE doc/code mismatch, parse_existing_csv year validation gap
