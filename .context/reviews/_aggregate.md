# Aggregate Review (Cycle 22) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-21 Status

All previous fixes confirmed still working. No regressions. Gate tests pass. C21-01 (SpecDerived stale fields), C21-02 (Sony RX/DSC naming), C21-03 (mpix preservation), and C21-04 (test coverage) fixes verified in code.

## Cross-Agent Agreement Matrix (Cycle 22 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| Year-change `elif` attached to wrong `if` (C21-01 regression) | code-reviewer, critic, verifier, tracer, debugger, test-engineer, architect | MEDIUM |
| Sony DSC hyphen inconsistency between Model Name and URL paths | code-reviewer, test-engineer | LOW |
| No test for year-change log in merge | test-engineer | LOW |
| No test for Sony DSC hyphen normalisation | test-engineer | LOW |
| Field preservation logic is ad-hoc/fragile | architect, critic | LOW (architectural) |

## Deduplicated New Findings (Ordered by Severity)

### C22-01: Year-change `elif` attached to wrong `if` — C21-01 regression

**Sources:** C22-01, C22-CR01, V22-01, T22-01, D22-01, TE22-01, A22-01
**Severity:** MEDIUM | **Confidence:** HIGH (7-agent consensus)

The C21-01 fix inserted SpecDerived field preservation code between the Spec year-preservation `if` and the year-change `elif`. This changed the `elif`'s parent from the year-preservation `if` to the SpecDerived pitch-preservation `if`.

**Before C21-01 (correct):**
```python
if new_spec.spec.year is None and existing_spec.spec.year is not None:
    new_spec.spec.year = existing_spec.spec.year
elif (new_spec.spec.year != existing_spec.spec.year):
    print("Year changed...")  # This fires when years differ
```

**After C21-01 (broken):**
```python
if new_spec.spec.year is None and ...:
    ...
if new_spec.pitch is None and existing_spec.pitch is not None:
    new_spec.pitch = existing_spec.pitch
elif (new_spec.spec.year != existing_spec.spec.year):
    print("Year changed...")  # Only fires when pitch is NOT None
```

**Impact:** Data correctness is NOT affected — years are still correctly set. Only the diagnostic log message is partially suppressed. When pitch is preserved from existing data, the year-change log is silently skipped.

**Fix:** Convert the `elif` to a standalone `if` after all preservation logic.

---

### C22-02: Sony DSC hyphen inconsistency between Model Name and URL paths

**Sources:** C22-02, TE22-02
**Severity:** LOW | **Confidence:** MEDIUM (2-agent consensus)

When the Sony name starts with "Sony" and falls through to URL-based parsing, hyphens in "DSC-HX400" become spaces. But when the name comes from "Model Name", the hyphen is preserved. This could create dedup mismatches.

**Fix:** Add `cleaned = re.sub(r'\bDSC-', 'DSC ', cleaned)` after the DSC uppercase normalizer.

---

### C22-03: No test for year-change log in merge

**Sources:** TE22-01
**Severity:** LOW | **Confidence:** HIGH

The merge function logs year changes, but no test verifies this diagnostic output. The C22-01 bug would have been caught if there was a test capturing stdout during merge.

---

### C22-04: No test for Sony DSC hyphen normalisation

**Sources:** TE22-02
**Severity:** LOW | **Confidence:** MEDIUM

Dependent on C22-02 fix. If a DSC-hyphen normalizer is added, it needs a test.

---

### C22-05: Field preservation logic is ad-hoc and fragile (architectural)

**Sources:** A22-01, C22-CR01
**Severity:** LOW | **Confidence:** HIGH (architectural, not a bug)

The merge function has 8 separate `if` statements for field preservation. This is fragile — inserting code in the middle (as C21-01 did) can break conditional chains. A generic helper would be more maintainable. However, this is a code quality improvement, not a correctness fix.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 5 (1 MEDIUM, 4 LOW)
- Cross-agent consensus findings (3+ agents): 1 (C22-01)
- 1 MEDIUM finding: Year-change `elif` misattachment
- 4 LOW findings: DSC hyphen, test gaps, architectural concern
