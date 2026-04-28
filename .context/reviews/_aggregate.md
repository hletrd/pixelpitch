# Aggregate Review (Cycle 44) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-43 Status

All previous fixes confirmed still working. No regressions in core logic. Gate tests pass. C43 fixes (GSMArena/CineD spec.size provenance, redundant pitch write clarification) all verified working correctly via live code execution in the verifier agent.

## Verification Results (Cycle 44)

- derive_spec with type='1/1.3', size=None → derived.size = (9.84, 7.40) from TYPE_SIZE — OK
- merge_camera_data preserves measured Geizhals spec.size when new has spec.size=None — OK
- test_merge_gsmarena_measured_preserved — exists and tests spec.size, derived.size, area
- test_merge_size_consistency — exists and tests C42-01 consistency fix
- PHONE_TYPE_SIZE and TYPE_SIZE import — already removed from gsmarena.py (C43-01 complete)

## Cross-Agent Agreement Matrix (Cycle 44 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| FORMAT_TO_MM dict in cined.py is dead code after C43-01 | CR44-01, CRIT44-01, ARCH44-01, TR44-01, DBG44-03, DOC44-01, TE44-02 | LOW |
| CineD fmt_m/fmt variables and `if size is None and fmt:` block are dead code | CR44-02, CRIT44-02, ARCH44-02, TR44-01, DBG44-03 | LOW |
| CineD docstring says FORMAT_TO_MM 'kept for regex coverage test' but no such test exists | CRIT44-03, DOC44-01 | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C44-01: FORMAT_TO_MM dict in cined.py is dead code after C43-01 fix — should be removed

**Sources:** CR44-01, CRIT44-01, ARCH44-01, TR44-01, DBG44-03, DOC44-01, TE44-02
**Severity:** LOW | **Confidence:** HIGH (7-agent consensus)

After C43-01 removed `size = FORMAT_TO_MM.get(fmt.lower())`, the FORMAT_TO_MM dict at module level (lines 37-51) is defined but never referenced by any executable code. It's only mentioned in comments and the docstring. The module docstring claims "The FORMAT_TO_MM table is kept for the regex coverage test only" but no test references it. This is dead code that could mislead future maintainers into thinking it's still used.

**Fix:** Remove the FORMAT_TO_MM dict entirely. Update the module docstring to remove the reference.

---

### C44-02: CineD fmt_m/fmt variables and `if size is None and fmt:` block are dead code after C43-01 fix

**Sources:** CR44-02, CRIT44-02, ARCH44-02, TR44-01, DBG44-03
**Severity:** LOW | **Confidence:** HIGH (5-agent consensus)

After C43-01 removed `FORMAT_TO_MM.get(fmt.lower())`, the format extraction regex (fmt_m, lines 92-97) and fmt variable assignment are computed but never used. The `if size is None and fmt:` block (lines 106-119) contains only a `pass` statement with comments. The entire format detection code path is dead code that wastes computation and misleads about the data flow.

**Fix:** Remove the fmt_m regex search, fmt assignment, and the `if size is None and fmt:` block with its comments. Also remove the format extraction regex (the `fmt_m` pattern) from the function.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 2 (C44-01, C44-02)
- Cross-agent consensus findings (3+ agents): 2 (C44-01 with 7 agents, C44-02 with 5 agents)
- Highest severity: LOW (both findings)
- Actionable findings: 2
- Verified safe / deferred: 0
