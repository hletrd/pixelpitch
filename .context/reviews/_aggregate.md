# Aggregate Review (Cycle 46) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-45 Status

All previous fixes confirmed still working. No regressions in core logic. Gate tests pass. C45 fixes (GSMArena decimal MP regex split fix, decimal MP tests) verified working correctly.

## Cross-Agent Agreement Matrix (Cycle 46 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| matched_sensors not preserved in merge_camera_data | CR46-01, CRIT46-01, V46-01, TR46-01, ARCH46-01, DBG46-01, TE46-01 | MEDIUM |
| LENS_RE dead code in gsmarena.py | CR46-02 | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C46-01: matched_sensors not preserved in merge_camera_data — data loss

**Sources:** CR46-01, CRIT46-01, V46-01, TR46-01, ARCH46-01, DBG46-01, TE46-01
**Severity:** MEDIUM | **Confidence:** HIGH (7-agent consensus, verified by live execution)

The `merge_camera_data` function preserves `type`, `size`, `pitch`, `mpix`, `year`, `area` fields from existing data when new data has `None`. However, `matched_sensors` is never checked for preservation. When `derive_spec` is called without `sensors_db` (or with an empty `sensors_db`), it returns `matched_sensors=[]`. The merge code treats `[]` as "we have data" (not `None`), so it overwrites existing sensor matches from the previous CSV with an empty list.

**Root cause:** `matched_sensors=[]` and `matched_sensors=None` are semantically different (empty after checking vs. not checked), but `derive_spec` always returns `[]` regardless of whether the sensors database was consulted.

**Failure scenario:**
1. Previous CSV has `matched_sensors=['IMX309', 'IMX366', 'IMX609']` for Canon R5
2. `sensors.json` is temporarily unavailable (missing/corrupt)
3. `derive_specs` -> `derive_spec(spec, {})` -> `matched_sensors=[]`
4. `merge_camera_data` overwrites existing matches with `[]`
5. CSV download loses all sensor match data

**Fix:**
1. In `derive_spec`: return `matched_sensors=None` when `sensors_db` is `None` or falsy, and `matched_sensors=[]` only when `sensors_db` was consulted but found no matches
2. In `merge_camera_data`: add `matched_sensors` preservation check: `if new_spec.matched_sensors is None and existing_spec.matched_sensors is not None: new_spec.matched_sensors = existing_spec.matched_sensors`
3. In `write_csv`: handle `matched_sensors=None` the same as `matched_sensors=[]` (write empty string)

---

### C46-02: LENS_RE dead code in gsmarena.py

**Sources:** CR46-02
**Severity:** LOW | **Confidence:** HIGH

The `LENS_RE` regex is defined at module level (`sources/gsmarena.py`, lines 45-50) but never referenced anywhere in the codebase. This is dead code similar to the C44-01 `FORMAT_TO_MM` removal in `cined.py`.

**Fix:** Remove the `LENS_RE` definition entirely.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 2 (C46-01, C46-02)
- Cross-agent consensus findings (3+ agents): 1 (C46-01 with 7 agents)
- Highest severity: MEDIUM (C46-01)
- Actionable findings: 2
- Verified safe / deferred: 0
