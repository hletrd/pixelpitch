# Aggregate Review (Cycle 32) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-31 Status

All previous fixes confirmed still working. No regressions. Gate tests pass (all checks). C31-01 through C31-04 all implemented and verified.

## Cross-Agent Agreement Matrix (Cycle 32 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| write_csv falsy checks silently drop 0.0 float values | code-reviewer, critic, verifier, tracer, debugger, test-engineer | LOW-MEDIUM |
| IR_MPIX_RE matches partial decimals without unit suffix | code-reviewer | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C32-01: write_csv uses truthy checks instead of None checks for float fields

**Sources:** CR32-01, CRIT32-01, V32-02, TR32-01, DBG32-01, TE32-01
**Severity:** LOW-MEDIUM | **Confidence:** HIGH (6-agent consensus)

Four fields in `write_csv` use Python truthiness (`if x`) instead of explicit None checks (`if x is not None`):

```python
area_str = f"{derived.area:.2f}" if derived.area else ""       # line 824
mpix_str = f"{spec.mpix:.1f}" if spec.mpix else ""             # line 825
pitch_str = f"{derived.pitch:.2f}" if derived.pitch else ""    # line 826
year_str = str(spec.year) if spec.year else ""                 # line 827
```

For float fields, `0.0` is falsy but is a valid float distinct from `None`. If any field is ever `0.0`, it would be written as empty string and read back as `None` by `parse_existing_csv`, causing silent data loss on CSV round-trip.

**Concrete scenario:**
1. `pixel_pitch(area, mpix)` returns `0.0` when `mpix <= 0`
2. `write_csv` writes `""` for pitch (because `bool(0.0) is False`)
3. `parse_existing_csv` reads `""` and produces `None`
4. Data lost: pitch changes from `0.0` to `None` on next build

**Verified:** Confirmed via test with `Spec(mpix=0.0)` — CSV row shows empty mpix field.

**Fix:** Replace truthy checks with explicit None checks:
```python
area_str = f"{derived.area:.2f}" if derived.area is not None else ""
mpix_str = f"{spec.mpix:.1f}" if spec.mpix is not None else ""
pitch_str = f"{derived.pitch:.2f}" if derived.pitch is not None else ""
year_str = str(spec.year) if spec.year is not None else ""
```

---

### C32-02: IR_MPIX_RE matches partial decimals without requiring unit suffix

**Sources:** CR32-02
**Severity:** LOW | **Confidence:** MEDIUM (1 agent)

The `IR_MPIX_RE` pattern `r"(\d+\.?\d*)"` matches any number without requiring a unit suffix (MP, Megapixel, etc.). Unlike the centralized `MPIX_RE` which requires a suffix, `IR_MPIX_RE` can match partial decimal numbers from malformed input (e.g., `.5` matches as `5`). The centralized `MPIX_RE` correctly rejects `.5` (returns None).

In practice, IR spec pages produce clean numeric values, so this is unlikely to cause real issues. But it's an inconsistency between the two regex patterns.

**Fix:** Add a suffix requirement or use the centralized `MPIX_RE` in the IR parser where appropriate.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 2 (1 LOW-MEDIUM, 1 LOW)
- Cross-agent consensus findings (3+ agents): 1 (C32-01 with 6 agents)
- 6-agent consensus: 1 (C32-01)
