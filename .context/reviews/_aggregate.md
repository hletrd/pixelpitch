# Aggregate Review (Cycle 33) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-32 Status

All previous fixes confirmed still working. No regressions. Gate tests pass (all checks). C32-01 implemented and verified.

## Cross-Agent Agreement Matrix (Cycle 33 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| derive_spec truthy check violates docstring for spec.pitch=0.0 | CR33-01, CRIT33-01, V33-02, TR33-01, ARCH33-01, DBG33-01, DOC33-01, TE33-01 | LOW-MEDIUM |
| sorted_by truthy checks sort 0.0 as -1 | CR33-02, V33-04, TE33-02 | LOW |
| prettyprint truthy checks display "unknown" for 0.0 | CR33-03 | LOW |
| Template truthy checks hide 0.0 as "unknown" | DES33-01, DBG33-02, V33-03, TE33-03 | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C33-01: Systemic truthy-vs-None inconsistency — derive_spec, sorted_by, prettyprint, template

**Sources:** CR33-01, CR33-02, CR33-03, CRIT33-01, V33-02, V33-03, V33-04, TR33-01, ARCH33-01, DBG33-01, DBG33-02, DES33-01, DOC33-01, TE33-01, TE33-02, TE33-03
**Severity:** LOW-MEDIUM | **Confidence:** HIGH (11-agent consensus on core issue)

The C32-01 fix addressed the write_csv serialization layer by replacing truthy checks with explicit `is not None` checks for float fields. However, the same truthy-vs-None pattern persists in FOUR other code locations, making the C32-01 fix incomplete:

1. **derive_spec (pixelpitch.py line 722):** `if spec.pitch:` — 0.0 pitch is overridden by computed value from area+mpix. The docstring says "spec.pitch always takes precedence" but this is violated for 0.0. This is the most significant instance because it corrupts data BEFORE it reaches write_csv, making the C32-01 CSV fix partially moot.

2. **sorted_by (pixelpitch.py lines 752-756):** `c.pitch if c.pitch else -1` — cameras with 0.0 values sort as -1 instead of 0.0.

3. **prettyprint (pixelpitch.py lines 772-778):** `if spec.mpix:` / `if derived.pitch:` — 0.0 displays as "unknown" instead of "0.0 MP" / "0.0 µm".

4. **Template (pixelpitch.html lines 76-89):** `{% if spec.pitch %}` / `{% if spec.spec.mpix %}` — 0.0 renders as "unknown" in HTML.

**Concrete scenario (derive_spec — most critical):**
1. Source parser produces `Spec(pitch=0.0, mpix=33.0, size=(35.9, 23.9))`
2. `derive_spec`: `if spec.pitch:` → False (0.0 is falsy)
3. Falls to elif: computes `pixel_pitch(858.61, 33.0)` = 5.12
4. Result: `derived.pitch=5.12` instead of `0.0`
5. The 0.0 value is lost BEFORE write_csv even runs — the C32-01 fix never gets a chance to preserve it

**Fix:** Replace all truthy checks with explicit `is not None` checks:
- `pixelpitch.py` line 722: `if spec.pitch is not None:`
- `pixelpitch.py` lines 752-756: `c.pitch if c.pitch is not None else -1`, etc.
- `pixelpitch.py` lines 772-778: `if spec.mpix is not None:`, `if derived.pitch is not None:`
- `pixelpitch.html` lines 76-89: `{% if spec.spec.mpix is not none %}`, `{% if spec.pitch is not none %}`

Add test coverage for:
- derive_spec with spec.pitch=0.0 (TE33-01)
- sorted_by with 0.0 values (TE33-02)
- Template rendering of 0.0 values (TE33-03)

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 1 (systemic truthy-vs-None issue across 4 code locations)
- Cross-agent consensus findings (3+ agents): 1 (C33-01 with 11 agents)
- Highest severity: LOW-MEDIUM
