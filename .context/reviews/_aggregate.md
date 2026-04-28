# Aggregate Review (Cycle 37) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-36 Status

All previous fixes confirmed still working. No regressions. Gate tests pass. C36-01/02/03/04/05 implemented and verified.

## Cross-Agent Agreement Matrix (Cycle 37 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| `derive_spec` does not validate NaN/inf size dimensions → produces nan area | CR37-02, CRIT37-01, V37-02, TR37-01, ARCH37-01, DBG37-01, TE37-01, TE37-02, DOC37-01 | MEDIUM |
| `0.0` pitch renders as "0.0 µm" not "unknown" — defense-in-depth | DES37-01 | LOW |
| Source parsers use bare `float()` but regex excludes NaN/inf | CRIT37-02, V37-03 | LOW (verified safe) |
| `match_sensors` division by near-zero width | CR37-03 | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C37-01: `derive_spec` does not validate NaN/inf size dimensions

**Sources:** CR37-02, CRIT37-01, V37-02, TR37-01, ARCH37-01, DBG37-01, TE37-01, TE37-02, DOC37-01
**Severity:** MEDIUM | **Confidence:** HIGH (9-agent consensus)

`derive_spec` computes `area = size[0] * size[1]` without checking that the size dimensions are finite. When `spec.size = (nan, 24.0)`:
1. `area = nan * 24.0 = nan`
2. `pixel_pitch(nan, mpix)` returns `0.0` (guarded)
3. `write_csv` writes `f"{nan:.2f}"` = `"nan"` to CSV
4. On re-read, `_safe_float("nan")` returns `None` — area changes from nan to None

The C36 fixes addressed NaN at the boundaries (CSV parser, openmvg) but not at the computation point. `derive_spec` should validate its inputs as defense-in-depth.

Source parser regex patterns (`[\d.]+`) naturally exclude NaN/inf strings, so the practical risk is limited to code-constructed Spec objects (tests, etc.). However, the architectural principle (validate at the computation point, not just the boundaries) warrants the fix.

**Fix:** Add `isfinite` validation in `derive_spec`:
```python
if size is not None and spec.mpix is not None:
    if isfinite(size[0]) and isfinite(size[1]) and size[0] > 0 and size[1] > 0:
        area = size[0] * size[1]
    else:
        size = None
        area = None
```

Also update the `derive_spec` docstring to mention NaN/inf handling.

Add tests for area being None when size has NaN dimensions, and for CSV round-trip of NaN area.

---

### C37-02: `0.0` pitch renders as "0.0 µm" instead of "unknown" — defense-in-depth

**Sources:** DES37-01, CR37-01
**Severity:** LOW | **Confidence:** HIGH

When `pixel_pitch` returns `0.0` for invalid inputs (NaN, inf, negative), the template renders "0.0 µm" instead of "unknown". A 0.0 µm pixel pitch is physically impossible. The JS `isInvalidData` function does not catch `pitch === 0` either.

**Fix:** Add `pitch === 0` check to JS `isInvalidData` function:
```javascript
if (pitch === 0) {
  return true;
}
```

---

### C37-03: Source parsers use bare `float()` but regex patterns exclude NaN/inf — verified safe

**Sources:** CRIT37-02, V37-03
**Severity:** LOW (verified safe) | **Confidence:** HIGH

All source parser regex patterns use `[\d.]+` or `\d` character classes which cannot match "nan" or "inf" strings. No fix required. Document as verified.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 3 (C37-01, C37-02, C37-03)
- Cross-agent consensus findings (3+ agents): 1 (C37-01 with 9 agents)
- Highest severity: MEDIUM (C37-01)
- Actionable findings: 2 (C37-01, C37-02)
- Verified safe (no action): 1 (C37-03)
