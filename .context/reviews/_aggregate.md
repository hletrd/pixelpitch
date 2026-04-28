# Aggregate Review (Cycle 36) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-35 Status

All previous fixes confirmed still working. No regressions. Gate tests pass. C35-01/02/03/04 implemented and verified.

## Cross-Agent Agreement Matrix (Cycle 36 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| `pixel_pitch` does not guard NaN/inf inputs | CR36-01, CRIT36-01, DBG36-02, V36-02, TR36-01, ARCH36-01, TE36-01, TE36-02, DES36-01 | MEDIUM |
| `parse_existing_csv` accepts NaN/inf from CSV strings | CR36-02, DBG36-01, V36-02, TR36-01, TE36-03 | MEDIUM |
| `openmvg.fetch` accepts inf sensor dimensions | CR36-03, V36-03 | LOW |
| `derive_spec` propagates NaN/inf from size to pitch | DBG36-01, TE36-04 | MEDIUM (subsumed by C36-01) |

## Deduplicated New Findings (Ordered by Severity)

### C36-01: `pixel_pitch` does not guard against NaN or inf inputs

**Sources:** CR36-01, CRIT36-01, DBG36-02, V36-02, TR36-01, ARCH36-01, TE36-01, TE36-02, DES36-01
**Severity:** MEDIUM | **Confidence:** HIGH (9-agent consensus)

The guard `if mpix <= 0 or area <= 0: return 0.0` does not reject NaN or inf:
- `float('nan') <= 0` is `False` — NaN bypasses the guard
- `float('inf') <= 0` is `False` — inf bypasses the guard
- `pixel_pitch(float('nan'), 10.0)` returns `nan` (not 0.0)
- `pixel_pitch(float('inf'), 10.0)` returns `inf` (not 0.0)

NaN propagates through `derive_spec` when size contains NaN: `area = nan * 24.0 = nan`, then `pixel_pitch(nan, mpix) = nan`.

The C35-01 fix (negative area guard) is incomplete — it uses comparison operators that do not catch NaN or inf.

**Concrete scenario:**
1. Corrupted CSV contains `nan` or `inf` for sensor dimensions
2. `parse_existing_csv` calls `float("nan")` which succeeds
3. `derive_spec` computes `area = nan * 24.0 = nan`
4. `pixel_pitch(nan, mpix)` returns `nan`
5. `write_csv` writes `nan` to CSV
6. Template renders "nan µm" in visible cell and `data-pitch="nan"` in HTML
7. JS `isInvalidData` does not catch NaN (`parseFloat("nan") || 0 = 0`)

**Fix:** Replace the `<= 0` guard with `math.isfinite` check:
```python
def pixel_pitch(area: float, mpix: float) -> float:
    if not math.isfinite(area) or not math.isfinite(mpix) or mpix <= 0 or area <= 0:
        return 0.0
    return 1000 * sqrt(area / (mpix * 10**6))
```

Also add `math.isfinite` checks in `parse_existing_csv` and `openmvg.fetch`.

---

### C36-02: `parse_existing_csv` accepts NaN and inf values from CSV strings

**Sources:** CR36-02, DBG36-01, V36-02, TR36-01, TE36-03
**Severity:** MEDIUM | **Confidence:** HIGH (5-agent consensus)

Python's `float()` accepts `"nan"`, `"inf"`, `"-inf"`, and `"NaN"` as valid inputs. The CSV parser uses bare `float()` calls for width, height, area, mpix, and pitch without checking for finite values. This allows NaN and inf to enter the data pipeline from corrupted or manually edited CSV files.

**Fix:** Add `math.isfinite` validation after each `float()` call in `parse_existing_csv`. Non-finite values should be treated as None:
```python
val = float(val_str)
result = val if math.isfinite(val) else None
```

---

### C36-03: `openmvg.fetch` accepts inf sensor dimensions

**Sources:** CR36-03, V36-03
**Severity:** LOW | **Confidence:** HIGH

The size guard `sw > 0 and sh > 0` passes for `inf` because `inf > 0` is True. While NaN is rejected (because `nan > 0` is False), inf dimensions produce `(inf, inf)` size which propagates through the pipeline.

**Fix:** Replace with `math.isfinite` check:
```python
size = (sw, sh) if sw and sh and math.isfinite(sw) and math.isfinite(sh) and sw > 0 and sh > 0 else None
```

---

### C36-04: JS `isInvalidData` does not catch NaN pitch values

**Sources:** DES36-01
**Severity:** LOW | **Confidence:** HIGH

When `data-pitch="nan"`, JS `parseFloat("nan") || 0` evaluates to `0`, which passes all validation checks. The row is NOT hidden by the "Hide possibly invalid data" filter. This is a defense-in-depth gap.

**Fix:** Add NaN check to `isInvalidData`:
```javascript
const pitch = parseFloat(row.attr('data-pitch'));
if (isNaN(pitch)) return true;
```

---

### C36-05: `pixel_pitch` docstring should mention NaN/inf handling

**Sources:** DOC36-01
**Severity:** LOW | **Confidence:** HIGH

After the fix, the docstring should document that NaN and inf inputs return 0.0.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 5
- Cross-agent consensus findings (3+ agents): 2 (C36-01 with 9 agents, C36-02 with 5 agents)
- Highest severity: MEDIUM (C36-01, C36-02)
