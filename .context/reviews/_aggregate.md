# Aggregate Review (Cycle 35) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-34 Status

All previous fixes confirmed still working. No regressions. Gate tests pass. C34-01/02/03 implemented and verified.

## Cross-Agent Agreement Matrix (Cycle 35 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| `derive_spec` crashes with ValueError on negative area (via pixel_pitch) | CR35-03, CRIT35-02, V35-03, TR35-01, ARCH35-01, DBG35-01, TE35-01, TE35-02 | MEDIUM |
| `_BOM` uses literal instead of escape sequence (comment-doc-code mismatch) | CR35-01, CRIT35-01, V35-02, TR35-02, ARCH35-02, DBG35-02, DOC35-01 | MEDIUM |
| Empty strings in matched_sensors from semicolon splitting | CR35-02, CRIT35-03, V35-04, DBG35-03, TE35-03 | LOW |
| openmvg produces positive mpix from negative pixel dimensions | CR35-04, V35-05, TE35-04 | LOW |
| Negative pitch/mpix render in template | DES35-01, DES35-02 | LOW |
| pixel_pitch docstring missing ValueError documentation | DOC35-02 | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C35-01: `derive_spec` crashes with ValueError on negative area — unhandled exception

**Sources:** CR35-03, CRIT35-02, V35-03, TR35-01, ARCH35-01, DBG35-01, TE35-01, TE35-02
**Severity:** MEDIUM | **Confidence:** HIGH (8-agent consensus)

`pixel_pitch(area, mpix)` calls `sqrt(area / (mpix * 10**6))`, which raises `ValueError` when `area < 0`. `derive_spec` calls `pixel_pitch` when `spec.pitch is None` and both area and mpix are known. If area is negative (from negative sensor dimensions), this crashes the build pipeline.

The existing guard in `pixel_pitch` is `if mpix <= 0: return 0.0`, which protects against zero/negative mpix but NOT against negative area.

**Concrete scenario:**
1. Source produces `Spec(size=(-5.0, 3.7), pitch=None, mpix=10.0)` or corrupted CSV has negative dimensions
2. `derive_spec` computes `area = -5.0 * 3.7 = -18.5`
3. Calls `pixel_pitch(-18.5, 10.0)`
4. `sqrt(-18.5 / 10_000_000)` raises `ValueError`
5. Unhandled exception crashes the entire build

**Fix:** Add guard in `pixel_pitch`:
```python
def pixel_pitch(area: float, mpix: float) -> float:
    if mpix <= 0 or area <= 0:
        return 0.0
    return 1000 * sqrt(area / (mpix * 10**6))
```

Also add test coverage for negative area in `pixel_pitch` and negative size in `derive_spec`.

---

### C35-02: `_BOM` uses literal character despite comment promising escape sequence

**Sources:** CR35-01, CRIT35-01, V35-02, TR35-02, ARCH35-02, DBG35-02, DOC35-01
**Severity:** MEDIUM | **Confidence:** HIGH (7-agent consensus)

The comment on `sources/__init__.py` lines 87-89 explicitly states that the escape sequence is used "rather than the literal character" to guard against editor stripping. But the actual code on line 90 uses the literal BOM character (raw bytes `ef bb bf` in the source file).

**Concrete scenario:**
1. An editor or CI pipeline normalizes UTF-8 source files
2. The invisible BOM literal on line 90 is silently stripped
3. `_BOM = ''` (empty string)
4. `strip_bom` never matches → BOM-prefixed CSVs not stripped
5. DictReader produces mangled headers → KeyError on every row → 0 records
6. Build silently produces empty/incomplete data with no error message

**Fix:** Replace the literal BOM character with the escape sequence `﻿` in the source file. The source file must contain the ASCII characters `\u`, `f`, `e`, `f`, `f` rather than the UTF-8 BOM bytes.

---

### C35-03: Empty strings in matched_sensors from semicolon splitting

**Sources:** CR35-02, CRIT35-03, V35-04, DBG35-03, TE35-03
**Severity:** LOW | **Confidence:** HIGH (5-agent consensus)

The matched_sensors field is split by semicolons:
```python
matched_sensors = sensors_str.split(";") if sensors_str else []
```

Leading/trailing/doubled semicolons produce empty strings: `;IMX455;` → `['', 'IMX455', '']`. These propagate through the CSV round-trip and could appear in rendered output.

**Fix:** Filter empty strings after split:
```python
matched_sensors = [s for s in sensors_str.split(";") if s] if sensors_str else []
```

---

### C35-04: openmvg produces positive mpix from negative pixel dimensions

**Sources:** CR35-04, V35-05, TE35-04
**Severity:** LOW | **Confidence:** HIGH

The mpix calculation uses `if pw and ph` (truthy check) which passes for negative integers. If `pw=-100` and `ph=-200`, their product is 20000, producing `mpix=20.0` — a physically meaningless value.

**Fix:** Replace truthy check with sign check:
```python
mpix = round(pw * ph / 1_000_000, 1) if pw > 0 and ph > 0 else None
```

---

### C35-05: Negative pitch/mpix render in template; NaN also renders

**Sources:** DES35-01, DES35-02
**Severity:** LOW | **Confidence:** HIGH

Negative values render as `-2.0 µm` and `-10.0 MP` in the template. The `isInvalidData` JS function does not check for negative values. NaN values render as `data-pitch="nan"` and "nan µm".

**Fix:** Add negative value check to `isInvalidData` JS function, and fix the data pipeline to reject negative values at the source (recommended). NaN defense is best handled by the `pixel_pitch` area guard (C35-01 fix).

---

### C35-06: `pixel_pitch` docstring does not document ValueError for negative area

**Sources:** DOC35-02
**Severity:** LOW | **Confidence:** HIGH

The `pixel_pitch` docstring does not mention that it raises `ValueError` when `area < 0`. If C35-01 is fixed by adding a guard, this becomes moot. If not, the docstring should document the exception.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 6
- Cross-agent consensus findings (3+ agents): 3 (C35-01 with 8 agents, C35-02 with 7 agents, C35-03 with 5 agents)
- Highest severity: MEDIUM (C35-01, C35-02)
