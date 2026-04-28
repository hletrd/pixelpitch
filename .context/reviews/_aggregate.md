# Aggregate Review (Cycle 34) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-33 Status

All previous fixes confirmed still working. No regressions. Gate tests pass (all checks). C33-01 implemented and verified.

## Cross-Agent Agreement Matrix (Cycle 34 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| match_sensors ZeroDivisionError with megapixels=0.0 | CR34-03, DBG34-01, V34-02, TR34-01, CRIT34-02, ARCH34-01, TE34-01 | MEDIUM |
| `list` command truthy check for spec.pitch | CR34-01, DBG34-02, V34-03, CRIT34-01 | LOW |
| match_sensors truthy checks for width/height | CR34-02, CRIT34-01, TE34-02 | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C34-01: match_sensors ZeroDivisionError with megapixels=0.0 — unhandled crash

**Sources:** CR34-03, DBG34-01, V34-02, TR34-01, CRIT34-02, ARCH34-01, TE34-01
**Severity:** MEDIUM | **Confidence:** HIGH (7-agent consensus)

The `match_sensors` function computes percentage differences for megapixel matching:

```python
if megapixels is not None and sensor_megapixels:
    megapixel_match = any(
        abs(megapixels - mp) / megapixels * 100 <= megapixel_tolerance
        for mp in sensor_megapixels
    )
```

If `megapixels=0.0`, the guard `megapixels is not None` is True (0.0 is not None), and the division by `megapixels` raises `ZeroDivisionError`. This exception is not caught anywhere and would crash the merge/render pipeline.

**Concrete scenario:**
1. Source parser produces `Spec(mpix=0.0)` from computation
2. `merge_camera_data` calls `match_sensors(size[0], size[1], spec.mpix, sensors_db)` for existing-only cameras
3. `0.0 is not None` → True → enters the division
4. `abs(0.0 - mp) / 0.0 * 100` → ZeroDivisionError
5. Unhandled exception crashes the entire build

**Fix:** Add `megapixels > 0` to the guard condition:
```python
if megapixels is not None and megapixels > 0 and sensor_megapixels:
```

Add test coverage for megapixels=0.0 (TE34-01).

---

### C34-02: `list` command truthy check skips cameras with pitch=0.0

**Sources:** CR34-01, DBG34-02, V34-03, CRIT34-01
**Severity:** LOW | **Confidence:** HIGH (4-agent consensus)

The `list` command at line 1170 uses `if spec.pitch:` to filter cameras before prettyprinting. If `spec.pitch=0.0`, the truthy check is False and the camera is silently omitted. This is the same class of truthy-vs-None bug fixed across 4 other locations in C33-01.

**Fix:** Replace with `if spec.pitch is not None:`

---

### C34-03: match_sensors truthy checks for width/height treat 0.0 as None

**Sources:** CR34-02, CRIT34-01, TE34-02
**Severity:** LOW | **Confidence:** HIGH

The guard clause in `match_sensors` uses truthy checks:
```python
if not sensors_db or not width or not height:
    return []
```

If width=0.0 or height=0.0, `not 0.0` is True, and the function returns []. While a sensor with 0.0 mm dimensions is physically meaningless, this conflates 0.0 with None in the same way as the C33-01 issue. For consistency and correctness, explicit None checks should be used.

Similarly, line 227: `if not sensor_width or not sensor_height:`

**Fix:** Replace with explicit None checks:
```python
if not sensors_db or width is None or height is None:
    return []
```
And line 227: `if sensor_width is None or sensor_height is None:`

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 3
- Cross-agent consensus findings (3+ agents): 3 (C34-01 with 7 agents, C34-02 with 4 agents, C34-03 with 3 agents)
- Highest severity: MEDIUM (C34-01)
