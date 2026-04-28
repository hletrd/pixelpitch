# Aggregate Review (Cycle 21) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-20 Status

All previous fixes confirmed still working. No regressions. Gate tests pass. C20-01 (pixel_pitch crash guard), C20-02 (Sony FX naming), and C20-03 (merge field preservation) fixes verified in code.

**However:** C20-03 fix is partially broken — see C21-01 below.

## Cross-Agent Agreement Matrix (Cycle 21 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| SpecDerived fields stale after merge (C20-03 regression) | code-reviewer, critic, verifier, tracer, debugger, test-engineer, architect, designer | HIGH |
| Sony RX/DSC/HX/WX/TX/QX misnamed by `.title()` | code-reviewer, critic, verifier, tracer, debugger, test-engineer | MEDIUM |
| mpix not preserved in merge when new has None | code-reviewer, critic, verifier, test-engineer | LOW |
| test_merge_field_preservation doesn't verify SpecDerived | code-reviewer, test-engineer | MEDIUM |

## Deduplicated New Findings (Ordered by Severity)

### C21-01: SpecDerived fields stale after merge — C20-03 regression

**Sources:** C21-01, C21-CR01, V21-01, T21-01, D21-01, TE21-01, A21-01, D21-01
**Severity:** HIGH | **Confidence:** HIGH (8-agent consensus)

The C20-03 fix added field preservation for `type`, `size`, and `pitch` at the `Spec` level (`new_spec.spec.*`). However, the `SpecDerived` fields (`new_spec.size`, `new_spec.area`, `new_spec.pitch`) were NOT updated to match. The Jinja2 template reads from `SpecDerived` fields, so the preserved values are invisible in the rendered HTML — cameras show "unknown" for sensor size and pixel pitch.

This affects 30.5% of cameras (532 with no size) and 32.5% (567 with no pitch) in the current dataset.

**Fix:** Add SpecDerived field preservation in `merge_camera_data`:
```python
if new_spec.size is None and existing_spec.size is not None:
    new_spec.size = existing_spec.size
if new_spec.area is None and existing_spec.area is not None:
    new_spec.area = existing_spec.area
if new_spec.pitch is None and existing_spec.pitch is not None:
    new_spec.pitch = existing_spec.pitch
```

---

### C21-02: Sony RX/DSC/HX/WX/TX/QX series cameras misnamed by `.title()`

**Sources:** C21-02, C21-CR02, V21-02, T21-02, D21-02, TE21-02
**Severity:** MEDIUM | **Confidence:** HIGH (6-agent consensus)

The C20-02 fix only addressed the FX series. The same `.title()` issue affects all Sony multi-letter uppercase series: RX, HX, WX, TX, QX, and DSC. These are all converted to lowercase second letter by `.title()` (e.g., "rx100" -> "Rx100" instead of "RX100").

**Fix:** Add general Sony uppercase series normalizers after the existing FX normalizer:
```python
cleaned = re.sub(r'\bRx(\d)', r'RX\1', cleaned)
cleaned = re.sub(r'\bHx(\d)', r'HX\1', cleaned)
cleaned = re.sub(r'\bWx(\d)', r'WX\1', cleaned)
cleaned = re.sub(r'\bTx(\d)', r'TX\1', cleaned)
cleaned = re.sub(r'\bQx(\d)', r'QX\1', cleaned)
cleaned = re.sub(r'\bDsc\b', r'DSC', cleaned)
```

---

### C21-03: mpix not preserved by merge when new data has mpix=None

**Sources:** C21-03, C21-CR03, V21-03, TE21-03
**Severity:** LOW | **Confidence:** HIGH (4-agent consensus)

The merge function preserves `type`, `size`, `pitch`, and `year` but NOT `mpix`. When a new source has `mpix=None` but existing data has `mpix=33.0`, the megapixel count is lost and the camera shows "unknown" resolution.

**Fix:** Add mpix preservation:
```python
if new_spec.spec.mpix is None and existing_spec.spec.mpix is not None:
    new_spec.spec.mpix = existing_spec.spec.mpix
```

---

### C21-04: test_merge_field_preservation doesn't verify SpecDerived fields

**Sources:** C21-04, TE21-01
**Severity:** MEDIUM | **Confidence:** HIGH (2-agent consensus)

The test only checks `spec.spec.*` (Spec-level) fields, not `spec.*` (SpecDerived-level) fields. This allowed the C21-01 bug to go undetected.

**Fix:** Add assertions for SpecDerived fields in the test.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 4 actionable (1 HIGH, 2 MEDIUM, 1 LOW)
- Cross-agent consensus findings (3+ agents): 3 (C21-01, C21-02, C21-03)
- 1 HIGH finding: SpecDerived stale fields after merge
- 2 MEDIUM findings: Sony RX/DSC naming, test gap for SpecDerived
- 1 LOW finding: mpix not preserved in merge
