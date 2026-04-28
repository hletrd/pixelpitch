# Aggregate Review (Cycle 40) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-39 Status

All previous fixes confirmed still working. No regressions in core logic. Gate tests pass. C39-01 implemented and verified — template renders "unknown" for negative/NaN/inf pitch/mpix using `> 0` guard.

## Cross-Agent Agreement Matrix (Cycle 40 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| `derive_spec` propagates `pixel_pitch` 0.0 sentinel — selectattr misclassification, CSV round-trip data loss | CR40-01, CRIT40-01, V40-02, TR40-01, ARCH40-01, DBG40-01, DES40-01 | MEDIUM |
| `write_csv` outputs inf/nan without validation | CR40-02, V40-03 | LOW |
| No test for `derive_spec` computed pitch=0.0 path | TE40-01 | LOW |
| No test for `write_csv` non-finite float output | TE40-02 | LOW |
| `derive_spec` docstring doesn't document 0.0 sentinel handling | DOC40-01 | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C40-01: `derive_spec` propagates `pixel_pitch` 0.0 sentinel as valid pitch — selectattr misclassification and CSV round-trip data loss

**Sources:** CR40-01, CRIT40-01, V40-02, TR40-01, ARCH40-01, DBG40-01, DES40-01
**Severity:** MEDIUM | **Confidence:** HIGH (7-agent consensus)

`pixel_pitch()` returns 0.0 as a sentinel for invalid inputs (negative, zero, NaN, inf). `derive_spec` propagates this 0.0 without converting to None. This causes three downstream issues:

1. **Template misclassification:** `selectattr('pitch', 'ne', None)` includes 0.0 (because 0.0 != None), so cameras with invalid computed pitch appear in the "with pitch" table showing "unknown" — they should be in the "without pitch" section.

2. **CSV round-trip data loss:** `write_csv` writes "0.00" for pitch. `parse_existing_csv` reads it back and rejects it (0.0 <= 0 positivity check), setting pitch=None. The value is silently lost on round-trip.

3. **Contract mismatch:** `pixel_pitch`'s output domain includes 0.0 as a special value, but `derive_spec`'s consumer contract expects `derived.pitch` to be either None or a positive finite value.

**Concrete scenario:**
```
Spec(name="Cam", size=(5.0, 3.7), pitch=None, mpix=0.0)
→ derive_spec computes: pixel_pitch(18.5, 0.0) = 0.0
→ derived.pitch = 0.0
→ Template: selectattr includes 0.0 → camera in wrong table section
→ write_csv: "0.00" for pitch
→ parse_existing_csv: rejects 0.0 → pitch=None (data loss on round-trip)
```

**Fix:** In `derive_spec`, after computing pitch from `pixel_pitch()`, convert 0.0 to None:

```python
if spec.pitch is not None:
    pitch = spec.pitch
elif spec.mpix is not None and area is not None:
    pitch = pixel_pitch(area, spec.mpix)
    if pitch == 0.0:  # 0.0 is a sentinel for invalid inputs
        pitch = None
else:
    pitch = None
```

Also update the `derive_spec` docstring to document this behavior.

---

### C40-02: `write_csv` outputs inf/nan strings for non-finite float values

**Sources:** CR40-02, V40-03
**Severity:** LOW | **Confidence:** HIGH

`write_csv` uses Python's default float formatting which produces "inf" and "nan" strings for non-finite values. While `parse_existing_csv` correctly rejects these on re-read (via `_safe_float`), other CSV consumers may not handle them. This is a defensive improvement, not a bug in current data flows (source parsers cannot produce inf/nan).

**Fix:** Add `isfinite` checks before formatting float fields in `write_csv`:
```python
mpix_str = f"{spec.mpix:.1f}" if spec.mpix is not None and isfinite(spec.mpix) else ""
pitch_str = f"{derived.pitch:.2f}" if derived.pitch is not None and isfinite(derived.pitch) else ""
area_str = f"{derived.area:.2f}" if derived.area is not None and isfinite(derived.area) else ""
```

---

### C40-03: No test for `derive_spec` computed pitch=0.0 path (mpix=0.0)

**Sources:** TE40-01
**Severity:** LOW | **Confidence:** HIGH

No test verifies that `derive_spec` with `spec.pitch=None` and `spec.mpix=0.0` correctly produces `derived.pitch=None` after the C40-01 fix.

**Fix:** Add test cases:
- `derive_spec` with `pitch=None, mpix=0.0` → `derived.pitch is None`
- `derive_spec` with `pitch=None, mpix=-1.0` → `derived.pitch is None`

---

### C40-04: No test for `write_csv` non-finite float output

**Sources:** TE40-02
**Severity:** LOW | **Confidence:** MEDIUM

No test verifies that `write_csv` does not write inf/nan strings to the CSV file.

**Fix:** Add test that write_csv with inf/nan mpix/pitch/area produces empty strings in the CSV.

---

### C40-05: `derive_spec` docstring doesn't document 0.0 sentinel handling

**Sources:** DOC40-01
**Severity:** LOW | **Confidence:** HIGH

The `derive_spec` docstring says pitch is computed from `pixel_pitch(area, mpix)` but doesn't mention that 0.0 sentinel returns are converted to None.

**Fix:** Update docstring to note: "When pixel_pitch returns 0.0 (invalid input sentinel), the computed pitch is set to None."

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 5 (C40-01 through C40-05)
- Cross-agent consensus findings (3+ agents): 1 (C40-01 with 7 agents)
- Highest severity: MEDIUM (C40-01)
- Actionable findings: 5
- Verified safe / deferred: 0
