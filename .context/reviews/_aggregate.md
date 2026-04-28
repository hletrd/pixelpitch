# Aggregate Review (Cycle 41) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-40 Status

All previous fixes confirmed still working. No regressions in core logic. Gate tests pass. C40-01 implemented and verified — `derive_spec` converts computed 0.0 sentinel to None.

## Cross-Agent Agreement Matrix (Cycle 41 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| `derive_spec` preserves invalid direct `spec.pitch` values (0.0, negative, NaN) — selectattr misclassification, CSV round-trip data loss | CR41-01, CRIT41-01, V41-02, TR41-01, ARCH41-01, DBG41-01, DES41-01 | MEDIUM |
| `write_csv` writes 0.0/negative mpix/pitch — `isfinite` guard insufficient for physical quantities | CR41-02, CRIT41-02, V41-03, DBG41-02 | LOW |
| `merge_camera_data` preserves `spec.pitch=0.0` from existing data — re-introduces sentinel | CR41-03, V41-04 | LOW |
| No test for `derive_spec` direct `spec.pitch=0.0` → None (existing test expects 0.0) | TE41-01 | LOW |
| No test for `write_csv` with 0.0/negative mpix/pitch | TE41-02 | LOW |
| `derive_spec` docstring doesn't document direct-path validation | DOC41-01 | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C41-01: `derive_spec` preserves invalid direct `spec.pitch` values (0.0, negative, NaN) — incomplete validation

**Sources:** CR41-01, CRIT41-01, V41-02, TR41-01, ARCH41-01, DBG41-01, DES41-01
**Severity:** MEDIUM | **Confidence:** HIGH (7-agent consensus)

The C40 fix added a 0.0-to-None conversion for the *computed* pitch path (when `spec.pitch is None` and pitch is derived from `pixel_pitch()`). However, the *direct* path — when `spec.pitch` is explicitly set to 0.0, negative, or NaN — is completely unguarded:

```python
if spec.pitch is not None:
    pitch = spec.pitch   # <-- no validation
```

This means:
- `Spec(pitch=0.0)` → `derived.pitch = 0.0` — passes through selectattr, wrong table section
- `Spec(pitch=-1.0)` → `derived.pitch = -1.0` — negative pitch in data model
- `Spec(pitch=nan)` → `derived.pitch = nan` — NaN in data model

The template `> 0` guard renders these as "unknown" in the cell, but the camera is still in the wrong section (selectattr includes 0.0 and -1.0 but NaN behavior varies).

**Concrete scenario:**
```
Spec(name="Cam", size=(5.0, 3.7), pitch=0.0, mpix=33.0)
→ derive_spec: pitch = spec.pitch = 0.0  (no validation)
→ selectattr('pitch', 'ne', None) includes it
→ Camera in "with pitch" table showing "unknown" — wrong section
```

**Fix:** In `derive_spec`, validate `spec.pitch` the same as computed pitch:

```python
if spec.pitch is not None:
    pitch = spec.pitch
    if not isfinite(pitch) or pitch <= 0:
        pitch = None
elif spec.mpix is not None and area is not None:
    pitch = pixel_pitch(area, spec.mpix)
    if pitch == 0.0:
        pitch = None
else:
    pitch = None
```

Also update the `derive_spec` docstring to document that both direct and computed paths produce the same contract: `derived.pitch` is either None or a positive finite value.

---

### C41-02: `write_csv` writes 0.0/negative mpix/pitch — `isfinite` guard insufficient for physical quantities

**Sources:** CR41-02, CRIT41-02, V41-03, DBG41-02
**Severity:** LOW | **Confidence:** HIGH

The C40 fix added `isfinite()` checks for mpix, pitch, and area in `write_csv`. However, `isfinite(0.0)` returns True and `isfinite(-1.0)` returns True, so these physically invalid values pass through to the CSV:

- `mpix=0.0` → written as "0.0" → `parse_existing_csv` rejects it → data loss on round-trip
- `mpix=-5.0` → written as "-5.0" → `parse_existing_csv` rejects it → data loss on round-trip
- `pitch=0.0` → written as "0.00" → `parse_existing_csv` rejects it → data loss on round-trip
- `pitch=-1.0` → written as "-1.00" → `parse_existing_csv` rejects it → data loss on round-trip

**Fix:** Replace `isfinite` checks with positivity checks (`> 0`) in `write_csv` for mpix, pitch, and area:

```python
mpix_str = f"{spec.mpix:.1f}" if spec.mpix is not None and spec.mpix > 0 else ""
pitch_str = f"{derived.pitch:.2f}" if derived.pitch is not None and derived.pitch > 0 else ""
area_str = f"{derived.area:.2f}" if derived.area is not None and derived.area > 0 else ""
```

This is consistent with `parse_existing_csv`'s positivity checks and ensures the CSV round-trip is lossless.

---

### C41-03: `merge_camera_data` preserves `spec.pitch=0.0` from existing data — re-introduces sentinel

**Sources:** CR41-03, V41-04
**Severity:** LOW | **Confidence:** MEDIUM

When merging, `merge_camera_data` preserves `spec.pitch` from existing data if new data has None. If the existing data has `spec.pitch=0.0` (e.g., from an older CSV that predates the positivity check), this 0.0 is preserved and then copied to `derived.pitch` via the consistency check at lines 471-473.

In practice, this is LOW severity because source parsers cannot produce `spec.pitch=0.0` and `parse_existing_csv` now rejects 0.0 pitch from CSV input. The only way 0.0 enters is through legacy data or direct API usage.

**Fix:** After CR41-01 fix is in place (derive_spec validates direct pitch), this becomes largely moot — `derive_spec` will convert 0.0 to None. However, `merge_camera_data` should also validate preserved pitch values for defense in depth.

---

### C41-04: No test for `derive_spec` direct `spec.pitch=0.0` → None (existing test expects 0.0)

**Sources:** TE41-01
**Severity:** LOW | **Confidence:** HIGH

The existing `test_derive_spec_zero_pitch` tests that `spec.pitch=0.0` is *preserved* as `derived.pitch=0.0`. After C41-01 is fixed, this test needs to be updated to expect `derived.pitch=None`. Additional test cases needed for `spec.pitch=-1.0` and `spec.pitch=nan`.

**Fix:** Update `test_derive_spec_zero_pitch` and add new test cases.

---

### C41-05: No test for `write_csv` with 0.0/negative mpix/pitch

**Sources:** TE41-02
**Severity:** LOW | **Confidence:** MEDIUM

No test verifies that `write_csv` does not write 0.0 or negative values for mpix/pitch to the CSV file. After C41-02 is fixed, tests should verify these produce empty strings in the CSV.

**Fix:** Add test cases for 0.0 and negative values in write_csv.

---

### C41-06: `derive_spec` docstring doesn't document direct-path validation

**Sources:** DOC41-01
**Severity:** LOW | **Confidence:** HIGH

The `derive_spec` docstring documents the computed-path sentinel handling but not the direct-path validation. After C41-01 is fixed, the docstring should be updated to document the uniform output contract.

**Fix:** Update docstring to note that both direct and computed paths produce the same contract: `derived.pitch` is either None or a positive finite value.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 6 (C41-01 through C41-06)
- Cross-agent consensus findings (3+ agents): 2 (C41-01 with 7 agents, C41-02 with 4 agents)
- Highest severity: MEDIUM (C41-01)
- Actionable findings: 6
- Verified safe / deferred: 0
