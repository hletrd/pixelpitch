# Verifier Review (Cycle 36) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-35 fixes, focusing on NEW issues

## V36-01: Gate tests pass — all checks verified

**Evidence:** Ran `python3 -m tests.test_parsers_offline` — all checks passed. C35 fixes verified working. No regressions.

## V36-02: NaN/inf values propagate through the pipeline — verified

**File:** `pixelpitch.py` (pixel_pitch, derive_spec, parse_existing_csv)
**Severity:** MEDIUM | **Confidence:** HIGH

**Evidence:** Tested directly:

```python
pp.pixel_pitch(float('nan'), 10.0)  # → nan (not 0.0)
pp.pixel_pitch(float('inf'), 10.0)  # → inf (not 0.0)
```

NaN propagates through derive_spec:
```python
spec = Spec(name='NaN', category='fixed', type=None, size=(float('nan'), 24.0), pitch=None, mpix=10.0, year=2020)
d = pp.derive_spec(spec)
# d.area = nan, d.pitch = nan
```

CSV accepts NaN/inf:
```python
pp.parse_existing_csv("0,Test,mirrorless,,36.00,24.00,864.00,nan,nan,2021,\n...")
# → mpix=nan, pitch=nan
```

Template renders NaN:
```python
d.pitch = float('nan')
html = template.render(specs=[d], ...)
# → "nan µm" in visible cell, data-pitch="nan"
```

## V36-03: openmvg inf guard missing — verified

**File:** `sources/openmvg.py`, line 96
**Severity:** LOW | **Confidence:** HIGH

**Evidence:** `float('inf') > 0` is True, so `sw > 0 and sh > 0` passes for inf. Verified that `float('nan') > 0` is False, so NaN IS rejected. Only inf passes through.

---

## Summary

- V36-01: Gate tests pass
- V36-02 (MEDIUM): NaN/inf propagate through pipeline, verified with direct testing
- V36-03 (LOW): openmvg inf guard missing, verified
