# Code Review (Cycle 34) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-33 fixes, focusing on NEW issues

## Previous Findings Status

C33-01 (systemic truthy-vs-None in derive_spec, sorted_by, prettyprint, template) fixed and verified. All gate tests pass.

## New Findings

### CR34-01: `list` command uses truthy check for spec.pitch — 0.0 pitch cameras silently omitted

**File:** `pixelpitch.py`, line 1170
**Severity:** LOW | **Confidence:** HIGH

The `list` command filters cameras with `if spec.pitch:` before prettyprinting:

```python
for spec in specs_sorted:
    if spec.pitch:
        prettyprint(spec)
```

If a camera has `spec.pitch=0.0`, the truthy check is False and the camera is silently omitted from the listing. This is inconsistent with the C33-01 fixes that already corrected derive_spec, sorted_by, prettyprint, and the template to use explicit `is not None` checks.

The `list` command is a CLI diagnostic path, so the impact is low, but it's the same class of truthy-vs-None bug that was fixed across 4 other locations.

**Fix:** Replace with `if spec.pitch is not None:`

---

### CR34-02: match_sensors uses truthy check for width/height — 0.0 dimensions silently rejected

**File:** `pixelpitch.py`, line 217
**Severity:** LOW | **Confidence:** HIGH

The guard clause in `match_sensors` uses truthy checks:

```python
if not sensors_db or not width or not height:
    return []
```

If width=0.0 or height=0.0, `not 0.0` is True, and the function returns []. While a sensor with width=0.0 mm is physically meaningless, the data model allows it and the function signature accepts `Optional[float]`. The truthy check conflates 0.0 with None.

Similarly, line 227:
```python
if not sensor_width or not sensor_height:
    continue
```

**Fix:** Replace with explicit None checks:
```python
if not sensors_db or width is None or height is None:
    return []
```

---

### CR34-03: match_sensors has potential ZeroDivisionError when megapixels=0.0

**File:** `pixelpitch.py`, line 238
**Severity:** MEDIUM | **Confidence:** HIGH

The megapixel matching uses a percentage calculation:
```python
megapixel_match = any(
    abs(megapixels - mp) / megapixels * 100 <= megapixel_tolerance
    for mp in sensor_megapixels
)
```

If `megapixels=0.0`, this divides by zero, raising `ZeroDivisionError`. The guard `if megapixels is not None and sensor_megapixels:` on line 236 does not protect against 0.0 because `0.0 is not None` is True.

**Concrete scenario:**
1. Source parser produces `Spec(mpix=0.0)` (e.g., from malformed data)
2. `match_sensors` is called with `megapixels=0.0`
3. `0.0 is not None` is True → enters the megapixel_match branch
4. `abs(0.0 - mp) / 0.0 * 100` → ZeroDivisionError
5. Unhandled exception crashes the merge pipeline

**Fix:** Add a guard for megapixels=0.0:
```python
if megapixels is not None and megapixels > 0 and sensor_megapixels:
```

---

## Summary

- CR34-01 (LOW): `list` command truthy check for spec.pitch omits 0.0 pitch cameras
- CR34-02 (LOW): match_sensors truthy checks for width/height reject 0.0 dimensions
- CR34-03 (MEDIUM): match_sensors ZeroDivisionError when megapixels=0.0
