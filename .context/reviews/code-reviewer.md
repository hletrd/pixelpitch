# Code Review (Cycle 39) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-38 fixes, focusing on NEW issues

## Previous Findings Status

All C38 findings confirmed fixed. Template renders "unknown" for 0.0 pitch/mpix. Gate tests pass.

## New Findings

### CR39-01: Template `!= 0.0` guard incomplete — negative/NaN/inf values render as numeric

**File:** `templates/pixelpitch.html`, lines 84-88 (pitch) and 76-80 (mpix)
**Severity:** MEDIUM | **Confidence:** HIGH

The C38-01 fix changed the template guard from `is not none` to `is not none and != 0.0`. This correctly handles the zero-sentinel case but misses negative, NaN, and inf values:

- `pitch = -1.0`: `-1.0 != 0.0` is True, so it renders as "-1.0 µm" — physically impossible
- `pitch = float('nan')`: `nan != 0.0` is True, renders as "nan µm" — malformed
- `mpix = -10.0`: `-10.0 != 0.0` is True, renders as "-10.0 MP" — physically impossible
- `mpix = float('nan')`: `nan != 0.0` is True, renders as "nan MP" — malformed
- `mpix = float('inf')`: `inf != 0.0` is True, renders as "inf MP" — malformed

Verified by rendering tests:
```
spec.pitch = -1.0 → template renders "-1.0 µm" (BUG)
spec.pitch = NaN  → template renders "nan µm" (BUG)
spec.mpix = -10.0 → template renders "-10.0 MP" (BUG)
spec.mpix = NaN   → template renders "nan MP" (BUG)
```

**Fix:** Change template guards to use positivity + finiteness checks:

For pitch (line 84):
```jinja2
{% if spec.pitch is not none and spec.pitch > 0 and spec.pitch is finite %}
```

Wait — Jinja2 doesn't have `is finite`. Instead, use the same pattern as Python's `isfinite` by adding a Jinja2 filter, or simply check `> 0` which covers both negative and zero (0.0 is not > 0, NaN comparisons return False, inf > 0 is True but inf pitch shouldn't occur through normal pipeline):

```jinja2
{% if spec.pitch is not none and spec.pitch > 0 %}
  {{ spec.pitch|round(1) }} µm
{% else %}
  <span class="text-muted">unknown</span>
{% endif %}
```

For mpix (line 76):
```jinja2
{% if spec.spec.mpix is not none and spec.spec.mpix > 0 %}
  {{ spec.spec.mpix|round(1) }} MP
{% else %}
  <span class="text-muted">unknown</span>
{% endif %}
```

The `> 0` check handles: None (skipped by `is not none`), 0.0 (not > 0), negative (not > 0), NaN (NaN > 0 is False). For inf, `inf > 0` is True, but inf mpix/pitch cannot enter the pipeline through normal code paths (source regexes only match digits, `pixel_pitch` returns 0.0 for inf inputs, `derive_spec` nullifies inf sizes). If defense-in-depth against inf is desired, add a Jinja2 custom filter for `isfinite`.

---

### CR39-02: `_safe_float` allows negative values through CSV pipeline

**File:** `pixelpitch.py`, lines 268-276
**Severity:** LOW | **Confidence:** HIGH

```python
def _safe_float(s: str) -> Optional[float]:
    """Parse a float string, returning None for NaN/inf/empty."""
    if not s:
        return None
    try:
        val = float(s)
        return val if isfinite(val) else None
    except (ValueError, TypeError):
        return None
```

The function correctly rejects NaN and inf (via `isfinite`), but allows negative values through. For fields like pitch, mpix, and area, negative values are physically meaningless. A CSV file containing negative values would pass through `parse_existing_csv` unfiltered, creating the entry vector for CR39-01.

Verified:
```
_safe_float("-1.0") → -1.0  (passes through)
_safe_float("-10.0") → -10.0 (passes through)
_safe_float("nan")   → None  (blocked)
_safe_float("inf")   → None  (blocked)
```

**Fix:** Option A — Add positivity check to `_safe_float` (but this would change its semantics as a general float parser). Option B (preferred) — Add validation in `parse_existing_csv` after `_safe_float` calls for fields that must be positive (width, height, area, mpix, pitch). Option C — Fix at template level only (CR39-01 fix) since source parsers can't produce negative values anyway.

---

### CR39-03: `data-pitch` attribute leaks invalid values in HTML source

**File:** `templates/pixelpitch.html`, line 50
**Severity:** LOW | **Confidence:** MEDIUM

```jinja2
<tr data-pitch="{{ spec.pitch or 0 }}"
```

When `spec.pitch` is negative (e.g., -1.0), the `or 0` coercion doesn't trigger because -1.0 is truthy. The resulting HTML contains `data-pitch="-1.0"`. Similarly, NaN produces `data-pitch="nan"`. The JS `isInvalidData` function correctly hides these rows, but the HTML source still contains invalid values.

**Fix:** Low priority since JS filters work correctly. If desired, change to:
```jinja2
<tr data-pitch="{{ spec.pitch if spec.pitch is not none and spec.pitch > 0 else 0 }}"
```

---

## Summary

- CR39-01 (MEDIUM): Template `!= 0.0` guard is incomplete — negative/NaN pitch and mpix render as numeric values
- CR39-02 (LOW): `_safe_float` allows negative values through CSV pipeline
- CR39-03 (LOW): `data-pitch` attribute leaks invalid values in HTML source
