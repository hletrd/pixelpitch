# Designer Review (Cycle 37) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-36 fixes

## Previous Findings Status

DES36-01 (NaN pitch renders as "nan µm") addressed. `pixel_pitch` returns `0.0` for NaN. JS `isInvalidData` now catches NaN via `isNaN()` check.

## New Findings

### DES37-01: `0.0` pitch cameras render as "0.0 µm" in the table — not "unknown"

**File:** `templates/pixelpitch.html`, lines 83-89
**Severity:** LOW | **Confidence:** HIGH

When `pixel_pitch` returns `0.0` for invalid inputs (negative area, NaN, inf), the template renders `{{ spec.pitch|round(1) }}` as `0.0 µm`. A zero pixel pitch is physically impossible, so this should arguably display as "unknown" (the same as `None` pitch).

The template logic is:
```jinja2
{% if spec.pitch is not none %}
  {{ spec.pitch|round(1) }} µm
{% else %}
  <span class="text-muted">unknown</span>
{% endif %}
```

Since `0.0 is not none` is True, the `0.0` value renders as a number, not "unknown". The JS `isInvalidData` filter does hide these rows when "Hide possibly invalid data" is checked (because `0 < 10` is not `> 10` and `0 < 0` is False, so the row passes... wait, actually `pitch = 0` is not caught by any existing JS check either).

Wait — let me re-check the JS. The `isInvalidData` function:
- `isNaN(parseFloat(row.attr('data-pitch')))` — 0 is not NaN, so this doesn't catch it
- `pitch > 10` — 0 is not > 10, so this doesn't catch it
- `pitch < 0` — 0 is not < 0, so this doesn't catch it

So a camera with `pitch=0.0` is NOT hidden by the "Hide possibly invalid data" filter, even though a 0.0 µm pixel pitch is physically impossible. This is a defense-in-depth gap similar to the NaN issue.

However, the practical impact is very low because:
1. `0.0` pitch can only occur from `pixel_pitch` returning `0.0` for invalid inputs
2. These cases are increasingly rare as the input validation improves
3. The camera data from sources is validated and shouldn't produce 0.0 pitch
4. Any 0.0 pitch that does appear would be immediately noticeable as "wrong"

**Fix options (defense-in-depth):**
1. Change `pixel_pitch` to return `None` instead of `0.0` for invalid inputs (requires template update)
2. Add `pitch === 0` check to JS `isInvalidData` function
3. Add a template-level check for `spec.pitch == 0.0` to render "unknown"

---

## Summary

- DES37-01 (LOW): `0.0` pitch renders as "0.0 µm" instead of "unknown" — defense-in-depth gap
