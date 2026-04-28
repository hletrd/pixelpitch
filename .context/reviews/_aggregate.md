# Aggregate Review (Cycle 39) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-38 Status

All previous fixes confirmed still working. No regressions in core logic. Gate tests pass. C38-01 implemented and verified — template renders "unknown" for 0.0 pitch/mpix.

## Cross-Agent Agreement Matrix (Cycle 39 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| Template `!= 0.0` guard incomplete — negative/NaN/inf pitch/mpix render as numeric | CR39-01, CRIT39-01, V39-02, TR39-01, ARCH39-01, DBG39-01, DES39-01, TE39-01 | MEDIUM |
| `_safe_float` allows negative values through CSV pipeline | CR39-02, TR39-01 | LOW |
| `data-pitch` attribute leaks invalid values in HTML source | CR39-03 | LOW |
| `_safe_float` docstring doesn't mention negative value handling | DOC39-01 | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C39-01: Template `!= 0.0` guard incomplete — negative/NaN/inf pitch/mpix render as numeric

**Sources:** CR39-01, CRIT39-01, V39-02, TR39-01, ARCH39-01, DBG39-01, DES39-01, TE39-01
**Severity:** MEDIUM | **Confidence:** HIGH (8-agent consensus)

The C38-01 fix changed the template guard from `is not none` to `is not none and != 0.0`. This correctly handles the 0.0-sentinel case but misses negative, NaN, and inf values:

- `pitch = -1.0`: renders "-1.0 µm" — physically impossible
- `pitch = NaN`: renders "nan µm" — malformed
- `mpix = -10.0`: renders "-10.0 MP" — physically impossible
- `mpix = NaN`: renders "nan MP" — malformed
- `mpix = inf`: renders "inf MP" — malformed

The `> 0` guard handles all these cases in a single condition:
- `0.0 > 0` → False → "unknown" (handles C38-01 case too)
- `-1.0 > 0` → False → "unknown"
- `NaN > 0` → False → "unknown"
- `5.12 > 0` → True → "5.12 µm" (correct)

**Fix:** In `templates/pixelpitch.html`:

For pitch (line 84):
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

For `data-pitch` attribute (line 50):
```jinja2
data-pitch="{{ spec.pitch if spec.pitch is not none and spec.pitch > 0 else 0 }}"
```

Also update `test_template_zero_pitch_rendering` to add negative pitch/mpix test cases.

---

### C39-02: `_safe_float` allows negative values through CSV pipeline

**Sources:** CR39-02, TR39-01
**Severity:** LOW | **Confidence:** HIGH

`_safe_float` only rejects NaN and inf (via `isfinite`), but allows negative values through. For physical-quantity fields (pitch, mpix, area, size dimensions), negative values are meaningless. A corrupted CSV with negative values would pass through `parse_existing_csv` unfiltered.

**Fix:** Add positivity validation in `parse_existing_csv` after `_safe_float` calls for fields that must be positive:
```python
# In parse_existing_csv, after getting width, height, area, mpix, pitch:
if width is not None and width <= 0:
    width = None
if height is not None and height <= 0:
    height = None
if area is not None and area <= 0:
    area = None
if mpix is not None and mpix <= 0:
    mpix = None
if pitch is not None and pitch <= 0:
    pitch = None
```

---

### C39-03: `data-pitch` attribute leaks invalid values in HTML source

**Sources:** CR39-03
**Severity:** LOW | **Confidence:** MEDIUM

When `spec.pitch` is negative, the `{{ spec.pitch or 0 }}` coercion doesn't trigger because negative values are truthy. The HTML source contains `data-pitch="-1.0"` or `data-pitch="nan"`. JS correctly hides these rows, but the HTML source still contains invalid values.

**Fix:** Change to: `data-pitch="{{ spec.pitch if spec.pitch is not none and spec.pitch > 0 else 0 }}"`

---

### C39-04: `_safe_float` docstring doesn't mention negative value handling

**Sources:** DOC39-01
**Severity:** LOW | **Confidence:** HIGH

The docstring says "returning None for NaN/inf/empty" but does not mention that negative values are allowed through.

**Fix:** Update docstring to document current behavior.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 4 (C39-01, C39-02, C39-03, C39-04)
- Cross-agent consensus findings (3+ agents): 1 (C39-01 with 8 agents)
- Highest severity: MEDIUM (C39-01)
- Actionable findings: 4
- Verified safe / deferred: 0
