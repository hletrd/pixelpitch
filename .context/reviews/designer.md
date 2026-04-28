# Designer Review (Cycle 33) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-32 fixes, focusing on NEW issues

## Previous Findings Status

All previous UI/UX findings remain deferred (F35-F39). No regressions.

## New Findings

### DES33-01: Template truthy checks hide 0.0 pitch/mpix values — shows "unknown" instead

**File:** `templates/pixelpitch.html`, lines 76-89
**Severity:** LOW | **Confidence:** HIGH

The template uses Jinja2 truthy checks for displaying pitch and mpix values:

```html
{% if spec.spec.mpix %}
  {{ spec.spec.mpix|round(1) }} MP
{% else %}
  <span class="text-muted">unknown</span>
{% endif %}
...
{% if spec.pitch %}
  {{ spec.pitch|round(1) }} µm
{% else %}
  <span class="text-muted">unknown</span>
{% endif %}
```

In Jinja2, `{% if 0.0 %}` evaluates to False. So a camera with mpix=0.0 or pitch=0.0 would display "unknown" instead of "0.0 MP" / "0.0 µm". This is inconsistent with the C32-01 fix that correctly preserves 0.0 in CSV. The data is present in the model but hidden from the user.

**Fix:** Use Jinja2's `is not none` test:
```html
{% if spec.spec.mpix is not none %}
  {{ spec.spec.mpix|round(1) }} MP
{% else %}
  <span class="text-muted">unknown</span>
{% endif %}
```

---

## Summary

- DES33-01 (LOW): Template truthy checks hide 0.0 pitch/mpix values — shows "unknown" instead
