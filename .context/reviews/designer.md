# Designer Review (Cycle 38) — UI/UX Review

**Reviewer:** designer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-37 fixes

## Previous Findings Status

DES37-01 (0.0 pitch renders as "0.0 µm" not "unknown") — partially fixed. The JS `isInvalidData` now catches `pitch === 0` and hides those rows. But this creates a new UX issue.

## New Findings

### DES38-01: Zero-pitch rows hidden by default but rendered as "0.0 µm" — inconsistent user experience

**File:** `templates/pixelpitch.html`, lines 157, 277-279, 84-88
**Severity:** MEDIUM | **Confidence:** HIGH

The "Hide possibly invalid data" toggle is checked by default (line 157). When checked, `isInvalidData` returns `true` for `pitch === 0` (line 277-279), hiding those rows. But the template still renders 0.0 pitch as "0.0 µm" (line 84-88), not "unknown".

The user experience is: if you uncheck "Hide possibly invalid data", you see "0.0 µm" pitch entries. If you check it, those entries vanish. This is inconsistent — either the data is valid (show it) or invalid (display "unknown" instead of a number).

A 0.0 µm pixel pitch is physically impossible for any real camera sensor. The correct UX treatment is to display "unknown" for these entries, just like `None` pitch values, and optionally hide them by default.

**Fix:** In the Jinja2 template, add a check for `spec.pitch == 0.0` to render "unknown" instead of "0.0 µm":
```jinja2
{% if spec.pitch is not none and spec.pitch != 0.0 %}
  {{ spec.pitch|round(1) }} µm
{% else %}
  <span class="text-muted">unknown</span>
{% endif %}
```

This makes the template consistent with the JS filter: both treat 0.0 pitch as "unknown/invalid".

---

### DES38-02: Scatter plot filter `pitch > 0` correctly excludes zero-pitch entries

**File:** `templates/pixelpitch.html`, line 372
**Severity:** N/A (verification only)
**Confidence:** HIGH

The scatter plot creation code has `if (!isNaN(pitch) && pitch > 0 && ...)` which correctly excludes zero-pitch entries from the plot. This is consistent with treating 0.0 pitch as invalid. No fix needed.

---

## Summary

- DES38-01 (MEDIUM): Template renders "0.0 µm" for zero pitch but JS hides those rows — inconsistent UX
- DES38-02: Scatter plot correctly excludes zero-pitch entries — verified
