# Debugger Review (Cycle 33) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-32 fixes, focusing on NEW issues

## Previous Findings Status

DBG32-01 (write_csv falsy checks) fixed in C32. DBG31-01/02 fixed in C31.

## New Findings

### DBG33-01: derive_spec silently overrides spec.pitch=0.0 with computed value

**File:** `pixelpitch.py`, line 722
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

**Failure mode:** If a source parser produces `Spec(pitch=0.0)`, derive_spec will NOT preserve the 0.0 value. Instead, it will compute pitch from area+mpix. The 0.0 value is lost before write_csv even runs. This means the C32-01 fix (which correctly preserves 0.0 in CSV) is only partially effective — if 0.0 never reaches write_csv because derive_spec already overrode it, the CSV fix is moot.

**Root cause:** Python truthiness check `if spec.pitch:` treats `0.0` as falsy, identical to `None`. The correct check is `if spec.pitch is not None`.

**Trigger:** A source parser sets pitch=0.0 (unlikely in practice but possible through computation or data corruption). derive_spec would compute a non-zero pitch from area+mpix and use that instead.

**Fix:** Replace `if spec.pitch:` with `if spec.pitch is not None:` in derive_spec.

---

### DBG33-02: Template renders 0.0 as "unknown" — display-level data loss

**File:** `templates/pixelpitch.html`, lines 76-89
**Severity:** LOW | **Confidence:** HIGH

**Failure mode:** If a camera has pitch=0.0 or mpix=0.0, the Jinja2 template's `{% if spec.pitch %}` evaluates to False, rendering "unknown" instead of the actual value. This is a display-level inconsistency: the data model and CSV serialization handle 0.0 correctly (C32-01), but the UI treats 0.0 the same as None.

**Fix:** Replace Jinja2 `{% if spec.pitch %}` with `{% if spec.pitch is not none %}`.

---

## Summary

- DBG33-01 (LOW-MEDIUM): derive_spec silently overrides spec.pitch=0.0 with computed value
- DBG33-02 (LOW): Template renders 0.0 as "unknown" — display-level data loss
