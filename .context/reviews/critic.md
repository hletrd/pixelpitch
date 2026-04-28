# Critic Review (Cycle 33) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-32 fixes, focusing on NEW issues

## Previous Findings Status

C32-01 (write_csv falsy checks) fixed and verified. C32-02 (IR_MPIX_RE) deferred.

## New Findings

### CRIT33-01: Systemic truthy-vs-None inconsistency — C32-01 fix was incomplete

**Files:** `pixelpitch.py` (derive_spec, sorted_by, prettyprint), `templates/pixelpitch.html`
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

The C32-01 fix correctly addressed the write_csv serialization layer, replacing truthy checks with explicit `is not None` checks. However, the same truthy-vs-None pattern persists in FOUR other locations:

1. **derive_spec (line 722):** `if spec.pitch:` — 0.0 pitch is overridden by computed value, violating the docstring's "takes precedence" guarantee. This is the most significant instance because it affects data correctness, not just display.

2. **sorted_by (lines 752-756):** `c.pitch if c.pitch else -1` — 0.0 sorts as -1 instead of 0.0.

3. **prettyprint (lines 772-778):** `if spec.mpix:` / `if derived.pitch:` — 0.0 displays as "unknown".

4. **Template (pixelpitch.html lines 76-89):** `{% if spec.pitch %}` / `{% if spec.spec.mpix %}` — 0.0 renders as "unknown" in HTML.

This is a cross-cutting consistency issue: the data model and CSV serialization now correctly handle 0.0 as distinct from None (C32-01), but the computation, sorting, and display layers still treat 0.0 as equivalent to None. The fix should be applied holistically.

**Fix:** Replace all truthy checks with explicit None checks across all four locations. In Jinja2 templates, use `{% if spec.pitch is not none %}` instead of `{% if spec.pitch %}`.

---

## Summary

- CRIT33-01 (LOW-MEDIUM): Systemic truthy-vs-None inconsistency — C32-01 fix incomplete across derive_spec, sorted_by, prettyprint, and templates
