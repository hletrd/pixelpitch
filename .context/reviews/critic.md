# Critic Review (Cycle 30) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-29 fixes, focusing on NEW issues

## Previous Findings Status

C29-01 through C29-04 all implemented. All previous fixes stable.

## New Findings

### CRIT30-01: GSMArena fetch() — no per-phone error resilience

**File:** `sources/gsmarena.py`, lines 246-252
**Severity:** MEDIUM | **Confidence:** HIGH

After the C29-02 fix, `imaging_resource.fetch()` and `apotelyt.fetch()` are safer against per-camera exceptions. However, `gsmarena.fetch()` was missed. Any unhandled exception in `fetch_phone()` (e.g., from unexpected HTML structure) aborts the entire GSMArena scrape. The CineD, IR, and Apotelyt fetch loops already have this pattern.

**Fix:** Add per-phone try/except in `gsmarena.fetch()`, consistent with the other sources.

---

### CRIT30-02: deduplicate_specs() manually reconstructs Spec objects — violates DRY

**File:** `pixelpitch.py`, lines 655-665 and 669-675
**Severity:** LOW | **Confidence:** HIGH

The `deduplicate_specs()` function creates new Spec objects field-by-field instead of using `dataclasses.replace()`. The C29-04 fix simplified `digicamdb.py` to a true alias, but the same DRY violation exists in `pixelpitch.py` itself. If Spec gains a new field, these reconstructions would silently drop it.

**Fix:** Use `dataclasses.replace(ref, name=unified_name, year=year)` and `dataclasses.replace(spec, name=name)`.

---

## Summary

- CRIT30-01 (MEDIUM): GSMArena fetch() — no per-phone error resilience
- CRIT30-02 (LOW): deduplicate_specs() manual Spec reconstruction — violates DRY
