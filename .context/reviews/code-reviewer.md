# Code Review (Cycle 30) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-29 fixes, focusing on NEW issues

## Previous Findings Status

All previous fixes confirmed still working. C29-01 through C29-04 all implemented. Gate tests pass.

## New Findings

### CR30-01: GSMArena fetch() loop lacks per-phone try/except

**File:** `sources/gsmarena.py`, lines 246-252
**Severity:** MEDIUM | **Confidence:** HIGH

The C29-02 fix added per-camera try/except to `imaging_resource.fetch()` and `apotelyt.fetch()`, but `gsmarena.fetch()` was missed. The same failure mode applies: if `fetch_phone()` raises an unhandled exception (e.g., from unexpected HTML structure or a future code change), the exception propagates through `fetch()`, aborting the entire GSMArena scrape. All phones processed so far are lost.

```python
# gsmarena.fetch() — lines 246-252:
for i, s in enumerate(slugs):
    spec = fetch_phone(s)  # <-- no try/except
    if spec:
        specs.append(spec)
```

For comparison, the CineD `fetch()` already has per-camera try/except (line 151-156), and the IR/Apotelyt `fetch()` functions were fixed in C29.

**Fix:** Add per-phone try/except to `gsmarena.fetch()`, logging the error and continuing.

---

### CR30-02: deduplicate_specs() manually reconstructs Spec objects — violates DRY

**File:** `pixelpitch.py`, lines 655-665 and 669-675
**Severity:** LOW | **Confidence:** HIGH

The `deduplicate_specs()` function creates new Spec objects field-by-field in two places:

1. Color-variant unification (lines 655-665):
```python
rest.append(
    Spec(unified_name, ref.category, ref.type, ref.size, ref.pitch, ref.mpix, year)
)
```

2. `remove_parens()` (lines 669-675):
```python
return Spec(name, spec.category, spec.type, spec.size, spec.pitch, spec.mpix, spec.year)
```

The C29-04 fix simplified `digicamdb.py` to a true alias, but the same DRY violation exists in `pixelpitch.py` itself. If Spec gains a new field (e.g., `sensor_brand`), these reconstructions would silently drop it.

**Fix:** Use `dataclasses.replace()` instead of manual field enumeration:
- Color-variant: `dataclasses.replace(ref, name=unified_name, year=year)`
- remove_parens: `dataclasses.replace(spec, name=name)`

---

## Summary

- CR30-01 (MEDIUM): GSMArena fetch() loop lacks per-phone try/except
- CR30-02 (LOW): deduplicate_specs() manually reconstructs Spec objects — violates DRY
