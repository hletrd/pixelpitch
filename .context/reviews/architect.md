# Architect — Cycle 53

**Date:** 2026-04-29
**HEAD:** `1c968dd`

## Architectural review

No new architectural risks this cycle.

The codebase remains a clean five-layer static site generator:

1. `models.py` — Spec / SpecDerived dataclasses.
2. `sources/*.py` — per-source scrapers.
3. `pixelpitch.py` parse/merge/write block — CSV round-trip.
4. `pixelpitch.py` render block — Jinja2 → static HTML.
5. `tests/*.py` — offline integration tests.

## Coupling check

`_safe_year`, `_safe_int_id`, `_safe_float` are pure, local to
`pixelpitch.py`, and depend only on `math.isfinite`. Coupling minimal.

## Layering check

No leakage. No source module imports parse helpers. Tests exercise
`parse_existing_csv` through its public API.

## F53-ARCH-01 (suggestion only)

If a fourth `_safe_*` helper appears, consider unifying. Three is
fine for now. Not filed.

## Verdict

No new architectural findings.
