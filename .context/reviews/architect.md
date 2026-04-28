# architect Review (Cycle 52)

**Date:** 2026-04-29
**HEAD:** 331c6f5

## Layering

- Entry: `pixelpitch.py` (CLI + render orchestration).
- Sources: `sources/<name>.py` modules with a `fetch()` callable,
  registered in `SOURCE_REGISTRY`.
- Models: `models.py` (Spec, SpecDerived).
- Tests: offline + live.
- Templates: Jinja2 in `templates/`.

The shape is sound for a static-site builder. No new layering smell.

## Architectural pattern emerging

The CSV round-trip (`write_csv` ↔ `parse_existing_csv`) has emerged as
the load-bearing serialization boundary across cycles 46-52. Each cycle
has hardened one column at a time:

- cycle 40: write-side finite/positive guards (mpix, area, pitch)
- cycle 50: matched_sensors round-trip + `;`-injection guard
- cycle 51: parse-side whitespace + dedup tolerance for matched_sensors
- cycle 52 (this cycle): year column parse tolerance — F52-01

A general "every CSV cell tolerates Excel hand-edit" invariant would
unify these. Out of scope as a refactor (F32 deferred) but worth
flagging as a target end-state.

## Carry-forward architectural items (deferred)

- F32: `pixelpitch.py` is 1314 LOC monolith — DEFERRED.
- F31: No source Protocol/base class — DEFERRED.
- C22-05: Field preservation logic is ad-hoc in `merge_camera_data` —
  DEFERRED.

## No new architectural findings this cycle.
