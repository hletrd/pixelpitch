# architect Review (Cycle 51)

**Date:** 2026-04-29
**HEAD:** 3b35dcc

## Layering

- Entry: `pixelpitch.py` (CLI + render orchestration).
- Sources: `sources/<name>.py` modules with a `fetch()` callable, registered in
  `SOURCE_REGISTRY`.
- Models: `models.py` (Spec, SpecDerived).
- Tests: offline + live.
- Templates: Jinja2 in `templates/`.

The shape is sound for a static-site builder. No new layering smell.

## Carry-forward architectural items (deferred)

- F32: `pixelpitch.py` is 1306 LOC monolith — DEFERRED (acceptable for solo-developer
  static site).
- F31: No source Protocol/base class — DEFERRED.
- C22-05: Field preservation logic is ad-hoc in `merge_camera_data` — DEFERRED (refactor
  risk > benefit).

## No new architectural findings this cycle.

The cycle-50 changes were small, orthogonal, and did not affect module boundaries.
