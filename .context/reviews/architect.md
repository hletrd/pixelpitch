# Architect — Cycle 50

**Date:** 2026-04-29
**HEAD:** `ed45eed`

## Architectural state

Layered structure:
1. `models.py` — pure data
2. `sources/` — fetchers (returns `Spec`)
3. `pixelpitch.py` — derivation, merge, render
4. `templates/` — view
5. CI — `.github/workflows/github-pages.yml`

## Findings

No new architectural findings this cycle. F32 (1291-line monolith) remains deferred; the file's growth this cycle is +0 LOC.

## Confirmations

- `SOURCE_REGISTRY` boundary is clean: each source returns `list[Spec]` and is independent of `pixelpitch.py` derivation logic.
- `merge_camera_data` is the single integration seam — well-documented, well-tested.
- CI now enforces both gates declared by the orchestrator (flake8 + tests). The pipeline's structural invariants match the orchestrator's contract.

## Architectural smells (deferred / acknowledged)

- F32: `pixelpitch.py` monolith — deferred, threshold for re-open at 1500 LOC; current 1291.
- C22-05: ad-hoc field preservation — deferred, threshold at 12+ if statements; currently ~10.
- F31: no Source Protocol — deferred.

## Summary

Architecture stable. No new risks introduced this cycle.
