# Code Reviewer — Cycle 50

**Date:** 2026-04-29
**HEAD:** `ed45eed` · **Gates:** flake8 0 errors · `tests.test_parsers_offline` PASS

## Inventory

- `pixelpitch.py` (1291 LOC) — main pipeline
- `models.py` (27 LOC) — Spec / SpecDerived dataclasses
- `sources/__init__.py` (110 LOC) — shared regex + http_get
- `sources/{apotelyt,cined,digicamdb,gsmarena,imaging_resource,openmvg}.py`
- `tests/test_parsers_offline.py` (2096 LOC), `tests/test_sources.py` (111 LOC)
- `templates/{index,pixelpitch,about}.html`
- `setup.cfg`, `.github/workflows/github-pages.yml`

## Findings

### F50-01 — `git pull --rebase || true` masks rebase failures (LOW / HIGH)
- File: `.github/workflows/github-pages.yml:108`
- Carry-forward of F49-02; not yet implemented. The subsequent `git push` would fail noisily on non-fast-forward, but the rebase step shows green misleadingly. Recommend dropping `|| true` or replacing with explicit error reporting.

### F50-03 — matched_sensors round-trip uses unescaped `;` delimiter (LOW / MEDIUM)
- File: `pixelpitch.py:373` (split) and `pixelpitch.py:920-922` (join)
- `write_csv` joins with `;`, `parse_existing_csv` splits on `;`. No escape. Currently safe because no sensor name in `sensors.json` contains `;`, but undocumented invariant. Either document the contract with an assertion in `write_csv`, or migrate to a safer delimiter.

## Confirmations
- F49-01 fully resolved (CI now runs flake8). Verified by reading `.github/workflows/github-pages.yml:46-50`.
- All cycle 1-48 fixes still in place.
- Field-preservation logic in `merge_camera_data` (lines 408-555) handles the C46-01 matched_sensors edge correctly.
- write_csv non-finite float guards (lines 916-918) work as documented.

## Confidence Summary
- HIGH: F50-01 logic; F49-01 resolved.
- MEDIUM: F50-03 (no current trigger; defense-in-depth).
