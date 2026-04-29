# Code-Reviewer Review (Cycle 58, orchestrator cycle 11)

**Date:** 2026-04-29
**HEAD:** `aef726b` (after C57-01 plan-completed marker)

## Inventory

- `pixelpitch.py` (1437 LOC) — primary module, all data-flow / merge / CLI / render.
- `models.py` — `Spec` and `SpecDerived` dataclasses.
- `sources/__init__.py`, `sources/apotelyt.py`, `sources/cined.py`,
  `sources/digicamdb.py`, `sources/gsmarena.py`,
  `sources/imaging_resource.py`, `sources/openmvg.py`.
- `tests/test_parsers_offline.py` (2595 LOC) — gate test file.
- `tests/test_sources.py`.
- Templates: `templates/index.html`, `templates/pixelpitch.html`,
  `templates/about.html`.
- CI / deploy: `.github/workflows/github-pages.yml`.

## Cycle 1–57 Status

All previously fixed findings remain fixed. Both gates pass at HEAD
`aef726b`:
- `flake8 .` → 0 errors.
- `python3 -m tests.test_parsers_offline` → all sections green
  (including the C57-01 `parse_existing_csv area recomputed from
  width*height` section).

## Cycle 58 New Findings

### F58-CR-01 (BUG): `source` CLI accepts negative or zero `--limit` silently — LOW

- **File:** `pixelpitch.py:1393-1399`
- **Detail:** `int(args[i + 1])` accepts `-1`, `0`, and any
  negative integer without validation. The parsed `limit` is
  passed through `kwargs["limit"]` to source modules where every
  consumer uses Python list slicing `urls[:limit]`
  (`apotelyt.py:162`, `cined.py:127`, `gsmarena.py:243`). With
  `limit=-1` slicing silently drops the last item; with
  `limit=0` slicing returns an empty list and the CSV is
  written empty. `openmvg.py:73` uses `i >= limit` which
  short-circuits at the first iteration when `limit <= 0`.
  Either way, the user gets confusing behavior with no error
  message.
- **Repro:** `python pixelpitch.py source openmvg --limit -1`
  exits silently with an empty `dist/camera-data-openmvg.csv`.
- **Fix:** Add `if limit <= 0: print(...); sys.exit(1)` to the
  `--limit` branch in `main()`. Match the existing
  `int(args[i + 1])` ValueError handler style.
- **Severity:** LOW. **Confidence:** HIGH.

### F58-CR-02 (cleanup, deferred): no test for `_safe_year` / `_safe_int_id` boundary at exactly 1900, 2100, 0, 1_000_000 — LOW

- **File:** `tests/test_parsers_offline.py` (gap)
- **Detail:** Existing parse-tolerance tests cover well below /
  above the bounds, but no test pins the exact boundary
  (`1900`, `2100`, `0`, `1_000_000` — `<=` operator, all
  inclusive). A future refactor to `<` would silently flip
  semantics.
- **Severity:** LOW. **Confidence:** MEDIUM.
- **Disposition:** Defer per repo policy on indirect coverage
  (analogous to F55-02).

### F58-CR-03 (style, no action): `_load_per_source_csvs` ignores `record_id` field after parse — INFO

- **File:** `pixelpitch.py:1093-1094`
- **Detail:** `parse_existing_csv` parses `id` from per-source
  CSVs, then `_load_per_source_csvs` immediately sets
  `d.id = None` to defer to `merge_camera_data`'s global id
  assignment. This is intentional and documented in the
  docstring, but the parse work for the id column is wasted.
- **Severity:** INFO. **Confidence:** HIGH.
- **Disposition:** No fix — wasted work is microscopic,
  documented intent is clear.

## Confidence summary

- 1 LOW actionable (F58-CR-01, reproducible silent no-op).
- 1 LOW deferred (F58-CR-02, indirect coverage suffices).
- 1 INFO (F58-CR-03, no fix).
