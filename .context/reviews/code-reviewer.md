# Code Review (Cycle 57)

**Reviewer:** code-reviewer
**Date:** 2026-04-29
**HEAD:** `01c31d8` (after C56-01 completion, gates green at HEAD)
**Gates:** `flake8 .` = 0; `python3 -m tests.test_parsers_offline` = green
(all sections including the new C56-01 size-less branch test).

## Inventory of code-relevant files (re-verified)

- `pixelpitch.py` (1416 lines), `models.py` (27 lines).
- `sources/__init__.py`, `apotelyt.py`, `cined.py`, `digicamdb.py`,
  `gsmarena.py`, `imaging_resource.py`, `openmvg.py`.
- `tests/test_parsers_offline.py` (2536 lines), `tests/test_sources.py`.
- `templates/*.html`.

All examined.

## New findings (cycle 57)

### F57-CR-01: `parse_existing_csv` size column and `area` column allowed to be inconsistent — LOW

- **File:** `pixelpitch.py:413-432`
- **Detail:** `parse_existing_csv` reads `width`, `height`, and `area`
  as three independent columns, then constructs both `Spec.size` and
  `derived.area` directly from each column, with no consistency check.
  A hand-edited CSV that has `width=23.6, height=15.6, area=999.0`
  will round-trip with the bogus area, and downstream consumers
  (template, write_csv) will emit it. `derive_spec` recomputes
  `area = width * height` when given a fresh Spec, but
  `parse_existing_csv` bypasses `derive_spec`.
- **Failure scenario:** A user hand-edits the CSV, intending to fix
  width/height, but forgets to clear `area`. The next deploy emits
  the stale area in the rendered HTML and CSV, defeating the visual
  sanity check that pitch ≈ 1000 * sqrt(area / mpix * 1e6).
- **Suggested fix:** When width and height are both present in
  `parse_existing_csv`, recompute `area = width * height` rather than
  trusting the column. Keep the parsed area only as a fallback when
  width or height is missing.
- **Severity:** LOW. **Confidence:** HIGH.

### F57-CR-02: `match_sensors` silently rejects sensor when megapixels and sensor_megapixels both present but disagree — LOW (intentional but undocumented)

- **File:** `pixelpitch.py:242-251`
- **Detail:** When both `megapixels` and `sensor_data["megapixels"]`
  are positive non-empty, `match_sensors` only appends if
  `megapixel_match` is True. On disagreement the sensor is silently
  rejected. This is intentional (rejection is more conservative than
  a size-only match) but the behaviour is not commented.
- **Suggested fix:** Add a single-line comment that states "when both
  megapixel sets are present and disagree, the sensor is rejected"
  to prevent future refactors from accidentally adding an
  `else: matches.append(...)` branch.
- **Severity:** LOW. **Confidence:** HIGH.

### F57-CR-03: `_load_per_source_csvs` size-less drop comment redundant with docstring — INFO

- **File:** `pixelpitch.py:1086-1088`
- **Detail:** Inline comment "Size unknown — matched_sensors is
  meaningless; honor derive_spec's 'not checked' sentinel" partly
  duplicates the C56-01 docstring update. Cosmetic only.
- **Severity:** INFO. **Confidence:** LOW.
- **Disposition:** No action; redundancy aids reader of the inner
  loop without forcing them to scroll up.

## Confirmed-still-good

- `_safe_int_id`, `_safe_year`, `_safe_float`: all hardened by
  C50-C53 plans. Tests pin Excel-coerced floats, NaN, inf, range
  guards.
- `merge_camera_data` matched_sensors preservation contract (C46):
  test pins both branches.
- `parse_existing_csv` BOM detection: pinned by C55-01 test.
- `_load_per_source_csvs` size-less branch: pinned by new C56-01
  test.

## Sweep for commonly missed issues

- Race conditions: single-threaded; N/A.
- Resource leaks: `read_text`, `write_text` use context managers
  via Path; OK.
- Error handling: try/except blocks all preserve the row or skip
  cleanly; no silent swallow detected.
- Comment/code drift: F57-CR-03 only.

## Confidence summary

- 1 LOW actionable (F57-CR-01: area consistency on parse-back).
- 1 LOW doc/comment (F57-CR-02: match_sensors disagreement
  comment).
- 1 INFO no-action (F57-CR-03).
