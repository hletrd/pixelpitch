# Code Review (Cycle 55)

**Reviewer:** code-reviewer
**Date:** 2026-04-29
**HEAD:** `f08c3c4` (after C54-01 completion)
**Gates:** flake8 = 0; `python3 -m tests.test_parsers_offline` = green.

## Inventory of code-relevant files (re-checked)

- `pixelpitch.py` (1414 lines), `models.py` (27 lines).
- `sources/__init__.py`, `apotelyt.py`, `cined.py`, `digicamdb.py`,
  `gsmarena.py`, `imaging_resource.py`, `openmvg.py`.
- `tests/test_parsers_offline.py` (2336 lines), `tests/test_sources.py`.
- `templates/*.html`.

## New findings (cycle 55)

### F55-CR-01: `parse_existing_csv` `has_id` schema detection ignores hand-edited leading-blank-cell — LOW

- **File:** `pixelpitch.py:371-372`
- **Detail:** `has_id = header[0] == "id"`. After strip_bom the BOM is
  gone (good). But if a hand-edited CSV has an inserted blank column
  before id (Excel sometimes does this), `has_id` becomes False and
  the parser silently maps to the no-id schema, mis-aligning every
  subsequent column.
- **Fix:** test `"id" in {h.strip().lower() for h in header}`.
- **Severity:** LOW. **Confidence:** MEDIUM (theoretical Excel edit).

### F55-CR-02: `_load_per_source_csvs` lazy-load idempotency relies on `{}` not being None — LOW (cosmetic)

- **File:** `pixelpitch.py:1076-1086`
- **Detail:** When `load_sensors_database()` returns `{}`, the
  `if sensors_db is None:` guard succeeds only the first time and
  subsequent rows correctly re-use the empty dict. Behavior is
  correct; comment could clarify.
- **Severity:** LOW (no fix needed). **Confidence:** HIGH.

### F55-CR-03: `merge_camera_data` mutates `existing_specs` items in-place — LOW (latent)

- **File:** `pixelpitch.py:622-628`
- **Detail:** Re-matching writes back to the caller's list (`existing_spec.matched_sensors = ...`). Today no caller reuses
  the input list, so no observable bug.
- **Fix:** Use a fresh `replace(existing_spec, matched_sensors=...)`.
- **Severity:** LOW. **Confidence:** HIGH (latent contract issue).

## Re-examined (still OK)

- `_safe_int_id` range guard.
- `_safe_year` range guard.
- `parse_existing_csv` matched_sensors strip+dedup.
- `write_csv` `;` delimiter guard.
- `merge_camera_data` matched_sensors tri-valued sentinel.
- CI flake8 gate.
- Rebase failure surfacing.
