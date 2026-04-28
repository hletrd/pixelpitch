# Code Review (Cycle 56)

**Reviewer:** code-reviewer
**Date:** 2026-04-29
**HEAD:** `e8d5414` (after C55-01 completion)
**Gates:** flake8 = 0; `python3 -m tests.test_parsers_offline` = green.

## Inventory of code-relevant files (re-checked)

- `pixelpitch.py` (1415 lines), `models.py` (27 lines).
- `sources/__init__.py`, `apotelyt.py`, `cined.py`, `digicamdb.py`,
  `gsmarena.py`, `imaging_resource.py`, `openmvg.py`.
- `tests/test_parsers_offline.py` (2456 lines), `tests/test_sources.py`.
- `templates/*.html`.

## New findings (cycle 56)

### F56-CR-01: `_load_per_source_csvs` lazy-load gate inside size-conditional drops cache when row has no size — LOW (latent)

- **File:** `pixelpitch.py:1071-1088`
- **Detail:** The new branch structure (post-C55-01) only attempts
  `load_sensors_database()` when `d.size is not None`. When the row
  has no size, `d.matched_sensors = None` is set unconditionally —
  even if a parsed cache value existed. This is the documented
  "size unknown means not checked" sentinel and matches `derive_spec`,
  so behavior is intentional. However, the docstring does not call
  out that a *cached* matched_sensors on a size-less row is also
  dropped. Cosmetic doc gap only.
- **Severity:** LOW. **Confidence:** HIGH.

### F56-CR-02: `merge_camera_data` size-mismatch warning emits when both come from same source — LOW (cosmetic)

- **File:** `pixelpitch.py:539-557`
- **Detail:** When existing CSV had `(23.6, 15.6)` and new fetch
  produced `(23.7, 15.7)` for the same camera (rounding drift across
  source updates), warning prints. Currently no tolerance — any
  mismatch warns. Most cases are sub-percent rounding noise.
- **Severity:** LOW (noise in CI logs only). **Confidence:** MEDIUM.
- **Disposition:** Defer; tolerance threshold would be subjective.

### F56-CR-03: `parse_existing_csv` silently truncates rows with extra columns past index 10 — LOW

- **File:** `pixelpitch.py:397-411`
- **Detail:** `sensors_str = values[10] if len(values) > 10 else ""`
  reads only column 10. If a future schema adds a column at index 11
  without updating the parser, the new column is ignored without
  warning. Acceptable since `write_csv`/`parse_existing_csv` are
  maintained together (per deferred F32 / C11-09).
- **Severity:** LOW. **Confidence:** HIGH.
- **Disposition:** Defer per existing C11-09 rationale.

## Re-examined (still OK at e8d5414)

- C55-01 `_load_per_source_csvs` empty-db cache fallback — code at
  `pixelpitch.py:1073-1087` matches the documented contract.
- `_safe_int_id` range guard, `_safe_year` range guard.
- `parse_existing_csv` matched_sensors strip+dedup.
- `write_csv` `;` delimiter guard.
- `merge_camera_data` matched_sensors tri-valued sentinel.
- CI flake8 gate.
- Rebase failure surfacing in CI.
- README enumerates smartphone.html/cinema.html (post-C55-01).
