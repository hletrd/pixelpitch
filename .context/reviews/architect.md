# Architect Review (Cycle 57)

**Reviewer:** architect
**Date:** 2026-04-29
**HEAD:** `01c31d8`

## System-level inventory

- Single 1416-line orchestrator (`pixelpitch.py`) calls
  scraper plugins (`sources/*.py`).
- Models in `models.py` (Spec, SpecDerived).
- Tests in `tests/`.
- Templates in `templates/`.

## Architectural risks

### F57-A-01: `area` is a derived-but-stored field, with two sources of truth — LOW (carry of F57-CR-01)

- **File:** `pixelpitch.py` (parse_existing_csv, derive_spec, write_csv).
- **Detail:** `area` is conceptually `width * height`, but stored
  independently in CSV. parse-time treats them as independent;
  derive-time treats `area` as derived. Adding a `derive_area()`
  helper used by both paths would dedup the contract.
- **Severity:** LOW. **Confidence:** HIGH.
- **Disposition:** Schedule the parse-time fix only; a refactor
  of derive_spec is not justified by this single bug.

### F57-A-02: `SpecDerived` constructor accepts `size=None` and `area>0` — partial defensive gap

- **Files:** `models.py`, `pixelpitch.py:447-451`.
- **Detail:** No invariant prevents constructing a SpecDerived where
  size is None but area is not None. `derive_spec` enforces this
  invariant; `parse_existing_csv` does not. The current path does
  set both to None when the column is missing, so the invariant
  holds in practice — but adding a `__post_init__` validator would
  prevent future breakage.
- **Severity:** LOW. **Confidence:** MEDIUM.
- **Disposition:** Defer; over-engineering for a stable
  data class.

### Carry-over
- F32 monolith → re-defer.
- F55-A-02 / F56-A-02 category list duplication → re-defer.

## Confidence summary

- 1 LOW actionable (F57-A-01, overlaps F57-CR-01).
- 1 LOW deferred (F57-A-02 invariant validation).
