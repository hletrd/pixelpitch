# code-reviewer Review (Cycle 52)

**Date:** 2026-04-29
**HEAD:** 331c6f5
**Gates:** flake8 0 errors; `python3 -m tests.test_parsers_offline` PASS

## Inventory

- Reviewed: `pixelpitch.py` (1314 LOC), `models.py`, `sources/*.py`, `tests/*.py`,
  `templates/*.html`, `.github/workflows/github-pages.yml`, `setup.cfg`,
  `requirements.txt`, prior cycles' aggregate, `.context/plans/deferred.md`.

## Verification of cycle 51 fixes

- F51-01 (whitespace tolerance) — FIXED at `a0ac8bc`. Code at lines
  377-381 now: `_raw = (s.strip() for s in sensors_str.split(";"))`
  / `list(dict.fromkeys(s for s in _raw if s))`. Test added at
  `d1b0ca1`.
- F51-02 (dedup on parse) — folded into the F51-01 fix; same test
  asserts duplicate removal.

## New findings

### F52-01: `year` column parser silently drops `2023.0`-style values — LOW / MEDIUM

- **File:** `pixelpitch.py:366-372`
- **Code:**
  ```python
  if year_str:
      try:
          y = int(year_str)
          if 1900 <= y <= 2100:
              year = y
      except ValueError:
          pass
  ```
- **Detail:** `int("2023.0")` raises `ValueError` and the year is silently
  dropped. `write_csv` (line 927) emits clean integer years today, so the
  internal round-trip works. But the same Excel hand-edit scenario that
  motivated F51-01 (whitespace tolerance) and F50-04 (matched_sensors
  round-trip preservation) applies to the year column too. Excel often
  coerces an integer year value to `2023.0` when the column type is
  guessed as numeric.
- **Failure scenario:** Maintainer opens `dist/camera-data.csv` in Excel
  → makes a small edit → saves → CI re-renders → year column blanks for
  every edited row.
- **Fix:** Tolerate trailing `.0` and surrounding whitespace. Try `int()`
  first; on `ValueError`, fall back to `int(float(year_str))`. Keep the
  1900-2100 range guard. Reject NaN/inf (covered by isfinite test).
- **Confidence:** MEDIUM
- **Severity:** LOW (no current trigger; defense-in-depth alongside
  F51-01 parser-tolerance change)

### F52-02: per-agent reviews and aggregate are uncommitted in working tree — LOW (process)

- **File:** `.context/reviews/*.md`
- **Detail:** All 12 review files appear modified-but-uncommitted at
  cycle start (per `git status`). The cycle's docs commit must include
  the refreshed reviews + aggregate, matching cycle-51's `331c6f5`
  pattern.
- **Severity:** LOW (process)
- **Confidence:** HIGH
