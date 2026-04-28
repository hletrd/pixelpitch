# code-reviewer Review (Cycle 51)

**Date:** 2026-04-29
**HEAD:** 3b35dcc (post-cycle-50 plan committed)
**Gates:** flake8 0 errors; `python3 -m tests.test_parsers_offline` PASS

## Inventory

- Reviewed: `pixelpitch.py` (1306 LOC), `models.py`, `sources/*.py`, `tests/*.py`,
  `templates/*.html`, `.github/workflows/github-pages.yml`, `setup.cfg`, `requirements.txt`,
  `.context/reviews/_aggregate.md` (cycle 50 baseline), `.context/plans/deferred.md`.

## New Findings

### F51-01: `parse_existing_csv` does not strip whitespace around `matched_sensors` tokens — LOW / MEDIUM
- **File:** `pixelpitch.py:373`
- **Detail:** `matched_sensors = [s for s in sensors_str.split(";") if s] if sensors_str else []`.
  The split does not call `.strip()` on each element. Today this is fine because `write_csv`
  produces tokens with no surrounding whitespace, but the pairing is only verified by the
  cycle-50 round-trip test on a single shape. If a hand-edited CSV (or future writer change)
  introduces ` IMX455` with a leading space, it would parse as a distinct token from `IMX455`
  and round-trip through the next CSV write, producing phantom tokens.
- **Failure scenario:** A user with the dist artifact open in Excel inserts a row with
  `IMX455; IMX571`. After save, `parse_existing_csv` produces `["IMX455", " IMX571"]`. The
  merge logic preserves these verbatim and they round-trip through the next CSV write. No
  crash, but the sensor list contains a whitespace-prefixed phantom.
- **Fix:** Add `.strip()` in the comprehension:
  `[s.strip() for s in sensors_str.split(";") if s.strip()]`.
- **Confidence:** MEDIUM
- **Severity:** LOW (no current trigger)

### F51-02: `write_csv` matched_sensors guard mutates the joined string but not the SpecDerived — LOW (informational)
- **File:** `pixelpitch.py:925-937`
- **Detail:** When an element of `derived.matched_sensors` contains `;`, the writer drops it
  from the *output string* but the in-memory `derived.matched_sensors` retains the offending
  element. This is the intended scope of the cycle-50 fix, but the comment at lines 920-924
  could note explicitly that the in-memory list is unchanged.
- **Fix:** Cosmetic comment addition only. Out of scope unless paired with another change.
- **Confidence:** HIGH
- **Severity:** LOW (informational; no behavior change requested)

## Repeated Findings (carry-forward)

None new. F50-01..04 are fully resolved at this cycle's HEAD (commits 5f2a3fd, 9dc88fa, 5b31802).
