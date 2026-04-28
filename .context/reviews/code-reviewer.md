# Code Reviewer — Cycle 48

**Date:** 2026-04-29
**Reviewer:** code-reviewer

## Inventory

Examined every Python source file:
- `pixelpitch.py` (1291 lines)
- `models.py` (27 lines)
- `sources/__init__.py` (111 lines)
- `sources/apotelyt.py` (184 lines)
- `sources/cined.py` (148 lines)
- `sources/digicamdb.py` (32 lines)
- `sources/gsmarena.py` (269 lines)
- `sources/imaging_resource.py` (302 lines)
- `sources/openmvg.py` (129 lines)
- `tests/test_parsers_offline.py` (2098 lines)
- `tests/test_sources.py` (111 lines)

## New Findings (Cycle 48)

### F48-01: Flake8 gate failures across the repo (33 errors)
- **Severity:** MEDIUM | **Confidence:** HIGH
- **Files affected:**
  - `pixelpitch.py:17` — F401 unused `dataclass` import
  - `pixelpitch.py:47` — E402 module-level import not at top of file
  - `pixelpitch.py:1240` — F541 f-string missing placeholders
  - `sources/__init__.py:30` — F401 unused `dataclass` import
  - `sources/apotelyt.py:37` — E303 too many blank lines (3)
  - `sources/cined.py:36` — E302 expected 2 blank lines, found 1
  - `tests/test_parsers_offline.py:17` — F401 unused `io`
  - `tests/test_parsers_offline.py:25` — E402 module-level import not at top
  - `tests/test_parsers_offline.py:392` — E231 missing whitespace after `,`
  - `tests/test_parsers_offline.py:594,666,845,977,1244,1822,1856` — F401 unused `models.SpecDerived` / `models.Spec` imports
  - `tests/test_parsers_offline.py:848,853,1232,1497,1550,1559,1568,1577,1586` — E127 continuation line over-indented
  - `tests/test_parsers_offline.py:1241` — F811 redefinition of unused `io`
  - `tests/test_parsers_offline.py:1271` — F841 local variable `merged2` assigned but never used
  - `tests/test_sources.py:23-25` — E402 module-level imports not at top
  - `tests/test_sources.py:32` — E231 missing whitespace after `:` and `,`
- **Why it's a problem:** The repo's `setup.cfg` declares flake8 with `max-line-length=160` as a gate. Currently 33 errors are emitted. The cycle GATES require lint to pass.
- **Failure scenario:** CI gate fails. Subsequent cycles inherit the failure.
- **Fix:** Remove unused imports, hoist `sys.path` insertion via `# noqa: E402` where genuinely needed; fix indentation; drop dead `merged2`; convert `f""` without placeholders to plain string; add whitespace; collapse extra blank lines.

### F48-02: Unused local `merged2` masks a likely missing assertion
- **File:** `tests/test_parsers_offline.py:1271`
- **Severity:** LOW | **Confidence:** MEDIUM
- **Why it's a problem:** F841 — the `merged2 = merge_camera_data(...)` call's result is never asserted. Either an assertion is missing or the call exists only for side-effects (not the case for `merge_camera_data` which returns a fresh dict).
- **Fix:** Drop the assignment if side-effect-only; otherwise add the missing assertion.

### F48-03: Duplicate `import io` (top-level + function-scope)
- **File:** `tests/test_parsers_offline.py:17` (top-level) and `tests/test_parsers_offline.py:1241` (inside test function)
- **Severity:** LOW | **Confidence:** HIGH
- **Why it's a problem:** F811 — top-level `io` import is unused, and a function-scoped re-import shadows it.
- **Fix:** Drop the unused top-level `io` import.

## Confirmation of prior status

- All correctness fixes from cycles 1–46 still pass the test gate.
- No new logic regressions found.

## Confidence Summary

| Finding | Severity | Confidence |
|---------|----------|------------|
| F48-01  | MEDIUM   | HIGH       |
| F48-02  | LOW      | MEDIUM     |
| F48-03  | LOW      | HIGH       |
