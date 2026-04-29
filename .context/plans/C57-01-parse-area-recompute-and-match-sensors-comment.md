# C57-01: recompute `area` in `parse_existing_csv` from width*height; comment `match_sensors` mpix-disagree branch

**Cycle:** 57 (orchestrator cycle 10)
**Status:** COMPLETED
**Findings addressed:** F57-01 (bug), F57-02 (comment),
F57-DOC-01 (docstring), F57-TE-01 (test).

## Implementation summary

- `fix(csv)` (commit `89d7053`): `parse_existing_csv` now
  recomputes `area = width * height` when both dimensions are
  present; falls back to the `area_str` column only when size
  is unavailable. Also adds a one-line comment to
  `match_sensors` clarifying the mpix-disagree rejection branch
  (F57-02). Docstring updated to document the area trust
  contract (F57-DOC-01).
- `test(csv)` (commit `1eb4700`): adds a new test section
  `parse_existing_csv area recomputed from width*height` pinning
  three branches (recompute, size-missing fallback, zero-area
  rejection) (F57-TE-01). Updates the pre-existing fixture
  assertion that trusted bogus area data.
- `docs(reviews)` (commit `56272ee`): cycle 57 reviews and plan.

Both gates pass post-fix:
- `flake8 .` → 0 errors.
- `python3 -m tests.test_parsers_offline` → all sections green
  including the new `parse_existing_csv area recomputed from
  width*height` section (9 new assertions).

## Background

After C56-01 the codebase is stable; both gates pass at HEAD
`01c31d8`. This cycle's review fan-out surfaced one actionable
LOW bug (F57-01) flagged by 8 of 11 reviewers — the
`parse_existing_csv` `area` column is trusted as-is even when
width and height are both present.

## Repro (verified)

```text
input:  '1,Foo,dslr,,23.6,15.6,999.0,24.0,3.85,2020,'
parsed: width=23.6 height=15.6 area=999.0
expected: 23.6 * 15.6 = 368.16
```

## Plan

### Step 1: recompute area from width*height when both present (F57-01)

In `pixelpitch.py:413-426`, change

```python
size = None
width = _safe_float(width_str)
height = _safe_float(height_str)
if width is not None and width <= 0:
    width = None
if height is not None and height <= 0:
    height = None
if width is not None and height is not None:
    size = (width, height)

area = _safe_float(area_str)
if area is not None and area <= 0:
    area = None
```

to recompute `area = width * height` when size is known. The
`area_str` column becomes a fallback only when size is missing.
This matches `derive_spec`'s area contract.

### Step 2: tighten `parse_existing_csv` docstring (F57-DOC-01)

Add a sentence to the `parse_existing_csv` docstring stating
that `area` is recomputed from `width * height` when both are
present (matching `derive_spec`); the `area` column is consulted
only as a fallback when size is unavailable.

### Step 3: comment `match_sensors` mpix-disagree branch (F57-02)

In `pixelpitch.py:242-251`, add a one-line comment ahead of the
`if megapixel_match:` block clarifying that when both megapixel
sets are present and disagree, the sensor is rejected (no
`else: matches.append`).

### Step 4: add round-trip test (F57-TE-01)

Add a section to `tests/test_parsers_offline.py`:

```
== parse_existing_csv area recomputed from width*height ==
```

that:
- parses a CSV row with width=23.6, height=15.6, area=999.0
  (deliberately bogus area)
- asserts the parsed `area` equals 23.6 * 15.6 (368.16)
- asserts the parsed `size` equals (23.6, 15.6)

And a second sub-section that:
- parses a CSV row with width="", height="", area="50.0"
- asserts `size is None`
- asserts `area == 50.0` (fallback path).

### Step 5: re-run gates

- `python3 -m flake8 .` → 0
- `python3 -m tests.test_parsers_offline` → all green
  (including the new section).

### Step 6: commit, then push

Two fine-grained commits:
1. `fix(csv): 🐛 recompute area from width*height in parse_existing_csv`
2. `test(csv): ✅ pin parse_existing_csv area recomputation`

Optionally a third for the `match_sensors` comment:
3. `docs(merge): 📝 comment match_sensors mpix-disagree rejection`

All commits GPG-signed via `-S`. `git pull --rebase` before
push.

## Exit criteria

- `parse_existing_csv` no longer trusts the `area` column when
  width and height are both present.
- Test pins both branches (recompute + fallback).
- `match_sensors` mpix-disagree rejection is commented.
- Both gates pass.
- Plan moves to STATUS: COMPLETED.
