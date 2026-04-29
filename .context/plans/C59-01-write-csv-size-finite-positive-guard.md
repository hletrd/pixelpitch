# C59-01: harden write_csv width/height columns with isfinite/positive guards

**Cycle:** 59 (orchestrator cycle 12)
**Status:** COMPLETED
**Findings addressed:** F59-01 (defensive-parity bug), F59-02
(test gap), F59-03 (docstring contract).

## Implementation summary

- `fix(csv)` (commit `1551b6e`): `write_csv` now guards the
  width/height columns with the same isfinite/positive checks
  used for area/mpix/pitch. Atomic-pair semantics: either both
  cells are populated or both are empty.
- `test(csv)` (commit `459229b`): adds the
  `write_csv width/height non-finite/non-positive guards`
  section with 18 assertions across inf, -inf, nan, zero,
  negative, mixed, sanity, and None cases.
- `docs(csv)` (commit `7bbde52`): write_csv docstring now
  documents the float-cell contract (which columns, empty-cell
  semantics, atomic-pair rule, round-trip guarantee with
  parse_existing_csv).

Both gates pass post-fix:
- `flake8 .` -> 0 errors.
- `python3 -m tests.test_parsers_offline` -> all sections green
  including the new
  `write_csv width/height non-finite/non-positive guards`
  section (18 new assertions).

## Background

After C58-01 the codebase is stable; both gates pass at HEAD
`fa0ae66`. This cycle's review fan-out surfaced one actionable
LOW defensive-parity finding (F59-01) flagged by 7 of 11
reviewers - the `write_csv` width/height columns rely on a
truthiness check (`if derived.size:`) that does not catch
non-finite (`inf`/`nan`) or non-positive (`<=0`) tuple
elements, while the area/mpix/pitch columns explicitly guard
each value with `isfinite(...) and ... > 0`.

Today upstream guards (`derive_spec` line 900,
`parse_existing_csv` line 430-433) prevent pathological
size tuples from reaching `write_csv`, so this is a
defensive-parity hygiene improvement rather than a live bug.
The fix co-locates the contract enforcement at the write
boundary, so a future regression in `derive_spec` or a new
direct-SpecDerived construction site does not leak
`inf`/`nan`/`0.00`/`-1.00` strings into the artifact CSV.

## Repro (verifier)

```python
from models import Spec, SpecDerived
from pixelpitch import write_csv
from pathlib import Path
import tempfile

spec = Spec(name="Bypass Cam", category="fixed", type=None,
            size=(0.0, 0.0), pitch=None, mpix=10.0, year=2020)
d = SpecDerived(spec=spec, size=(0.0, 0.0), area=0.0,
                pitch=None, matched_sensors=[], id=0)

with tempfile.NamedTemporaryFile("w+", suffix=".csv",
                                 delete=False) as f:
    p = Path(f.name)
write_csv([d], p)
# CSV row contains: "0,Bypass Cam,fixed,,0.00,0.00,,,,2020,"
# The width/height cells are populated as "0.00" rather than
# being empty.
```

## Plan

### Step 1: harden write_csv width/height (F59-01)

In `pixelpitch.py:1018-1019`, replace the truthy-only check
with an explicit isfinite/positive guard mirroring the
area/mpix/pitch guards:

```python
if (derived.size
        and isfinite(derived.size[0]) and derived.size[0] > 0
        and isfinite(derived.size[1]) and derived.size[1] > 0):
    width_str = f"{derived.size[0]:.2f}"
    height_str = f"{derived.size[1]:.2f}"
else:
    width_str = ""
    height_str = ""
```

Same pattern as F40-01 (write_csv finite-guard for pitch /
mpix). The atomic-pair semantic ("either both populated or
both empty") is the simplest rule and matches how
`parse_existing_csv` reads the columns (it requires both for
`size` to be non-None at line 434-435).

### Step 2: update the write_csv docstring (F59-03)

In `pixelpitch.py:1000-1001`, expand the docstring to document
the float-cell contract:

```python
def write_csv(specs: list[SpecDerived], output_file: Path) -> None:
    """Write camera specs to a CSV file using the csv module for proper escaping.

    Float-cell contract: the numeric columns
    ``sensor_width_mm``, ``sensor_height_mm``, ``sensor_area_mm2``,
    ``megapixels``, and ``pixel_pitch_um`` are guarded against
    non-finite (``inf``/``nan``) and non-positive (``<= 0``)
    values - those rows write an empty cell instead of
    ``"inf"``/``"nan"``/``"0.0"``/``"-1.0"``. ``parse_existing_csv``
    relies on this contract for round-trip safety.
    """
```

### Step 3: add a regression test (F59-02)

Add a new section to `tests/test_parsers_offline.py`:

```
== write_csv width/height non-finite/non-positive guards ==
```

The test constructs synthetic SpecDerived bypassing
`derive_spec`, calls `write_csv`, and asserts the CSV row's
width/height cells are empty for `inf`, `nan`, `0.0`,
negative, and mixed cases:

- `derived.size = (float("inf"), 24.0)` -> both cells empty
- `derived.size = (35.9, float("nan"))` -> both cells empty
- `derived.size = (float("-inf"), 24.0)` -> both cells empty
- `derived.size = (0.0, 0.0)` -> both cells empty
- `derived.size = (-1.0, -1.0)` -> both cells empty
- Sanity: `derived.size = (35.9, 23.9)` -> "35.90","23.90"
- Sanity: `derived.size = None` -> both cells empty

The test mirrors `test_write_csv_nonfinite_guards` and
`test_write_csv_zero_negative_guards` for parity.

### Step 4: re-run gates

- `python3 -m flake8 .` -> 0
- `python3 -m tests.test_parsers_offline` -> all green
  (including the new section).

### Step 5: commit, then push

Three fine-grained commits, all GPG-signed via `-S`:

1. `fix(csv): :bug: guard write_csv width/height against non-finite/non-positive` (use the bug gitmoji)
2. `test(csv): :white_check_mark: pin write_csv size-column finite/positive guards`
3. `docs(csv): :memo: document write_csv float-cell contract in docstring`

`git pull --rebase` before push.

## Exit criteria

- `write_csv` no longer writes "inf"/"nan"/"0.00"/"-1.00" for
  the width/height columns; cells become empty.
- Docstring documents the float-cell contract.
- Test pins all five non-finite/non-positive cases plus the
  two sanity cases.
- Both gates pass.
- Plan moves to STATUS: COMPLETED.

## Deferred (per F59 deferred-fix rules)

- F59-04 (`_load_per_source_csvs` "missing" log line wording)
  - informational, no behavior change. Defer.
