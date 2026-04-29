# Verifier Review (Cycle 59, orchestrator cycle 12)

**Date:** 2026-04-29
**HEAD:** `fa0ae66`

## Gate verification

- `python3 -m flake8 .` -> 0 errors. PASS.
- `python3 -m tests.test_parsers_offline` -> all sections green.
  PASS.

No regression vs cycle 58 (HEAD `aef726b` -> `fa0ae66` is
docs-only).

## F59-CR-01 reproduction (defensive-parity gap)

Constructed a SpecDerived bypassing `derive_spec`:

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
print(p.read_text())
```

Output (relevant row):

```
0,Bypass Cam,fixed,,0.00,0.00,,,,2020,
```

The `0.00,0.00` columns demonstrate the gap: `derived.size =
(0.0, 0.0)` is truthy (non-empty tuple), so `if derived.size`
in `write_csv` writes `"0.00"` for both width and height even
though the values are non-positive. The `area_str` column
correctly remains empty (caught by the `derived.area > 0`
guard). The `mpix_str` correctly writes `"10.0"`. So the
asymmetry is real: width/height accept invalid values that
area/mpix/pitch reject.

In normal operation the bypass is not reachable (derive_spec
filters non-positive size, parse_existing_csv too). But the
defensive parity matters for:

1. Future regressions in derive_spec (the guard at line 900
   could be inadvertently removed by a refactor).
2. New code paths constructing SpecDerived directly (e.g., a
   future source module that calls write_csv on its own
   derived list without going through derive_spec).
3. Test fixtures and synthetic SpecDerived inputs.

Fix: harden the width/height write at the boundary, mirroring
the area/mpix/pitch guards.

## Confirmation matrix

- F59-01: confirmed reproducible with the synthetic
  SpecDerived above. Behavior is "writes 0.00,0.00 to CSV";
  desired behavior is "writes empty cells".

## Cycle 1-58 fix verification

All previous cycle fixes verified at HEAD via the gate:

- C58-01 (`--limit -1` rejection): `--limit -1` -> SystemExit
  with non-zero status, error to stdout. PASS.
- C57-01 (area recompute from width*height in
  parse_existing_csv): test section green. PASS.
- All earlier cycles: gate green.
