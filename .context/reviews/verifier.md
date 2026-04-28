# Verifier Review (Cycle 37) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-36 fixes

## V37-01: Gate tests pass — all checks verified

**Evidence:** Ran `python3 -m tests.test_parsers_offline` — all checks passed. C36 fixes verified working. No regressions.

## V37-02: `derive_spec` with NaN size tuple produces nan area then 0.0 pitch — verified

**File:** `pixelpitch.py` lines 726-733
**Severity:** MEDIUM | **Confidence:** HIGH

**Evidence:** Tested directly:
```python
import pixelpitch as pp
from models import Spec

spec = Spec(name='NaN Size', category='fixed', type=None,
            size=(float('nan'), 24.0), pitch=None, mpix=10.0, year=2020)
d = pp.derive_spec(spec)
# d.area = nan (not None)
# d.pitch = 0.0 (pixel_pitch guard catches it)
```

The test `test_derive_spec_negative_size` already tests `size=(nan, 24.0)` and expects `pitch=0.0`, which passes. However, the area is `nan` rather than `None`, which is inconsistent. The template only reads `derived.pitch` (not `derived.area`), so this doesn't cause visible rendering issues. But `write_csv` would write `area_str = f"{nan:.2f}"` = `"nan"` to CSV, and `_safe_float("nan")` on re-read would return `None`. So there's a round-trip asymmetry: `area=nan` writes as `"nan"` which reads back as `None`.

## V37-03: Source parser regex patterns naturally exclude NaN/inf — verified

**Files:** `sources/cined.py`, `sources/apotelyt.py`, `sources/imaging_resource.py`, `sources/gsmarena.py`
**Severity:** N/A (verification only)
**Confidence:** HIGH

**Evidence:** Examined all regex patterns used for numeric extraction:
- `SIZE_MM_RE = re.compile(r"([\d.]+)\s*[x×]\s*([\d.]+)\s*mm")` — `[\d.]+` cannot match "nan" or "inf"
- `PITCH_UM_RE = re.compile(r"([\d.]+)\s*(?:µm|um|...)")` — same
- `MPIX_RE = re.compile(r"([\d.]+)\s*(?:effective\s+)?(?:Mega ?pixels?|MP)")` — same
- `IR_SENSOR_SIZE_RE = re.compile(r"([\d.]+)\s*mm\s*[x×]\s*([\d.]+)\s*mm")` — same
- `IR_PITCH_RE = re.compile(r"([\d.]+)\s*microns?")` — same
- `IR_MPIX_RE = re.compile(r"(\d+\.?\d*)")` — same
- `RES_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")` — matches only 3-5 digit integers

All regex patterns use `[\d.]+` or `\d` character classes which cannot match "nan" or "inf" strings. The source parsers are safe from NaN/inf injection via regex-extracted values.

---

## Summary

- V37-01: Gate tests pass
- V37-02 (MEDIUM): `derive_spec` produces `area=nan` (not None) for partially-NaN size, causing round-trip asymmetry
- V37-03: Source parser regex patterns verified safe from NaN/inf injection
