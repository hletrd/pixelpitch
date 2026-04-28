# Verifier Review (Cycle 27) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-26 fixes

## V27-01: Gate tests pass — all checks verified

**Evidence:** Ran `python3 -m tests.test_parsers_offline` — all checks passed. C26-01 and C26-02 fixes verified working.

## V27-02: PITCH_UM_RE missing "um" — verified discrepancy

**File:** `sources/__init__.py`, line 66 vs `sources/gsmarena.py`, line 50
**Severity:** LOW | **Confidence:** HIGH

**Evidence:**
```python
from sources import PITCH_UM_RE as shared
import re

gsmarena_pattern = re.compile(r'([\d.]+)\s*(?:µm|μm|um)', re.IGNORECASE)

# Shared pattern does NOT match "um":
shared.search('1.09um')   # Returns None

# GSMArena local pattern DOES match "um":
gsmarena_pattern.search('1.09um')  # Returns match

# Both match "µm" and "μm":
shared.search('5.12µm')          # Returns match
gsmarena_pattern.search('5.12µm')  # Returns match
```

The shared PITCH_UM_RE is missing "um" which the GSMArena local pattern handles. No current data path uses the shared pattern against "um" text (Geizhals uses µm/μm), so no data is lost at runtime.

## V27-03: parse_existing_csv accepts year=0 — verified

**File:** `pixelpitch.py`, line 336
**Severity:** LOW | **Confidence:** HIGH

**Evidence:**
```python
import pixelpitch as pp

# Year=0 is accepted and displayed
csv_test = 'id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,megapixels,pixel_pitch_um,year,matched_sensors\n0,Test,mirrorless,,36.00,24.00,864.00,45.0,5.00,0,\n'
parsed = pp.parse_existing_csv(csv_test)
# parsed[0].spec.year == 0 (displays as "0" on website)

# Year=-1 is also accepted
csv_test2 = 'id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,megapixels,pixel_pitch_um,year,matched_sensors\n0,Test,mirrorless,,36.00,24.00,864.00,45.0,5.00,-1,\n'
parsed2 = pp.parse_existing_csv(csv_test2)
# parsed2[0].spec.year == -1 (displays as "-1" on website)
```

No current source produces year=0 or negative years (`parse_year()` only matches 19xx/20xx), but the CSV parser has no guard.

---

## Summary

- V27-01: All gate tests pass
- V27-02 (LOW): PITCH_UM_RE missing "um" — verified
- V27-03 (LOW): parse_existing_csv accepts year=0 — verified
