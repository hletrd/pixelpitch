# Verifier Review (Cycle 28) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-27 fixes

## V28-01: Gate tests pass — all checks verified

**Evidence:** Ran `python3 -m tests.test_parsers_offline` — all checks passed. C27-01 and C27-02 fixes verified working.

## V28-02: imaging_resource.py pitch float() missing ValueError guard — verified

**File:** `sources/imaging_resource.py`, line 238
**Severity:** MEDIUM | **Confidence:** HIGH

**Evidence:**
```python
from sources.imaging_resource import IR_PITCH_RE
m = IR_PITCH_RE.search("5.1.2 microns")
# m.group(1) == "5.1.2"
float("5.1.2")  # raises ValueError
```

The `size` (line 229) and `mpix` (line 246) float() calls are wrapped in try/except ValueError, but `pitch` (line 238) is not. This is an inconsistency in the C26-02 fix.

## V28-03: CineD year regex produces unvalidated years — verified

**File:** `sources/cined.py`, line 114
**Severity:** LOW | **Confidence:** HIGH

**Evidence:**
```python
import re
m = re.search(r"Release Date.{0,40}?(\d{4})", "Release Date: SKU1234 model", re.IGNORECASE)
# m.group(1) == "1234" — matches any 4-digit number, not just 19xx/20xx
```

The C27-02 fix added year range validation to `parse_existing_csv()`, but `cined._parse_camera_page()` produces years via `int(year_m.group(1))` without range validation. A CineD page with text "Release Date: model1234" could produce year=1234.

The `parse_year()` fallback on line 114 validates 19xx/20xx, but the primary regex path does not.

## V28-04: Apotelyt PITCH_RE missing "um" — DRY inconsistency verified

**File:** `sources/apotelyt.py`, line 35
**Severity:** LOW | **Confidence:** HIGH

**Evidence:**
```python
from sources.apotelyt import PITCH_RE
m = PITCH_RE.search("5.12um")
# Returns None — Apotelyt PITCH_RE does not match "um"
```

After the C27-01 fix added "um" to the shared `PITCH_UM_RE`, the local `PITCH_RE` in `apotelyt.py` was not updated. This is a DRY inconsistency.

---

## Summary

- V28-01: All gate tests pass
- V28-02 (MEDIUM): imaging_resource.py pitch ValueError guard missing — verified
- V28-03 (LOW): CineD year regex unvalidated — verified
- V28-04 (LOW): Apotelyt PITCH_RE missing "um" — DRY inconsistency verified
