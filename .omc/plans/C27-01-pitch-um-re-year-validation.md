# Plan: Cycle 27 Findings — PITCH_UM_RE "um" Fix & CSV Year Validation

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** CR27-01, CRIT27-01, V27-02, TR27-01, ARCH27-01, DBG27-01, DOC27-01, TE27-01, CR27-02, CRIT27-02, V27-03, TR27-02, DBG27-02, TE27-02

---

## Task 1: Add "um" to shared PITCH_UM_RE and fix doc/code mismatch — C27-01

**Finding:** C27-01 (8-agent consensus: code-reviewer, critic, verifier, tracer, architect, debugger, document-specialist, test-engineer)
**Severity:** LOW | **Confidence:** HIGH
**Files:** `sources/__init__.py`, `pixelpitch.py`, `tests/test_parsers_offline.py`

### Problem

The comment in `pixelpitch.py` line 44 claims PITCH_UM_RE matches "um":
```
# PITCH_UM_RE matches µm, μm (Greek mu), "microns", "um", and HTML entities.
```

But the actual regex in `sources/__init__.py` line 66 does NOT include `um`:
```python
PITCH_UM_RE = re.compile(r"([\d.]+)\s*(?:µm|microns?|μm|&micro;m|&#0?956;m)", re.IGNORECASE)
```

Meanwhile, GSMArena has its own `PITCH_RE` at line 50 that includes `um`. The shared pattern should be a true superset of all local patterns.

### Implementation

1. In `sources/__init__.py` line 66, add `um` to the PITCH_UM_RE alternation:
   ```python
   PITCH_UM_RE = re.compile(r"([\d.]+)\s*(?:µm|um|microns?|μm|&micro;m|&#0?956;m)", re.IGNORECASE)
   ```

2. The comment in `pixelpitch.py` line 44 already claims "um" support, so no change needed there (the code will now match the documentation).

3. Add test in `tests/test_parsers_offline.py`:
   ```python
   # PITCH_UM_RE handles lowercase ASCII "um"
   result = pp.parse_sensor_field('CMOS 5.12um')
   expect("PITCH handles um", result["pitch"], 5.12, tol=0.01)
   ```

4. Run gate tests to verify no regressions.

### Verification — DONE

- Gate tests (`python -m tests.test_parsers_offline`) — all checks passed
- New test for "um" matching passed: `PITCH handles um: got=5.12 want=5.12 (±0.01)`
- Existing PITCH_UM_RE tests (µm, μm, microns) still pass
- Commit: 1b27a4e

---

## Task 2: Add year range validation in parse_existing_csv — C27-02

**Finding:** C27-02 (6-agent consensus: code-reviewer, critic, verifier, tracer, debugger, test-engineer)
**Severity:** LOW | **Confidence:** MEDIUM
**Files:** `pixelpitch.py`, `tests/test_parsers_offline.py`

### Problem

The CSV parser accepts any integer for the year column without range validation. `year = int(year_str) if year_str else None` accepts year=0, year=-1, year=99999 etc. These would display verbatim on the website. No current source produces invalid years, but this is a defensive hardening gap.

### Implementation

1. In `pixelpitch.py` line 336, replace:
   ```python
   year = int(year_str) if year_str else None
   ```
   with:
   ```python
   year = None
   if year_str:
       try:
           y = int(year_str)
           if 1900 <= y <= 2100:
               year = y
       except ValueError:
           pass
   ```

2. Add tests in `tests/test_parsers_offline.py`:
   ```python
   # Year validation: year=0 should be None
   csv_year0 = 'id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,megapixels,pixel_pitch_um,year,matched_sensors\n0,Test,mirrorless,,36.00,24.00,864.00,45.0,5.00,0,\n'
   parsed_y0 = pp.parse_existing_csv(csv_year0)
   expect("year=0 rejected", parsed_y0[0].spec.year, None)

   # Year validation: negative year should be None
   csv_neg = 'id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,megapixels,pixel_pitch_um,year,matched_sensors\n0,Test,mirrorless,,36.00,24.00,864.00,45.0,5.00,-1,\n'
   parsed_neg = pp.parse_existing_csv(csv_neg)
   expect("year=-1 rejected", parsed_neg[0].spec.year, None)

   # Year validation: valid year still works
   csv_valid = 'id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,megapixels,pixel_pitch_um,year,matched_sensors\n0,Test,mirrorless,,36.00,24.00,864.00,45.0,5.00,2021,\n'
   parsed_valid = pp.parse_existing_csv(csv_valid)
   expect("year=2021 accepted", parsed_valid[0].spec.year, 2021)
   ```

3. Run gate tests to verify no regressions.

### Verification — DONE

- Gate tests (`python -m tests.test_parsers_offline`) — all checks passed
- New year validation tests passed:
  - `year=0 rejected: got=None want=None`
  - `year=-1 rejected: got=None want=None`
  - `year=99999 rejected: got=None want=None`
  - `year=2021 accepted: got=2021 want=2021`
- Existing CSV parsing tests still pass
- Commit: 83910a5

---

## Deferred Findings

None. All findings are scheduled for implementation.
