# Verifier Review (Cycle 25) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-24 fixes

## V25-01: Gate tests pass — all 222 checks verified

**Evidence:** Ran `python3 -m tests.test_parsers_offline` — all 222 checks passed.

## V25-02: SIZE_RE inconsistency verified — Unicode × not matched

**File:** `pixelpitch.py`, line 42
**Severity:** MEDIUM | **Confidence:** HIGH

**Evidence:**
```python
SIZE_RE = re.compile(r"([\d\.]+)x([\d\.]+)mm")
SIZE_RE.search('36.0×24.0mm')   # Returns None (Unicode ×)
SIZE_RE.search('36.0 x 24.0mm') # Returns None (spaces)
SIZE_RE.search('36.0x24.0mm')   # Returns ('36.0', '24.0') ✓
```

The shared `SIZE_MM_RE` handles both cases correctly. The Geizhals parser (`parse_sensor_field`) uses the limited `SIZE_RE`.

## V25-03: PITCH_RE inconsistency verified — Greek mu not matched

**File:** `pixelpitch.py`, line 43
**Severity:** MEDIUM | **Confidence:** HIGH

**Evidence:**
```python
PITCH_RE = re.compile(r"([\d\.]+)µm")
PITCH_RE.search('5.12μm')       # Returns None (Greek mu U+03BC)
PITCH_RE.search('5.12 microns') # Returns None
PITCH_RE.search('5.12µm')       # Returns ('5.12',) ✓ (micro sign U+00B5)
```

The shared `PITCH_UM_RE` handles both cases correctly. The Geizhals parser uses the limited `PITCH_RE`.

## V25-04: ValueError risk verified — malformed float in SIZE_RE match

**File:** `pixelpitch.py`, line 556
**Severity:** MEDIUM | **Confidence:** MEDIUM

**Evidence:**
```python
SIZE_RE.search('36.0.1x24.0mm')  # matches group(1)="36.0.1"
float("36.0.1")                  # ValueError: could not convert string to float
```

The regex `[\d\.]+` allows multiple dots. While this exact input is unlikely from real Geizhals HTML, the lack of a ValueError guard means any malformed match crashes the entire category fetch.

---

## Summary

- V25-01: All gate tests pass (222/222)
- V25-02 (MEDIUM): SIZE_RE misses Unicode × — verified
- V25-03 (MEDIUM): PITCH_RE misses Greek mu — verified
- V25-04 (MEDIUM): ValueError risk on malformed float — verified
