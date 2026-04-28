# Plan: Cycle 33 Findings — Systemic Truthy-vs-None Fix

**Created:** 2026-04-28
**Status:** PENDING
**Source Reviews:** CR33-01, CR33-02, CR33-03, CRIT33-01, V33-02, V33-03, V33-04, TR33-01, ARCH33-01, DBG33-01, DBG33-02, DES33-01, DOC33-01, TE33-01, TE33-02, TE33-03

---

## Task 1: Fix derive_spec truthy check for spec.pitch — C33-01 (core)

**Finding:** C33-01 (11-agent consensus)
**Severity:** LOW-MEDIUM | **Confidence:** HIGH
**Files:** `pixelpitch.py` line 722

### Problem

The `derive_spec` function uses `if spec.pitch:` (truthy check) instead of `if spec.pitch is not None:` (explicit None check). If `spec.pitch=0.0`, the truthy check is False, and pitch is computed from area+mpix instead, violating the docstring's "spec.pitch always takes precedence" guarantee.

This is the most critical fix because it corrupts data BEFORE write_csv runs, making the C32-01 CSV fix partially moot.

### Implementation

1. In `pixelpitch.py`, `derive_spec()` function, line 722:
   - Change `if spec.pitch:` to `if spec.pitch is not None:`

---

## Task 2: Fix sorted_by truthy checks for sort keys — C33-01 (sorting)

**Finding:** CR33-02, V33-04, TE33-02
**Severity:** LOW | **Confidence:** HIGH
**Files:** `pixelpitch.py` lines 752-756

### Problem

`sorted_by` uses truthy checks for sort keys. 0.0 values sort as -1 instead of 0.0.

### Implementation

1. In `pixelpitch.py`, `sorted_by()` function, lines 752-756:
   - Change `c.pitch if c.pitch else -1` to `c.pitch if c.pitch is not None else -1`
   - Change `c.area if c.area else -1` to `c.area if c.area is not None else -1`
   - Change `c.spec.mpix if c.spec.mpix else -1` to `c.spec.mpix if c.spec.mpix is not None else -1`

---

## Task 3: Fix prettyprint truthy checks for display — C33-01 (console)

**Finding:** CR33-03
**Severity:** LOW | **Confidence:** HIGH
**Files:** `pixelpitch.py` lines 772-778

### Problem

`prettyprint` uses truthy checks for mpix and pitch display. 0.0 values display as "unknown".

### Implementation

1. In `pixelpitch.py`, `prettyprint()` function:
   - Change `if spec.mpix:` to `if spec.mpix is not None:`
   - Change `if derived.pitch:` to `if derived.pitch is not None:`

---

## Task 4: Fix template truthy checks for HTML rendering — C33-01 (UI)

**Finding:** DES33-01, DBG33-02, V33-03, TE33-03
**Severity:** LOW | **Confidence:** HIGH
**Files:** `templates/pixelpitch.html` lines 76-89

### Problem

Jinja2 template uses `{% if spec.pitch %}` and `{% if spec.spec.mpix %}` which evaluate 0.0 as falsy, showing "unknown" instead of the actual value.

### Implementation

1. In `templates/pixelpitch.html`:
   - Change `{% if spec.spec.mpix %}` to `{% if spec.spec.mpix is not none %}`
   - Change `{% if spec.pitch %}` to `{% if spec.pitch is not none %}`

---

## Task 5: Add test coverage for 0.0 value handling — C33-01 (tests)

**Finding:** TE33-01, TE33-02, TE33-03
**Severity:** LOW-MEDIUM | **Confidence:** HIGH
**Files:** `tests/test_parsers_offline.py`

### Implementation

1. Add derive_spec test (TE33-01):
   - Create Spec with pitch=0.0 and mpix=33.0, size=(35.9, 23.9)
   - Assert derive_spec preserves pitch=0.0 (not computing from area+mpix)

2. Add sorted_by test (TE33-02):
   - Create cameras with pitch=0.0 and verify they sort at 0.0, not -1

3. Add template rendering test (TE33-03):
   - Render pixelpitch.html template with pitch=0.0 and mpix=0.0
   - Assert HTML contains "0.0" and not "unknown" for those fields

---

## Verification

- Gate tests (`python3 -m tests.test_parsers_offline`) — all checks must pass
- New derive_spec test must pass (0.0 pitch preserved)
- Existing CSV round-trip test must still pass
- No regressions in any existing tests

---

## Deferred Findings

### C32-02: IR_MPIX_RE matches partial decimals without unit suffix

- **File:** `sources/imaging_resource.py`, line 47
- **Original Severity:** LOW | **Confidence:** MEDIUM (1 agent)
- **Reason for deferral:** In practice, IR spec pages produce clean numeric values for the "Effective Megapixels" field. The `.5` matching `5` scenario requires malformed HTML stripping that is extremely unlikely. The centralized `MPIX_RE` does not have this issue. Adding a suffix requirement would require changing the IR parser's approach since it operates on a pre-extracted field value (not raw text with units). The risk of introducing a regression by changing the regex outweighs the theoretical benefit.
- **Re-open if:** An IR spec page produces an incorrect megapixel value due to partial decimal matching, or during a parser consistency pass.
