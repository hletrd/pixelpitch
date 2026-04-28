# Plan: Cycle 34 Findings — match_sensors ZeroDivisionError + Residual Truthy Fixes

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** CR34-01, CR34-02, CR34-03, DBG34-01, DBG34-02, V34-02, V34-03, TR34-01, CRIT34-01, CRIT34-02, ARCH34-01, TE34-01, TE34-02

---

## Task 1: Fix match_sensors ZeroDivisionError with megapixels=0.0 — C34-01 (critical) — DONE

Commit: 843ad70

**Finding:** C34-01 (7-agent consensus)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `pixelpitch.py` line 238

### Problem

The `match_sensors` function divides by `megapixels` to compute a percentage difference. If `megapixels=0.0`, the guard `megapixels is not None` is True (0.0 is not None), and the division raises `ZeroDivisionError`. This would crash the entire merge/render pipeline.

### Implementation

1. In `pixelpitch.py`, `match_sensors()` function, line 236:
   - Change `if megapixels is not None and sensor_megapixels:` to `if megapixels is not None and megapixels > 0 and sensor_megapixels:`

---

## Task 2: Fix `list` command truthy check for spec.pitch — C34-02 — DONE

Commit: 698d96c

**Finding:** CR34-01, DBG34-02, V34-03, CRIT34-01
**Severity:** LOW | **Confidence:** HIGH
**Files:** `pixelpitch.py` line 1170

### Problem

The `list` command uses `if spec.pitch:` which treats pitch=0.0 as falsy, silently omitting cameras with 0.0 pitch. Same class of truthy-vs-None bug fixed across 4 other locations in C33-01.

### Implementation

1. In `pixelpitch.py`, `main()` function, line 1170:
   - Change `if spec.pitch:` to `if spec.pitch is not None:`

---

## Task 3: Fix match_sensors truthy checks for width/height — C34-03 — DONE

Commit: de6f4f4

**Finding:** CR34-02, CRIT34-01, TE34-02
**Severity:** LOW | **Confidence:** HIGH
**Files:** `pixelpitch.py` lines 217, 227

### Problem

The guard clause in `match_sensors` uses truthy checks for width/height. If width=0.0 or height=0.0, `not 0.0` is True, and the function returns [] or skips the sensor. This conflates 0.0 with None.

### Implementation

1. In `pixelpitch.py`, `match_sensors()` function, line 217:
   - Change `if not sensors_db or not width or not height:` to `if not sensors_db or width is None or height is None:`

2. In `pixelpitch.py`, `match_sensors()` function, line 227:
   - Change `if not sensor_width or not sensor_height:` to `if sensor_width is None or sensor_height is None:`

---

## Task 4: Add test coverage for match_sensors edge cases — C34-01/C34-03 (tests) — DONE

Commit: 5a01f56 (also includes additional zero/negative dimension guards)

**Finding:** TE34-01, TE34-02
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `tests/test_parsers_offline.py`

### Implementation

1. Add match_sensors test with megapixels=0.0 (TE34-01):
   - Call match_sensors with megapixels=0.0
   - Assert no ZeroDivisionError
   - Assert returns a list (size-only match)

2. Add match_sensors test with width=0.0 or height=0.0 (TE34-02):
   - Call match_sensors with width=0.0 or height=0.0
   - Assert returns [] (None-like behavior for 0.0 dimensions)

---

## Verification

- Gate tests (`python3 -m tests.test_parsers_offline`) — all checks must pass
- New match_sensors 0.0 tests must pass
- Existing tests must not regress

---

## Deferred Findings

### C32-02: IR_MPIX_RE matches partial decimals without unit suffix

- **File:** `sources/imaging_resource.py`, line 47
- **Original Severity:** LOW | **Confidence:** MEDIUM (1 agent)
- **Reason for deferral:** In practice, IR spec pages produce clean numeric values for the "Effective Megapixels" field. The `.5` matching `5` scenario requires malformed HTML stripping that is extremely unlikely. The centralized `MPIX_RE` does not have this issue. Adding a suffix requirement would require changing the IR parser's approach since it operates on a pre-extracted field value (not raw text with units). The risk of introducing a regression by changing the regex outweighs the theoretical benefit.
- **Re-open if:** An IR spec page produces an incorrect megapixel value due to partial decimal matching, or during a parser consistency pass.
