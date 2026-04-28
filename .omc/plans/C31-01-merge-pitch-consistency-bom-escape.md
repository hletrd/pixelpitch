# Plan: Cycle 31 Findings — Merge Pitch Consistency, BOM Escape, Docstring, Positional Args

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** CR31-01, CRIT31-01, V31-02, TR31-01, ARCH31-01, DBG31-01, TE31-01, CR31-02, CRIT31-02, V31-03, DBG31-02, ARCH31-02, CR31-03, DOC31-01

---

## Task 1: Fix merge_camera_data spec/derived pitch inconsistency — C31-01

**Finding:** C31-01 (7-agent consensus: code-reviewer, critic, verifier, tracer, architect, debugger, test-engineer)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `pixelpitch.py` lines 413-432, `tests/test_parsers_offline.py`

### Problem

When `merge_camera_data` preserves `spec.pitch` from existing data (because new has None), it does NOT update `derived.pitch` if `derived.pitch` was already computed from area+mpix. The template and write_csv both read `derived.pitch`, so the computed approximation silently overwrites the authoritative measurement.

### Implementation

1. In `pixelpitch.py`, after all Spec field preservation (line 432), add a consistency check: if `new_spec.spec.pitch is not None` and `new_spec.pitch != new_spec.spec.pitch`, set `new_spec.pitch = new_spec.spec.pitch`. This ensures the authoritative `spec.pitch` (whether original or preserved from existing) always takes precedence over the computed `derived.pitch`.

2. Add a test case to `test_merge_field_preservation` in `tests/test_parsers_offline.py`:
   - Create existing: spec with pitch=2.0 (direct measurement)
   - Create new: spec with pitch=None but size+mpix set (so derive_spec computes derived.pitch from area+mpix)
   - After merge, assert `derived.pitch == spec.pitch` (2.0, not the computed value)

### Verification — DONE

- Gate tests (`python -m tests.test_parsers_offline`) — all checks passed
- New test case passes
- Commit: 8579ce1

---

## Task 2: Replace BOM literal with escape sequence and centralize — C31-02, C31-05

**Finding:** C31-02 (5-agent consensus: code-reviewer, critic, verifier, debugger, architect) + C31-05
**Severity:** LOW-MEDIUM | **Confidence:** HIGH
**Files:** `pixelpitch.py` line 276, `sources/openmvg.py` line 67, `sources/__init__.py`

### Problem

Both files use `csv_content[0] == "﻿"` with the literal BOM character (U+FEFF). If the source file is re-encoded by a tool that normalizes Unicode, the literal disappears and the comparison silently breaks.

### Implementation

1. Add `strip_bom(text: str) -> str` to `sources/__init__.py`
2. Replace BOM handling in `pixelpitch.py` (parse_existing_csv) with call to `strip_bom()`
3. Replace BOM handling in `sources/openmvg.py` (fetch) with call to `strip_bom()`
4. Update imports as needed

### Verification — DONE

- Gate tests (`python -m tests.test_parsers_offline`) — all checks passed
- BOM-related tests still pass
- Commit: 8478efb

---

## Task 3: Add docstring to derive_spec() — C31-04

**Finding:** C31-04 (document-specialist)
**Severity:** LOW | **Confidence:** HIGH
**Files:** `pixelpitch.py` lines 680-704

### Problem

The function has non-obvious priority logic for pitch but no docstring.

### Implementation

1. Add a docstring to `derive_spec()` explaining pitch priority: direct `spec.pitch` takes precedence over computation from area+mpix.

### Verification — DONE

- Gate tests pass
- Commit: 58c9573

---

## Task 4: Convert positional Spec/SpecDerived args to keyword args — C31-03

**Finding:** C31-03 (code-reviewer)
**Severity:** LOW | **Confidence:** MEDIUM
**Files:** `pixelpitch.py`, all source modules

### Problem

Spec and SpecDerived are constructed with positional args in many places. If field order changes, these silently produce wrong objects.

### Implementation

1. Convert all `Spec(...)` and `SpecDerived(...)` calls to use keyword arguments
2. Cover: `pixelpitch.py` lines 346-347, 625; `sources/gsmarena.py` lines 159-167; `sources/imaging_resource.py` lines 265-273; `sources/apotelyt.py` lines 148-156; `sources/cined.py` lines 122-130; `sources/openmvg.py` lines 112-120

### Verification — DONE

- Gate tests pass
- Commit: 96b8215

---

## Deferred Findings

None. All findings are scheduled for implementation.
