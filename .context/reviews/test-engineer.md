# Test Engineer Review (Cycle 26) — Test Coverage, Flaky Tests, TDD

**Reviewer:** test-engineer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-25 fixes

## Previous Findings Status

TE25-01 (SIZE_RE/PITCH_RE test gaps) and TE25-02 (ValueError guard test) both addressed in C25-01/C25-02.

## New Findings

### TE26-01: No test for MPIX_RE "MP" or "Mega pixels" format handling in parse_sensor_field / extract_specs

**File:** `tests/test_parsers_offline.py`
**Severity:** MEDIUM | **Confidence:** HIGH

The C25-01 fix centralized SIZE_MM_RE and PITCH_UM_RE and added tests for their expanded format support (Unicode x, Greek mu, "microns"). However, MPIX_RE was not centralized and no tests verify that it handles "MP" or "Mega pixels" formats.

If MPIX_RE is centralized (as suggested by CR26-01), tests should verify:
```python
# MPIX_RE handles "MP" abbreviation
result = pp.parse_sensor_field('CMOS 36.0x24.0mm, 33.0 MP')
expect("MPIX_RE handles MP", result.get("mpix"), None)  # Currently fails — mpix not returned

# MPIX_RE handles "Mega pixels" format
result2 = pp.parse_sensor_field('CMOS 36.0x24.0mm, 33.0 Mega pixels')
expect("MPIX_RE handles Mega pixels", result2.get("mpix"), None)  # Currently fails
```

Note: `parse_sensor_field()` does not extract mpix — that's done in `extract_specs()` using a separate `MPIX_RE.search()`. Tests should be added for `extract_specs()` or the raw `MPIX_RE` pattern.

### TE26-02: No test for ValueError guard in source modules

**File:** `tests/test_parsers_offline.py`
**Severity:** LOW | **Confidence:** MEDIUM

No tests exist for the case where source module float() calls encounter malformed input. If ValueError guards are added to source modules, tests should verify:
- Apotelyt: malformed sensor size → size=None, other fields preserved
- GSMArena: malformed pitch → pitch=None, other fields preserved
- Cined: malformed sensor size → size=None, other fields preserved

---

## Summary

- TE26-01 (MEDIUM): No test for MPIX_RE "MP" or "Mega pixels" format handling
- TE26-02 (LOW): No test for ValueError guard in source modules
