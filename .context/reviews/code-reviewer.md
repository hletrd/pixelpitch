# Code Review (Cycle 32) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-31 fixes, focusing on NEW issues

## Previous Findings Status

C31-01 through C31-04 all implemented and verified. All previous fixes stable.

## New Findings

### CR32-01: write_csv uses truthy checks instead of None checks for float fields

**File:** `pixelpitch.py`, lines 824-827
**Severity:** LOW-MEDIUM | **Confidence:** HIGH

Four fields in `write_csv` use Python truthiness (`if x`) instead of explicit None checks (`if x is not None`):

```python
area_str = f"{derived.area:.2f}" if derived.area else ""       # line 824
mpix_str = f"{spec.mpix:.1f}" if spec.mpix else ""             # line 825
pitch_str = f"{derived.pitch:.2f}" if derived.pitch else ""    # line 826
year_str = str(spec.year) if spec.year else ""                 # line 827
```

For float fields (area, mpix, pitch), the value `0.0` is falsy but is a valid float. If any field is ever `0.0`, it would be written as an empty string to CSV and read back as `None` by `parse_existing_csv`, causing silent data loss on CSV round-trip.

**Concrete scenario:**
1. Camera with `spec.mpix=0.0` (e.g., from a parser bug or edge case)
2. `write_csv` writes `""` for mpix (because `bool(0.0) is False`)
3. `parse_existing_csv` reads `""` and produces `None`
4. Data lost: the camera's mpix field changes from `0.0` to `None` on next build

**Likelihood:** Very low in practice — no real camera has 0.0 MP/area/pitch. But `pixel_pitch()` can return `0.0` when `mpix <= 0`, and this value would be silently dropped.

**Fix:** Replace truthy checks with explicit None checks:
```python
area_str = f"{derived.area:.2f}" if derived.area is not None else ""
mpix_str = f"{spec.mpix:.1f}" if spec.mpix is not None else ""
pitch_str = f"{derived.pitch:.2f}" if derived.pitch is not None else ""
year_str = str(spec.year) if spec.year is not None else ""
```

---

### CR32-02: IR_MPIX_RE can match partial decimal numbers from malformed input

**File:** `sources/imaging_resource.py`, line 47
**Severity:** LOW | **Confidence:** MEDIUM

The `IR_MPIX_RE` pattern is `r"(\d+\.?\d*)"` — it matches any integer or decimal number. Unlike the centralized `MPIX_RE` which requires a unit suffix (MP, Megapixel, etc.), `IR_MPIX_RE` has no suffix requirement.

When applied to text containing a leading-dot decimal (e.g., `".5"` from malformed HTML stripping), it matches `"5"` instead of rejecting the input or matching `"0.5"`. The centralized `MPIX_RE` does NOT have this issue — `MPIX_RE.search('.5')` returns `None`.

**Concrete scenario:** If the "Effective Megapixels" field on an IR spec page contains malformed text like "approx. 5" (with a trailing dot), the regex would match `"5"` from the decimal portion, not from the intended number. However, in practice, the IR spec pages consistently produce clean numeric values.

**Fix:** Add a suffix requirement or use a more restrictive pattern. At minimum, require that the matched number is preceded by a non-dot character or start of string.

---

## Summary

- CR32-01 (LOW-MEDIUM): write_csv falsy checks silently drop 0.0 float values
- CR32-02 (LOW): IR_MPIX_RE matches partial decimals without unit suffix
