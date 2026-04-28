# Plan: Cycle 32 Findings — write_csv Falsy Check Fix

**Created:** 2026-04-28
**Status:** PENDING
**Source Reviews:** CR32-01, CRIT32-01, V32-02, TR32-01, DBG32-01, TE32-01

---

## Task 1: Fix write_csv truthy checks to use explicit None checks — C32-01

**Finding:** C32-01 (6-agent consensus: code-reviewer, critic, verifier, tracer, debugger, test-engineer)
**Severity:** LOW-MEDIUM | **Confidence:** HIGH
**Files:** `pixelpitch.py` lines 824-827, `tests/test_parsers_offline.py`

### Problem

Four fields in `write_csv` use Python truthiness (`if x`) instead of explicit None checks (`if x is not None`):

```python
area_str = f"{derived.area:.2f}" if derived.area else ""       # line 824
mpix_str = f"{spec.mpix:.1f}" if spec.mpix else ""             # line 825
pitch_str = f"{derived.pitch:.2f}" if derived.pitch else ""    # line 826
year_str = str(spec.year) if spec.year else ""                 # line 827
```

For float fields (area, mpix, pitch), the value `0.0` is falsy but is a valid float distinct from `None`. If any field is ever `0.0`, it would be written as empty string and read back as `None` by `parse_existing_csv`, causing silent data loss on CSV round-trip.

### Implementation

1. In `pixelpitch.py`, `write_csv()` function, replace the four truthy checks with explicit None checks:
   ```python
   area_str = f"{derived.area:.2f}" if derived.area is not None else ""
   mpix_str = f"{spec.mpix:.1f}" if spec.mpix is not None else ""
   pitch_str = f"{derived.pitch:.2f}" if derived.pitch is not None else ""
   year_str = str(spec.year) if spec.year is not None else ""
   ```

2. Add a test case to `test_csv_round_trip` in `tests/test_parsers_offline.py`:
   - Create a Spec with `mpix=0.0` and `derived.pitch=0.0`
   - Write to CSV, read back
   - Assert mpix and pitch are preserved as `0.0` (not `None`)

### Verification

- Gate tests (`python -m tests.test_parsers_offline`) — all checks must pass
- New test case must pass
- Existing CSV round-trip test must still pass

---

## Deferred Findings

### C32-02: IR_MPIX_RE matches partial decimals without unit suffix

- **File:** `sources/imaging_resource.py`, line 47
- **Original Severity:** LOW | **Confidence:** MEDIUM (1 agent)
- **Reason for deferral:** In practice, IR spec pages produce clean numeric values for the "Effective Megapixels" field. The `.5` matching `5` scenario requires malformed HTML stripping that is extremely unlikely. The centralized `MPIX_RE` does not have this issue. Adding a suffix requirement would require changing the IR parser's approach since it operates on a pre-extracted field value (not raw text with units). The risk of introducing a regression by changing the regex outweighs the theoretical benefit.
- **Re-open if:** An IR spec page produces an incorrect megapixel value due to partial decimal matching, or during a parser consistency pass.
