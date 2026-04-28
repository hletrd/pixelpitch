# Plan: Cycle 25 Findings — Regex Consistency & ValueError Guard

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** CR25-01, CRIT25-01, V25-02, V25-03, TR25-02, ARCH25-01, DBG25-02, TE25-01, CR25-02, CRIT25-02, V25-04, TR25-01, DBG25-01, TE25-02, DOC25-01

---

## Task 1: Centralize SIZE_RE and PITCH_RE by importing shared patterns from sources/__init__.py — C25-01

**Finding:** C25-01 (8-agent consensus: code-reviewer, critic, verifier, tracer, architect, debugger, test-engineer, document-specialist)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `pixelpitch.py`, `sources/__init__.py`, `tests/test_parsers_offline.py`

### Problem

The Geizhals-specific regex patterns in `pixelpitch.py` are significantly less robust than the shared patterns in `sources/__init__.py`:

- `SIZE_RE = re.compile(r"([\d\.]+)x([\d\.]+)mm")` — only ASCII `x`, no spaces, no `×`
- `SIZE_MM_RE` (sources/__init__.py line 65) — matches `x`, `×`, spaces, case-insensitive
- `PITCH_RE = re.compile(r"([\d\.]+)µm")` — only micro sign `µ` (U+00B5)
- `PITCH_UM_RE` (sources/__init__.py line 66) — matches `µ`, `μ`, "microns", "um", HTML entities

`TYPE_FRACTIONAL_RE` was already centralized (imported from sources). `SIZE_RE` and `PITCH_RE` should follow the same pattern.

### Implementation

1. In `pixelpitch.py`, replace local `SIZE_RE` and `PITCH_RE` definitions (lines 42-43) with imports from `sources`:
   ```python
   from sources import TYPE_FRACTIONAL_RE, SIZE_MM_RE, PITCH_UM_RE
   ```
   Remove the local `SIZE_RE` and `PITCH_RE` definitions.

2. Update all references in `pixelpitch.py`:
   - `parse_sensor_field()` (line 554): `SIZE_RE.search()` → `SIZE_MM_RE.search()`
   - `parse_sensor_field()` (line 559): `PITCH_RE.search()` → `PITCH_UM_RE.search()`
   - `extract_specs()` (line 592): `MPIX_RE.search()` stays (no shared equivalent for "Megapixel")
   - Any other references

3. Note: `SIZE_MM_RE` captures groups are (width, height) same as `SIZE_RE`. `PITCH_UM_RE` captures the pitch value same as `PITCH_RE`. The group indices are compatible.

4. Update `parse_sensor_field()` docstring to mention expanded format support.

5. Add tests in `test_parsers_offline.py`:
   - `parse_sensor_field` with Unicode × separator
   - `parse_sensor_field` with spaces around x
   - `parse_sensor_field` with Greek mu μ
   - `parse_sensor_field` with "microns" suffix

### Verification — DONE
- Gate tests (`python3 -m tests.test_parsers_offline`) — all 230 checks passed
- New test cases pass (Unicode ×, spaces, Greek μ, "microns")
- Commit: 17a7b2d

---

## Task 2: Add ValueError guard in parse_sensor_field — C25-02

**Finding:** C25-02 (6-agent consensus: code-reviewer, critic, verifier, tracer, debugger, test-engineer)
**Severity:** MEDIUM | **Confidence:** MEDIUM
**Files:** `pixelpitch.py`, `tests/test_parsers_offline.py`

### Problem

`parse_sensor_field()` calls `float(size_match.group(1))` and `float(pitch_match.group(1))` without try/except. The regex `[\d.]+` allows multiple dots (e.g., `"36.0.1"`), and `float()` raises `ValueError` on such input. This exception propagates up and can cause the entire Geizhals category to be dropped.

### Implementation

1. Wrap the float() calls in `parse_sensor_field()` with try/except ValueError:

   ```python
   # Extract sensor dimensions
   size_match = SIZE_MM_RE.search(sensor_text)
   if size_match:
       try:
           result["size"] = (float(size_match.group(1)), float(size_match.group(2)))
       except ValueError:
           result["size"] = None

   # Extract pixel pitch
   pitch_match = PITCH_UM_RE.search(sensor_text)
   if pitch_match:
       try:
           result["pitch"] = float(pitch_match.group(1))
       except ValueError:
           result["pitch"] = None
   ```

2. Add test in `test_parsers_offline.py`:
   ```python
   # Malformed dimension — should not crash, return None for size
   result_bad = pp.parse_sensor_field('CMOS 36.0.1x24.0mm')
   expect("malformed size returns None for size", result_bad["size"], None)
   ```

### Verification — DONE
- Gate tests (`python3 -m tests.test_parsers_offline`) — all 230 checks passed
- New test cases pass (malformed size returns None, malformed pitch returns None)
- Commit: 17a7b2d

---

## Deferred Findings

### C25-03: parse_sensor_field docstring format limitations
- **File:** `pixelpitch.py`, lines 530-539
- **Original Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** Addressed as part of Task 1 (C25-01 fix). The docstring was updated to include Unicode × and Greek μ examples.
- **Status:** RESOLVED — docstring now shows `"CMOS 36.0×24.0mm, 5.12μm"` as an example.
