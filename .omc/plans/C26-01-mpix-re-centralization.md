# Plan: Cycle 26 Findings — MPIX_RE Centralization & Source ValueError Guards

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** CR26-01, CRIT26-01, V26-02, TR26-01, ARCH26-01, DBG26-01, TE26-01, DOC26-01, CR26-02, CRIT26-02, V26-03, TR26-02, DBG26-02, TE26-02

---

## Task 1: Centralize MPIX_RE by importing shared pattern from sources/__init__.py — C26-01

**Finding:** C26-01 (8-agent consensus: code-reviewer, critic, verifier, tracer, architect, debugger, test-engineer, document-specialist)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `pixelpitch.py`, `sources/__init__.py`, `tests/test_parsers_offline.py`

### Problem

The C25-01 fix centralized SIZE_MM_RE and PITCH_UM_RE (imported from sources/__init__.py), but MPIX_RE was not similarly centralized, despite the C25-01 aggregate review explicitly flagging it. The implementation plan incorrectly stated "no shared equivalent" but `sources/__init__.py` line 67 exports `MPIX_RE` that is a superset of the local pattern.

Current centralization status:
- `TYPE_FRACTIONAL_RE` — centralized ✓
- `SIZE_MM_RE` — centralized ✓
- `PITCH_UM_RE` — centralized ✓
- `MPIX_RE` — NOT centralized ✗

### Implementation

1. In `pixelpitch.py`, remove the local `MPIX_RE` definition (line 42):
   ```python
   MPIX_RE = re.compile(r"([\d\.]+)\s*Megapixel")
   ```

2. Update the import line (line 48) to include `MPIX_RE`:
   ```python
   from sources import TYPE_FRACTIONAL_RE, SIZE_MM_RE, PITCH_UM_RE, MPIX_RE
   ```

3. Verify that `extract_specs()` line 596 (`MPIX_RE.search(mpix_text)`) still works — the shared pattern has the same group(1) capture group format. The shared pattern `([\d.]+)\s*(?:effective\s+)?(?:Mega ?pixels?|MP)` captures the numeric value in group(1), same as the local pattern. The only difference is the regex now also matches "MP", "Mega pixels" etc.

4. Add tests in `test_parsers_offline.py`:
   - `MPIX_RE` handles "MP" abbreviation
   - `MPIX_RE` handles "Mega pixels" format
   - `MPIX_RE` still handles "Megapixel" format (existing behavior)
   - Note: `parse_sensor_field()` doesn't extract mpix. Tests should target the `MPIX_RE` pattern directly or test through `extract_specs()`.

5. Update docstring if applicable.

### Verification — DONE
- Gate tests (`python -m tests.test_parsers_offline`) — all checks passed
- New test cases pass (MP, Mega pixels, Megapixel, Megapixels, effective prefix, lowercase mp)
- Commit: cf27a79

---

## Task 2: Add ValueError guards in source module float() calls — C26-02

**Finding:** C26-02 (6-agent consensus: code-reviewer, critic, verifier, tracer, debugger, test-engineer)
**Severity:** LOW | **Confidence:** MEDIUM
**Files:** `sources/cined.py`, `sources/apotelyt.py`, `sources/gsmarena.py`, `sources/imaging_resource.py`, `tests/test_parsers_offline.py`

### Problem

The C25-02 fix added ValueError guards in `pixelpitch.py`'s `parse_sensor_field()`. Source modules that parse the same data types still call `float()` on regex matches without any guard.

Affected locations:
1. `sources/cined.py` line 98: `size = (float(s.group(1)), float(s.group(2)))`
2. `sources/apotelyt.py` lines 119-120: `size = (float(m.group(1)), float(m.group(2)))`
3. `sources/apotelyt.py` line 123: `pitch = float(m.group(1))`
4. `sources/apotelyt.py` line 129: `mpix = float(m.group(1))`
5. `sources/gsmarena.py` line 130: `mpix = float(mp_match.group(1))`
6. `sources/gsmarena.py` line 133: `pitch = float(pitch_match.group(1))`
7. `sources/imaging_resource.py` line 228: `size = (float(m.group(1)), float(m.group(2)))`

### Implementation

For each affected source module, wrap float() calls in try/except ValueError, setting the affected field to None:

**sources/apotelyt.py:**
```python
# Before:
size = (float(m.group(1)), float(m.group(2)))
# After:
try:
    size = (float(m.group(1)), float(m.group(2)))
except ValueError:
    size = None

# Before:
pitch = float(m.group(1))
# After:
try:
    pitch = float(m.group(1))
except ValueError:
    pitch = None

# Before:
mpix = float(m.group(1))
# After:
try:
    mpix = float(m.group(1))
except ValueError:
    mpix = None
```

**sources/cined.py:**
```python
# Before:
size = (float(s.group(1)), float(s.group(2)))
# After:
try:
    size = (float(s.group(1)), float(s.group(2)))
except ValueError:
    size = None
```

**sources/gsmarena.py:**
```python
# Before:
mpix = float(mp_match.group(1)) if mp_match else None
# After:
mpix = None
if mp_match:
    try:
        mpix = float(mp_match.group(1))
    except ValueError:
        mpix = None

# Before:
pitch = float(pitch_match.group(1)) if pitch_match else None
# After:
pitch = None
if pitch_match:
    try:
        pitch = float(pitch_match.group(1))
    except ValueError:
        pitch = None
```

**sources/imaging_resource.py:**
```python
# Before:
size = (float(m.group(1)), float(m.group(2)))
# After:
try:
    size = (float(m.group(1)), float(m.group(2)))
except ValueError:
    size = None
```

### Verification — DONE
- Gate tests (`python -m tests.test_parsers_offline`) — all checks passed
- Existing test fixtures still parse correctly
- Commit: 3f20e6f

---

## Deferred Findings

None. All findings are scheduled for implementation.
