# Verifier Review (Cycle 26) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-25 fixes

## V26-01: Gate tests pass — all checks verified

**Evidence:** Ran `python3 -m tests.test_parsers_offline` — all checks passed.

## V26-02: MPIX_RE not centralized — verified discrepancy

**File:** `pixelpitch.py`, line 42 vs `sources/__init__.py`, line 67
**Severity:** MEDIUM | **Confidence:** HIGH

**Evidence:**
```python
# pixelpitch.py line 42:
MPIX_RE = re.compile(r"([\d\.]+)\s*Megapixel")

# sources/__init__.py line 67:
MPIX_RE = re.compile(r"([\d.]+)\s*(?:effective\s+)?(?:Mega ?pixels?|MP)", re.IGNORECASE)
```

The local pattern in pixelpitch.py only matches "Megapixel" (case-sensitive). The shared pattern handles "MP", "Mega pixels", "Megapixels" etc.

```python
import re
MPIX_LOCAL = re.compile(r"([\d\.]+)\s*Megapixel")
MPIX_SHARED = re.compile(r"([\d.]+)\s*(?:effective\s+)?(?:Mega ?pixels?|MP)", re.IGNORECASE)

MPIX_LOCAL.search("33.0 MP")         # Returns None
MPIX_SHARED.search("33.0 MP")        # Returns ("33.0",)
MPIX_LOCAL.search("33.0 Mega pixels")  # Returns None
MPIX_SHARED.search("33.0 Mega pixels")  # Returns ("33.0",)
MPIX_LOCAL.search("33.0 Megapixel")  # Returns ("33.0",) ✓
MPIX_SHARED.search("33.0 Megapixel") # Returns ("33.0",) ✓
```

The shared pattern is a strict superset of the local pattern. It is exported in `__all__` (line 89) and can be imported.

## V26-03: ValueError guard missing in source modules — verified

**File:** `sources/cined.py` line 98, `sources/apotelyt.py` lines 119-129, `sources/gsmarena.py` lines 130/133, `sources/imaging_resource.py` line 228
**Severity:** LOW | **Confidence:** MEDIUM

**Evidence:** Confirmed by code inspection that all four source modules call `float()` on regex match groups without try/except ValueError guards. The regex pattern `[\d.]+` allows multi-dot values like "36.0.1" which `float()` rejects.

```python
# sources/cined.py line 98:
size = (float(s.group(1)), float(s.group(2)))  # no try/except

# sources/apotelyt.py line 119-120:
size = (float(m.group(1)), float(m.group(2)))  # no try/except

# sources/gsmarena.py line 130:
mpix = float(mp_match.group(1)) if mp_match else None  # no try/except
```

However, these modules are called from `fetch_one()` / `_phone_to_spec()` which are called in loops within `fetch()`. If a ValueError propagates from one camera, it would be caught by the module-level exception handler (e.g., `except Exception as ex` in cined.py line 149), losing only that camera.

---

## Summary

- V26-01: All gate tests pass
- V26-02 (MEDIUM): MPIX_RE not centralized — verified
- V26-03 (LOW): ValueError guard missing in source modules — verified
