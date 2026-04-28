# Aggregate Review (Cycle 26) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-25 Status

All previous fixes confirmed still working. No regressions. Gate tests pass (all checks). C25-01 (SIZE_RE/PITCH_RE centralization) and C25-02 (ValueError guard) implemented and verified.

## Cross-Agent Agreement Matrix (Cycle 26 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| MPIX_RE not centralized — incomplete C25-01 DRY fix | code-reviewer, critic, verifier, tracer, architect, debugger, test-engineer, document-specialist | MEDIUM |
| ValueError guard missing in source modules | code-reviewer, critic, verifier, tracer, debugger, test-engineer | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C26-01: MPIX_RE not centralized — incomplete DRY resolution from C25-01

**Sources:** CR26-01, CRIT26-01, V26-02, TR26-01, ARCH26-01, DBG26-01, TE26-01, DOC26-01
**Severity:** MEDIUM | **Confidence:** HIGH (8-agent consensus)

The C25-01 aggregate review explicitly mentioned MPIX_RE: "Update the MPIX_RE in pixelpitch.py similarly (it only matches 'Megapixel', not 'MP' or 'Mega pixels')." However, the implementation plan incorrectly stated "MPIX_RE.search() stays (no shared equivalent for 'Megapixel')" — but `sources/__init__.py` line 67 DOES export a shared `MPIX_RE` that is a superset of the local pattern.

Current centralization status after C25-01:
- `TYPE_FRACTIONAL_RE` — centralized (imported from sources) ✓
- `SIZE_MM_RE` — centralized (imported from sources) ✓
- `PITCH_UM_RE` — centralized (imported from sources) ✓
- `MPIX_RE` — NOT centralized (local definition in pixelpitch.py line 42) ✗

The local `MPIX_RE` in pixelpitch.py only matches "Megapixel" (case-sensitive). The shared `MPIX_RE` in sources/__init__.py matches "MP", "Mega pixels", "Megapixel", "Megapixels" etc. (case-insensitive).

**Impact:** If Geizhals changes "Megapixel" to "MP" or "Mega pixels" in their HTML, megapixel values are silently lost. Camera shows "unknown" resolution and computed pixel pitch is lost.

**Fix:** Import `MPIX_RE` from `sources` in `pixelpitch.py`, replacing the local definition on line 42. Update `extract_specs()` line 596 to use the imported pattern. Add test cases for "MP" and "Mega pixels" formats. Update docstring if applicable.

---

### C26-02: ValueError guard missing in source module float() calls

**Sources:** CR26-02, CRIT26-02, V26-03, TR26-02, DBG26-02, TE26-02
**Severity:** LOW | **Confidence:** MEDIUM (6-agent consensus)

The C25-02 fix added ValueError guards only in `pixelpitch.py`'s `parse_sensor_field()`. Source modules that parse the same data types from HTML still call `float()` on regex matches without any guard:

1. `sources/cined.py` line 98: `size = (float(s.group(1)), float(s.group(2)))`
2. `sources/apotelyt.py` lines 119-120: `size = (float(m.group(1)), float(m.group(2)))`
3. `sources/apotelyt.py` line 123: `pitch = float(m.group(1))`
4. `sources/apotelyt.py` line 129: `mpix = float(m.group(1))`
5. `sources/gsmarena.py` line 130: `mpix = float(mp_match.group(1))`
6. `sources/gsmarena.py` line 133: `pitch = float(pitch_match.group(1))`
7. `sources/imaging_resource.py` line 228: `size = (float(m.group(1)), float(m.group(2)))`

**Impact:** A malformed value from any source (e.g., "36.0.1" in a size field) raises ValueError and the individual camera record is lost. The blast radius is smaller than C25-02 (only one camera, not an entire category) because source modules process one camera at a time in loops with outer exception handlers.

**Fix:** Wrap float() calls in try/except ValueError, setting the affected field to None. This is consistent with the `parse_sensor_field()` pattern and allows remaining fields to still be extracted.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 2 (1 MEDIUM, 1 LOW)
- Cross-agent consensus findings (3+ agents): 2 (C26-01: 8-agent, C26-02: 6-agent)
- 1 MEDIUM finding: MPIX_RE not centralized — incomplete C25-01 DRY resolution
- 1 LOW finding: ValueError guard missing in source modules
