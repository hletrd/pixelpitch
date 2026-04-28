# Aggregate Review (Cycle 31) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-30 Status

All previous fixes confirmed still working. No regressions. Gate tests pass (all checks). C30-01 and C30-02 both implemented and verified.

## Cross-Agent Agreement Matrix (Cycle 31 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| merge_camera_data spec/derived pitch inconsistency | code-reviewer, critic, verifier, tracer, architect, debugger, test-engineer | MEDIUM |
| BOM literal character fragility in two files | code-reviewer, critic, verifier, debugger, architect | LOW-MEDIUM |
| Spec/SpecDerived positional args in parsers | code-reviewer | LOW |
| derive_spec() missing docstring | document-specialist | LOW |
| BOM check duplication across modules | architect | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C31-01: merge_camera_data can leave spec.pitch and derived.pitch inconsistent

**Sources:** CR31-01, CRIT31-01, V31-02, TR31-01, ARCH31-01, DBG31-01, TE31-01
**Severity:** MEDIUM | **Confidence:** HIGH (7-agent consensus)

When `merge_camera_data` preserves `spec.pitch` from existing data (because new has None), it does NOT update `derived.pitch` if `derived.pitch` was already computed from area+mpix in the new data. The template and write_csv both read `derived.pitch`, so the computed approximation silently overwrites the authoritative measurement.

**Concrete scenario:**
1. openMVG provides camera "X" with `spec.pitch=None`, `spec.size=(5.0, 3.7)`, `spec.mpix=10.0`
2. `derive_spec()` computes `derived.pitch ~= 1.36`
3. Existing CSV has same camera with `spec.pitch=2.0`, `derived.pitch=2.0` (direct Geizhals measurement)
4. Merge preserves `spec.pitch=2.0` from existing (correct) but `derived.pitch` stays at 1.36 (wrong)
5. Template displays 1.36, write_csv persists 1.36 — the 2.0 measurement is permanently lost

**Fix:** After all Spec field preservation in merge_camera_data, if `spec.pitch` is not None, set `derived.pitch = spec.pitch` regardless of what derived.pitch was computed to be.

---

### C31-02: BOM check uses literal U+FEFF character instead of escape sequence

**Sources:** CR31-02, CRIT31-02, V31-03, DBG31-02, ARCH31-02
**Severity:** LOW-MEDIUM | **Confidence:** HIGH (5-agent consensus)

**Files:** `pixelpitch.py` line 276; `sources/openmvg.py` line 67

Both files compare `csv_content[0] == "﻿"` using the literal BOM character (U+FEFF). If the source file is re-encoded by a tool that normalizes Unicode, the literal disappears and the comparison silently breaks, causing BOM-prefixed CSVs to produce 0-row parses.

**Fix:** Replace literal BOM with the escape sequence `'﻿'` in both files. Use `csv_content.startswith('﻿')` for clarity.

---

### C31-03: Spec and SpecDerived constructed with positional args in parsers

**Sources:** CR31-03
**Severity:** LOW | **Confidence:** MEDIUM (1 agent)

**Files:** `pixelpitch.py` lines 346-347 and 625; also all source modules

If the dataclass field order changes, positional construction silently produces wrong objects. The C30-02 fix addressed `deduplicate_specs()` with `dataclasses.replace()`, but parser code paths still use positional args.

**Fix:** Use keyword arguments for Spec and SpecDerived construction in all parser code paths.

---

### C31-04: derive_spec() missing docstring — pitch priority logic undocumented

**Sources:** DOC31-01
**Severity:** LOW | **Confidence:** HIGH (1 agent)

**File:** `pixelpitch.py`, lines 680-704

The function has non-obvious priority logic: `derived.pitch` is set to `spec.pitch` when available, otherwise computed from area+mpix. This is critical for understanding the merge consistency issue but is undocumented.

**Fix:** Add a docstring explaining the pitch priority.

---

### C31-05: BOM check duplication across two modules

**Sources:** ARCH31-02
**Severity:** LOW | **Confidence:** HIGH (1 agent, subsumed by C31-02)

The BOM-stripping logic is duplicated in `pixelpitch.py` and `sources/openmvg.py`. The fix for C31-02 should also centralize the BOM check into a shared utility.

**Fix:** Extract `strip_bom(text: str) -> str` into `sources/__init__.py` and call it from both files.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 5 (1 MEDIUM, 1 LOW-MEDIUM, 3 LOW)
- Cross-agent consensus findings (3+ agents): 2 (C31-01 with 7 agents, C31-02 with 5 agents)
- 7-agent consensus: 1 (C31-01)
- 5-agent consensus: 1 (C31-02)
