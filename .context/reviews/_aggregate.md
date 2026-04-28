# Aggregate Review (Cycle 43) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-42 Status

All previous fixes confirmed still working. No regressions in core logic. Gate tests pass. C42 fixes (merge size consistency, CLI --limit guard, docstring update) all verified.

## Cross-Agent Agreement Matrix (Cycle 43 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| GSMArena/CineD set spec.size from lookup tables — merge can't preserve measured Geizhals values | CR43-02, CR43-02b, SR43-01, CRIT43-01, V43-02, TR43-01, ARCH43-01, DBG43-01, DBG43-02, DES43-01, DOC43-01, DOC43-02, TE43-01, TE43-02 | MEDIUM |
| C42-01 fix writes derived.pitch redundantly — consistency check already handles it | CR43-01, CRIT43-02, V43-03, DBG43-03 | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C43-01: GSMArena/CineD set spec.size from lookup tables — merge cannot preserve measured Geizhals values (silent data loss)

**Sources:** CR43-02, CR43-02b, SR43-01, CRIT43-01, V43-02, TR43-01, ARCH43-01, DBG43-01, DBG43-02, DES43-01, DOC43-01, DOC43-02, TE43-01, TE43-02
**Severity:** MEDIUM | **Confidence:** HIGH (14-agent consensus)

When GSMArena provides a sensor type like "1/1.3", `_phone_to_spec` sets `spec.size` from the `PHONE_TYPE_SIZE` lookup table (e.g., `(9.84, 7.40)`). Similarly, CineD sets `spec.size` from `FORMAT_TO_MM` when only a format class is known. These lookup-table values are stored as `spec.size`, which `merge_camera_data` treats as authoritative measured data.

The merge condition `if new_spec.spec.size is None and existing_spec.spec.size is not None` only preserves existing measured values when `spec.size is None`. Because GSMArena/CineD set `spec.size` from lookup tables, the condition is False and the Geizhals measured value is never preserved.

**Concrete scenario (verified by code trace):**
```python
# Geizhals: measured size from product page
existing = pp.derive_spec(Spec(name="Samsung S25 Ultra", category="smartphone",
                                type="1/1.3", size=(9.76, 7.30), pitch=None, mpix=200.0, year=2025))

# GSMArena: size from TYPE_SIZE lookup (NOT None)
new = pp.derive_spec(Spec(name="Samsung S25 Ultra", category="smartphone",
                           type="1/1.3", size=(9.84, 7.40), pitch=None, mpix=200.0, year=2025))

merged = pp.merge_camera_data([new], [existing])
# merged.spec.size = (9.84, 7.40)  ← WRONG (TYPE_SIZE lookup, not measured)
# Measured Geizhals value (9.76, 7.30) is LOST
```

**Impact:** The template displays the TYPE_SIZE approximation instead of the measured Geizhals value. The CSV stores wrong values. On the next merge cycle, the correct value is permanently lost.

**Fix:** Two parts:
1. **GSMArena:** Change `_phone_to_spec` to NOT set `spec.size` from `PHONE_TYPE_SIZE`. Instead, set only `spec.type` and leave `spec.size = None`. Let `derive_spec` compute `derived.size` from `spec.type` using the TYPE_SIZE lookup.
2. **CineD:** Change `_parse_camera_page` to NOT set `spec.size` from `FORMAT_TO_MM`. Instead, set only `spec.type` (mapping format names to fractional-inch types where possible) and leave `spec.size = None`. For formats that don't map to fractional-inch types (e.g., "Super 35", "APS-C"), leave both `spec.size = None` and `spec.type = None` — the data will show "unknown" for size, which is more honest than showing an approximation as if it were measured.

Note: For CineD, not all FORMAT_TO_MM entries have corresponding TYPE_SIZE entries (e.g., "Super 35" is not a fractional-inch type). For these, we need a different approach: either add them to TYPE_SIZE (which represents a different measurement convention), or accept that CineD format-class sizes will show as "unknown" in the template when no Geizhals data exists. The latter is more honest.

---

### C43-02: C42-01 fix writes derived.pitch redundantly — the pitch consistency check already handles it

**Sources:** CR43-01, CRIT43-02, V43-03, DBG43-03
**Severity:** LOW | **Confidence:** HIGH (4-agent agreement)

The C42-01 fix at `pixelpitch.py` line 467 includes `new_spec.pitch = existing_spec.pitch`. But lines 498-501 already handle derived.pitch consistency by checking if `spec.pitch != derived.pitch` and overriding. Since spec.pitch is preserved from existing at lines 468-473, the consistency check at 498-501 will overwrite derived.pitch if there's a mismatch. The write at line 467 is redundant and creates a subtle ordering dependency.

Not a correctness bug — the final value is correct. But the redundant write is misleading and makes the code harder to reason about.

**Fix:** Remove `new_spec.pitch = existing_spec.pitch` from line 467 in the C42-01 fix block. The existing pitch consistency logic at lines 490-501 already ensures derived.pitch tracks spec.pitch.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 2 (C43-01, C43-02)
- Cross-agent consensus findings (3+ agents): 2 (C43-01 with 14 agents, C43-02 with 4 agents)
- Highest severity: MEDIUM (C43-01)
- Actionable findings: 2
- Verified safe / deferred: 0
