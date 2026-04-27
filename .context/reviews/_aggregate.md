# Aggregate Review (Cycle 14) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-13 Status

All previous fixes confirmed still working. No regressions.
Deferred items: see `.context/plans/deferred.md`.

## Cross-Agent Agreement Matrix (Cycle 14 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| openMVG DSLR misclassification → duplicates | code-reviewer, critic, verifier, tracer, architect, debugger, designer, test-engineer | MEDIUM |
| UTF-8 BOM in CSV → 0-row parse failure | code-reviewer, critic, verifier, tracer, architect, debugger, security-reviewer, test-engineer | MEDIUM |
| CineD FORMAT_TO_MM unreachable entries | code-reviewer, critic, verifier, tracer, debugger, test-engineer | LOW |
| PHONE_TYPE_SIZE mutable alias of TYPE_SIZE | code-reviewer | LOW |
| openMVG docstring doesn't document heuristic limitation | document-specialist | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C14-01: openMVG classifies all interchangeable-lens cameras as "mirrorless" — causes visible duplicates
**Sources:** C14-01, CR14-01, V14-01, T14-01, A14-01, D14-01, UX14-01, TE14-02
**Severity:** MEDIUM | **Confidence:** HIGH (8-agent consensus)

openMVG's category heuristic (`size[0] >= 20 → mirrorless`) misclassifies all DSLRs as "mirrorless". When the same DSLR also appears in Geizhals data with `category="dslr"`, `create_camera_key` produces different keys and `merge_camera_data` preserves both entries. This results in visible duplicate entries on the "All Cameras" page.

**Concrete failure:** Canon EOS 5D appears twice: once under "mirrorless" (openMVG) and once under "dslr" (Geizhals). The same applies to potentially dozens of DSLRs in the openMVG dataset.

**Fix:** Add a DSLR name-pattern heuristic to openMVG's category assignment (e.g., if name matches Canon EOS \d+D, Nikon D\d+, Pentax K-\d+, etc.).

---

### C14-02: UTF-8 BOM in camera-data.csv causes complete parse failure
**Sources:** C14-02, CR14-02, V14-02, T14-02, A14-02, D14-02, S14-01, TE14-01
**Severity:** MEDIUM | **Confidence:** HIGH (8-agent consensus)

If `camera-data.csv` or any source CSV is saved with a UTF-8 BOM (e.g., from Excel's "CSV UTF-8" save option), `parse_existing_csv` fails completely: the BOM makes `header[0]` equal to `"﻿id"` instead of `"id"`, causing schema detection to fail, column misalignment, and 0 rows parsed. The entire render pipeline proceeds with no existing data.

**Concrete failure:** A developer opens `camera-data.csv` in Excel, saves as "CSV UTF-8", and the next build produces a site missing all preserved cameras.

**Fix:** Strip BOM at the entry point of `parse_existing_csv`:
```python
if csv_content and csv_content[0] == '﻿':
    csv_content = csv_content[1:]
```

---

### C14-03: CineD `FORMAT_TO_MM` has unreachable entries — dead code
**Sources:** C14-03, CR14-03, V14-03, T14-03, D14-03, TE14-03
**Severity:** LOW | **Confidence:** HIGH (6-agent consensus)

Three entries in `FORMAT_TO_MM` are unreachable because the regex in `_parse_camera_page` doesn't capture those string patterns: `"super35"` (regex requires space), `"1 inch"` (regex matches `1"` or `1-inch` only), `"2/3-inch"` (regex matches `2/3"` only).

**Fix:** Extend regex to capture missing variants, or remove unreachable entries.

---

### C14-04: `PHONE_TYPE_SIZE` is a mutable alias of `TYPE_SIZE`
**Sources:** C14-04
**Severity:** LOW | **Confidence:** HIGH

`gsmarena.PHONE_TYPE_SIZE` is a direct reference to `pixelpitch.TYPE_SIZE`. Any mutation to `PHONE_TYPE_SIZE` would corrupt the central table. The code comment warns against mutation but the type system doesn't enforce it.

**Fix:** Use `dict(SENSOR_TYPE_SIZE)` (copy) or `MappingProxyType` (read-only view).

---

### C14-05: openMVG docstring doesn't document category heuristic limitation
**Sources:** DS14-01
**Severity:** LOW | **Confidence:** HIGH

The openMVG module docstring describes coverage but doesn't mention that the category heuristic misclassifies DSLRs. Developers reading the docstring won't understand why duplicate entries may appear.

**Fix:** Add a note to the docstring about the category heuristic limitation.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics
- Total distinct new findings: 5 (2 MEDIUM, 3 LOW)
- Cross-agent consensus findings (3+ agents): 3
- All cycle 1-13 fixes remain intact
- Remaining deferred items: see `.context/plans/deferred.md`
