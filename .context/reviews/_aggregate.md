# Aggregate Review (Cycle 15) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-14 Status

All previous fixes confirmed still working. No regressions in previously-fixed items.
Deferred items: see `.context/plans/deferred.md`.

**IMPORTANT:** The C14-01 DSLR regex fix introduced new bugs that are now being caught in this cycle.

## Cross-Agent Agreement Matrix (Cycle 15 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| Canon EOS xxxD regex false negative → misclassified as mirrorless | code-reviewer, critic, verifier, tracer, architect, debugger, designer, test-engineer | MEDIUM |
| Samsung NX regex false positive → mirrorless classified as DSLR | code-reviewer, critic, verifier, tracer, architect, debugger, designer, test-engineer | MEDIUM |
| Geizhals rangefinder misclassification → 43 triple-category duplicates | code-reviewer, critic, verifier, tracer, architect, debugger, designer, test-engineer | MEDIUM |
| openMVG CSV DictReader has no BOM defense | code-reviewer, security-reviewer, tracer, debugger, test-engineer | LOW |
| Sigma SD regex misses 2-digit models | code-reviewer, critic, verifier, tracer, debugger | LOW |
| openMVG docstring doesn't warn about regex bugs | document-specialist | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C15-01: Canon EOS xxxD DSLRs misclassified as mirrorless by incomplete regex — regression from C14-01
**Sources:** C15-01, CR15-01, V15-01, T15-01, A15-01, D15-01, UX15-02, TE15-02
**Severity:** MEDIUM | **Confidence:** HIGH (8-agent consensus)

The C14-01 DSLR regex `Canon\s+EOS[-\s]+\dD` only matches single-digit xD models (5D, 6D, 7D, 1D). It misses the entire Canon Rebel/xxxD series (250D, 800D, 850D), the xxD series (70D, 80D, 90D), and the xxxxD series (1200D, 2000D, 4000D) — all of which are DSLRs.

Verified against dist data: 5 Canon EOS xxxD cameras are misclassified as "mirrorless" in the current data.

**Concrete failure:** Canon EOS 250D appears on the All Cameras page under "Mirrorless" (from openMVG) alongside its correct "DSLR" classification (from Geizhals).

**Fix:** Change `Canon\s+EOS[-\s]+\dD` to `Canon\s+EOS[-\s]+\d+D` (allow multiple digits before the final D).

---

### C15-02: Samsung NX pattern incorrectly classifies mirrorless cameras as DSLR — regression from C14-01
**Sources:** C15-02, CR15-01, V15-02, T15-02, D15-02, UX15-02, TE15-01
**Severity:** MEDIUM | **Confidence:** HIGH (7-agent consensus)

The C14-01 DSLR regex includes `Samsung\s+NX\d{3}` which matches Samsung NX100, NX200, NX300, NX500. But **all Samsung NX cameras are mirrorless**. Samsung never made a DSLR under the NX brand. The comment "some were DSLR-style" confuses body styling with camera type.

Verified: at least 1 Samsung NX camera is misclassified as "dslr" in the current dist data.

**Concrete failure:** Samsung NX300 appears on the DSLR page instead of the Mirrorless page.

**Fix:** Remove `Samsung\s+NX\d{3}` from `_DSLR_NAME_RE`.

---

### C15-03: Geizhals rangefinder (Messsucher) category misclassifies 43 cameras → triple-category duplicates
**Sources:** C15-04, CR15-02, V15-03, T15-03, A15-02, D15-03, UX15-01, TE15-03
**Severity:** MEDIUM | **Confidence:** HIGH (8-agent consensus)

Geizhals's "Messsucher" filter returns 53 cameras, but only ~10 are actual rangefinders (Leica M series). The other 43 are interchangeable-lens cameras that also appear under "DSLR" or "Mirrorless" categories. Since `create_camera_key` uses `name + category`, each camera generates up to 3 separate entries.

Current dist data has exactly 43 cameras with triple-category duplicates (dslr + mirrorless + rangefinder). This is the most user-visible data-quality issue on the site.

**Concrete failure:** Canon EOS 5D Mark IV appears 3 times on the All Cameras page: once as "DSLR", once as "Mirrorless", once as "Rangefinder".

**Fix:** Add category normalization during merge: when a camera already exists in Geizhals dslr or mirrorless data, discard the rangefinder duplicate. Only true rangefinders (Leica M, Voigtlander Bessa, etc.) should remain in the rangefinder category.

---

### C15-04: openMVG CSV DictReader has no BOM defense — potential 0-record parse
**Sources:** C15-05, S15-01, T15-04, D15-04, TE15-04
**Severity:** LOW | **Confidence:** HIGH (5-agent consensus)

If the openMVG GitHub CSV ever contains a UTF-8 BOM, `csv.DictReader` would produce mangled field names (e.g., `"﻿CameraMaker"` instead of `"CameraMaker"`), causing `KeyError` on every row and 0 records returned. The existing BOM defense in `parse_existing_csv` does not cover this code path.

**Concrete failure:** If the openMVG CSV repository adds a BOM, the next CI build produces a site missing all openMVG cameras.

**Fix:** Strip BOM from the CSV body before passing to `DictReader` in `openmvg.fetch()`.

---

### C15-05: Sigma SD regex `\d?` misses 2-digit models (SD10, SD14, SD15)
**Sources:** C15-03, CR15-03, V15-04, T15-01, D15-01 (partial)
**Severity:** LOW | **Confidence:** HIGH

The pattern `Sigma\s+SD\d?` only matches 0-1 digits after "SD", missing Sigma SD10, SD14, and SD15 — all Foveon-sensor DSLRs.

**Fix:** Change `Sigma\s+SD\d?` to `Sigma\s+SD\d+`.

---

### C15-06: openMVG docstring doesn't warn about DSLR regex limitations
**Sources:** DS15-01
**Severity:** LOW | **Confidence:** HIGH

The docstring added in C14-05 describes the DSLR heuristic but doesn't warn about its known limitations (Canon xxxD, Samsung NX). The Samsung NX comment in the code is also misleading.

**Fix:** Update the docstring and the Samsung NX comment.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics
- Total distinct new findings: 6 (3 MEDIUM, 3 LOW)
- Cross-agent consensus findings (3+ agents): 4
- All cycle 1-14 fixes remain intact
- 2 MEDIUM findings are REGRESSIONS from the C14-01 fix
- Remaining deferred items: see `.context/plans/deferred.md`
