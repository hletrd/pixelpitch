# Aggregate Review (Cycle 16) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-15 Status

All previous fixes confirmed still working. No regressions in previously-fixed items.

## Cross-Agent Agreement Matrix (Cycle 16 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| sensor_size_from_type crashes on invalid input (1/0, 1/, 1/0.0) | code-reviewer, security-reviewer, critic, verifier, tracer, architect, debugger, test-engineer | MEDIUM |
| merge_camera_data doesn't dedup among new_specs | code-reviewer, perf-reviewer, critic, verifier, tracer, architect, debugger, test-engineer, designer | MEDIUM |
| Pentax DSLR regex misses 10+ models | code-reviewer, critic, verifier, tracer, debugger, designer, test-engineer | LOW |
| digicamdb alias creates duplicate source CSVs | code-reviewer, critic, architect | LOW |
| http_get doesn't catch OSError subclasses | security-reviewer, debugger, test-engineer | LOW |
| No tests for sensor_size_from_type invalid inputs | test-engineer | LOW |
| No tests for merge_camera_data duplicate new_specs | test-engineer | LOW |
| Docstring gaps for sensor_size_from_type and merge_camera_data | document-specialist | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C16-01: `sensor_size_from_type` crashes on invalid fractional sensor types (1/0, 1/0.0, 1/)
**Sources:** C16-01, S16-01, CR16-01, V16-01, T16-01, A16-03, D16-01, TE16-01, DS16-01
**Severity:** MEDIUM | **Confidence:** HIGH (9-agent consensus)

The function computes `1 / float(typ[2:])` for types starting with `1/` that are not in the lookup table. If `typ` is `"1/0"`, `"1/0.0"`, or `"1/"`, this raises `ZeroDivisionError` or `ValueError`. These exceptions propagate through `derive_spec` -> `derive_specs`, crashing the entire render pipeline.

Reproduced: `Spec(name='Test', category='fixed', type='1/0', size=None, ...)` -> `derive_spec()` -> `ZeroDivisionError`.

**Fix:** Wrap the computation in a try/except block in `sensor_size_from_type` and return None on any arithmetic or conversion error. Add tests.

---

### C16-02: `merge_camera_data` does not deduplicate among `new_specs` — duplicate entries when same camera appears in multiple sources with same category
**Sources:** C16-02, P16-01, CR16-02, V16-02, T16-02, A16-01, D16-02, TE16-02, UX16-01, DS16-02
**Severity:** MEDIUM | **Confidence:** HIGH (10-agent consensus)

When two entries in `new_specs` have the same `create_camera_key` result (e.g., a camera from both Geizhals DSLR and openMVG DSLR), both are appended to `merged_specs` without deduplication. This produces visible duplicate rows on the All Cameras page.

Reproduced: Two specs with same name+category in new_specs -> 2 entries in merged result instead of 1.

This is a regression risk from the C15-01 fix: now that openMVG correctly classifies Canon EOS xxxD cameras as DSLR, they overlap with Geizhals DSLR data.

**Fix:** Track seen keys among new_specs within the merge loop. When a duplicate key is encountered, merge/replace instead of appending.

---

### C16-03: Pentax DSLR regex misses models without hyphen (K3, K5, K7, K1) and letter-suffix models (KP, KF, K-r, K-x) and multi-digit models (K-30, K-50, K-70, K100D, etc.)
**Sources:** C16-03, CR16-03, V16-03, T16-03, D16-03, UX16-02, TE16-03
**Severity:** LOW | **Confidence:** HIGH (7-agent consensus)

The Pentax pattern `Pentax\s+K[-\s]\d` requires a hyphen/space between K and the first digit, and only matches a single digit. It misses at least 10 Pentax DSLR models. The `Pentax\s+\d{1,2}D` pattern also misses `Pentax 645Z`.

**Fix:** Change `Pentax\s+K[-\s]\d` to `Pentax\s+K[-\s]?\d+\w?` or more precisely `Pentax\s+K[-\s]?\d+[A-Za-z]*`. Also add `Pentax\s+645[A-Z]` for the medium-format line.

---

### C16-04: `digicamdb` source is a pure alias for openMVG — potential duplicate source CSVs
**Sources:** C16-04, CR16-04, A16-02
**Severity:** LOW | **Confidence:** HIGH (3-agent consensus)

The digicamdb module delegates to openmvg.fetch(). If both source CSVs exist, every camera appears twice in the data, compounding C16-02.

**Fix:** Remove digicamdb from SOURCE_REGISTRY (it's a no-op alias), or add a guard to skip if openmvg CSV already exists.

---

### C16-05: `http_get` does not catch `OSError` subclasses (ConnectionResetError, SSLError)
**Sources:** S16-02, D16-04, TE16-04
**Severity:** LOW | **Confidence:** MEDIUM (3-agent consensus)

While urllib typically wraps these as URLError, edge cases exist where the underlying socket error leaks through. Adding `OSError` to the except clause would make the function more robust.

**Fix:** Add `OSError` to the except clause in `http_get`.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics
- Total distinct new findings: 5 (2 MEDIUM, 3 LOW)
- Cross-agent consensus findings (3+ agents): 5
- All cycle 1-15 fixes remain intact
- 2 MEDIUM findings are NEW (not regressions from previous fixes)
- C16-02 is partially triggered by the C15-01 fix (openMVG now correctly classifies Canon xxxD as DSLR, creating more overlap with Geizhals)
