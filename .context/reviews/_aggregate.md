# Aggregate Review (Cycle 17) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-16 Status

All previous fixes confirmed still working. No regressions in previously-fixed items. Gate tests pass (98 checks).

## Cross-Agent Agreement Matrix (Cycle 17 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| Pentax KP/KF/K-r/K-x still misclassified (C16-03 fix incomplete) | code-reviewer, critic, verifier, tracer, debugger, designer, test-engineer | MEDIUM |
| Nikon Df missed by DSLR regex | code-reviewer, critic, verifier, tracer, debugger, designer, test-engineer | LOW |
| GSMArena SENSOR_FORMAT_RE doesn't match Unicode quotes | code-reviewer, tracer, debugger, test-engineer, document-specialist | LOW |
| openMVG docstring says "Pentax K-*" but regex is broader | document-specialist | LOW |
| sensors_db loaded unconditionally in merge | perf-reviewer | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C17-01: Pentax KP, KF, K-r, K-x still misclassified as mirrorless — C16-03 fix incomplete
**Sources:** C17-01, CR17-01, V17-01, T17-01, D17-01, UX17-01, TE17-01, A17-01 (context), DS17-01 (doc)
**Severity:** MEDIUM | **Confidence:** HIGH (8-agent consensus)

The C16-03 fix changed `Pentax\s+K[-\s]\d` to `Pentax\s+K[-\s]?\d+[A-Za-z]*`. This regex still requires at least one digit after K[-\s]?. Four Pentax DSLR models have no digit in that position:
- **Pentax KP** — letter directly after K
- **Pentax KF** — letter directly after K
- **Pentax K-r** — hyphen + letter (no digit)
- **Pentax K-x** — hyphen + letter (no digit)

All four are DSLRs. Verified: `_DSLR_NAME_RE.search()` returns None for all four.

**Fix:** Change `Pentax\s+K[-\s]?\d+[A-Za-z]*` to `Pentax\s+K[-\s]?[\dA-Za-z]+[A-Za-z]*` to allow letters or digits after K[-\s]?. Add test cases for Pentax KP and KF.

---

### C17-02: Nikon Df missed by DSLR regex
**Sources:** C17-02, CR17-02, V17-02, T17-02, D17-02, UX17-02, TE17-02
**Severity:** LOW | **Confidence:** HIGH (7-agent consensus)

The Nikon pattern `Nikon\s+D\d{1,4}` requires at least one digit after D. Nikon Df has no digit — it's a letter-only suffix. The Df is a well-known retro-style DSLR.

**Fix:** Add `|Nikon\s+Df` to the regex alternation. Add test case.

---

### C17-03: GSMArena SENSOR_FORMAT_RE doesn't match Unicode curly quotes
**Sources:** C17-03, T17-03, D17-03, TE17-03, DS17-02
**Severity:** LOW | **Confidence:** MEDIUM (5-agent consensus)

The regex `r'(1/[\d.]+)"'` requires ASCII double-quote. Some web pages use Unicode right double quotation mark (U+2033). The central `TYPE_FRACTIONAL_RE` handles this correctly. When GSMArena pages use curly quotes, the sensor type is lost silently.

**Fix:** Change the regex to `r'(1/[\d.]+)(?:\"|″)'` or reuse `TYPE_FRACTIONAL_RE`.

---

### C17-04: openMVG docstring says "Pentax K-*" but regex is now broader
**Sources:** DS17-01
**Severity:** LOW | **Confidence:** HIGH

After the C16-03 fix, the regex matches K3, K5, K7 (no hyphen) as well as K-1, K-30 (with hyphen). The docstring still says "Pentax K-*" which implies only hyphenated models.

**Fix:** Update docstring to say "Pentax K-mount (K3, K-1, KP, etc.)".

---

### C17-05: sensors_db loaded unconditionally in merge_camera_data
**Sources:** P17-01
**Severity:** LOW | **Confidence:** HIGH

`load_sensors_database()` is always called even when no existing-only cameras need sensor matching. Minor inefficiency — the file is small and the parse is fast.

**Fix (if desired):** Lazy-load — only call when existing-only cameras are found.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics
- Total distinct new findings: 5 (1 MEDIUM, 4 LOW)
- Cross-agent consensus findings (3+ agents): 3
- All cycle 1-16 fixes remain intact
- 1 MEDIUM finding is a carry-over from incomplete C16-03 fix (Pentax letter-only models)
- Gate tests pass (98 checks)
