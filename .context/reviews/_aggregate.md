# Aggregate Review (Cycle 38) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-37 Status

All previous fixes confirmed still working. No regressions in core logic. Gate tests pass. C37-01 and C37-02 implemented and verified.

## Cross-Agent Agreement Matrix (Cycle 38 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| Template renders "0.0 µm" for zero pitch but JS `isInvalidData` hides those rows — UX contradiction | CR38-01, CRIT38-01, V38-02, TR38-01, ARCH38-01, DBG38-01, DES38-01, TE38-01 | MEDIUM |
| `match_sensors` latent ZeroDivisionError if guard changes | CR38-02 | LOW |

## Deduplicated New Findings (Ordered by Severity)

### C38-01: Template renders "0.0 µm" for zero pitch but JS hides those rows — UX contradiction introduced by C37-02

**Sources:** CR38-01, CRIT38-01, V38-02, TR38-01, ARCH38-01, DBG38-01, DES38-01, TE38-01
**Severity:** MEDIUM | **Confidence:** HIGH (8-agent consensus)

The C37-02 fix added `if (pitch === 0) { return true; }` to the JS `isInvalidData` function, hiding zero-pitch rows by default. However, the Jinja2 template still renders `0.0` pitch as "0.0 µm" (not "unknown"). This creates a three-layer disagreement:

1. **Python `pixel_pitch`**: returns `0.0` as sentinel for "invalid input"
2. **Jinja2 template**: treats `0.0` as a valid number, renders "0.0 µm"
3. **JS `isInvalidData`**: treats `0.0` as invalid, hides the row

The user experience: rows with `pitch=0.0` are hidden by default (toggle is checked), but if the user unchecks "Hide possibly invalid data", they see "0.0 µm" displayed as if it were a legitimate measurement. A 0.0 µm pixel pitch is physically impossible.

The same issue applies to `mpix=0.0` — rendered as "0.0 MP" in the template but representing invalid data.

**Fix:** Update the Jinja2 template to render "unknown" for `pitch=0.0` and `mpix=0.0`, consistent with JS treating these as invalid:
```jinja2
{% if spec.pitch is not none and spec.pitch != 0.0 %}
  {{ spec.pitch|round(1) }} µm
{% else %}
  <span class="text-muted">unknown</span>
{% endif %}
```

Similarly for mpix:
```jinja2
{% if spec.spec.mpix is not none and spec.spec.mpix != 0.0 %}
  {{ spec.spec.mpix|round(1) }} MP
{% else %}
  <span class="text-muted">unknown</span>
{% endif %}
```

Also update `test_template_zero_pitch_rendering` to verify that 0.0 pitch/mpix render as "unknown" (not as numbers).

---

### C38-02: `match_sensors` latent ZeroDivisionError risk — currently guarded

**Sources:** CR38-02
**Severity:** LOW | **Confidence:** MEDIUM

The `abs(megapixels - mp) / megapixels` expression would divide by zero if `megapixels=0.0`. The outer guard `megapixels > 0` prevents this, but if that guard were ever relaxed, a ZeroDivisionError would occur. This is a theoretical risk only — no fix needed now.

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics

- Total distinct new findings: 2 (C38-01, C38-02)
- Cross-agent consensus findings (3+ agents): 1 (C38-01 with 8 agents)
- Highest severity: MEDIUM (C38-01)
- Actionable findings: 1 (C38-01)
- Verified safe / deferred: 1 (C38-02)
