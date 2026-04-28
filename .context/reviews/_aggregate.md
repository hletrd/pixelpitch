# Aggregate Review (Cycle 20) — Deduplicated, Merged Findings

**Date:** 2026-04-28
**Reviewers:** code-reviewer, perf-reviewer, security-reviewer, critic, verifier, test-engineer, tracer, architect, debugger, designer, document-specialist

## Cycle 1-19 Status

All previous fixes confirmed still working. No regressions. Gate tests pass (105 checks). C19-01 (tablesorter column indices) and C19-02 (env var error handling) fixes verified in code.

## Cross-Agent Agreement Matrix (Cycle 20 New Findings)

| Finding | Flagged By | Highest Severity |
|---------|-----------|-----------------|
| `pixel_pitch()` ZeroDivisionError when mpix=0 | code-reviewer, critic, verifier, tracer, debugger, test-engineer | MEDIUM |
| Sony FX series misnamed by `_parse_camera_name` | code-reviewer, critic, verifier, tracer, debugger, test-engineer | MEDIUM |
| Merge doesn't preserve type/size/pitch from existing | code-reviewer, critic, verifier, architect | LOW |
| Stale CSV duplicates (259 pairs) | code-reviewer | LOW (data artifact) |

## Deduplicated New Findings (Ordered by Severity)

### C20-01: `pixel_pitch()` crashes with ZeroDivisionError when mpix=0.0
**Sources:** C20-01, C20-02, CR20-02, V20-01, T20-01, D20-01, TE20-01
**Severity:** MEDIUM | **Confidence:** HIGH (7-agent consensus)

The `pixel_pitch()` function divides by `mpix * 10**6` without checking for zero or negative values. When `mpix=0.0`, ZeroDivisionError. When `mpix<0`, ValueError (sqrt of negative). The `derive_spec()` function has no try/except guard, so this crashes the entire render pipeline.

**Concrete failure scenario:** A source HTML page contains "0.0 Megapixels" (placeholder or data error). The regex matches, producing mpix=0.0. `derive_spec` calls `pixel_pitch(area, 0.0)` and the entire `python pixelpitch.py` command crashes.

**Fix:** Add guard in `pixel_pitch()`:
```python
def pixel_pitch(area: float, mpix: float) -> float:
    if mpix <= 0:
        return 0.0
    return 1000 * sqrt(area / (mpix * 10**6))
```

---

### C20-02: Sony FX series cameras misnamed by `_parse_camera_name`
**Sources:** C20-03, CR20-01, V20-02, T20-02, D20-02, TE20-02
**Severity:** MEDIUM | **Confidence:** HIGH (6-agent consensus)

The `_parse_camera_name()` function applies `.title()` to URL slugs, converting "fx3" to "Fx3" instead of "FX3". The function has special-case replacement for "Sony Zv " -> "Sony ZV-" but no equivalent for Sony FX series. This causes cameras like FX3, FX6, FX30 to appear with incorrect capitalization.

**Impact:** Users searching for "Sony FX3" won't find it. If other sources produce the correct name "Sony FX3", the merge treats them as different cameras (different merge keys), creating duplicate entries.

**Fix:** Add FX-series normalization:
```python
cleaned = re.sub(r'\bFx(\d)', r'FX\1', cleaned)
```

---

### C20-03: `merge_camera_data` doesn't preserve type/size/pitch from existing data when new data has None
**Sources:** C20-04, CR20-03, V20-03, A20-01
**Severity:** LOW | **Confidence:** HIGH (4-agent consensus)

The merge function has explicit logic to preserve `year` from existing data when new data has `year=None`. However, `type`, `size`, and `pitch` have no such preservation. When a new spec has `type=None` but existing had `type='1/2.3'`, the type is lost.

**Fix:** Add field-level merge logic similar to year:
```python
if new_spec.spec.type is None and existing_spec.spec.type is not None:
    new_spec.spec.type = existing_spec.spec.type
if new_spec.spec.size is None and existing_spec.spec.size is not None:
    new_spec.spec.size = existing_spec.spec.size
if new_spec.spec.pitch is None and existing_spec.spec.pitch is not None:
    new_spec.spec.pitch = existing_spec.spec.pitch
```

---

### C20-04: 259 duplicate (name, category) pairs in dist/camera-data.csv
**Sources:** C20-05
**Severity:** LOW | **Confidence:** HIGH (data artifact, not code bug)

The current CSV has 259 duplicate pairs from previous CI runs before the merge dedup logic was fixed. Running `merge_camera_data([], existing)` reduces 1742 records to 1472 with 0 duplicates. The next CI deployment will clean this up.

**No code fix needed.**

---

## AGENT FAILURES

No agents failed. All reviews completed successfully.

## Summary Statistics
- Total distinct new findings: 3 actionable (2 MEDIUM, 1 LOW) + 1 data artifact
- Cross-agent consensus findings (3+ agents): 3 (C20-01, C20-02, C20-03)
- 2 MEDIUM findings: pixel_pitch crash, Sony FX misnaming
- 1 LOW finding: merge field preservation
- 1 data artifact: stale CSV duplicates (no code fix needed)
