# Architect Review (Cycle 56)

**Reviewer:** architect
**Date:** 2026-04-29
**HEAD:** `e8d5414`

## Findings

### F56-A-01: two refresh paths still divergent on size-less rows — LOW (gated)

- **Files:** `pixelpitch.py:613-628` (merge existing-only branch),
  `pixelpitch.py:1071-1087` (`_load_per_source_csvs`).
- **Detail:** Post-C55-01, both branches preserve cache when
  sensors_db is empty. They still disagree on size-less rows:
  - merge existing-only: leaves matched_sensors untouched
  - _load_per_source_csvs: forces matched_sensors = None
  Both are correct under their own contracts (merge preserves
  whatever existed; per-source-load honors `derive_spec`'s
  "not-checked when size unknown" sentinel). Folding into a shared
  helper requires a `force_none_if_no_size: bool` parameter.
- **Severity:** LOW. **Confidence:** HIGH.
- **Disposition:** Defer per F55-06 rationale (refactor risk in a
  monolith file outweighs marginal cleanup).

### F56-A-02: render_html category lists still duplicated — LOW (carry-over)

- **File:** `pixelpitch.py:1148-1164`
- Carry-over from F55-A-02. Adding a new category requires edits
  in several places. Not worse this cycle.
- **Severity:** LOW. **Confidence:** HIGH.
- **Disposition:** Defer.

## No high-severity architectural risks.
