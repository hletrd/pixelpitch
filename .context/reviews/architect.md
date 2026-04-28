# Architect Review (Cycle 55)

**Reviewer:** architect
**Date:** 2026-04-29
**HEAD:** `f08c3c4`

## Findings

### F55-A-01: two divergent "refresh sensors" implementations — LOW

- **Files:** `pixelpitch.py:615-628` (merge existing-only branch),
  `pixelpitch.py:1074-1086` (`_load_per_source_csvs`).
- **Detail:** Both paths compute matched_sensors against current
  sensors.json but disagree on the empty-db fallback (preserve vs
  drop). A small helper `_refresh_matched_sensors(d, sensors_db)`
  could converge them.
- **Severity:** LOW. **Confidence:** HIGH.

### F55-A-02: render_html category lists are duplicated — LOW

- **File:** `pixelpitch.py:1148-1164`
- **Detail:** Geizhals categories and source-only categories are
  hardcoded in multiple spots. Adding a new category requires
  edits in several places.
- **Severity:** LOW (refactor opportunity). **Confidence:** HIGH.

## No high-severity architectural risks.
