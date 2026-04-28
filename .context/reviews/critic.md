# Critic Review (Cycle 55)

**Reviewer:** critic
**Date:** 2026-04-29
**HEAD:** `f08c3c4`

## Overview

54+ cycles of fixes have hardened the repository. Both gates pass.
This cycle's job is to surface what is still genuinely improvable
without manufacturing churn.

## Findings

### F55-CRIT-01: `_load_per_source_csvs` drops cached matched_sensors when sensors.json fails — LOW

- **File:** `pixelpitch.py:1041-1086`
- **Detail:** When `load_sensors_database()` returns `{}` (file
  missing or invalid), every parsed row's matched_sensors is reset
  to `None`. The per-source CSV's pre-computed cache is discarded
  even though it would be a sensible fallback. By contrast,
  `merge_camera_data`'s existing-only branch preserves the cached
  matched_sensors when sensors_db is empty (it just skips the
  re-match). The two paths disagree.
- **Fix:** When sensors_db is empty in `_load_per_source_csvs`,
  leave the parsed matched_sensors untouched (treat the per-source
  CSV column as a cache fallback).
- **Severity:** LOW (sensors.json failure is rare).
- **Confidence:** MEDIUM.

### F55-CRIT-02: `merge_camera_data` size-preservation has two redundant paths — LOW (cleanup)

- **File:** `pixelpitch.py:539-557` and `pixelpitch.py:576-577`
- **Detail:** Spec.size preservation overrides derived.size when
  they disagree. The later derived.size preservation is
  unreachable when the first executed; reachable only when
  new_spec.spec.size was non-None. Logic correct, code hard to
  read.
- **Fix:** Optional restructure into a single decision tree.
- **Severity:** LOW. **Confidence:** MEDIUM.

### F55-CRIT-03: `tests/test_parsers_offline.py` is 2336-line monolith — LOW

- Same class as deferred F32 (`pixelpitch.py` monolith).
  Recommend deferral on identical grounds.

## No HIGH/CRITICAL findings.
