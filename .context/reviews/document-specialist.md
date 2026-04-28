# Document Specialist — Cycle 54

**HEAD:** `93851b0`

## Doc/code mismatch sweep

### F54-DOC-01 — `_load_per_source_csvs` docstring says "caches" but treats them as authoritative — LOW

- **File:** `pixelpitch.py:1028-1053`
- **Severity:** LOW | **Confidence:** HIGH
- **Detail:** Docstring (lines 1031-1033) says: "These are produced
  by `python pixelpitch.py source <name>` runs and serve as caches
  between deployments." But the function trusts the file's
  matched_sensors column verbatim — see F54-01. The doc and code
  disagree on whether per-source CSVs are caches or authoritative.
- **Fix:** Resolved when F54-01 implementation lands.

### F54-DOC-02 — `merge_camera_data` docstring does not document matched_sensors preservation — LOW (cosmetic)

- **File:** `pixelpitch.py:475-497`
- **Severity:** LOW | **Confidence:** HIGH
- **Detail:** The behavior added in C46 (preserve existing
  matched_sensors when new is None, treat [] as authoritative)
  is implemented correctly but not described in the docstring.
- **Fix:** One-paragraph addition. Cosmetic.

## Confirmed clean

- `_safe_year` and `_safe_int_id` docstrings were synced with code
  in C52-C53. Re-checked, accurate.
- README.md and templates carry no library-version claims that
  could drift.
