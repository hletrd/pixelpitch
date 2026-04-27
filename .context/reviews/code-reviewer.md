# Code Review (Cycle 16) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-15 fixes, focusing on NEW issues missed or introduced by previous fixes

## Previously Fixed (Cycles 1-15) — Confirmed Resolved

All previous fixes remain intact. C15-01 (Canon xxxD regex), C15-02 (Samsung NX removal), C15-03 (rangefinder dedup), C15-04 (BOM defense), C15-05 (Sigma SD regex), C15-06 (docstring update) all verified as correctly applied and tested.

## New Findings

### C16-01: `sensor_size_from_type` crashes on invalid fractional sensor types (1/0, 1/0.0, 1/)
**File:** `pixelpitch.py`, lines 152-165 (`sensor_size_from_type`)
**Severity:** MEDIUM | **Confidence:** HIGH

The function computes `1 / float(typ[2:])` for types starting with `1/` that are not in the lookup table. If `typ` is `"1/0"` or `"1/0.0"`, this raises `ZeroDivisionError`. If `typ` is `"1/"`, this raises `ValueError` (empty string to float). Both exceptions are uncaught and propagate through `derive_spec` -> `derive_specs`, crashing the entire render pipeline.

The `SENSOR_TYPE_RE` regex `(1/[\d\.]+)\"` can match `1/0`, `1/0.0` from source HTML. If a source provides `type="1/0"` and no size, the crash path is triggered.

**Concrete failure:** A malformed GSMArena or Geizhals page containing `1/0"` in the sensor field would crash `python pixelpitch.py` during the render step, producing no output.

**Fix:** Wrap the computation in a try/except block and return None on any arithmetic or conversion error.

---

### C16-02: `merge_camera_data` does not deduplicate among `new_specs` — duplicate entries when same camera appears in multiple sources with the same category
**File:** `pixelpitch.py`, lines 349-407 (`merge_camera_data`)
**Severity:** MEDIUM | **Confidence:** HIGH

The merge loop iterates `new_specs` linearly and checks each against `existing_by_key`. If two entries in `new_specs` have the same `create_camera_key` result (e.g., a camera from both Geizhals mirrorless and openMVG mirrorless), both are appended to `merged_specs`. There is no deduplication among `new_specs` themselves.

Verified by test: feeding two specs with the same name+category into `merge_camera_data([], ...)` produces 2 entries instead of 1.

**Concrete failure:** If a camera like "Canon EOS 250D" appears in both the Geizhals DSLR category and the openMVG source CSV (also classified as DSLR by the C15-01 fix), the All Cameras page shows it twice.

**Fix:** Before appending to `merged_specs`, track seen keys among new_specs. If a key has already been processed in the current batch, skip or merge instead of appending again.

---

### C16-03: Pentax DSLR regex misses models without hyphen (K3, K5, K7, K1) and letter-suffix models (KP, KF, K-r, K-x) and medium-format (645D, 645Z)
**File:** `sources/openmvg.py`, line 47 (`Pentax\s+K[-\s]\d`)
**Severity:** LOW | **Confidence:** HIGH

The Pentax pattern `Pentax\s+K[-\s]\d` requires a hyphen or space between K and the first digit. It misses:
- Pentax K3, K5, K7, K1 (no hyphen: 4 models)
- Pentax KP, KF (letter suffix, not digit)
- Pentax K-r, K-x (letter after hyphen)
- Pentax K-30, K-50, K-70 (2 digits after hyphen — `\d` only matches 1)
- Pentax K100D, K200D, K10D, K20D (2+ digits + D suffix)

The `Pentax\s+\d{1,2}D` pattern only matches 1-2 digit models (645D) but the `\d` before D in names like `K100D` is part of the K-mount naming, not the medium-format line. `Pentax 645Z` is also missed (Z suffix instead of D).

**Concrete failure:** Pentax K3, K5, K7, KP, KF, K-r, K-x (all DSLRs) would be classified as mirrorless by the openMVG heuristic if they appear in the dataset with sensor width >= 20mm.

**Fix:** Change `Pentax\s+K[-\s]\d` to `Pentax\s+K[-\s]?\d+\w?` or more precisely `Pentax\s+K[-\s]?\d+[A-Za-z]*` to cover all K-mount naming variants.

---

### C16-04: `digicamdb` source is an alias for openMVG but registered as a separate source — potential for identical duplicate CSVs
**File:** `sources/digicamdb.py`, line 21; `pixelpitch.py`, line 985 (`SOURCE_REGISTRY`)
**Severity:** LOW | **Confidence:** HIGH

The digicamdb module delegates entirely to `openmvg.fetch()`, returning the same data. Both are registered in `SOURCE_REGISTRY`. If a user runs `python pixelpitch.py source digicamdb --out dist`, the resulting `camera-data-digicamdb.csv` would contain identical records to `camera-data-openmvg.csv`. When both are loaded by `_load_per_source_csvs`, every camera from these sources appears twice in `new_specs`, compounding the C16-02 merge dedup bug.

The CI workflow only fetches `openmvg`, so this only triggers on manual fetch. But it's still a design flaw.

**Fix:** Either remove `digicamdb` from `SOURCE_REGISTRY` (since it's a no-op alias), or make it a redirect that skips CSV creation if `camera-data-openmvg.csv` already exists.

---

## Summary
- NEW findings: 4 (2 MEDIUM, 2 LOW)
- C16-01: sensor_size_from_type crashes on 1/0 etc. — MEDIUM
- C16-02: merge_camera_data doesn't dedup among new_specs — MEDIUM
- C16-03: Pentax DSLR regex misses many models — LOW
- C16-04: digicamdb alias creates duplicate source CSVs — LOW
