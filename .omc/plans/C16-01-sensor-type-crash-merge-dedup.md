# Plan: Cycle 16 Findings — Sensor Type Crash, Merge Dedup, Pentax Regex, http_get Exception, digicamdb Alias

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** C16-01 through C16-05 (aggregate)

---

## Task 1: Fix `sensor_size_from_type` crash on invalid fractional sensor types — DONE
**Finding:** C16-01 (9-agent consensus)
**Severity:** MEDIUM | **File:** `pixelpitch.py`, lines 152-165

### Problem
The function computes `1 / float(typ[2:])` for types starting with `1/` not in the lookup table. On `1/0`, `1/0.0`, `1/` this raises ZeroDivisionError or ValueError, crashing the entire render pipeline through derive_spec -> derive_specs.

### Fix
Wrap the computation in a try/except block catching ZeroDivisionError and ValueError, returning None on error. Update docstring.

### Verification
- Add test cases for `sensor_size_from_type("1/0")`, `sensor_size_from_type("1/")`, `sensor_size_from_type("1/0.0")`
- Verify they return None instead of crashing
- Run `python3 -m tests.test_parsers_offline`

---

## Task 2: Fix `merge_camera_data` duplicate entries from same key in new_specs — DONE
**Finding:** C16-02 (10-agent consensus)
**Severity:** MEDIUM | **File:** `pixelpitch.py`, lines 349-407

### Problem
When two entries in new_specs have the same create_camera_key (e.g., camera from both Geizhals DSLR and openMVG DSLR), both are appended to merged_specs without deduplication. This produces visible duplicate rows on the All Cameras page.

### Fix
Track seen keys among new_specs within the merge loop. When a duplicate key is encountered, merge/replace instead of appending. Update docstring.

### Verification
- Add test case for duplicate keys in new_specs
- Verify merge_camera_data produces 1 entry, not 2
- Run `python3 -m tests.test_parsers_offline`

---

## Task 3: Fix Pentax DSLR regex to cover models without hyphen and letter-suffix models — DONE
**Finding:** C16-03 (7-agent consensus)
**Severity:** LOW | **File:** `sources/openmvg.py`, line 47

### Problem
`Pentax\s+K[-\s]\d` misses Pentax K3, K5, K7, K1 (no hyphen), KP, KF (letter suffix), K-r, K-x (letter after hyphen), K-30, K50, K100D etc. Also `Pentax\s+\d{1,2}D` misses Pentax 645Z.

### Fix
Change `Pentax\s+K[-\s]\d` to `Pentax\s+K[-\s]?\d+[A-Za-z]*` and `Pentax\s+\d{1,2}D` to `Pentax\s+\d{1,2}[DZ]`.

### Verification
- Add "Pentax K3" and "Pentax 645Z" to the test CSV in test_openmvg_csv_parser
- Verify they are classified as "dslr"
- Run `python3 -m tests.test_parsers_offline`

---

## Task 4: Add OSError to http_get exception handling — DONE
**Finding:** C16-05 (3-agent consensus)
**Severity:** LOW | **File:** `sources/__init__.py`, lines 48-61

### Problem
http_get catches URLError, HTTPError, TimeoutError but not OSError subclasses like ConnectionResetError and SSLError. While urllib typically wraps these, edge cases exist.

### Fix
Add OSError to the except clause in http_get.

### Verification
- Run `python3 -m tests.test_parsers_offline` (no regression)
- The fix is a safety improvement; testing would require mocking network errors

---

## Task 5: Remove digicamdb from SOURCE_REGISTRY or add duplicate guard — DONE
**Finding:** C16-04 (3-agent consensus)
**Severity:** LOW | **File:** `sources/digicamdb.py`; `pixelpitch.py`, line 985

### Problem
digicamdb is a pure alias for openMVG. If both source CSVs exist, every camera appears twice. This compounds C16-02.

### Fix
Remove digicamdb from SOURCE_REGISTRY since it provides no unique data. The module can remain as documentation.

### Verification
- Verify `python pixelpitch.py source digicamdb` raises ValueError (unknown source)
- Run `python3 -m tests.test_parsers_offline`

---

## Deferred Findings

No new deferred findings from cycle 16. All 5 findings have implementation tasks above.

Existing deferred items remain in `.context/plans/deferred.md`.
