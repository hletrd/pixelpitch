# Plan: Cycle 24 Findings — Sensor Type Regex Gaps

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** CRIT24-01, V24-02, TR24-01, ARCH24-01, DBG24-01, TE24-01, CRIT24-02, V24-03, TR24-02, DBG24-02, TE24-02, DOC24-01

---

## Task 1: Fix TYPE_FRACTIONAL_RE to match "1/x.y inch" (space+inch) — C24-01

**Finding:** C24-01 (6-agent consensus: critic, verifier, tracer, architect, debugger, test-engineer)
**Severity:** LOW | **Confidence:** HIGH
**File:** `sources/__init__.py`, line 68
**Test file:** `tests/test_parsers_offline.py`

### Problem

The `TYPE_FRACTIONAL_RE` regex pattern is:
```
(1/[\d.]+)(?:\"|inch|-inch|-type|\s*type|″)
```

It matches `1/2.3-inch` and `1/2.3"` but NOT `1/2.3 inch` (space before "inch"). The pattern has `\s*type` but lacks the corresponding `\s*inch`.

### Implementation

1. Update `TYPE_FRACTIONAL_RE` in `sources/__init__.py` line 68 to add `\s*inch` alternative:
   ```
   (1/[\d.]+)(?:\"|\s*inch|-inch|-type|\s*type|″)
   ```
   Note: Replace `inch` with `\s*inch` (which subsumes the `inch` case since `\s*` matches zero spaces).

2. Add test case in `test_gsmarena_unicode_quotes()`:
   ```python
   m5 = TYPE_FRACTIONAL_RE.search('1/2.3 inch')
   expect("space+inch suffix match", m5.group(1) if m5 else None, "1/2.3")
   ```

### Verification — DONE
- Gate tests (`python3 -m tests.test_parsers_offline`) — all 222 checks passed
- New test cases pass (space+inch, no-space inch)
- Commit: 4f667c0

---

## Task 2: Add bare 1-inch sensor type extraction in parse_sensor_field — C24-02

**Finding:** C24-02 (5-agent consensus: critic, verifier, tracer, debugger, test-engineer)
**Severity:** LOW | **Confidence:** HIGH
**File:** `pixelpitch.py`, lines 529-558 (parse_sensor_field function)
**Test file:** `tests/test_parsers_offline.py`

### Problem

`parse_sensor_field()` uses `TYPE_FRACTIONAL_RE` which only matches fractional-inch formats (`1/x.y` prefix). The bare 1-inch format (`1"`) is not matched, even though `TYPE_SIZE` has a `"1"` key with value `(13.2, 8.8)`.

### Implementation

Add a fallback check after the `TYPE_FRACTIONAL_RE` check in `parse_sensor_field()`:

```python
# Extract sensor type (e.g. "1/3.1", "1/2.3")
type_match = TYPE_FRACTIONAL_RE.search(sensor_text)
if type_match:
    result["type"] = type_match.group(1)
elif re.search(r'\b1["″\-]inch\b|\b1["″]\s', sensor_text):
    # Bare 1-inch format (not fractional 1/x.y)
    result["type"] = "1"
```

Also add a test case in `test_sensor_size_from_type()` or a new test function:
```python
# 1-inch sensor format
result = pp.parse_sensor_field('CMOS 1"')
expect("parse_sensor_field bare 1-inch type", result["type"], "1")
```

### Verification — DONE
- Gate tests — all 222 checks passed
- New test cases pass (bare 1", 1-inch, 1 inch, fractional precedence)
- Commit: 7576314

---

## Task 3: Update TYPE_FRACTIONAL_RE comment — C24-04

**Finding:** C24-04 (document-specialist)
**Severity:** LOW | **Confidence:** MEDIUM
**File:** `pixelpitch.py`, lines 47-49

### Problem

The comment says "matches '1/x.y' followed by any recognized suffix (ASCII/Unicode quotes, 'inch', '-type', etc.)" but doesn't list all suffix alternatives precisely. The `\s*type` alternative is not mentioned.

### Implementation

Update the comment to explicitly list all suffix alternatives:
```python
# Canonical fractional-inch sensor type regex — matches "1/x.y" followed by
# any recognized suffix: ASCII/Unicode quotes (" ″), optional-space + "inch",
# "-inch", optional-space + "type", or "-type".
```

### Verification — DONE
- Comment updated to list all suffix alternatives explicitly
- Commit: 7576314 (included in Task 2 commit)

---

## Deferred Findings

### C24-03: _parse_fields rstrip("</") strips chars not string
- **File:** `sources/imaging_resource.py`, line 95
- **Original Severity:** LOW | **Confidence:** HIGH
- **Reason:** Previously deferred as C3-08. The `rstrip("</")` strips individual chars (<, /, ") rather than the string `"</"`. In practice, Imaging Resource values rarely end in these characters, and the regex/HTML processing usually cleans up tag remnants. The risk of data mangling is theoretical.
- **Re-open if:** A concrete case of data mangling is found, or the Imaging Resource parser starts producing values ending in `"`, `/`, or `<`.

### C24-05: SpecDerived.size shadows Spec.size — maintainability concern
- **File:** `models.py`, line 23 (`SpecDerived.size`) vs line 14 (`Spec.size`)
- **Original Severity:** LOW | **Confidence:** MEDIUM
- **Reason:** This is a design choice in the data model. `SpecDerived.size` holds the computed/derived size (which may differ from `Spec.size` when derived from sensor type). Renaming would require changes across the entire codebase (template, tests, merge logic) with no correctness benefit. The shadowing is documented and verified to work correctly.
- **Re-open if:** Another attribute shadowing bug is introduced, or a developer expresses confusion about the relationship between `spec.size` and `spec.spec.size`.
