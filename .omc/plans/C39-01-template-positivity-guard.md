# Plan: Cycle 39 Findings — Template Positivity Guard & CSV Pipeline Validation

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** CR39-01, CRIT39-01, V39-02, TR39-01, ARCH39-01, DBG39-01, DES39-01, TE39-01, CR39-02, CR39-03, DOC39-01

---

## Task 1: Replace `!= 0.0` with `> 0` in template pitch/mpix guards — C39-01 (core)

**Finding:** C39-01 (8-agent consensus)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `templates/pixelpitch.html`, lines 76-80, 84-88

### Problem

The C38-01 fix added `!= 0.0` checks for pitch and mpix in the Jinja2 template. This handles zero values but misses negative and NaN values, which also render as physically impossible or malformed strings ("-1.0 µm", "nan µm", "-10.0 MP", "nan MP").

The `> 0` guard handles all invalid cases in a single condition:
- `0.0 > 0` → False → "unknown" (handles the C38-01 case)
- `-1.0 > 0` → False → "unknown" (new fix)
- `NaN > 0` → False → "unknown" (new fix)
- `5.12 > 0` → True → "5.12 µm" (correct)

### Implementation

1. In `templates/pixelpitch.html`, line 76, change:
   ```jinja2
   {% if spec.spec.mpix is not none and spec.spec.mpix != 0.0 %}
   ```
   To:
   ```jinja2
   {% if spec.spec.mpix is not none and spec.spec.mpix > 0 %}
   ```

2. In `templates/pixelpitch.html`, line 84, change:
   ```jinja2
   {% if spec.pitch is not none and spec.pitch != 0.0 %}
   ```
   To:
   ```jinja2
   {% if spec.pitch is not none and spec.pitch > 0 %}
   ```

3. In `templates/pixelpitch.html`, line 50, change:
   ```jinja2
   data-pitch="{{ spec.pitch or 0 }}"
   ```
   To:
   ```jinja2
   data-pitch="{{ spec.pitch if spec.pitch is not none and spec.pitch > 0 else 0 }}"
   ```
   This addresses C39-03 (data-pitch attribute leaks invalid values).

---

## Task 2: Add positivity validation in `parse_existing_csv` — C39-02

**Finding:** C39-02
**Severity:** LOW | **Confidence:** HIGH
**Files:** `pixelpitch.py`, `parse_existing_csv` function (around lines 340-357)

### Problem

`_safe_float` allows negative values through. For physical-quantity fields (width, height, area, mpix, pitch), negative values are meaningless. A corrupted CSV with negative values passes through `parse_existing_csv` unfiltered.

### Implementation

In `parse_existing_csv`, after the `_safe_float` calls (around lines 341-348), add positivity checks:

```python
# Reject non-positive physical quantities
if width is not None and width <= 0:
    width = None
if height is not None and height <= 0:
    height = None
if area is not None and area <= 0:
    area = None
if mpix is not None and mpix <= 0:
    mpix = None
if pitch is not None and pitch <= 0:
    pitch = None
```

This ensures that size becomes None if either dimension is non-positive (since `if width is not None and height is not None` on line 343 would still create a tuple with a None element — need to also update the size construction):

```python
size = None
width = _safe_float(width_str)
height = _safe_float(height_str)
# Reject non-positive dimensions
if width is not None and width <= 0:
    width = None
if height is not None and height <= 0:
    height = None
if width is not None and height is not None:
    size = (width, height)

area = _safe_float(area_str)
if area is not None and area <= 0:
    area = None
mpix = _safe_float(mpix_str)
if mpix is not None and mpix <= 0:
    mpix = None
pitch = _safe_float(pitch_str)
if pitch is not None and pitch <= 0:
    pitch = None
```

---

## Task 3: Update `_safe_float` docstring — C39-04

**Finding:** DOC39-01
**Severity:** LOW | **Confidence:** HIGH
**Files:** `pixelpitch.py`, lines 268-269

### Implementation

Change docstring from:
```python
"""Parse a float string, returning None for NaN/inf/empty."""
```
To:
```python
"""Parse a float string, returning None for NaN/inf/empty.

Note: negative values are returned as-is; callers that require
positive-only values (e.g. pitch, mpix, area) must apply their
own ``val <= 0`` check.
"""
```

---

## Task 4: Add tests for negative/NaN pitch/mpix template rendering — TE39-01

**Finding:** TE39-01
**Severity:** LOW | **Confidence:** HIGH
**Files:** `tests/test_parsers_offline.py`

### Implementation

Add test cases to `test_template_zero_pitch_rendering` or create a new test function:

```python
def test_template_negative_pitch_rendering():
    """Verify pixelpitch.html template renders negative pitch/mpix as 'unknown'."""
    section("template negative value rendering")
    import pixelpitch as pp
    from models import Spec

    spec = Spec(name="Neg Cam", category="fixed", type=None,
                size=(5.0, 3.7), pitch=-1.0, mpix=-10.0, year=2020)
    d = pp.derive_spec(spec)
    d.id = 0

    from datetime import datetime, timezone
    date = datetime.now(timezone.utc)

    html = pp._get_env().get_template("pixelpitch.html").render(
        title="Test", specs=[d], page="fixed", date=date,
    )

    # Negative mpix must render as "unknown", not "-10.0 MP"
    expect("template: negative mpix renders as unknown",
           "-10.0 MP" not in html, True)
    expect("template: negative mpix shows unknown text",
           "unknown" in html, True)

    # Negative pitch must render as "unknown", not "-1.0 µm"
    expect("template: negative pitch renders as unknown",
           "-1.0 µm" not in html, True)
```

Also add test for `parse_existing_csv` with negative values:

```python
def test_parse_existing_csv_negative_values():
    """Verify parse_existing_csv rejects negative physical quantities."""
    section("parse_existing_csv negative value rejection")
    import pixelpitch as pp

    csv_neg = (
        "id,name,category,type,sensor_width_mm,sensor_height_mm,sensor_area_mm2,"
        "megapixels,pixel_pitch_um,year,matched_sensors\n"
        "0,Test,mirrorless,,-5.00,-3.00,15.00,-10.0,-1.00,2021,\n"
    )
    parsed = pp.parse_existing_csv(csv_neg)
    expect("negative CSV: row count", len(parsed), 1)
    expect("negative CSV: size is None", parsed[0].size, None)
    expect("negative CSV: area is None", parsed[0].area, None)
    expect("negative CSV: mpix is None", parsed[0].spec.mpix, None)
    expect("negative CSV: pitch is None", parsed[0].pitch, None)
```

---

## Verification

- Gate tests (`python3 -m tests.test_parsers_offline`) — all checks must pass
- Template with `pitch=0.0` still renders "unknown" (C38-01 fix preserved)
- Template with `pitch=-1.0` renders "unknown" (C39-01 new fix)
- Template with `mpix=-10.0` renders "unknown" (C39-01 new fix)
- `data-pitch` attribute uses 0 for invalid values (C39-03)
- `parse_existing_csv` rejects negative physical quantities (C39-02)
- New tests for negative value rendering pass (TE39-01)

---

## Deferred Findings

None. All findings are scheduled for implementation.
