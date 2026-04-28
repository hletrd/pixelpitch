# Code Review (Cycle 21) — Code Quality, Logic, SOLID, Maintainability

**Reviewer:** code-reviewer
**Date:** 2026-04-28
**Scope:** Full repository re-review after cycles 1-20 fixes, focusing on NEW issues

## C21-01: `merge_camera_data` preserves `Spec` fields but NOT `SpecDerived` fields — stale data in rendered HTML

**File:** `pixelpitch.py`, lines 403-410
**Severity:** HIGH | **Confidence:** HIGH (reproduced)

The C20-03 fix added field preservation for `type`, `size`, and `pitch` at the `Spec` level (`new_spec.spec.*`). However, the `SpecDerived` fields (`new_spec.size`, `new_spec.area`, `new_spec.pitch`) are NOT updated to match. The Jinja2 template reads from `SpecDerived` fields, not `Spec` fields, so the preserved values are invisible in the rendered HTML.

**Concrete failure scenario:**
1. Camera "Sony A7 IV" has `size=(35.9, 23.9)`, `pitch=5.12` in the existing CSV
2. A new source (e.g., IR) has the same camera with `size=None`, `pitch=None`
3. After merge: `spec.spec.size=(35.9, 23.9)` (preserved), but `spec.size=None` (NOT updated)
4. Template reads `spec.size` -> shows "unknown" instead of "35.9 × 23.9 mm"
5. Similarly, `spec.spec.pitch=5.12` but `spec.pitch=None` -> shows "unknown" instead of "5.12 µm"

**Evidence:**
```python
spec_new = Spec(name='Test Cam', category='fixed', type='1/2.3', size=None, pitch=None, mpix=10.0, year=2020)
derived_new = pp.derive_spec(spec_new)
spec_existing = Spec(name='Test Cam', category='fixed', type='1/2.3', size=(5.0, 3.7), pitch=2.0, mpix=10.0, year=2020)
derived_existing = pp.derive_spec(spec_existing)
merged = pp.merge_camera_data([derived_new], [derived_existing])
m = merged[0]
# m.spec.size = (5.0, 3.7)  <- preserved at Spec level (correct)
# m.size = None               <- NOT updated at SpecDerived level (BUG)
# m.spec.pitch = 2.0          <- preserved at Spec level (correct)
# m.pitch = None              <- NOT updated at SpecDerived level (BUG)
```

**Fix:** Also preserve `SpecDerived` fields alongside `Spec` fields in `merge_camera_data`:
```python
if new_spec.size is None and existing_spec.size is not None:
    new_spec.size = existing_spec.size
if new_spec.area is None and existing_spec.area is not None:
    new_spec.area = existing_spec.area
if new_spec.pitch is None and existing_spec.pitch is not None:
    new_spec.pitch = existing_spec.pitch
```

---

## C21-02: Sony RX/DSC/HX/WX/TX/QX series cameras misnamed — same `.title()` issue as C20-02 but broader

**File:** `sources/imaging_resource.py`, lines 158-165
**Severity:** MEDIUM | **Confidence:** HIGH (reproduced)

The C20-02 fix added `re.sub(r'\bFx(\d)', r'FX\1', cleaned)` to fix the FX naming issue. However, the same `.title()` issue affects ALL Sony multi-letter uppercase series: RX, HX, WX, TX, QX, and DSC. These are all converted to lowercase second letter by `.title()` (e.g., "rx100" -> "Rx100" instead of "RX100").

**Concrete failure scenario:** A user searches for "Sony RX100 VII" on the All Cameras page and doesn't find it because it's listed as "Sony Rx100 Vii". If another source (Apotelyt) produces the correct name "Sony RX100 VII", the merge treats them as different cameras (different keys), creating a duplicate entry.

**Evidence:**
```python
name = imaging_resource._parse_camera_name(
    {'Model Name': 'Sony RX100 VII'},
    'https://www.imaging-resource.com/cameras/sony-rx100-vii-review/specifications/'
)
# Returns "Sony Rx100 Vii" instead of "Sony RX100 VII"
```

**Fix:** Add a general Sony uppercase series normalizer:
```python
# After the existing FX normalizer:
cleaned = re.sub(r'\bFx(\d)', r'FX\1', cleaned)
# Add:
cleaned = re.sub(r'\bRx(\d)', r'RX\1', cleaned)
cleaned = re.sub(r'\bHx(\d)', r'HX\1', cleaned)
cleaned = re.sub(r'\bWx(\d)', r'WX\1', cleaned)
cleaned = re.sub(r'\bTx(\d)', r'TX\1', cleaned)
cleaned = re.sub(r'\bQx(\d)', r'QX\1', cleaned)
cleaned = re.sub(r'\bDsc\b', r'DSC', cleaned)
```

---

## C21-03: `mpix` field is NOT preserved by merge when new data has `mpix=None`

**File:** `pixelpitch.py`, lines 403-410
**Severity:** LOW | **Confidence:** HIGH

The merge function preserves `type`, `size`, `pitch`, and `year` from existing data when new data has None. However, `mpix` has no such preservation logic. When a new source has `mpix=None` but existing data has `mpix=33.0`, the megapixel count is lost and the camera shows "unknown" resolution.

**Concrete failure scenario:** Camera "Canon R5" has `mpix=45.0` in the existing CSV. A new source fetch returns the camera without effective megapixels. After merge, `mpix=None` -> camera shows "unknown" resolution.

**Fix:** Add mpix preservation:
```python
if new_spec.spec.mpix is None and existing_spec.spec.mpix is not None:
    new_spec.spec.mpix = existing_spec.spec.mpix
```

---

## C21-04: `test_merge_field_preservation` only validates `Spec` fields, not `SpecDerived` fields

**File:** `tests/test_parsers_offline.py`, lines 676-718
**Severity:** MEDIUM | **Confidence:** HIGH

The test added in C20-03 checks `merged_t[0].spec.type`, `merged_s[0].spec.size`, and `merged_p[0].spec.pitch` (Spec-level fields). It does NOT check `merged_t[0].size`, `merged_s[0].area`, or `merged_p[0].pitch` (SpecDerived-level fields). This is why the C21-01 bug went undetected — the test passes but doesn't verify the fields that the template actually uses.

**Fix:** Add assertions for SpecDerived fields:
```python
expect("merge: preserves derived.size from existing",
       merged_s[0].size, (5.0, 3.7), tol=0.01)
expect("merge: preserves derived.area from existing",
       merged_s[0].area, 18.5, tol=0.01)
expect("merge: preserves derived.pitch from existing",
       merged_p[0].pitch, 2.0, tol=0.01)
```

---

## Summary

- C21-01 (HIGH): SpecDerived fields stale after merge — data shows as "unknown" in rendered HTML
- C21-02 (MEDIUM): Sony RX/DSC/HX/WX/TX/QX naming — same .title() issue as C20-02 but broader
- C21-03 (LOW): mpix not preserved by merge when new data has None
- C21-04 (MEDIUM): test_merge_field_preservation doesn't verify SpecDerived fields
