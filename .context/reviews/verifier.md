# Verifier Review (Cycle 21) — Evidence-Based Correctness Check

**Reviewer:** verifier
**Date:** 2026-04-28

## V21-01: SpecDerived stale fields after merge — verified with live data
**Severity:** HIGH | **Confidence:** HIGH (reproduced with live CSV)

Verified against the actual `dist/camera-data.csv` (1742 records). 532 cameras (30.5%) have no size and 567 (32.5%) have no pitch. When these cameras are merged with new data that also has missing fields, the C20-03 fix preserves `Spec` fields but the `SpecDerived` fields remain None, causing "unknown" display in the rendered HTML.

**Evidence:**
```
spec_new = Spec(name='Test Cam', category='fixed', type='1/2.3', size=None, pitch=None, mpix=10.0, year=2020)
derived_new = pp.derive_spec(spec_new)
# derived_new.size=None, derived_new.area=None, derived_new.pitch=None

spec_existing = Spec(name='Test Cam', category='fixed', type='1/2.3', size=(5.0, 3.7), pitch=2.0, mpix=10.0, year=2020)
derived_existing = pp.derive_spec(spec_existing)
# derived_existing.size=(5.0, 3.7), derived_existing.area=18.5, derived_existing.pitch=2.0

merged = pp.merge_camera_data([derived_new], [derived_existing])
# merged[0].spec.size=(5.0, 3.7) — preserved (Spec level)
# merged[0].size=None — NOT preserved (SpecDerived level)
# merged[0].spec.pitch=2.0 — preserved (Spec level)
# merged[0].pitch=None — NOT preserved (SpecDerived level)
```

The template accesses `spec.size` (SpecDerived) and `spec.pitch` (SpecDerived), not `spec.spec.size` (Spec) or `spec.spec.pitch` (Spec). So the preserved values are invisible to users.

---

## V21-02: Sony RX naming — verified misnaming
**Severity:** MEDIUM | **Confidence:** HIGH (reproduced)

Tested: `_parse_camera_name({'Model Name': 'Sony RX100 VII'}, url)` returns `'Sony Rx100 Vii'` instead of `'Sony RX100 VII'`. The `.title()` method capitalizes only the first letter of each word, converting 'rx' to 'Rx' instead of 'RX'.

**Evidence:** Direct Python execution confirms misnaming for RX100, RX10, RX0, RX1, HX400, WX350, TX30, QX1, DSC-RX100.

---

## V21-03: mpix not preserved — verified data loss
**Severity:** LOW | **Confidence:** HIGH (reproduced)

Tested: Merging new spec with `mpix=None` against existing spec with `mpix=33.0` loses the megapixel count. The camera shows "unknown" resolution in the rendered HTML.

**Evidence:**
```
spec_new = Spec(name='Test', category='mirrorless', type=None, size=(35.9, 23.9), pitch=5.12, mpix=None, year=2020)
merged = pp.merge_camera_data([pp.derive_spec(spec_new)], [derived_existing])
# merged[0].spec.mpix = None (not preserved)
```

---

## Summary

- V21-01 (HIGH): SpecDerived stale fields after merge — verified, template shows "unknown"
- V21-02 (MEDIUM): Sony RX/DSC/HX/WX/TX/QX misnaming — verified
- V21-03 (LOW): mpix not preserved in merge — verified
