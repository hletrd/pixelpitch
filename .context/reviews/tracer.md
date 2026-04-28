# Tracer Review (Cycle 21) — Causal Tracing of Suspicious Flows

**Reviewer:** tracer
**Date:** 2026-04-28

## T21-01: Data flow from merge preservation to template rendering — stale SpecDerived fields traced

**Trace:**
1. `merge_camera_data()` preserves `spec.type` from existing -> `new_spec.spec.type = existing_spec.spec.type`
2. `merge_camera_data()` preserves `spec.size` from existing -> `new_spec.spec.size = existing_spec.spec.size`
3. `merge_camera_data()` preserves `spec.pitch` from existing -> `new_spec.spec.pitch = existing_spec.spec.pitch`
4. BUT: `new_spec.size` (SpecDerived) remains None
5. BUT: `new_spec.area` (SpecDerived) remains None
6. BUT: `new_spec.pitch` (SpecDerived) remains None
7. Template accesses `spec.size` (SpecDerived) -> None -> "unknown"
8. Template accesses `spec.pitch` (SpecDerived) -> None -> "unknown"
9. Template accesses `spec.spec.mpix` (Spec) -> None if not preserved -> "unknown"

**Root cause:** The C20-03 fix preserved fields at the Spec level but ignored the SpecDerived level. The data model has two layers: Spec (input) and SpecDerived (computed). The merge function must preserve both.

**Fix point:** Add SpecDerived field preservation after Spec field preservation in `merge_camera_data`.

---

## T21-02: Sony RX naming corruption flow traced

**Trace:**
1. IR spec page: `_parse_camera_name({'Model Name': 'Sony RX100 VII'}, url)`
2. Name starts with "Sony" -> enters Sony-specific branch
3. URL slug extracted: "sony-rx100-vii" -> `.replace("-", " ")` -> "sony rx100 vii"
4. `.title()` -> "Sony Rx100 Vii" (lowercase 'x' capitalized, 'ii' -> 'Ii')
5. Roman numeral normalizer: "Ii" -> "II" -> "Sony Rx100 Vii" -> "Sony Rx100 VII"
6. FX normalization: no match
7. No RX normalization -> returns "Sony Rx100 VII" (wrong 'x' case)

**Root cause:** `.title()` capitalizes every word-initial letter including 'x' in 'rx'. Only FX was normalized, not RX/DSC/HX/WX/TX/QX.

**Fix point:** Add RX/DSC/HX/WX/TX/QX normalizations after the existing FX normalization.

---

## Summary

- T21-01: SpecDerived stale fields — full data flow traced, fix at merge_camera_data
- T21-02: Sony RX naming — same pattern as FX, fix at _parse_camera_name
