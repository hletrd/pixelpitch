# Critic Review (Cycle 21) — Multi-Perspective Critique

**Reviewer:** critic
**Date:** 2026-04-28

## C21-CR01: C20-03 fix is fundamentally broken — SpecDerived fields are stale

**Severity:** HIGH | **Confidence:** HIGH

The C20-03 fix added field preservation for `type`, `size`, and `pitch` at the `Spec` level. But this is the WRONG layer. The Jinja2 template reads from `SpecDerived` attributes (`spec.size`, `spec.pitch`, `spec.area`), not from `Spec` attributes (`spec.spec.size`, `spec.spec.pitch`). The preserved values exist in the data structure but are invisible to users.

This means the C20-03 fix was essentially a no-op for the rendered output. The data preservation it claims to implement does not actually work. Any camera that relies on field preservation from existing data will still show "unknown" for sensor size and pixel pitch.

**Amplification:** This affects 30.5% of cameras (532 with no size) and 32.5% (567 with no pitch) in the current dataset. These cameras are being shown as "unknown" when they could display the preserved data.

---

## C21-CR02: Sony FX fix is incomplete — all Sony uppercase series are affected

**Severity:** MEDIUM | **Confidence:** HIGH

The C20-02 fix added `re.sub(r'\bFx(\d)', r'FX\1', cleaned)` for FX series cameras. But the same `.title()` issue affects every Sony camera series with a two-letter uppercase prefix: RX, HX, WX, TX, QX, and DSC. The FX fix was a band-aid for a systemic issue with `.title()` and Sony naming conventions.

**Amplification:** If IR has review pages for any RX-series cameras (RX100, RX10, etc.), those cameras are misnamed and may create duplicate entries when merged with data from Apotelyt or GSMArena.

---

## C21-CR03: Incomplete field preservation — mpix is missing

**Severity:** LOW | **Confidence:** HIGH

The merge function preserves `type`, `size`, `pitch`, and `year` but NOT `mpix`. This is an inconsistent design choice — why preserve some fields and not others? The original year-preservation logic was added for a specific reason (different sources report different years). The same reasoning applies to mpix (different sources may or may not report effective megapixels).

---

## Summary

- C21-CR01 (HIGH): C20-03 SpecDerived stale fields — fix is a no-op for rendered output
- C21-CR02 (MEDIUM): Sony FX fix incomplete — all uppercase series affected
- C21-CR03 (LOW): mpix not preserved — inconsistent with other field preservation
