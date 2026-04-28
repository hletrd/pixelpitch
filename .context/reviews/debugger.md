# Debugger Review (Cycle 20) — Latent Bugs, Failure Modes, Regressions

**Reviewer:** debugger
**Date:** 2026-04-28

## D20-01: `pixel_pitch` ZeroDivisionError — confirmed latent bug
**Severity:** MEDIUM | **Confidence:** HIGH (reproduced)

Reproduced: `pixel_pitch(864.0, 0.0)` raises ZeroDivisionError. `pixel_pitch(864.0, -1.0)` raises ValueError. The `derive_spec` function has no try/except guard.

**Failure mode:** If any source HTML contains "0.0 Megapixels" or a negative number, the entire render crashes. The CI workflow has `continue-on-error: true` for individual source fetches, but the render step (`python pixelpitch.py`) does NOT have this protection. A single bad spec crashes the whole deployment.

---

## D20-02: Sony FX naming — confirmed data quality bug
**Severity:** MEDIUM | **Confidence:** HIGH (reproduced)

Reproduced: `_parse_camera_name({'Model Name': 'Sony FX3'}, url)` returns "Sony Fx3". The `.title()` method capitalizes 'x' in 'fx'.

---

## Summary

- D20-01 (MEDIUM): pixel_pitch crash — latent bug, will trigger on bad source data
- D20-02 (MEDIUM): Sony FX naming — data quality bug
