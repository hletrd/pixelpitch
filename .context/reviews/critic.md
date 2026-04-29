# Critic Review (Cycle 59, orchestrator cycle 12)

**Date:** 2026-04-29
**HEAD:** `fa0ae66`

## Posture

After 58 cycles of focused hardening, the repo is in a mature
state. Both gates pass; the architecture is intentional and
stable. This cycle's critic surface is narrow.

## New findings

### F59-CRIT-01 (defensive-parity gap, LOW): width/height write asymmetric vs. area/mpix/pitch

- **File:** `pixelpitch.py:1018-1019`
- **Cross-references:** F59-CR-01 (code-reviewer flagged the same
  defensive-parity concern from a SOLID/maintenance angle;
  critic flags it from the "what does the artifact contract
  say?" angle).
- **Detail:** The CSV artifact is the durable interface between
  consecutive builds. Every other numeric float field
  (`area`, `mpix`, `pitch`) explicitly hardens against
  `inf`/`nan`/`<= 0` writes. The width/height columns rely on
  upstream guards (in `derive_spec` and `parse_existing_csv`).
  The contract is "the CSV will never contain non-finite or
  non-positive numeric values for sensor dimensions" - but the
  enforcement of that contract lives in two places that can
  drift, instead of being co-located at the write boundary.
  Hardening at the write boundary makes the contract local
  and self-evident.
- **Severity:** LOW. **Confidence:** HIGH.

### F59-CRIT-02 (carry-over): per-build noise from missing per-source CSVs

- **File:** `pixelpitch.py:1085`
- **Detail:** Echoes F59-CR-02. Same disposition: defer
  (informational).

## Carry-over (still applicable)

- F32 monolith, F58-A-02 argparse drift, F56-A-02 / F57-A-02 /
  F58-A-02 category list duplication - repo policy.
- F55-04, F55-05, F56-DOC-03, F57-DOC-03, F58-DOC-02 - repo
  policy.
- C10-07 redirect SSRF, C10-08 debug port - repo policy.
- F35..F40 UI carry-overs - re-confirmed by designer this cycle.

## Bottom line

One actionable defensive-parity finding (F59-01 = F59-CR-01 =
F59-CRIT-01). All other surface area is already either fixed or
explicitly deferred per repo policy.
