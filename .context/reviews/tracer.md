# Tracer Review (Cycle 59, orchestrator cycle 12)

**Date:** 2026-04-29
**HEAD:** `fa0ae66`

## Causal trace: where can a non-finite/non-positive size reach `write_csv`?

Producers of `SpecDerived.size`:

1. `derive_spec` (line 900-906): explicitly guards
   `isfinite(size[0]) and isfinite(size[1]) and size[0] > 0 and
   size[1] > 0`. Sets size = None if not.
2. `parse_existing_csv` (line 430-435): rejects width/height
   <= 0 by setting them to None. Note: also calls `_safe_float`
   which already rejects non-finite. So `size` after parse is
   either None or `(positive finite, positive finite)`.
3. `merge_camera_data` (line 569): `new_spec.size =
   existing_spec.size`. Inherits whatever invariants the input
   already satisfies.

Consumers:

- `write_csv` (line 1018-1019): `if derived.size: ...`. Only
  truthy-check; no element-level isfinite/positive guard.

## Hypothesis 1: are there any code paths that construct SpecDerived directly?

Searched the codebase for `SpecDerived(...)` construction sites:

- `derive_spec` (line 929-930): the canonical path; guarded.
- `parse_existing_csv` (line 470-472): guarded by upstream
  width/height guards.
- Tests: a few synthetic SpecDerived constructions (e.g.,
  test_write_csv_*).

No production code path constructs SpecDerived(size=non-finite).
**Conclusion:** F59-CR-01 is a defensive-parity gap, not a
live bug. Hardening it is a hygiene improvement.

## Hypothesis 2: does the upstream guard cover all cases?

`derive_spec` line 900 uses `isfinite(size[0]) and isfinite(size[1])
and size[0] > 0 and size[1] > 0`. This covers:
- inf, -inf -> rejected
- nan -> rejected
- 0.0 -> rejected
- negative -> rejected

Comprehensive. The single-point-of-enforcement is fine *for
the current code*. F59-CR-01 hardens the symmetry at the write
boundary, so a future regression (or a new direct-construction
caller) is also caught.

## Cross-agent agreement

F59-01 is flagged by code-reviewer (CR-01), critic (CRIT-01),
verifier (reproduced), test-engineer (TE-01 paired test gap),
and debugger (D-01). Five-reviewer agreement - high signal for
a LOW-severity finding.

## Carry-over

No new findings; all previous tracer findings still resolved
or deferred per repo policy.
