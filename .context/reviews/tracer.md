# Tracer — Cycle 49

**Date:** 2026-04-29

## Causal traces

### Trace A: A new flake8 violation reaches master

Hypothesis: Without CI gate, a flake8 violation in PR merges to master.

Evidence chain:
1. Local developer runs flake8 — clean (cycle 48 baseline).
2. PR adds a new module with unused imports (F401).
3. CI runs `python -m tests.test_parsers_offline` — passes (no lint check).
4. PR merges. Master accumulates the violation.
5. Monthly scheduled run deploys without failure.
6. Next review-plan-fix cycle discovers the regression.

Conclusion: F49-08 is real and reachable.

### Trace B: matched_sensors preservation across CSV round-trip

Verified post-C46-01 fix correctly preserves matched_sensors when sensors_db is unavailable. No regression.

## Summary

Single actionable causal finding: F49-08 (gate enforcement gap).
