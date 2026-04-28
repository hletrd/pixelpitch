# Plan: Cycle 30 Findings — GSMArena Fetch Resilience & deduplicate_specs DRY

**Created:** 2026-04-28
**Status:** COMPLETED
**Source Reviews:** CR30-01, CRIT30-01, TR30-01, ARCH30-01, DBG30-01, TE30-01, CR30-02, CRIT30-02, ARCH30-02

---

## Task 1: Add per-phone try/except to GSMArena fetch() — C30-01

**Finding:** C30-01 (6-agent consensus: code-reviewer, critic, tracer, architect, debugger, test-engineer)
**Severity:** MEDIUM | **Confidence:** HIGH
**Files:** `sources/gsmarena.py`

### Problem

The C29-02 fix added per-camera try/except to `imaging_resource.fetch()` and `apotelyt.fetch()`, but `gsmarena.fetch()` was missed. If `fetch_phone()` raises an unhandled exception (e.g., from unexpected HTML structure), the entire GSMArena scrape aborts. The CineD, IR, and Apotelyt `fetch()` functions all have per-camera try/except.

### Implementation

1. In `sources/gsmarena.py` lines 246-252, wrap `fetch_phone(s)` in try/except, logging the error and continuing.

### Implementation

1. In `sources/gsmarena.py` lines 246-252, wrapped `fetch_phone(s)` in try/except, logging the error and continuing.

### Verification — DONE

- Gate tests (`python -m tests.test_parsers_offline`) — all checks passed
- Commit: 57362f9

**Finding:** C30-02 (3-agent consensus: code-reviewer, critic, architect)
**Severity:** LOW | **Confidence:** HIGH
**Files:** `pixelpitch.py`

### Problem

The `deduplicate_specs()` function creates new Spec objects field-by-field in two places (color-variant unification and remove_parens). If Spec gains a new field, these reconstructions would silently drop it. The C29-04 fix addressed the same pattern in `digicamdb.py`.

### Implementation

1. Changed `from dataclasses import dataclass` to `from dataclasses import dataclass, replace`
2. Replaced `Spec(unified_name, ref.category, ...)` with `replace(ref, name=unified_name, year=year)` in color-variant unification
3. Replaced `Spec(name, spec.category, ...)` with `replace(spec, name=name)` in remove_parens

### Verification — DONE

- Gate tests (`python -m tests.test_parsers_offline`) — all checks passed
- `test_deduplicate_specs` tests all pass unchanged
- Commit: 48f3e77

---

## Deferred Findings

None. All findings are scheduled for implementation.
