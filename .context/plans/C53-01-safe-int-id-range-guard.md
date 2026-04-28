# C53-01: `_safe_int_id` range guard + tolerance test extension

**Cycle:** 53
**Status:** COMPLETED
**Findings addressed:** F53-01, F53-02, F53-DOC-01

## Implementation

- `252f959` fix(csv): 🛡️ add range guard to _safe_int_id (reject 1e308 big-ints)
- `cfb72c3` test(csv): ✅ extend id parse-tolerance with nan/inf/1e308/negative/range rows

Both gates pass at `cfb72c3`:
- `flake8 .` → 0 errors.
- `python3 -m tests.test_parsers_offline` → all sections green
  (10 id-tolerance rows, all expected ids match).

## Background

Cycle 52 introduced `_safe_year` and `_safe_int_id` as
parse-tolerance helpers in `pixelpitch.py`. The two helpers were
intended to be symmetric (both defending the CSV round-trip against
Excel hand-edits in `dist/camera-data.csv`), but the implementation
diverged:

- `_safe_year` has both an `isfinite` check AND a 1900-2100 range
  guard. It correctly rejects `"3000"`, `"1e308"`, `"nan"`, `"inf"`.
- `_safe_int_id` has only the `isfinite` check. It rejects `"nan"`
  and `"inf"` but accepts `"1e308"`, returning a 309-digit Python
  big-int because `int(float("1e308"))` is finite (largest finite
  IEEE 754 double ≈1.797e308).

The 309-digit id propagates through `merge_camera_data` until
`main()` reassigns sequential ids before `write_csv`. The committed
CSV is therefore safe, but the original id-to-row mapping for that
row is permanently lost, and any code reading `spec.id` between
parse and reassignment sees garbage.

The `_safe_int_id` docstring (line 324) claims "Same Excel-hand-edit
class as `_safe_year`", which is false in the absence of a range
guard. Doc/code mismatch.

## Repo policy applied

Per `~/.claude/CLAUDE.md` (no project-level CLAUDE.md):

- All commits GPG-signed (`git commit -S`).
- Conventional Commits + gitmoji.
- One commit per fine-grained change.
- `git pull --rebase` before push.
- No `--no-verify`, no Co-Authored-By.
- Run gates (`flake8 .`, `python3 -m tests.test_parsers_offline`)
  before push.

## Plan

### Step 1 — Refresh per-agent reviews and aggregate

DONE in this cycle's review pass. The 11 review files and
`_aggregate.md` have been refreshed for cycle 53.

### Step 2 — Implement F53-01 + F53-DOC-01

**File:** `pixelpitch.py`, lines 318-337.

Add a post-conversion range guard to `_safe_int_id`:

```python
def _safe_int_id(s: str) -> Optional[int]:
    """Parse a record_id string tolerantly, returning None for invalid values.

    Accepts ``"5"``, ``"5.0"``, and ``" 5 "``. Returns None on any
    parse failure rather than raising — callers (parse_existing_csv)
    already drop ids on error, but raising would skip the entire row
    via the broad except. Same Excel-hand-edit class as ``_safe_year``;
    rejects non-finite floats and out-of-range integers (anything
    outside ``[0, 1_000_000]``) so an Excel-coerced ``"1.0E+308"``
    cannot propagate a 309-digit big-int through merge_camera_data.
    """
    if not s:
        return None
    try:
        n = int(s)
    except (ValueError, TypeError):
        try:
            f = float(s)
        except (ValueError, TypeError):
            return None
        if not isfinite(f):
            return None
        n = int(f)
    if 0 <= n <= 1_000_000:
        return n
    return None
```

The upper bound `1_000_000` is comfortably above any plausible
sequential id (current count ~1000, sensors_db ~200) while small
enough to reject scientific-notation Excel coercions. Negative ids
are also rejected (sequential reassignment never produces them).

### Step 3 — Extend tolerance tests (F53-02)

**File:** `tests/test_parsers_offline.py`.

Extend the existing year-tolerance and id-tolerance sections with
new rows asserting `None` for `"nan"`, `"inf"`, `"-inf"`, `"1e308"`.

Year tolerance section: add 4 assertions.
Id tolerance section: add 4 assertions (note: existing test only
covers `"5"`, `"5.0"`, garbage, empty — gap is exactly the
scientific-notation edge that would now be rejected by the
range guard).

### Step 4 — Run gates and commit

- `flake8 .` → must exit 0.
- `python3 -m tests.test_parsers_offline` → must exit 0 with all
  sections green.

Commits (all GPG-signed, fine-grained):

1. `fix(csv): 🛡️ add range guard to _safe_int_id (reject "1e308" big-ints)`
   - Modifies `pixelpitch.py` only.
2. `test(csv): ✅ extend year/id parse-tolerance with nan/inf/1e308 rows`
   - Modifies `tests/test_parsers_offline.py` only.
3. `docs(reviews): 📝 add cycle 53 reviews and plan for safe_int_id range guard`
   - Adds 11 review files + `_aggregate.md` + this plan.

### Exit criteria

- Both gates pass at HEAD.
- `_safe_int_id("1e308")` returns None.
- `_safe_int_id("5")` returns 5.
- `_safe_int_id("5.0")` returns 5.
- `_safe_int_id("-3")` returns None (was -3 before; sequential ids
  never go negative, so this tightening is safe).
- Plan archived (status COMPLETED) only after every commit lands.

## Deferred items recorded for cycle 53

No new deferrals. F53-03 (test assert messages do not encode
rejection reason) is a recommendation, not a finding — it does not
require a deferral entry.

The cycle 51/52 deferred queue (`.context/plans/deferred.md`)
remains unchanged. Re-validation of those entries remains a future
cycle's process task (F51-03).
