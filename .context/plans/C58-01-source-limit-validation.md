# C58-01: validate `--limit` is a positive integer in `source` CLI

**Cycle:** 58 (orchestrator cycle 11)
**Status:** COMPLETED
**Findings addressed:** F58-01 (bug), F58-02 (help-text doc),
F58-03 (test).

## Implementation summary

- `fix(cli)` (commit `b012ae9`): `main()` now rejects
  `--limit <= 0` with a clear error message and SystemExit(1),
  preventing silent slicing-based truncation/empty-output by
  source consumers.
- `docs(cli)` (commit `ef7177f`): `--help` now documents the
  positive-integer constraint for `--limit`.
- `test(cli)` (commit `f44c17e`): adds a new test section
  `source CLI rejects non-positive --limit` pinning all three
  branches (negative, zero, positive). Uses
  `unittest.mock.patch` to short-circuit `fetch_source` so the
  test stays offline.

Both gates pass post-fix:
- `flake8 .` → 0 errors.
- `python3 -m tests.test_parsers_offline` → all sections green
  including the new `source CLI rejects non-positive --limit`
  section (9 new assertions).

## Background

After C57-01 the codebase is stable; both gates pass at HEAD
`aef726b`. This cycle's review fan-out surfaced one actionable
LOW bug (F58-01) flagged by 5 of 11 reviewers — the
`source` CLI accepts negative or zero `--limit` values
silently, producing empty / truncated CSV output without an
error signal.

## Repro (verifier)

```text
$ python pixelpitch.py source openmvg --limit -1
# silently writes an empty dist/camera-data-openmvg.csv,
# exits 0, no error to stderr.
```

Tracing:
- `apotelyt`/`cined`/`gsmarena` use `urls[:limit]`. With
  `limit=-1` slicing drops the last URL; with `limit=0`
  slicing returns empty.
- `openmvg` uses `if i >= limit: break` which short-circuits
  immediately for `limit <= 0`.

## Plan

### Step 1: validate `--limit` value (F58-01)

In `pixelpitch.py:1393-1399`, after the `int(args[i + 1])`
parse, add a check:

```python
if a == "--limit" and i + 1 < len(args):
    try:
        limit = int(args[i + 1])
    except ValueError:
        print(f"Error: --limit requires an integer, got '{args[i + 1]}'")
        sys.exit(1)
    if limit <= 0:
        print(f"Error: --limit must be a positive integer, got {limit}")
        sys.exit(1)
```

Match the style of the existing ValueError handler.

### Step 2: update `--help` text (F58-02)

In `pixelpitch.py:1422-1426`, update the `source` help
string to mention the positive-integer constraint:

```python
print(
    "  source <name> [--limit N] [--out DIR]\n"
    "                Fetch from an alternative source.\n"
    "                --limit N must be a positive integer.\n"
    f"                Available: {', '.join(sorted(SOURCE_REGISTRY))}"
)
```

### Step 3: add validation test (F58-03)

Add a section to `tests/test_parsers_offline.py`:

```
== source CLI rejects non-positive --limit ==
```

that:
- patches `sys.argv` to `["pixelpitch.py", "source", "openmvg",
  "--limit", "-1"]`, calls `main()`, asserts `SystemExit` with
  non-zero status.
- patches with `--limit 0`, asserts the same.
- patches with `--limit 5` (without actually fetching — mock
  `fetch_source` to a no-op), asserts no `SystemExit`.

The test must not actually invoke source fetching (would be
network-dependent). Use `unittest.mock.patch` on
`pixelpitch.fetch_source` to short-circuit.

### Step 4: re-run gates

- `python3 -m flake8 .` → 0
- `python3 -m tests.test_parsers_offline` → all green
  (including the new section).

### Step 5: commit, then push

Three fine-grained commits:
1. `fix(cli): 🐛 reject non-positive --limit in source command`
2. `docs(cli): 📝 document --limit positive-integer constraint in --help`
3. `test(cli): ✅ pin --limit validation in source command`

All commits GPG-signed via `-S`. `git pull --rebase` before
push.

## Exit criteria

- `--limit -1` and `--limit 0` exit with non-zero status
  and a clear error message.
- `--help` documents the positive-integer constraint.
- Test pins both rejection branches.
- Both gates pass.
- Plan moves to STATUS: COMPLETED.

## Deferred (per F58 deferred-fix rules)

- F58-04 (`--out --limit` typo tolerance) — covered by
  F58-A-02 architectural refactor when accepted.
- F58-05 (argparse migration) — same class as F32 monolith.
- F58-06 (boundary tests at exact range edges) — analogous
  to F55-02 (indirect coverage suffices).
