# Plan C48-01: Flake8 Gate Cleanup

**Status:** completed
**Priority:** P1 (gate-blocking)
**Findings addressed:** F48-01, F48-02, F48-03 (cross-agent consensus from cycle 48 aggregate)

## Repo Rules Consulted

Read in this order before planning:
- `CLAUDE.md` (none at repo root)
- `.context/reviews/_aggregate.md` (cycle 48)
- `setup.cfg` — flake8 max-line-length=160, no other rule excludes
- Cycle GATES: `flake8` and `python -m tests.test_parsers_offline`

The repo has no rule explicitly authorizing blanket `noqa` suppression. Therefore:
- Real lint issues (unused imports, dead variable, f-string typo, indentation, whitespace, blank-line spacing) → fix at root.
- E402 on `sys.path.insert` followed by local imports → suppress with targeted `# noqa: E402` (project-pattern is to keep `sys.path` manipulation at the top of the script for self-contained execution).

## Problem

Flake8 reports 33 errors across the repo:

| Code | Count | Description |
|------|-------|-------------|
| F401 | 11 | Unused imports |
| F541 | 1 | f-string missing placeholders |
| F811 | 1 | Redefinition of unused `io` |
| F841 | 1 | Unused local `merged2` |
| E127 | 9 | Continuation line over-indented |
| E231 | 3 | Missing whitespace after `,`/`:` |
| E302 | 1 | Expected 2 blank lines |
| E303 | 1 | Too many blank lines |
| E402 | 5 | Module-level import not at top |

The orchestrator GATES require flake8 to pass. Test gate already passes.

## Implementation Steps

### Step 1: Fix unused imports (F401, F811)
- [x] `pixelpitch.py:17` — drop unused `from dataclasses import dataclass`
- [x] `sources/__init__.py:30` — drop unused `from dataclasses import dataclass`
- [x] `tests/test_parsers_offline.py:17` — drop unused top-level `import io` (function-local re-import at line 1241 stays)
- [x] `tests/test_parsers_offline.py:594,666,845,977,1244,1822,1856` — drop unused `models.SpecDerived` / `models.Spec` re-imports inside test functions

### Step 2: Fix dead local (F841)
- [x] `tests/test_parsers_offline.py:1271` — investigate; if the test intended an assertion on `merged2`, add it; otherwise drop the assignment.

### Step 3: Fix f-string typo (F541)
- [x] `pixelpitch.py:1240` — convert `f""` to `""` if no placeholder is wanted, or add the missing `{}` substitution.

### Step 4: Fix whitespace and indentation (E127, E231, E302, E303)
- [x] `sources/apotelyt.py:37` — collapse extra blank line (3 → 2)
- [x] `sources/cined.py:36` — add the missing blank line (1 → 2)
- [x] `tests/test_parsers_offline.py:392` — add whitespace after `,`
- [x] `tests/test_sources.py:32` — add whitespace after `:` and `,`
- [x] `tests/test_parsers_offline.py:848,853,1232,1497,1550,1559,1568,1577,1586` — fix continuation indentation (E127). Reformat the affected multi-line expressions so continuation lines align with opening bracket or use hanging indent.

### Step 5: Suppress E402 only where the `sys.path` shim is intentional
- [x] `pixelpitch.py:47` — add `# noqa: E402` to the line(s) imported after `sys.path.insert`
- [x] `tests/test_parsers_offline.py:25` — same
- [x] `tests/test_sources.py:23-25` — same

### Step 6: Run full gate suite
- [x] `python3 -m flake8 . --exclude=.git,__pycache__,dist,downloaded_files,.context,.omc,templates` — must report 0 errors.
- [x] `python3 -m tests.test_parsers_offline` — must still pass.

### Step 7: Commit each logical group separately

GPG-signed commits in conventional+gitmoji format, fine-grained:
- [x] `chore(lint): 🧹 drop unused dataclass and io imports`
- [x] `chore(lint): 🧹 drop unused models.SpecDerived/Spec test imports`
- [x] `chore(lint): 🧹 fix whitespace and blank-line lint errors`
- [x] `chore(lint): 🧹 fix continuation indentation E127`
- [x] `fix(test): 🐛 add missing assertion or drop dead merged2 local`
- [x] `chore(lint): 🧹 convert empty f-string to plain string`
- [x] `chore(lint): 🧹 mark sys.path-shim imports with noqa: E402`

## Exit Criteria

- [x] `flake8` reports 0 errors with the cycle's exclude list.
- [x] `python3 -m tests.test_parsers_offline` continues to pass.
- [x] No `# noqa` suppressions added except for the documented E402 cases.
- [x] All commits GPG-signed, conventional+gitmoji, pushed.

## Risk Assessment

LOW. Pure lint cleanup with no behavior change. The single risk is the F841 `merged2` resolution: if the test intended an assertion, removing the variable would lose that intent. Step 2 explicitly inspects the surrounding code before deciding.
