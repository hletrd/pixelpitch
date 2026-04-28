# Plan C49-01: Add flake8 Gate to CI Workflow

**Status:** completed
**Priority:** P1 (gate-enforcement gap)
**Findings addressed:** F49-01 (cross-agent consensus from cycle 49 aggregate, flagged by 7 agents)

## Repo Rules Consulted

Read in order before planning:
- `CLAUDE.md` (none at repo root; user-global instructions only)
- `.context/reviews/_aggregate.md` (cycle 49)
- `setup.cfg` — `[flake8] max-line-length = 160`
- `.github/workflows/github-pages.yml` — current CI pipeline
- Cycle GATES: `flake8` and `python -m tests.test_parsers_offline`

The repo has no rule prohibiting CI changes. Adding a flake8 step is purely additive (no behavior change for non-lint code paths).

## Problem

The orchestrator declares two GATES:
1. `flake8` (config in `setup.cfg`, max-line-length 160)
2. `python -m tests.test_parsers_offline`

The CI workflow (`.github/workflows/github-pages.yml`) only runs gate #2. Cycle 48 spent meaningful effort fixing 33 flake8 errors at root, but without CI enforcement that work has a half-life of one merge cycle. F49-01 was flagged by 7 of 11 reviewers (code-reviewer, critic, verifier, test-engineer, architect, tracer, debugger) — strongest cross-agent consensus this cycle.

## Implementation Steps

### Step 1: Add a flake8 step to the CI workflow

After the existing "Run offline tests" step, add a "Run flake8 lint" step that runs the same command and exclusion list the orchestrator uses:

```yaml
- name: Run flake8 lint
  run: |
    . .venv/bin/activate
    pip install flake8
    python -m flake8 . --exclude=.git,__pycache__,dist,downloaded_files,.context,.omc,templates
```

Notes:
- Place AFTER offline tests so test failures (cheaper to debug) surface first.
- Install flake8 inline rather than adding it to `requirements.txt` (it's a dev tool, not a runtime dep). If `requirements.txt` already lists flake8, drop the inline install.
- The exclude list mirrors the orchestrator's local invocation for parity.

### Step 2: Verify

- [x] Confirm the workflow YAML still parses (no indentation errors). — `python3 -c "import yaml; yaml.safe_load(...)"` PASS.
- [x] Confirm `python3 -m flake8 . --exclude=...` returns 0 errors locally before commit.
- [x] Confirm test gate still passes locally.

### Step 3: Commit

GPG-signed commit, conventional + gitmoji:
- `ci(workflow): 👷 enforce flake8 gate in github-pages workflow`

Body:
```
Add a flake8 step to the github-pages CI workflow so the lint gate is
enforced on master rather than only locally during review cycles.

Closes F49-01 (cycle 49 aggregate, 7-agent consensus).
```

## Exit Criteria

- [x] CI workflow contains a flake8 step.
- [x] Local `python3 -m flake8 . --exclude=...` reports 0 errors.
- [x] Local `python3 -m tests.test_parsers_offline` reports PASS.
- [x] Workflow YAML is valid (PyYAML parse PASS).
- [x] Commit is GPG-signed and follows conventional commits + gitmoji.

## Out of Scope

- F49-02 (`git pull --rebase || true`) — separate plan if pursued; LOW impact on idempotent monthly workflow.
- F49-04 (sensor-DB lookup index) — performance polish; defer to future cycle.
