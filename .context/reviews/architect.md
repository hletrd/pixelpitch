# Architect — Cycle 49

**Date:** 2026-04-29

## Architectural state

- Single 1.3K-line orchestrator (`pixelpitch.py`) — CLI, scrape coordination, merge engine, CSV I/O, HTML render.
- Per-source modules in `sources/` follow uniform `fetch(limit=...) -> list[Spec]` contract.
- Shared regex/helpers exposed from `sources/__init__.py`.

## Findings

### F49-11: CI quality-gate definition does not match orchestrator GATES (MEDIUM / HIGH)
- **Detail:** Orchestrator declares `GATES: flake8 + tests.test_parsers_offline`. CI declares only the test gate. Architectural inconsistency that erodes CI's role as the source of truth for "does master build?"
- **Architectural fix:** Bring CI in line with orchestrator GATES (add flake8 step).
- **Confidence:** HIGH

### Carried-forward architectural items

F31, F32, C22-05 remain validly deferred per documented exit criteria.

## Summary

Architecture stable. Single new finding: process gap between local and remote gates (F49-11, same surface as F49-06/F49-08).
