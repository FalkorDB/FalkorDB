---
name: test
description: Run FalkorDB's test suites - Rust unit tests, Python e2e/function/MVCC/concurrency tests, openCypher TCK compliance tests, and flow tests. Use when asked to run, narrow down, or debug a failing test, or to decide which suite covers a change.
allowed-tools: Bash
---

# Test

All Python suites require the virtualenv. Locally it's `./venv` (create once
with `python3 -m venv venv && pip install -r tests/requirements.txt`); inside
the devcontainer/CI it lives at `/data/venv`. The activation lines below fall
back between the two so they work in either environment. The module must be
built first (`cargo build`, see the `build` skill) for anything that loads
`libfalkordb.{so,dylib}`.

## Rust unit tests (default choice for Rust-only changes)

```bash
cargo test -p graph
```

## Python e2e / function / MVCC / concurrency tests

```bash
source venv/bin/activate 2>/dev/null || source /data/venv/bin/activate
pytest tests/test_e2e.py tests/test_functions.py -vv       # e2e + function tests
pytest tests/test_mvcc.py tests/test_concurrency.py -vv     # MVCC + concurrency tests
```
Run a single file/suite by narrowing the `pytest` args, e.g.
`pytest tests/test_mvcc.py -vv`.

## TCK tests (openCypher language-compliance suite)

```bash
source venv/bin/activate 2>/dev/null || source /data/venv/bin/activate
TCK_DONE=tck_done.txt pytest tests/tck/test_tck.py -s              # all currently-passing scenarios
TCK_INCLUDE=<path> pytest tests/tck/test_tck.py -s                 # only scenarios under <path>
```
- `tck_done.txt` is the checked-in allow-list of scenarios expected to pass;
  a scenario newly failing there is a regression, one newly passing should be
  added to the file.
- Common `TCK_INCLUDE` paths: `tests/tck/features/expressions/list`,
  `tests/tck/features/expressions/map`, `tests/tck/features/clauses/match`,
  `tests/tck/features/clauses/return`.

## Flow tests (query flows against a running module instance)

```bash
cargo build        # debug build required first (flow.sh defaults to target/debug)
./flow.sh
```
`flow.sh` supports env vars for narrowing/debugging a run:
- `TEST="tests/flow/test_function_calls" ./flow.sh` — one file
- `TEST="tests/flow/test_function_calls:testFunctionCallsFlow.test89_JOIN" ./flow.sh` — one test
- `FAIL_FAST=1` — stop on first failure, run serially (`--parallelism 1`)
- `RELEASE=1` — run against `target/release` instead of `target/debug`
- Progress is tracked in `flow_tests_done.txt` / `flow_tests_todo.txt`; logs
  land in `tests/flow/logs`.

## Notes

- CI (`rust-pr.yml`) runs all of the above per PR/push, plus fuzzing and
  coverage — see the `fuzz`, `profile`, and `coverage` skills for those.
- Some flow tests against the replica (`-svc`) are known-flaky under
  ASAN/coverage instrumentation due to a pre-existing concurrency race
  (`atomic_refcell` "already mutably borrowed"); if only those fail under
  those configurations, re-run before treating it as a regression.

If tests fail, analyze the output and help the user understand what went wrong.
