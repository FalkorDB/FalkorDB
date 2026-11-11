---
name: profile
description: Profile FalkorDB performance - generate samply flamegraphs and run the benchmark suite. Use when investigating a performance regression/bottleneck or comparing benchmark results. For code-coverage reports use the coverage skill.
allowed-tools: Bash
---

# Profile

## 1. Flamegraphs with samply

For finding performance bottlenecks in query execution (full reference:
`docs/profiling.md`):

```bash
cargo install --locked samply                     # one-time
samply setup                                      # one-time, macOS: lets samply attach
# [profile.release] has no debuginfo, so a plain --release build has no line
# info. [profile.profiling] is release plus symbols:
cargo build --profile profiling

# record redis-server with the module loaded, run your query/workload
# against it, then stop the server
samply record redis-server --loadmodule ./target/profiling/libfalkordb.dylib
```
On stop, samply opens the Firefox Profiler UI in your browser. Use
`libfalkordb.so` on Linux.

Don't add debuginfo with `RUSTFLAGS="-C debuginfo=2"`: on Linux, `RUSTFLAGS`
replaces the link flag in `.cargo/config.toml` instead of adding to it, and
the link fails. Without `samply setup`, `samply record -p` (what `bench
profile` runs) waits forever on macOS instead of failing.

## 2. Benchmark suite

Use the per-query harness in `bench/` — see the **`bench`** skill for the full
loop and `bench/README.md` for the details. In short:

```bash
cargo build --release
uv sync --project bench                        # once
uv run --project bench bench measure           # every query -> bench/results/current.csv
uv run --project bench bench compare           # regression gate vs your own baseline
```

To profile one benchmark query rather than a workload you drive by hand,
`bench profile "<query name>"` records the server with samply while that query
runs in a loop — it reuses the harness's own server, so you name the query once.
It loads `target/release/` unless you pass `--module
target/profiling/libfalkordb.dylib`, so pass that for named frames.

In CI, labelling a PR **`benchmark-cov`** runs the same harness against the PR
and its base and posts a per-query comparison, plus deterministic callgrind
instruction counts (`.github/workflows/benchmark-cov.yml`). Those counts are
PR-vs-base only — the C engine cannot be measured under callgrind, because it
busy-waits on a worker thread valgrind schedules arbitrarily; the vs-C comparison
runs on allocated bytes instead. It is on-demand only, so a suspected regression
is compared against the PR base rather than against a stored series.

Treat samply flamegraphs above as the primary tool for *locating* a bottleneck
and `bench/` for *quantifying* it.

## Notes

- For line/region/function code-coverage reports (LLVM
  `instrument-coverage`, `lcov`, devcontainer/docker flow) see the
  `coverage` skill.
