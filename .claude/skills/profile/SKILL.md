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
cargo install samply                              # one-time
# [profile.release] has `debug` commented out in Cargo.toml, so a plain
# --release build has no line info. Add debuginfo for readable flamegraphs:
RUSTFLAGS="-C debuginfo=2" cargo build --release

# record redis-server with the module loaded, run your query/workload
# against it, then stop the server
samply record redis-server --loadmodule ./target/release/libfalkordb.dylib
```
On stop, samply opens the Firefox Profiler UI in your browser. Use
`libfalkordb.so` on Linux.

## 2. Benchmark suite

Use the per-query harness in `bench/` — see the **`bench`** skill for the full
loop and `bench/README.md` for the details. In short:

```bash
cargo build --release
python3 bench/run_bench.py     # 317 queries -> bench/results/current.csv
python3 bench/compare.py       # regression gate vs your own baseline
```

In CI, labelling a PR **`benchmark-cov`** runs the same harness against the PR
and its base and posts a per-query comparison, plus deterministic callgrind
instruction counts against the production C engine
(`.github/workflows/benchmark-cov.yml`). It is on-demand only — there is no
trend history, so a suspected regression is compared against the PR base
rather than against a stored series.

The macro throughput/latency A/B pipeline and its `gh-pages` dashboards were
removed (they depended on a `gh-pages` branch and GCE-runner variables that do
not exist here). For a macro reading, run
[`FalkorDB/benchmark`](https://github.com/FalkorDB/benchmark) locally against
two containers. Treat samply flamegraphs above as the primary tool for
*locating* a bottleneck and `bench/` for *quantifying* it.

## Notes

- For line/region/function code-coverage reports (LLVM
  `instrument-coverage`, `lcov`, devcontainer/docker flow) see the
  `coverage` skill.
