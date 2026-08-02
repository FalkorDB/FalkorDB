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

There is **no benchmark or profiling CI in this repository** — the A/B
pipeline, the flame-graph workflow and their `gh-pages` dashboards were
removed, because they depended on a `gh-pages` branch and GCE-runner
variables that do not exist here. So there is no trend history to compare a
suspected regression against.

To measure a change, run the [`FalkorDB/benchmark`](https://github.com/FalkorDB/benchmark)
tool locally against two containers (before/after, or vs the C engine's
`docker.io/falkordb/falkordb-server:edge`), and treat samply flamegraphs
above as the primary tool for locating a bottleneck.

## Notes

- For line/region/function code-coverage reports (LLVM
  `instrument-coverage`, `lcov`, devcontainer/docker flow) see the
  `coverage` skill.
