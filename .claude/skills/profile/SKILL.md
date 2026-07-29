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

CI runs the A/B FalkorDB-vs-FalkorDB benchmark pipeline
(`.github/workflows/_benchmark.yml`, driven by the `FalkorDB/benchmark`
tool) and publishes the trend to
<https://falkordb.github.io/falkordb-rs-next-gen/benchmark/> — compare
against that history when checking for a regression.

## Notes

- For line/region/function code-coverage reports (LLVM
  `instrument-coverage`, `lcov`, devcontainer/docker flow) see the
  `coverage` skill.
