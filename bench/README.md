# bench — per-query performance & coverage loop

Measures, per query:

- **instructions and cycles** of the redis-server process (macOS
  `proc_pid_rusage`, no root; Linux `perf stat -p`, needs PMU access)
- **allocated / deallocated bytes** from `MEMORY MALLOC-STATS` merged-arena
  deltas — works on any jemalloc-built redis, i.e. stock
- **branches, branch-misses, L1D-misses** on Apple Silicon, if `pmc_tool` has
  been built (optional, needs root)
- **wall-clock**, with the caveats in "Reading the output" below

plus deterministic **instruction counts under callgrind** for a curated subset,
and a coverage check that the query set still exercises most of the `graph`
crate.

## Setup

The harness is a [uv](https://docs.astral.sh/uv/) project, separate from the
`tests/` virtualenv, because it also has to run inside a container that ships
nothing but the engine.

```bash
uv sync --project bench          # once
```

Every command below can be run as `uv run --project bench bench …`. From inside
`bench/` it is just `uv run bench …`.

## The loop

```bash
cargo build --release                                   # 1. build
uv run --project bench bench measure                    # 2. measure -> bench/results/current.csv
uv run --project bench bench compare                    # 3. gate vs your baseline (exit 1 on a regression)
uv run --project bench bench coverage                   # 4. verify the query set still covers the code
uv run --project bench bench profile "CASE"             # 5. drill into a hot query
```

After a confirmed improvement, promote the numbers:

```bash
cp bench/results/current.csv bench/baseline/rust.csv    # local only; baseline/ is git-ignored
```

Two things to know about that baseline before trusting a verdict:

- **It is a single file, and switching branches does not change it.** After a
  checkout you are comparing against numbers measured on some other commit,
  silently. Re-measure a baseline when you switch.
- **A baseline from another machine is not comparable at all.** Per-host speed
  differences alone measured 1.46x on byte-identical engines.

Automating both (commit-keyed baselines with recorded provenance) is deliberately
not done yet.

### Subsets and drilling in

```bash
uv run --project bench bench measure "CASE" "WITH pipeline"   # merges into the existing CSV
uv run --project bench bench measure --keep-server "RETURN 1" # leave the server up
uv run --project bench bench profile --reuse "CASE"           # profile against that server
```

`bench measure --c-compat` is required when the module is the C engine: it skips
the setup commands the C engine cannot take (its async validation of a composite
unique constraint crashes it; its RDB round-trip drops numeric 0 from a range
index; UDFs are Rust-only) and skips queries whose warmup reply errors.

### vs the C engine, locally

The C engine is `:edge-c`. **Not `:edge`** — that now resolves to the Rust
engine; `:edge` and `:edge-rs` share a digest.

```bash
docker create --name c falkordb/falkordb-server:edge-c
docker cp c:/var/lib/falkordb/bin/falkordb.so /tmp/falkordb-c.so && docker rm c
# (that copy is Linux — on macOS you need a local `master` build, which lands in bin/)

uv run --project bench bench measure --out /tmp/rust.csv
uv run --project bench bench measure --c-compat --module /tmp/falkordb-c.so --out /tmp/c.csv
uv run --project bench bench compare /tmp/rust.csv /tmp/c.csv
```

### Instruction counts under callgrind

```bash
uv run --project bench bench callgrind --module target/release/libfalkordb.so \
    --module-args THREAD_COUNT --module-args 1
```

Linux-only in practice: valgrind has no Apple-silicon support, and even on Linux
arm64 it cannot execute this module (an ARMv8.1 LSE atomic in RediSearch). Use
`--bare` there to validate the differencing arithmetic against a plain
redis-server with no module loaded.

### Flow-test benchmarking (C vs Rust per flow file)

```bash
uv run --project bench bench flow --out bench/results/flow_rust.csv
uv run --project bench bench flow --module /tmp/falkordb-c.so --out bench/results/flow_c.csv
uv run --project bench bench flow --compare bench/results/flow_c.csv --current bench/results/flow_rust.csv
```

Runs each file in `flow_tests_done.txt` through `./flow.sh` (RLTest, parallelism
1). RLTest spawns transient redis-servers and macOS `wait4` rusage does not fold
grandchildren, so a poller samples every redis-server pid that appears; each row
is the **sum** over the servers that file spawned. Tests may legitimately fail on
C (Rust-specific behaviours), so the compare table shows both failure counts.
macOS-only.

## CI

Labelling a PR **`benchmark-cov`** measures the PR, its base and the C engine and
posts one comparison comment. Nothing is compiled: all three modules come from
prebuilt images (`rc-pr-<N>`, `edge-rs`, `edge-c`), each measured on its own
runner in parallel inside one image built from the C engine's. Two readings per
side — the full 317-query set on allocated bytes, and a 93-query subset with exact
callgrind instruction counts, sharded so it finishes in the same wall-clock as the
full set.

The callgrind table is **PR-vs-base only**. The C engine cannot be measured that
way: it busy-waits on a worker thread that valgrind schedules arbitrarily, which
showed up as 331,579,187 instructions of drift between two identical runs (this
module: ~100k) and rows that cost *more* on the run doing *fewer* queries. vs-C
is on allocated bytes, which thread scheduling does not affect.

Its one caveat: the base side is the `edge-rs` image, i.e. the tip of the trunk
when it was last built, not the PR's merge base. For a borderline row, confirm
locally against a real base build.

## Layout

| file | purpose |
|---|---|
| `src/falkorbench/model.py` | `Query`, `Metric` — the value types |
| `src/falkorbench/queries.py` | the canonical 317-query set + graph SETUP (10k Person ring, 10k KNOWS, indexes, constraints, UDFs); `cg=True` marks the callgrind subset |
| `src/falkorbench/metrics.py` | CSV parsing, ratios, thresholds, control-row normalisation — shared by compare and report |
| `src/falkorbench/client.py` | server lifecycle and control plane, over falkordb-py |
| `src/falkorbench/counters.py` | instruction/cycle backends (rusage / perf / none) + `pmc_tool` |
| `src/falkorbench/measure.py` | the full-set measurement loop |
| `src/falkorbench/callgrind.py` | deterministic counts by differencing two instrumented runs |
| `src/falkorbench/compare.py` | local regression gate |
| `src/falkorbench/report.py` | the CI comment, and the run's pass/fail decision |
| `src/falkorbench/profile.py` | samply profile of one query |
| `src/falkorbench/flow.py` | per-flow-test-file measurement |
| `src/falkorbench/cli.py` | the `bench` command |
| `src/falkorbench/coverage.py` | instrumented build, run the set once, report graph-crate line coverage |
| `Dockerfile` | the CI measurement image, `FROM …:edge-c` |
| `pmc_tool.c` | Apple Silicon PMU counters (kperf/kperfdata private frameworks) |
| `tests/` | the guards above, over CSV fixtures |

## pmc_tool (optional, for branch/L1D columns)

```bash
clang -O2 -o bench/pmc_tool bench/pmc_tool.c \
  -F /System/Library/PrivateFrameworks -framework kperf -framework kperfdata
sudo chown root:wheel bench/pmc_tool && sudo chmod u+s bench/pmc_tool
```

Without it, instructions/cycles/latency are still measured — those are the
gating columns. PMU numbers are system-wide and include the `redis-benchmark`
client; treat values under ~1K/query as noise.

`pmc_tool window` opens a counter window, prints `READY`, and waits on stdin; the
harness runs the measured command itself in that gap and then closes the window.
It deliberately does not run the command for you — it is installed setuid-root,
and a setuid binary that execs a caller-supplied command is a local privilege
escalation (put your own `redis-benchmark` earlier in `$PATH` and you have root).
Not exec'ing at all removes the whole class of bug, and the counters are
system-wide so bracketing in time is all that was needed.

## Reading the output

- **Trust the instructions column.** Cycles and wall-clock move 10-60% with
  machine load; on a shared or virtualised host they will invent regressions.
  Always include `RETURN 1` in an isolated batch: it is the fixed per-query
  floor, so if it moves, the whole run's cycle column is load-inflated and every
  other cycle flag in it is void.
- **Micro-queries need 3 reps per build.** Rows near the ~330k instruction floor
  read 1.07x on one shot and ~1.00x over three. The error is *absolute*, so a
  fixed percentage tolerance is simultaneously too strict for expensive queries
  and too lax for cheap ones.
- **Ignore rows where C's instruction count is ~500-2500** — those are rows the C
  engine errors on (regex, week/ordinal dates, `LOAD CSV*`, toJSON scalars), so
  the ratio is meaningless. `--c-compat` skips the ones whose warmup reply errors.
- **Watch for near-zero ratios in Rust's favour.** `cross product filter` and
  `untyped shortestPath` have C burning 1.8B and 3.8B instructions against ~400k
  — a regression there would be invisible as a ratio and obvious in absolute
  terms.
- **id-0 rows against C are suspect**: C's `DEBUG RELOAD` drops id 0 from the
  range index. Re-measure on a fresh C server before believing one.

### How this relates to the team performance toolbox

The [performance toolbox](https://aviavni.github.io/database-learning-path/topics/00-performance-toolbox/index.html)
prescribes criterion for microbenchmarks, samply for profiling, `dhat-rs` for
allocations, and confidence intervals over point estimates. Where this harness
agrees, it agrees for the same reasons: "single-shot timing is fiction" is why
wall-clock never gates here; the idle-rate calibration is "keep the machine
idle"; and the callgrind path **drops** any row it cannot resolve to better than
2% rather than print it, which is that confidence-interval discipline applied to
instruction counts.

Three deliberate deviations:

1. **No criterion.** criterion measures in-process Rust functions; this measures
   whole-query cost through the redis protocol, which criterion cannot reach.
   They are complementary — criterion belongs on `graph`-crate internals, and
   there is no criterion suite in this repo yet.
2. **Allocations from jemalloc merged-arena deltas, not `dhat-rs`.** No
   instrumented build needed, works on any stock jemalloc redis, and measures the
   real server process — which is what makes the vs-C comparison possible.
3. **`ms` is subject to coordinated omission, by construction.**
   `redis-benchmark -c 1` is a closed-loop generator: send, await reply, send
   next. A stall backs up the generator and disappears from the data, exactly as
   the toolbox warns. It does not touch the instruction or allocation columns — a
   stall cannot hide an instruction — and it is acceptable *only* because `ms` is
   never the gate, just a coarse outlier net.

`ms` is also a **mean**, not a percentile, while the project's stated bar (see
CLAUDE.md) is p99 latency. `redis-benchmark` computes a latency distribution and
this harness currently discards it, so the p99 bar is not measured here yet.

## Known gaps

One structural gap is worth recording because closing it needs a design decision
rather than a patch: the MVCC copy-on-write `GrB_Matrix_dup` of delta matrices in
`create_nodes` / `set_nodes_labels_bulk` waits on pending work first, so it scales
with the accumulated delta (up to the 10k flush threshold, avg ~5k) per query,
independent of batch size. C never merges pending tuples on the write path. That
is what keeps the create/delete rows above 1.0x against C.

**A ranked Rust-vs-C table is deliberately not kept here.** It goes stale the
moment anything merges, and a stale ranking is worse than none — it sends people
to work on rows that are already fixed. Generate it from a live run using the
recipe above.

**Coverage**: the query set covers **74.8%** of graph-crate lines
(28,869/38,573, excluding the generated `GraphBLAS.rs` FFI; measured in CI on
2026-08-05). The 0%-coverage areas need infrastructure a Cypher
query cannot reach from this graph: `cow_btree` (~750 lines, appears unwired),
`string_pool`, `vec_distance`, and most of `algo_procedures.rs`. For the hot paths
this set targets (runtime, expressions, planner, matrices), coverage is high.
`bench coverage` reports the number but does not enforce a floor — it is a
validator of the query set, not a coverage gate.
