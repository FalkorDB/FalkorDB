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
prebuilt images (`rc-pr-<N>-valkey`, `edge-rs`, `edge-c`), each measured on its own
runner in parallel inside one image built from the C engine's. Two readings per
side — the full query set on allocated bytes, and the `cg=True` subset with exact
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
| `src/falkorbench/queries.py` | the canonical query set + graph SETUP (10k Person ring, 10k KNOWS, indexes, constraints, UDFs); `cg=True` marks the callgrind subset |
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
| `src/falkorbench/ldbc/` | LDBC SNB Interactive v1 — dataset fetch/prepare, loader, parameters, runner, and the vendored query texts |
| `Dockerfile` | the CI measurement image, `FROM …:edge-c` |
| `pmc_tool.c` | Apple Silicon PMU counters (kperf/kperfdata private frameworks) |
| `tests/` | the guards above, over CSV fixtures |

## pmc_tool (optional, for branch/L1D columns)

```bash
clang -O2 -o bench/pmc_tool bench/pmc_tool.c \
  -F /System/Library/PrivateFrameworks -framework kperf -framework kperfdata
sudo chown root:wheel bench/pmc_tool && sudo chmod u+s bench/pmc_tool
```

Without it, instructions, cycles, allocated bytes and wall-clock are still
measured, and the branch/L1D columns stay empty. With it, those columns are
filled in **and gated** like the rest (`GATED_BY_DEFAULT` in `metrics.py`; only
`ms` is left out). PMU numbers are system-wide and include the `redis-benchmark`
client, so on a busy machine they can fail the gate on their own: treat values
under ~1K/query as noise, and narrow the gate with `--metrics
instr,cycles,alloc_bytes,dealloc_bytes` when they do.

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
- **Micro-queries need 3 reps per build.** Rows near the instruction floor (what
  `RETURN 1` costs: about 175k on Apple Silicon)
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

## LDBC SNB (Interactive v1)

A second, complementary workload: the 14 **complex reads** of the LDBC Social
Network Benchmark, run against the real SNB dataset. The micro-benchmark set
above measures narrow operations on a synthetic ring; these measure whole
realistic queries — multi-hop traversals, aggregation, `OPTIONAL MATCH`,
variable-length paths — on a graph with realistic degree skew. A planner
regression that a single-op query cannot see tends to show up here.

```bash
bench ldbc fetch --sf 0.1                 # download + prepare (~18 MB)
bench ldbc run --sf 0.1                   # load, then measure all 14
bench ldbc run --sf 1 IC1 IC13            # a subset, at SF1 (~230 MB)
bench ldbc run --params ./substitution_parameters   # official parameters
```

Results go to `results/ldbc_sf<SF>.csv` (p50/p95/p99 per query) and are
deliberately **not** merged into `results/current.csv`: LDBC's metric is
response time, and mixing it into the file the `compare` thresholds gate would
distort the micro-benchmark gate.

### This is not an auditable LDBC result

Do not publish these as an "LDBC score". A real result requires the official
Java driver with its validation and workload-generation phases, LDBC
membership, and an audit commissioned from a certified auditor. What this gives
you is an internal, repeatable number on a standard dataset and standard
queries.

### Five queries differ from the reference text

Each departure is annotated in the `.cypher` file next to the line it replaced,
and listed in `ldbc/queries.py::REWRITES` so a run always prints them. All five
were confirmed against a running engine, and each was checked against the C
engine too so that a genuine bug is not filed away as a dialect gap:

| query | why it could not run as written |
|---|---|
| IC1, IC13 | `MATCH path = shortestPath(...)` → *"FalkorDB currently only supports shortestPaths in WITH or RETURN clauses"*. Moved into `WITH`. |
| IC14 | `allShortestPaths()` with inline endpoint patterns → *"Source and destination must already be resolved"*. Endpoints pre-bound in a preceding `MATCH`. Also, a pattern comprehension's `WHERE` cannot see the enclosing list comprehension's variable, so the per-relationship weights are computed over `UNWIND`ed relationships with their endpoints hoisted into plain variables. |
| IC10 | `datetime({epochMillis: ...}).month` → *"Unknown function 'datetime'"*. The loader derives `birthdayMonth`/`birthdayDay` instead. |
| IC7 | `not((liker)-[:KNOWS]-(person))` → *"Type mismatch: expected Boolean or Null but was List"*. A traversal pattern is not a boolean-valued expression in a projection, and `exists()` explicitly refuses traversal patterns, so this becomes `size(<pattern>) = 0`. |

In their upstream form, IC1, IC7, IC13 and IC14 fail identically on the C
engine, so those four are dialect gaps rather than regressions. With the
rewrites, C runs IC7 but still not IC1, IC13 or IC14, nor IC10 (the nightly
check below lists each error). IC6 and IC9 were rewritten too at first,
and were not dialect gaps: both ran correctly on C and hit two Rust
wrong-result bugs ([#2556](https://github.com/FalkorDB/FalkorDB/issues/2556),
[#2557](https://github.com/FalkorDB/FalkorDB/issues/2557)). Both are fixed, and
both queries run the upstream text again. That distinction only exists because
every failure was re-checked against `falkordb/falkordb-server:edge-c` before
being classified — without that step two real regressions would have been
written off as dialect gaps, and two dialect gaps filed as false bugs.

`CREATE CONSTRAINT ... ASSERT n.id IS UNIQUE` is also unsupported, so upstream's
`indices.cypher` becomes `GRAPH.CONSTRAINT CREATE` calls — each of which
additionally requires its exact-match index to already exist, and validates
asynchronously (the loader waits for `OPERATIONAL` rather than assuming it;
note `db.constraints()` reports `UNDER CONSTRUCTION`, never `PENDING`).

The CSVs are pipe-separated. FalkorDB has no `DELIMITER` keyword — the
standard-Cypher spelling is `FIELDTERMINATOR`, and it goes *after* `AS <var>`.

### The nightly check: `bench ldbc check`

The nightly perf run (epic #3974) compares main's newest image with the latest
release through several independent checks; this is the LDBC one. It is one
command, with the contract every nightly check shares:

```bash
uv run --project bench bench ldbc check \
  --candidate ghcr.io/falkordb/falkordb-server@sha256:… \
  --baseline falkordb/falkordb-server@sha256:… \
  --out out --mode nightly \
  [--candidate-label main@4c10889] [--baseline-label 6.2.0] \
  [--candidate-commit <40 hex>] [--baseline-commit <40 hex>]
```

Both arguments are images pinned by digest, and nothing is compiled: each image
runs as its own server in docker (its own redis or valkey and allocator),
side by side on this machine, on ports docker picks (never 6379). The dataset is
bind-mounted read-only as the module's `IMPORT_FOLDER`. On the VM the servers
are pinned to `PERF_SERVER_CPUS`; locally they run unpinned and the result says
`runner: local`.

**How it measures.** The dataset is loaded into the baseline, then the
candidate. Parameters are sampled once, from the baseline, and both sides run
the same rows (`raw/params.json`). Per parameter row the two servers run back to
back, in two rounds with the order swapped, after one warm-up call per query
and side. Each latency is one `GRAPH.RO_QUERY` on a connection that never
re-sends (redis-py's retries are off), with the query's server-side `TIMEOUT`.

**What it reports** (`<out>/result.json`, contract schema 1), per query:

| finding | metric | when |
|---|---|---|
| `IC3 p50`, `IC3 p95` | `p50_ms`, `p95_ms` | the ratio candidate/baseline over the rows both sides finished crosses 2x **and** the difference is at least 0.5 ms (p50) / 2 ms (p95). A crossing is measured again (two more swapped rounds); it is a `regression` only when the re-run crosses too, otherwise `ok` with the re-run's numbers. Wall clock, hence the wide band. |
| `IC9 rows` | `rows` (`mismatch`) | the two sides return a different number of rows for the same parameter row. A regression at once, and the query gets no latency finding: the two sides did different work (6.0.4's IC9 "ran" 93x faster by returning nothing, #2557). |
| `IC3 completed` | `completed` | judged per parameter row: the candidate timed out, errored or lost its server on a row the baseline finished is a regression at once; rows failing on both sides are `info`. A parameter row that timed out is skipped on both sides from then on, and a side stops a query after two timeouts. |
| `server up` | `completed` | a server that went down at any point, also after its last reply (its container exited): the candidate's is a regression, the baseline's makes the result an `error`. |
| `IC6 returns rows` | `invariant` | zero rows on every parameter row, on both sides: it measured nothing (`info`). |
| `IC14 same rows` | `invariant` | the counts agree but the row contents differ, between two images of the same engine (`info`; see below). |
| `load` | `throughput` | entities loaded per second, per side: one sample each, so `info` only. |

Row contents are compared as a digest of the sorted rows with doubles rounded
to nine significant digits (`raw/samples.csv`), so tie order and the engines'
different float spelling do not count. IC14 returns every shortest path, and
paths of equal weight have no defined order, which the sorted digest absorbs.
A difference is reported as `info` only, until it has proven stable across
nights.

**Thresholds.** The ratio and the floors come from three A/A runs (main@4c1088949
against itself twice, C v4.22.4 against itself once; nightly mode, unpinned,
on an n4-standard-16 dev VM). The worst ratio either way was 1.19x (IC11 p95),
the worst p50 difference on a query under 5 ms 0.061 ms, and the worst p95
difference on a query under 10 ms 0.65 ms; 2x is 1.68x above the first, 0.5 ms
8x the second, 2 ms 3x the third. The judge's pinned A/A runs on the VM
workflow are the independent check of these values.

**Queries an engine cannot run** are left out rather than failing the check: a
query on which the *baseline* errors on every parameter row is reported as one
`info` finding (`completed`, baseline 0) and is not compared. On the C engine
that is IC1, IC10, IC13 and IC14, measured on v4.22.4 (C against itself, every
parameter row):

| query | the C engine's error |
|---|---|
| IC1, IC13 | *"FalkorDB does not currently support undirected shortestPath traversals"* |
| IC14 | *"allShortestPaths(...) does not support a minimal length different from 1"* |
| IC10 | *"Type mismatch: expected Null but was Pointer"* |

The same query failing on the *candidate* only is a regression.

**Verdict.** `regression` (confirmed) if any finding is one; otherwise `error`
when something could not be measured (an image not pinned by digest, a pull or
a dataset download that failed, a baseline that did not start, load or stay
up, no query that finished on both sides, a deadline that cut any query's
rounds); otherwise `pass`. The summary counts as compared only the queries
with at least one parameter row finished on both sides. `result.json` is written
first as an `error` and replaced at the end, so a run stopped from outside
still leaves one. The command exits 0 whenever `result.json` was written.

**Modes** set only the length, never the method:

| mode | dataset | parameter rows per query | rounds (+ re-run) | per-query timeout | deadline |
|---|---|---|---|---|---|
| `queue` | SF0.1 (18 MB) | 10 | 2 (+2) | 30 s | 25 min |
| `nightly` | SF0.1 (18 MB) | 25 | 2 (+2) | 60 s | 35 min |
| `long` | SF1 (230 MB) | 25 | 2 (+2) | 300 s | 150 min |

How long it takes, measured on an n4-standard-16 dev VM (unpinned), with the
settings above: main@4c1088949 against itself took 13.5 min in `nightly` and
7.7 min in `queue`. Earlier runs, before the per-round budget was raised to
300 s: C v4.22.4 against itself 3.7 min, 6.0.4 against 6.2.0 (nine
regressions confirmed, so nine re-runs) 20.6 min. Most of it is the two loads
(about 70 s each) and the queries that time out (IC3 on every row, and one IC14
row, on the Rust engine). The 35-minute deadline stops new measurement well
inside the 45 minutes the check gets in `tests/perf/checks.yaml`. `long` (SF1)
is accepted and set as above but not yet run end to end.

**Artifacts** under `<out>/raw/`: `samples.csv` (every request: phase, round,
query, parameter row, side, outcome, latency, rows, digest), `summary.csv`
(per query and side), `params.json`, `load.json`, `explain/<side>/<IC>.txt`
(`GRAPH.EXPLAIN` of every query on both sides), `<side>-server.log`, and
`run.json` (mode, thresholds, engines, rewrites, notes).

**Tests.** `bench/tests/test_ldbc.py` covers the analysis, the measurement
loop (with scripted servers) and every result shape, holding each result.json
to the contract's validator (`tests/perf/validate.py`, from #3986; set
`PERF_VALIDATE_PY` to use another copy, and the tests that need it skip without
it). `perf-contract.yml` (#3986) runs them with
`uv run --project bench pytest bench/tests`.

### Substitution parameters

LDBC generates parameters chosen so the queries hit a representative spread of
the data; the cost of these queries varies by orders of magnitude with how
well-connected the chosen person is, so the parameter set matters as much as the
query text. Pass an official set with `--params DIR`.

Without it, the harness samples the loaded graph with a fixed seed. Those runs
are deterministic and comparable **to each other**, but not to published LDBC
numbers — the CSV and the console output both say which source was used, and
that caveat should travel with any number taken from it.

Sampling deliberately draws from where the data is dense: people from the
most-connected end of the `KNOWS` degree distribution, and dates from the last
quarter of the corpus. IC3 gets a much wider window than the rest — it counts
friends who are *not* resident in either country yet posted from both inside the
window, and at SF0.1 a 30-day window returns 0 rows where 365 returns 1 and 730
returns 2. Both engines agree on those counts, so it is a property of the
dataset, not of either engine; `durationDays` is an explicit LDBC parameter, so
widening it stays inside the query's own contract. A uniformly drawn person is usually isolated and a
uniformly drawn date window usually lands in the sparse early history; either
returns nothing in microseconds, which would report the benchmark as fast while
measuring almost none of the work it exists to measure. `runner.problems()`
treats "returned zero rows on every run" as a failure for the same reason.

### What the first run found

Running the 14 complex reads against SF0.1 was worth doing for the bugs alone.
All fourteen now produce a measurement — IC3 needs `--timeout` above the default
— and four of them (IC1, IC10, IC13, IC14) do not run on the C engine at all.
Every query except IC3 and IC10 is in the same order of magnitude as C or
faster. Four defects came out of the exercise, each confirmed against the C
engine before being filed:

| issue | what |
|---|---|
| [#2555](https://github.com/FalkorDB/FalkorDB/issues/2555) | aggregation over a bare map property returns empty — `collect`→`[]`, `count`→`0`, silently |
| [#2556](https://github.com/FalkorDB/FalkorDB/issues/2556) | inline property filter under `UNWIND` raises a spurious type mismatch |
| [#2557](https://github.com/FalkorDB/FalkorDB/issues/2557) | `WHERE` on a node matched from an `UNWIND`-bound anchor silently returns zero rows |
| [#2558](https://github.com/FalkorDB/FalkorDB/issues/2558) | a variable-length traversal loses its indexed anchor when the `MATCH` has a second pattern — 19,000x |

**#2558 is the one that matters.** On the same parameter row, returning the
identical 2 rows, IC3 takes **136.72 ms** on C and **271,817.66 ms** on Rust —
1,988x — and IC10 takes 86 s. Both plans abandon a unique-index seed in favour
of scanning the unbound side. No other measured query is affected, so a single
planner decision accounts for the entire Rust-vs-C gap on this workload.

Three of the four are **silent** — wrong or empty results with no error. They
were caught only because the runner treats "zero rows on every run" as a failure
rather than as a fast query. A harness that reported latency alone would have
called that run a success.

Following the file-level convention above, a ranked latency table is not kept
here; regenerate it from a live run. `results/ldbc_sf<SF>.csv` is written per run
and deliberately not merged into `current.csv` — these runtimes are orders of
magnitude larger than the micro-benchmark's and would distort its regression
thresholds.
