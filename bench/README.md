# bench — per-query performance & coverage loop

Measures per-query **instructions and cycles of the redis-server process**
(macOS `proc_pid_rusage`, no root) plus optional system-wide **branches,
branch-misses, L1D-misses** (Apple Silicon PMU via `pmc_tool`, needs root)
and per-query **jemalloc allocated/deallocated bytes** (`MEMORY MALLOC-STATS`
merged-arena deltas, works on any jemalloc-built redis, i.e. stock).
Includes a regression gate against a stored baseline and a coverage check
that the query set exercises most of the `graph` crate.

## The loop

```bash
cargo build --release                      # 1. build
python3 bench/run_bench.py                 # 2. measure -> bench/results/current.csv
python3 bench/compare.py                   # 3. gate vs bench/baseline/rust.csv (exit 1 on any metric regression)
python3 bench/compare.py bench/results/current.csv bench/baseline/c.csv   # vs legacy C engine
bench/coverage.sh                          # 4. verify query set still covers the code
bench/profile.sh case "MATCH (p:Person) RETURN sum(CASE WHEN p.id % 3 = 0 THEN 1 ELSE 3 END)"   # 5. drill into a hot query
# ...optimize, goto 1. After a confirmed improvement:
cp bench/results/current.csv bench/baseline/rust.csv   # local only; baseline/ is git-ignored
```

`run_bench.py` accepts query names to re-run a subset (rows are merged into
the existing CSV): `python3 bench/run_bench.py "CASE" "WITH pipeline"`.
`--keep-server` leaves the server + graph up for `profile.sh`.
`--c-compat` is required when the module is the C FalkorDB: it skips the
composite unique constraint in setup (its async validation crashes the C
server) and skips queries whose warmup reply errors.

## Flow-test benchmarking (C vs Rust per flow file)

```bash
python3 bench/flow_bench.py --out bench/results/flow_rust.csv
# The C module: build the `master` branch of this repo (lands under bin/), or
# pull it out of the published image:
#   docker create --name c falkordb/falkordb-server:edge
#   docker cp c:/var/lib/falkordb/bin/falkordb.so /tmp/falkordb-c.so && docker rm c
# (that copy is Linux — on macOS you need a local `master` build)
python3 bench/flow_bench.py --module bin/macos-arm64v8-release/falkordb.so \
    --out bench/results/flow_c.csv
python3 bench/flow_bench.py --compare bench/results/flow_c.csv --current bench/results/flow_rust.csv
```

Runs each file in `flow_tests_done.txt` through `./flow.sh` (RLTest,
parallelism 1). RLTest spawns transient redis-servers and macOS `wait4`
rusage does not fold grandchildren, so a poller thread samples every
redis-server pid that appears during the run; each row is the **sum of
server-side instructions / cycles / lifetime-peak memory** over the servers
that file spawned, plus wall time and pass/fail counts. Tests may
legitimately fail on C (Rust-specific behaviors) — the compare table shows
both failure counts.

## Files

| file | purpose |
|---|---|
| `queries.py` | canonical 317-query set + graph SETUP (10k Person ring, 10k KNOWS, index on id) |
| `run_bench.py` | start server, build graph, measure, write CSV (`--once` for coverage) |
| `compare.py` | ratio table + regression gate on every metric (cycles/instr/branches/alloc/dealloc +10%, br_miss/l1d_miss/ms +25%); `--threshold` overrides all, `--metrics` restricts the set |
| `flow_bench.py` | per-flow-test-file server instr/cycles/peak-mem, C vs Rust compare |
| `coverage.sh` | instrumented build, run set once, report graph-crate line coverage |
| `profile.sh` | `sample`-based hot-stack profile of one query |
| `pmc_tool.c` | PMU counter tool (kperf/kperfdata private frameworks) |
| `baseline/` | your own baselines, git-ignored — see below |

## pmc_tool (optional, for branch/L1D columns)

```bash
clang -O2 -o bench/pmc_tool bench/pmc_tool.c \
  -F /System/Library/PrivateFrameworks -framework kperf -framework kperfdata
sudo chown root:wheel bench/pmc_tool && sudo chmod u+s bench/pmc_tool
```

Without it, `run_bench.py` still measures instructions/cycles/latency
(those are the regression-gate columns). PMU numbers are system-wide and
include the redis-benchmark client; treat values under ~1K/query as noise.

`pmc_tool window` opens a counter window, prints `READY`, and waits on stdin;
the harness runs the measured command itself in that gap and then closes the
window. It deliberately does not run the command for you — it is installed
setuid-root, and a setuid binary that execs a caller-supplied command is a
local privilege escalation (put your own `redis-benchmark` earlier in `$PATH`
and you have root). Not exec'ing at all removes the whole class of bug, and
the counters are system-wide so bracketing in time is all that was needed.

## Findings (2026-07-26 baseline, M3 Pro)

Query set restructured for issue isolation (single feature per query first,
mixed clauses after; create and delete measured separately). Both baselines
re-measured full-set in identical order — ratios are only apples-to-apples
when both engines see the same capacity/ordering context.

The isolation restructure immediately found three delta-scaling bugs:
- Aggregations without a registered `batch_agg` (percentileDisc/Cont,
  stDev/P) deep-cloned their collected-values accumulator every row.
  Fixed by registering batch fns: percentileDisc 194x→2.0x, percentileCont
  192x→2.0x, stDev 207x→**0.66x**, stDevP 213x→**0.69x** vs C cycles.
- The small-delete path in `Graph::delete_nodes` interleaved
  `node_labels_matrix.iter()` (which waits on the pending delta) with
  `remove()` per node — a GraphBLAS pending-tuple merge per node,
  O(deleted × |delta|). Fixed with a read-phase/write-phase split:
  delete 100 49.9x→**1.71x**, write 100 1.74x→**0.95x**,
  write 10 1.08x→**0.92x**.
- `import_node_attrs` re-read `node_labels_matrix` per created node when
  any index exists; the first `iter` per query forced the pending-delta
  merge. Fixed by passing labels from `Pending::set_labels`:
  create 100 7.7x→**4.79x**, create 10k 6.1x→**2.99x**,
  create node 3.1x→**2.48x**, write 100 →**0.85x**, write 1k →1.21x.

One known structural gap is worth recording because fixing it needs a design
decision rather than a patch: the MVCC COW `GrB_Matrix_dup` of delta matrices
in `create_nodes` / `set_nodes_labels_bulk` waits on pending work first, so it
scales with the accumulated delta (up to the 10k flush threshold, avg ~5k) per
query, independent of batch size. C never merges pending tuples on the write
path. That is what keeps the create/delete rows above 1.0x.

**A ranked Rust-vs-C table is deliberately NOT kept here.** It goes stale the
moment anything merges, and a stale ranking is worse than none — it sends
people to work on rows that are already fixed. Generate it from a live run:

```bash
python3 bench/run_bench.py --out /tmp/rust.csv
python3 bench/run_bench.py --c-compat --module <c-module> --out /tmp/c.csv
python3 bench/compare.py /tmp/rust.csv /tmp/c.csv | sort -k4 -rn | head -20
```

See `.claude/skills/bench/SKILL.md` for how to read that output — which columns
to trust, and which C rows are artifacts rather than measurements.

**Coverage**: the pre-restructure set covered **44.0%** of graph-crate lines
(excluding the generated GraphBLAS.rs FFI). The remaining 0%-coverage areas
need infrastructure a Cypher query can't reach from this graph: fulltext /
vector index scans, `load_csv`, JS UDFs (`udf/*`), constraints, `cow_btree`
(~750 lines, appears unwired), `string_pool`, `vec_distance`, and ~97% of
`algo_procedures.rs` (only pageRank/BFS/WCC are called). For the hot paths
this set targets (runtime, expressions, planner, matrices), coverage is high.
