# Bench loop

Repeatable per-query performance + regression + coverage loop. All scripts
live in `bench/` (see `bench/README.md` for findings and details).

## Full loop (run in this order)

```bash
# 1. build the release module
cargo build --release

# 2. measure all 317 queries -> bench/results/current.csv (~10 min)
python3 bench/run_bench.py

# 3. regression gate vs baseline — every metric (cycles/instr/branches/alloc_bytes/
#    dealloc_bytes at +10%, br_miss/l1d_miss/ms at +25%); exit 1 on any breach.
#    --threshold X overrides all metrics, --metrics cycles restricts the gate.
python3 bench/compare.py

# 4. optional: compare against the legacy C engine baseline
python3 bench/compare.py bench/results/current.csv bench/baseline/c.csv

# 5. coverage check: query set should stay ~70% of graph-crate lines
bash bench/coverage.sh
```

After a confirmed improvement, promote the new numbers:

```bash
cp bench/results/current.csv bench/baseline/rust.csv   # local only
```

## Drilling into one query

```bash
# re-measure a subset (rows merge into existing CSV by query name)
python3 bench/run_bench.py "CASE" "WITH pipeline"

# keep the server + graph up, then sample-profile a query
python3 bench/run_bench.py --keep-server "RETURN 1"
bash bench/profile.sh case "MATCH (p:Person) RETURN sum(CASE WHEN p.id % 3 = 0 THEN 1 ELSE 3 END)"
```

`profile.sh` args: `<out-name> "<cypher>" [GRAPH.QUERY|GRAPH.RO_QUERY] [port]`.
Output lands in `bench/results/sample_<name>.txt`; the "Sort by top of stack"
section lists the hot leaves.

## Operational notes

- `run_bench.py` starts its own redis-server on :6399 and builds the graph
  (10k Person ring + KNOWS edges, index on id). If a server is already up
  with the graph, pass `--reuse --port <p>`. It refuses to start if the port
  is busy.
- Instructions/cycles come from `proc_pid_rusage` (per-process, no root) —
  these are the regression-gate columns. Branch/L1D columns need
  `bench/pmc_tool` (setuid root); without it they're left empty and that's
  fine. Rebuild it with:
  ```bash
  clang -O2 -o bench/pmc_tool bench/pmc_tool.c \
    -F /System/Library/PrivateFrameworks -framework kperf -framework kperfdata
  sudo chown root:wheel bench/pmc_tool && sudo chmod u+s bench/pmc_tool
  ```
- PMU numbers are system-wide and include the redis-benchmark client; use
  the RETURN 1 row as the client floor, treat <1K/query as noise.
- `coverage.sh` uses port 6401 and an instrumented debug build; it exits
  non-zero if any query errors, so it doubles as query validation.
- Queries and graph setup are canonical in `bench/queries.py`; add new
  queries there and they flow to benchmark, compare, and coverage. The
  sized "write N" queries must stay LAST in the list — they inflate node
  capacity / matrix dimension to max(N) and would slow every full-graph
  query measured after them (algo.pageRank went 150x when they ran first).
- Run-to-run noise is ~1-2% on cycles but micro-queries (<200k cycles) can
  flag ±15% on cycles; the instruction ratio is the stable signal — trust
  it over a cycles-only flag. Adjust the gate with `--threshold`.

## Finding the current improvement targets

Deliberately not listed here. This section previously carried a ranked
Rust-vs-C table and it was wrong twice in one day — it still named MERGE bound
pattern after #777 fixed it, and it predated `arithmetic` and `CASE` overtaking
C. A hardcoded ranking in a doc is stale the moment anything merges, and a
stale ranking is worse than none: it sends people to work on rows that are
already fixed.

Generate it from a live run instead. Labelling a PR `benchmark-cov` measures
`main` and the PR and posts the comparison, but **not the C engine** — that job
runs on a macOS runner and there is no macOS build of the C module to load. For
a vs-C reading, run it locally:

```bash
python3 bench/run_bench.py --out /tmp/rust.csv                     # this build
python3 bench/run_bench.py --c-compat --module <c-module> --out /tmp/c.csv
python3 bench/compare.py /tmp/rust.csv /tmp/c.csv | sort -k4 -rn | head -20
```

### Reading the output

- **Trust the instructions column.** Cycles and wall-clock move 10-60% with
  machine load; on a shared or virtualised host they will invent regressions.
  Always include `RETURN 1` in an isolated batch: it is the fixed per-query
  floor, so if it moves, the whole run's cycle column is load-inflated and
  every other cycle flag in it is void.
- **Micro-queries need 3 reps per build.** Rows near the ~240k instr floor read
  1.04x on one shot and 1.00x over three.
- **Ignore rows where C's instruction count is ~500-2500** — those are rows the
  C engine errors on (regex, week/ordinal dates, `LOAD CSV*`, toJSON scalars),
  so the ratio is meaningless. `--c-compat` skips the ones whose warmup reply
  errors.
- **Watch for near-zero ratios in Rust's favour.** `cross product filter` and
  `untyped shortestPath` have C burning 1.8B and 3.8B instructions against
  ~400k — a regression there would be invisible as a ratio and obvious in
  absolute terms.
- **id-0 rows against C are suspect**: C's `DEBUG RELOAD` drops id 0 from the
  range index. Re-measure on a fresh C server before believing one.
