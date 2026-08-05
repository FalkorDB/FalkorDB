# Bench loop

Repeatable per-query performance + regression + coverage loop. The harness is a
uv project under `bench/` — see `bench/README.md` for details.

## Full loop (run in this order)

```bash
# 0. once: install the harness's dependencies
uv sync --project bench

# 1. build the release module
cargo build --release

# 2. measure all 317 queries -> bench/results/current.csv (~2 min)
uv run --project bench bench measure

# 3. regression gate vs your baseline; exits 1 on a breach
uv run --project bench bench compare

# 4. coverage check: reports graph-crate line coverage, and fails if any query
#    stopped working (it runs the whole set once, so it doubles as validation)
uv run --project bench bench coverage
```

The gate's thresholds live in `bench/src/falkorbench/metrics.py` (`THRESHOLDS`):
the deterministic counters — instructions, cycles, branches, allocated and
deallocated bytes — are gated at +10%, the noisy miss counters at +25%. Wall-clock
is handled separately: it is first divided by the `RETURN 1` control row to cancel
the per-host speed difference, and only flagged past ±50%, because across two
machines a raw `ms` ratio is a noise detector rather than a measurement.
`--threshold X` overrides every metric; `--metrics instr,cycles` narrows the gate.

After a confirmed improvement, promote the numbers:

```bash
cp bench/results/current.csv bench/baseline/rust.csv   # local only, baseline/ is git-ignored
```

Two traps in that baseline, neither currently automated away:

- It is one file and a branch switch does not change it, so after a checkout you
  compare against numbers from some other commit — silently.
- A baseline from a different machine is not comparable at all; per-host speed
  differences alone measured 1.46x on byte-identical engines.

## Drilling into one query

```bash
# re-measure a subset (rows merge into the existing CSV by query name)
uv run --project bench bench measure "CASE" "WITH pipeline"

# keep the server + graph up, then profile a query against it
uv run --project bench bench measure --keep-server "RETURN 1"
uv run --project bench bench profile --reuse "CASE"
```

`bench profile` records with samply and writes
`bench/results/profile_<name>.json.gz`; `--open-ui` opens the Firefox Profiler
instead of only saving. It names the query once — it reuses the harness's own
server, pid and query text.

## Operational notes

- `bench measure` starts its own redis-server on :6399 and builds the graph (10k
  Person ring + KNOWS edges, indexes, constraints, UDFs). If a server is already
  up with the graph, pass `--reuse --port <p>`. It refuses to start if the port is
  busy. `--reuse` does not skip the graph build unless you also pass `--no-setup`
  — otherwise it would measure an empty database and report numbers that look
  real.
- Instructions/cycles come from `proc_pid_rusage` on macOS and `perf stat -p` on
  Linux. Neither works on a virtualised host with no PMU, in which case those
  columns are left **empty rather than zero** — a zero would read as a real
  measurement. Branch/L1D columns additionally need `bench/pmc_tool` (setuid
  root); without it they stay empty and that is fine.
- `bench coverage` uses port 6401 and an instrumented debug build. It reports a
  percentage but does not enforce a floor — it is a validator of the query set,
  not a coverage gate.
- Queries and graph setup are canonical in `bench/src/falkorbench/queries.py`; add
  queries there and they flow to measurement, comparison and coverage. Set
  `cg=True` to include one in the callgrind subset — that means it must run on the
  reduced CG_SETUP graph (1,000 :Person, a :KNOWS ring, 5,000 :Tmp) and must not
  drain a pool faster than it is refilled. `bench/tests/test_queries.py` enforces
  both, plus the rule that the sized "write N" queries stay LAST in the list —
  they inflate node capacity / matrix dimension to max(N) and would slow every
  full-graph query measured after them (algo.pageRank went 150x when they ran
  first).
- Run-to-run noise is ~1-2% on instructions and wider on cycles; micro-queries
  (<400k instr) can read 1.07x on a single shot. The instruction ratio over three
  reps is the stable signal.

## Finding the current improvement targets

Generate them from a live run — a hardcoded ranking in a doc is stale the moment
anything merges, and a stale ranking sends people to work on rows that are already
fixed.

Labelling a PR `benchmark-cov` measures the PR, its base and the C engine and
posts one comparison comment. Nothing is compiled: all three modules come from
prebuilt images (`rc-pr-<N>`, `edge-rs` and `edge-c`), each measured on its own
runner in parallel inside one image built from the C engine's. Two readings per
side — the full 317-query set on allocated bytes, and a 93-query subset with exact
callgrind instruction counts.

The callgrind table is **PR-vs-base only**. The C engine cannot be measured that
way: it busy-waits on a worker thread that valgrind schedules arbitrarily, which
showed up as 331,579,187 instructions of drift between two identical runs (this
module: ~100k) and rows that cost *more* on the run doing *fewer* queries. vs-C is
on allocated bytes, which thread scheduling does not affect.

Its one caveat: the base side is the `edge-rs` image, i.e. the tip of the trunk
when it was last built, not the PR's merge base. For a borderline row, confirm
locally against a real base build before acting on it.

For a vs-C reading locally you need a C module. The C engine is **`:edge-c`**, not
`:edge` — `:edge` now resolves to the Rust engine, sharing a digest with
`:edge-rs`. Build the `master` branch of this repo (it lands under `bin/`), or on
Linux copy one out of the image:

```bash
docker create --name c falkordb/falkordb-server:edge-c
docker cp c:/var/lib/falkordb/bin/falkordb.so /tmp/falkordb-c.so && docker rm c

uv run --project bench bench measure --out /tmp/rust.csv
uv run --project bench bench measure --c-compat --module /tmp/falkordb-c.so --out /tmp/c.csv
uv run --project bench bench compare /tmp/rust.csv /tmp/c.csv
```

### Reading the output

See the "Reading the output" section of `bench/README.md` — which columns to
trust, which C rows are artifacts rather than measurements, and how this harness
relates to (and deliberately deviates from) the team performance-toolbox
guidance. That is the single place it is written down.
