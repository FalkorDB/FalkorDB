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
uv sync --project bench          # once: the harness and the dev group (pytest, ruff, pyyaml)
```

uv runs it on the Python `bench/.python-version` pins (3.12), on a laptop, a VM,
the CI image and the runners alike. The soak and fuzz checks need one more
dependency group each: `uv run --project bench --group soak …`, `--group fuzz`.

Every command below can be run as `uv run --project bench bench …`. From inside
`bench/` it is just `uv run bench …`. The tests:

```bash
uv run --project bench --group dev python -I -B -m pytest bench/tests
```

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

## Result contract (nightly perf checks)

Each nightly perf check (micro-hw, pokec, soak-a, soak-b, fuzz, ldbc) compares a
candidate image with a baseline image and writes one `result.json`. The nightly
merges them into one `nightly.json`, which the reporter posts.
`src/falkorbench/contract.py` defines those two files and checks them: it
rejects a result or nightly that cannot be reported as it stands, i.e. a missing
or malformed field the reporter reads, or a verdict that contradicts its
findings. It is standard library only, so the reporting path never depends on a
package.

### The format

`result.json` (one per check):

```json
{
  "schema": 1, "check": "micro-hw", "verdict": "pass | regression | error", "confirmed": false,
  "summary": "one line in plain words",
  "candidate": {"image": "repo@sha256:<64 hex>", "commit": "<40 hex or null>", "label": "main@4c1088949"},
  "baseline":  {"image": "repo@sha256:<64 hex>", "commit": null, "label": "6.2.1"},
  "machine": {"runner": "gce", "type": "n4-standard-16", "cpu": "<model>", "cores": 16},
  "started": "2026-11-24T00:05:00Z", "finished": "2026-11-24T00:41:00Z",
  "findings": [{"name": "…", "metric": "p50_ms", "candidate": 13.3, "baseline": 0.4, "ratio": 33.25,
                "threshold": 1.5, "direction": "higher_is_worse", "status": "regression"}],
  "error": null, "artifacts": ["raw/rows.csv"]
}
```

`nightly.json` holds `schema`, `date`, `run_url`, `candidate`, `baseline`,
`previous_green_commit`, `status` (green, red, error or skipped), `skip_reason`
and `checks`: one entry per planned check, its `result.json` or, when it has
none, a stub `{"check", "verdict": "error", "error"}`.

The rules the contract adds to "every field present and well formed"
(`contract.CODES` lists every problem it can report):

* images are pinned by digest; commits are 40 hex or null;
* a `regression` verdict is confirmed and has a regression finding; `pass` and
  `error` carry none (a regression outranks an error); `error` has a reason and
  any other verdict has `error: null`;
* a run that did not finish (`completed` 0) or a broken invariant (`invariant`
  0) is a regression, unless the baseline broke too (then `info`); a `rows`
  finding with direction `mismatch` whose sides differ is a regression;
* `incomplete` is optional: null, or a non-empty one-line string saying what was not
  measured or was cut (a regression outranks an error, so this is where a regressed
  result says its run was partial); validate checks only that it is a string or null;
* no `NaN` or `Infinity` (Python's `json.dump` writes them for a 0 baseline);
* at most 50 findings;
* a nightly's `status` is red if any check regressed, else error if any check
  errored, else green; `skipped` (with a reason) exactly when `checks` is empty;
  each check appears once, and a stub can only say error.

Keys the format does not name are ignored.

### Checking a file

```sh
uv run --project bench bench validate out/result.json          # any number of files
uv run --project bench bench validate --kind nightly nightly.json
```

It prints `<file>: <code>: <json-path>: <message>` per problem, or
`<file>: ok (<kind>)`, and exits 0 (all good), 1 (a problem) or 2 (usage). From
Python: `from falkorbench import contract`, then
`contract.validate_file(path, kind)` returns `(kind, [Problem(code, path, message)])`;
`validate_document` takes a parsed document, and `IMAGE_RE`, `COMMIT_RE`,
`METRICS`, `VERDICTS` and the other constants are the format's own.

### Acceptance pairs

`acceptance.yaml` lists, per check, image pairs whose right answer is known: the
same image compared with itself must pass; 6.2.0 against main@4c1088949 must
flag the rows #3963 fixed; 6.0.4 against 6.0.1 must catch the 3-hop read that
hangs. A check is accepted when an independent run of every pair (the acceptance
run) comes out as listed. The file's header defines each field.

`tests/test_contract.py` runs the checker over the example files
(`tests/fixtures/valid/`: 7 results and one nightly, plus the nightlies and
one-field broken variants `tests/fixtures/examples.py` builds from them) and
checks `acceptance.yaml` against the format and the query set.
`.github/workflows/perf-contract.yml` runs `bench/tests` on pull requests that
touch `bench/`.

### Shared check kit

`src/falkorbench/kit.py` holds what two or more checks would otherwise each
write: the flags, `result.json`, the threshold test and the docker calls. It is
functions only and standard library only. Each check keeps its own measuring
and judging. `args` is anything with the flags as attributes: argparse's
namespace from `add_args`, or `types.SimpleNamespace(**kwargs)` in a click command.

* Flags: `add_args(parser)`; `input_errors(args) -> [str]` (an image that is not
  pinned by digest, a bad commit, a label that is not one line, an unknown mode);
  `side(args, "candidate")`; `default_label(image)`; `out_dir(out) -> Path`
  (absolute path, `raw/` created); `machine()` (the four fields of a well-formed
  `$PERF_MACHINE_JSON`, else this host as `local`); `utc_now()`.
* Findings: `finding(name, metric, candidate, baseline, status, *, direction,
  threshold, ratio)` (it computes the ratio and turns NaN into null);
  `correctness_status(candidate, baseline)` (the 0/1 rule);
  `crossed(candidate, baseline, threshold, floor, direction) -> +1 | -1 | 0`.
  A row crosses when worse/better > threshold *and* the difference > floor, both
  strict. `floor` is required: with a zero baseline the ratio is infinite, so the
  floor alone decides. A missing, negative or non-finite value never crosses.
* Result: `select(findings)` keeps regressions, improvements, then at most 10
  info rows, 50 in all, in the order given. `ok` rows stay in `raw/rows.csv`; a
  status the contract does not know raises. `verdict(findings, errors)` is
  regression > error > pass. `result(check, args, *, started, findings, summary,
  errors, artifacts, machine_info)` also reports `input_errors`. Then
  `error_result(check, args, started, error)` and `write_result(out, result)`
  (temp file + rename).
* Always a result: `run_check(check, args, work, build, cleanup=None)`.
  * It writes an error result first, so a SIGKILL still leaves one.
  * The first SIGTERM or SIGINT raises `Stopped`, a `BaseException`, so a
    check's `except Exception` cannot swallow it. Every later signal is ignored,
    so the check's own teardown and the final write finish.
  * `Stopped` (even one the check swallowed) or any exception, `sys.exit`
    included, becomes the error passed to `build(error)`. `build` turns what was measured so far into the result, so a
    confirmed regression still outranks the stop.
  * If `build` fails, says pass despite the error, or its result cannot be
    written, `run_check` writes `error_result` instead.
  * Then `cleanup()` runs and the old signal handlers come back. `args.out` is
    made absolute.
  * `stopping()` says whether a signal arrived, so a check can skip graceful stops.
* Docker (`DockerError` on any failure): `pull(image, attempts=3)`;
  `image_env(image)`; `start(image, name, *, env, volumes, labels, cpus,
  entrypoint, command, extra, port, host_network, foreground, output) ->
  Container(name, port, proc)`.
  * By default `start` runs detached with `-p 127.0.0.1::6379` and reads the
    port back with `docker port`, so docker picks a free one (no race, never 6379).
  * A server that exits at start comes back with `port` 0, and its container is
    kept for `state` and `logs`.
  * `host_network` takes the server's own port, which must not be 6379 or
    already be in use. `foreground` keeps `docker run` as a child process.
  * `ping(port)` and `wait_ping(port, timeout_s, container)`: one PING each
    time, never re-sent. The wait keeps to its timeout and stops early once the
    container is no longer running. A foreground `docker run` that docker itself
    refused (exit 125) raises `DockerError`: no server ran.
  * `logs(name, tail)` (stdout and stderr interleaved), `state(name)`
    (`running`, `exited 139`, `gone`), `host_pid(name)`, `remove(*names)` and
    `remove_labelled("key=value")`.

```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bench example")
    kit.add_args(parser)
    args = parser.parse_args(argv)
    started, run_id = kit.utc_now(), str(os.getpid())
    cpus = os.environ.get("PERF_SERVER_CPUS")
    up, ports, p50 = {}, {}, []  # filled as work() goes; build() judges what it reached

    def work():
        if kit.input_errors(args):
            return  # result() reports them; nothing is pulled
        for side in kit.SIDES:
            image = getattr(args, side)
            kit.pull(image)
            server = kit.start(image, f"ex-{side}-{run_id}", labels={"ex": run_id}, cpus=cpus)
            up[side], ports[side] = int(kit.wait_ping(server.port, 120, server)), server.port
        if all(up.values()):
            for order in (kit.SIDES, kit.SIDES[::-1]):  # the re-run swaps the order
                times = {side: measure_p50_ms(ports[side]) for side in order}  # the check's own
                p50.append((times["candidate"], times["baseline"]))

    def build(error):
        findings = []
        if up:  # a server that never answered is completed 0, even if the run then failed
            c, b = up.get("candidate"), up.get("baseline")
            status = kit.correctness_status(c, b)
            findings.append(
                kit.finding("up", "completed", c, b, status, direction="lower_is_worse")
            )
        if p50:
            crossings = [kit.crossed(c, b, 1.5, 2.0) for c, b in p50]
            status = {1: "regression", -1: "improvement", 0: "ok"}[crossings[0]]
            if crossings[0] and crossings[1:] != [crossings[0]]:
                status = "info"  # the re-run did not repeat it: not confirmed
            findings.append(kit.finding("q1", "p50_ms", *p50[0], status, threshold=1.5))
        summary = f"{len(p50)} passes compared"
        return kit.result(
            "ex", args, started=started, findings=findings, summary=summary, errors=[error]
        )

    kit.run_check("ex", args, work, build, cleanup=lambda: kit.remove_labelled(f"ex={run_id}"))
    return 0  # result.json was written
```

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
| `src/falkorbench/contract.py` | the nightly perf checks' `result.json` / `nightly.json` format and its checker (`bench validate`) |
| `src/falkorbench/kit.py` | what the nightly perf checks share: flags, `result.json`, the threshold test, docker |
| `src/falkorbench/coverage.py` | instrumented build, run the set once, report graph-crate line coverage |
| `Dockerfile` | the CI measurement image, `FROM …:edge-c` |
| `pmc_tool.c` | Apple Silicon PMU counters (kperf/kperfdata private frameworks) |
| `acceptance.yaml` | the nightly perf checks' acceptance pairs |
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
