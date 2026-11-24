# `pokec`: FalkorDB/benchmark on the Pokec dataset

A check of the nightly perf contract (`tests/perf/README.md`): it runs the
mixed Cypher workload of [FalkorDB/benchmark](https://github.com/FalkorDB/benchmark)
on the Pokec social graph, against the candidate and baseline images, on one
machine and in one run, and writes `<out>/result.json`.

It exists because of 6.0.2–6.0.4: the 3-hop read
`MATCH (s:User {id:$id})-->()-->()-->(n:User) WHERE n.age >= 18 RETURN DISTINCT n.id`
planned from the age index and took over 20 s instead of ~20 ms, and under 5%
writes the server stopped answering for 16–30 s at a time (Nov 23 run,
`#3821` not backported). No other check runs a mixed workload under concurrent
writes, so this is the check that sees a query plan going wrong *and* what that
does to everybody else on the server.

```
python3 -I tests/perf/pokec/pokec.py --candidate <image@sha256:…> --baseline <image@sha256:…> \
    --out <dir> --mode <queue|nightly|long> \
    [--candidate-label …] [--baseline-label …] [--candidate-commit …] [--baseline-commit …]
```

Standard library only. It needs Docker, git, `uv`, network access, and on a
machine without a C toolchain, root (or passwordless sudo) to `apt-get` one.

## Modes

`--mode` changes the length, never the method.

| mode | dataset | reps a side, each workload | re-run of a workload with a row to confirm | fails on | measured duration (n4-standard-16) |
| --- | --- | --- | --- | --- | --- |
| `queue` | medium | 1 | 1 rep a side | a run or a query that does not finish, a new PING freeze (> 1 s) | not measured (phase 3) |
| `nightly` | medium | 3 | 2 reps a side | the above, plus a latency or throughput row past 1.3x | see "Durations" below |
| `long` | medium, then large | 3 | 2 reps a side | as nightly | not measured (phase 3) |

medium is 100,000 users and 1,768,515 `Friend` edges; large is 1,632,803 and
30,622,564. `queue` reports every latency row as `info` and never fails on one.

### Durations

Measured through `_perf-vm.yml` (run 45389482775, a fresh n4-standard-16 per
pair, so each includes `apt-get`, rustup and a 224–270 s build of the
benchmark):

| pair | verdict | runs | re-run | the command's wall time |
| --- | --- | --- | --- | --- |
| main@4c1088949 vs itself | pass | 12 | none | 17.4 min |
| 6.0.4 vs 6.0.1 | regression | 12 (all six 6.0.4 runs stopped at their deadline: 180 s read-only, 227 s with writes) | none (the candidate did not finish) | 31.4 min |
| C v4.22.4 vs itself | pass | 12 | none | 22.1 min |

A healthy run takes 24–31 s read-only and 71–130 s under 5% writes. A re-run
adds 2 reps a side of a workload, about 2 or 9 min. `timeout_minutes: 75`
leaves room for the slowest of these plus a re-run of both workloads.

## What one run does

1. `docker run` the image by digest, pinned to `PERF_SERVER_CPUS`
   (`--cpuset-cpus`), on a free port (never 6379), with
   `FALKORDB_ARGS` = the image's own `FALKORDB_ARGS` plus `THREAD_COUNT 10`.
   Every image measured so far ships
   `MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000`.
2. Bulk-load the dataset into a fresh server (graph `falkor`,
   `-j INTEGER -s -i User:id -i User:age`), wait until the indexes are
   `OPERATIONAL`, and check the counts: a load that does not produce exactly
   the dataset's users and edges is a run that did not finish.
3. Start a PING probe on its own connection: one PING every 200 ms, each timed
   from its own send; a reply that takes 30 s is a timeout, and a PING still
   waiting when the run ends is recorded with how long it has waited.
4. `benchmark run --vendor falkor --parallel 8 --mps 2000` with the
   workload's query file, under a deadline (below).
5. Parse `metrics.prom` (per-query p50/p95/p99, global p50/p95/p99, timeout
   and failure rates), `meta.json` (the query phase's elapsed time) and the
   run's log (every failed request, with its query), save the server's log,
   remove the container.

The client side (bulk loader, benchmark, probe) runs on `PERF_CLIENT_CPUS`:
`_perf-vm.yml` starts the command under `taskset`.

Order: rep 1 runs the baseline first and every later rep swaps, so the two
sides interleave. With 3 reps (`nightly`, `long`) the order changes at least
once; with 1 rep (`queue`) it changes only when a re-run is needed, since the
re-run starts with the candidate. Before the first run each image
is started once, empty, to record its server and module (`raw/servers.json`),
read back `THREAD_COUNT`, and test the timeout (below).

## Workload

Two query files are generated once per invocation (`benchmark generate-queries`,
10,000 queries, the `baseline` profile), and every run of both sides uses the
same file: generation is unseeded, so the files are kept in `raw/`.

* **read-only**: write ratio 0.
* **5% writes**: write ratio 0.05 (≈ 500 writes over 10 write types).

About 46 read query types, ~200 requests each. The four whole-graph algorithm
queries (`algo_pagerank_summary`, `algo_max_flow_single_pair`,
`algo_msf_summary`, `algo_harmonic_summary`) are switched off with the
benchmark's own `--enable-algo-*=false` flags. With them on, main@4c1088949
against itself, read-only, took 589 s for one run instead of 23.6 s. 354 of
10,000 requests were abandoned at the client's own 10 s guard, and the global
p99 was 3.9 s: harmonic centrality's p50 was 4.1 s and it checks the timeout
only after it finishes. The Nov 23 run's 58.7 s for 10,000 queries fits only
without them, and 6.0.x and C v4.22 lack some of them.

## Tools, pinned

| what | pin | how the check gets it |
| --- | --- | --- |
| FalkorDB/benchmark | `c366437bf462070d9da96b75634c975d57b01707` | `git fetch` of that commit, `cargo build --release --locked` with Rust 1.100.0 (rustup, minimal profile); needs `build-essential pkg-config protobuf-compiler cmake`, which it `apt-get`s when missing |
| falkordb-bulk-loader | `d4e6718ebdc26b5dec4169d7dbb5a66f211e2701` | `git fetch`, `uv sync --frozen --no-dev` (its uv.lock: falkordb-py 1.7.1, redis-py 8.1.0) |
| Pokec medium | `pokec_medium_import.cypher`, sha256 `3614ffa9…c664` | downloaded from memgraph's public S3 bucket, converted to CSV in Python (the Nov 23 conversion, `bench_capacity = 1 + (31·src + 17·dst) mod 20` on every edge) |
| Pokec large | `pokec_large.setup.cypher.gz`, sha256 `9b6ebec0…e8d3` | same |

`--tools-dir` (default `$PERF_POKEC_TOOLS`, else `~/.cache/falkordb-perf/pokec`)
keeps all of it between runs. A GCE VM is fresh every night, so the nightly
pays for the build: 4 min 11 s on the 8 client CPUs of an n4-standard-16. The
dataset download and conversion took 8 s.

## Clients never re-send

Read from the source, not assumed:

* **FalkorDB/benchmark c366437.** Its vendored falkordb-rs has a `RetryPolicy`
  that is disabled by default (`vendor/falkordb-rs/src/retry.rs`), and the
  benchmark never sets one. The client is a pool of 8 redis 1.8.2 multiplexed
  connections (`ConnectionStrategy::Pooled`, `get_async_connection`) with
  `response_timeout` None: a command is written once and its reply, or its
  error, is returned. The per-query timeout is server-side:
  `FALKOR_QUERY_TIMEOUT_MS`, sent with every query.
  A tokio guard at that timeout + 5 s drops the request (`Elapsed`), which is
  counted as failed, not sent again. So every latency is one request.
* **What the benchmark leaves out.** A failed or timed-out request is not in
  its percentiles, only in a per-query rate. This check counts every failed
  request per query from the run's log, so a query that stops finishing cannot
  vanish from p99 (below). It also checks those counts against `metrics.prom`'s
  rates, and errors out if they disagree (a log format the parser no longer
  reads). Two query types can send the same text: `vertex_on_label_property`
  and `vertex_on_label_property_index` share a template, so the same random id
  collides. A failure logged under such a text is split between the two from
  their `metrics.prom` rates; timeouts and errors mixed across both types are an
  error, since the log does not say which type got which.
* **Every request has to end.** The benchmark ignores its workers' join
  errors, so a worker that died after taking a request could leave a run at
  exit 0 with the request unaccounted for. The check counts the requests that
  ended (`operations_total`, per query type) and treats a run whose count falls
  short of the generated file as not finished.
* **Latency starts at dequeue.** The benchmark times a request from the moment
  a worker takes it to its reply (closed loop, 8 workers), not from its
  scheduled time. Every image measured finished slower than 2000 queries/s, so
  no worker waited for its slot.
* **The bulk loader** uses redis-py 8.1, whose default socket timeout plus
  `Retry` would re-send a `GRAPH.BULK` batch slower than the timeout. The check
  passes `?socket_timeout=3600` in the URL, which redis-py parses, and checks
  the node and edge counts after every load.
* **The PING probe and the check's own queries** use a RESP client in
  `pokec.py`: one request, one reply, no retries.

## Timeouts

`FALKOR_QUERY_TIMEOUT_MS` is 15,000 (the benchmark's default is 180,000). The
slowest normal query on main@4c1088949, `all_shortest_paths_len`, has a p99 of
3.5–3.7 s read-only, so 15 s is about 4x that. 6.0.4's 3-hop read needs more
than 20 s alone. With 180 s, one such request held a 6.0.x server, writer and
GIL included, for minutes and the run never finished. With 15 s it ends as a
counted timeout, and the run goes on to name it.

Every image ships `TIMEOUT 1000`. Before measuring, the check sends a read that
would run for hours with `timeout 2500` and requires that it timed out between
2.2 s and 4.0 s. On main, 6.0.4 and C v4.22.4 it stopped at 2.50 s, and at
1.00 s with no timeout argument. A stop outside that window means another limit
applied, and a server whose own `TIMEOUT` falls inside the window cannot tell
the two apart. Either way the per-query timeout is not shown to apply on that
image, and the check is an `error`.

## Did not finish

A run did not finish when the server did not start or answer, the load failed or
produced the wrong counts, the benchmark exited non-zero, or the run passed its
deadline. The deadline is 3x the other side's slowest finished run of the same
workload (at least 180 s on medium, 900 s on large). With nothing to compare yet
it is 900 s (medium) or 3600 s (large). Measured A/A runs took 23–24 s
(read-only) and 92 s (5% writes) of wall time, so a run stopped at its deadline
was at least 3x slower than the other side.

## Findings

Names end in `[<workload>]` (`[large, <workload>]` for the large dataset). The
expansion queries read `3-hop expansion, age >= 18
(aggregate_expansion_3_with_filter)`, so the 6.0.4 acceptance pair's `3-hop` pattern
matches its rows.

| finding | metric | when it is a regression |
| --- | --- | --- |
| `benchmark run finished [w]` | `completed` (1 if every run of the side finished) | candidate 0, baseline 1: at once |
| `server started [dataset]` | `completed` | the candidate image does not answer PING within 120 s: at once |
| `<query> [w]` | `completed` (1 if every request of that query returned) | candidate 0, baseline 1. An error reply counts at once (a correctness failure). A timeout counts once it happens in 2 separate runs of the side. In 1 run only, the workload is re-run, and if it does not happen again the row is `info` |
| `all queries [w]`, `<query> [w]` | `p50_ms` `p95_ms` `p99_ms` (median over the side's finished runs) | past 1.3x **and** past an absolute floor (p50 1 ms, p95 5 ms, p99 15 ms), in the first pass **and** in the re-run. Under 5% writes only `all queries` and a read query's p50 (below) |
| `throughput [w]` | `throughput` (10,000 / the query phase's seconds; lower is worse) | below 1/1.3 in both passes |
| `PING max [w]` | `ping_max_ms`: the slowest PING reached in at least 2 of the side's runs, finished or not (the second-highest run maximum) | the candidate froze (a PING over 1 s) in at least 2 runs and its value is above 1.3x the baseline's. One freeze re-runs the workload; freezes as long on the baseline are `info` |

Both sides 0 on a `completed` row is `info` (contract v1.4), and the summary says
so.

### Thresholds measured

From four A/A runs, each 3 runs a side on n4-standard-16 with the
`_perf-vm.yml` CPU split: main@4c1088949 and C v4.22.4, once on the dev VM and
once in the `_perf-vm.yml` acceptance run 45389482775.

* **Read-only, per query.** p50 never moved: no row's worst split of its six
  readings (median of the top three over median of the bottom three) passed
  1.3x. p95 and p99 of the millisecond queries did. In the acceptance run's A/As, a
  worst split moved `indexed_in_list_predicate` p95 by 4.0 ms on C, and
  `aggregate_expansion_4_with_filter` p99 by 14.7 ms (1.47x) on main.
* **How the floors were set.** A false regression needs the first pass (3 runs
  against 3) and the re-run (2 against 2) to cross. For each gated row, the
  check's rule was applied to every 3-vs-3 and 2-vs-2 split of its six
  readings, and the two crossing probabilities multiplied. Floors of p95 3 ms
  and p99 10 ms gave 0.02–0.04 expected false regressions per night in the two
  acceptance-run A/As, which is roughly one false red night a month. p95 5 ms and
  p99 15 ms gave none in all four, so those are the floors.
* **5% writes, per query.** A read's tail there is the writer gate (#3816), not
  the query: whether 5% of a query's ~200 requests waited behind a writer decides
  its p95. One image against itself read `indexed_or_predicate` p95 at 5.9, 22.7
  and 43.0 ms in three runs. A write's latency is mostly its wait for the write
  lock: `single_vertex_write` p50 411–557 ms, `merge_user_insert_path` p50
  222–332 ms on one image. So under writes only a read query's p50 gates; the
  rest is in `raw/rows.csv`.
* **Whole workload.** Global p95 and p99 stayed within ±3% of each other in
  both workloads (5% writes p95 235.5–245.8 ms, p99 925.7–950.3 ms), p50 within
  0.1 ms, and the query phase within ±2% (read-only 27.8–28.1 s, 5% writes
  76.2–78.2 s), so those rows gate in both workloads.

A workload is re-run (2 reps a side in `nightly` and `long`, 1 in `queue`)
when a gated row crossed in the first pass, a query timed out in one candidate
run only (also when the baseline timed out once too), or one candidate run
froze. It is not re-run when a candidate run of
that workload already did not finish: that is a regression already, and each
more run would spend another deadline.

`findings` holds every regression and improvement, then up to 10 `info` rows,
worst first, at most 50. Every comparison is in `raw/rows.csv`, every run in
`raw/runs.csv`.

## Verdict

* `regression` when any finding is (`confirmed: true`: each regression above was
  repeated, or is a correctness or did-not-finish failure that needs no re-run);
* else `error` when the check could not set up (tools, dataset, image pull, the
  baseline image not starting, the timeout test), was stopped by a signal, a
  baseline run did not finish, a side has no finished run of a workload, a row
  crossed in the first pass and the re-run gave no value to repeat it with (no
  finished re-run, or none of that query's requests returned), a finished run
  had no PING answered, or no run was made at all;
* else `pass`. Queries that failed on both sides are named in the summary.

`artifacts` lists only the files this run wrote (an `--out` reused by hand may
hold an earlier run's; `_perf-vm.yml` starts from an empty one).

`result.json` is written at the start and rewritten after every run as an
`error` checkpoint ("the check did not finish"), so a check killed by the
workflow's timeout still leaves one, and a regression confirmed before the kill
still says `regression`.

## Server binaries differ between images

Each image runs its own server. 6.2.0 and main@4c1088949 set
`FALKORDB_SERVER=valkey` (valkey-server 9.2.0). 6.0.x and the C images run
redis-server. So 6.0.4 vs 6.0.1 is Redis against Redis, and any 6.2.0/main
against 6.0.x pair also changes the server. `raw/servers.json` records each
side's server name, version, executable and module.

## C images

`falkordb/falkordb-server:v4.22.x` (Redis, module at
`/var/lib/falkordb/bin/falkordb.so`) runs the same way. A query a side does not
support ends in error replies, which on both sides is an `info` row, not an
error for the whole check.

## Artifacts

* `raw/check.log`: what the check did, with times; `raw/setup.log`: tool setup.
* `raw/runs.csv`, `raw/rows.csv`, `raw/servers.json`, `raw/notes.txt`.
* `raw/queries-<dataset>-<workload>.jsonl`: the generated query files.
* `raw/runs/<dataset>-<workload>-<side>-p<pass>r<rep>/`: `load.log`, `run.log`,
  `ping.csv`, `server.log`, `bench/falkor/{metrics.prom,meta.json}`.

## Running it

On the VM, `_perf-vm.yml` runs the `checks.yaml` entry. Elsewhere, run it
unpinned (`runner: local` in `machine`); a local result is not a contract run.
`--reps N` and `--workloads read|write5` (hidden) shorten a development run.

Tests: `python -m pytest tests/perf/pokec` (no Docker; runs in
`perf-contract.yml`). They check every result they build with
`tests/perf/validate.py`.

## Limits

* One query mix on one dataset size: a regression in a query shape the
  benchmark does not send, or that only shows on a bigger graph (medium is
  100k nodes), is not seen here (the size probe and `long` cover some of it).
* The workload is closed-loop with 8 clients: a freeze delays at most 8
  requests, so the PING probe, not the latency rows, is what sees it.
* Per-query tails, and every row of a write query, are not gated under writes
  (above); a query that regresses only there shows in the whole-workload rows or
  not at all.
* 1.3x with 3 reps a side and a re-run: a smaller change, or one under the
  floors (a 5 ms p99 that becomes 19 ms), passes.
