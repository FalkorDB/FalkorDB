# The fuzz check: a seeded soak of long reads, writes and cheap reads

`bench fuzz` (this package, `falkorbench.fuzz`) is one check of the perf nightly
(`bench/checks.yaml`, name `fuzz`, `runner: vm`). It starts each image's own server, loads it with long reads, open-loop writes and
cheap reads on several graphs while a separate process PINGs it every 10 ms, and reports which of
its checks held on the candidate and on the baseline.

```
uv run --project bench --frozen --group fuzz bench fuzz --candidate <image@sha256:…> --baseline <image@sha256:…> \
    --out <dir> --mode <queue|nightly|long> \
    [--candidate-label …] [--baseline-label …] [--candidate-commit …] [--baseline-commit …]
```

It needs Docker, uv and both images pinned by digest. It writes `<dir>/result.json` (contract
schema 1) whatever happens, including when it could not measure (`verdict: error`), and exits 0
once that file is written.

## Modes

| mode | what runs | time on an n4-standard-16 |
| --- | --- | --- |
| `queue` | the control epoch: 4 runs (candidate, baseline, baseline, candidate), each a fresh server and a 300 s epoch, plus 2 more runs only to confirm a timing break | about 30 min, 45 with a repeat |
| `nightly` | the control epoch, then graph-build timing: 4 fresh-server builds at scale 0.25 in the same order (2 more to confirm) | about 35 min, 50 with repeats |
| `long` | both, then seeded random epochs on both sides (one server-hour per side per seed hour, order swapped every hour, 3 epochs of 300 s each; an hour pair starts only if its estimate fits before 3.5 h after the start, and then runs whole on both sides) and the repros; then replays of up to 2 seeds where the candidate broke a check the baseline held | about 3.5–4 h |

`--mode` sets only the length. The checks and the way they are judged are the same in every mode.
`long` runs on one image flavour per call: to cover the Valkey and the Redis images, the caller
runs the check once per flavour.

The `nightly` time includes a 6.2.0-like candidate whose build is slow (about a minute more per
control run). `timeout_minutes` in `checks.yaml` is 90.

## What one run is

A run is one server-hour on one side:

1. A fresh container from the image, started with the image's own default server,
   `${FALKORDB_SERVER:-redis}-server` from the image's config env, and `--loadmodule /var/lib/falkordb/bin/falkordb.so
   THREAD_COUNT <n> CACHE_SIZE 200 EFFECTS_THRESHOLD 0`, pinned to `PERF_SERVER_CPUS`, on a free
   port (never 6379). Beside it, a second container from the same image with no module: the plain
   server that tells a stall of the host from a freeze of FalkorDB. It is pinned to
   `PERF_CLIENT_CPUS`.
2. The graphs are built (`model.py`: 14 labels from a halving series, 2.4M nodes and
   3.6M edges at scale 1, edges wired by node id). Each cheap read is then timed alone, 7 times,
   on a quiet server (its calibrated cost).
3. One or more epochs. The drivers run under `taskset` on the client CPUs (the VM workflow does
   that). No client re-sends: no socket timeout and no retries on measured requests (redis-py 8
   retries by default, so retries are off explicitly), and every client speaks RESP2. Reads and
   writes are open loop, sent at their scheduled time whether or not earlier ones have returned,
   and timed from that schedule.
   * long reads, in their own streams: a full edge scan that visits every edge and returns about
     3% of them, or a full count followed by one count per relationship type (TIMEOUT 6 s each),
     with an upsert sent 1 ms after half of them;
   * cheap reads with their own TIMEOUT (1 s): keyed lookups, a filtered sort and a distinct
     (44 requests/s per graph at multiplier 1); and an unindexed lookup that reads every node of
     the largest label (1/s, TIMEOUT 0.6 s; too slow to be judged as cheap on a full-size graph);
   * writes on busy graphs: MERGE upserts and a bulk SET (12 requests/s at multiplier 1), Poisson,
     or held back while a long read runs ("holdoff").
4. After each epoch the graphs are checked against the ledger of what the writes committed.

**The control epoch** is fixed: two graphs, graph 0 at scale 1.0 is busy (one long-read stream,
Poisson writes, so writes keep arriving while a long read holds the graph), graph 1 at scale 0.1
is quiet (reads only), THREAD_COUNT 32, 300 s. A writer that blocks the server while it waits
for a long read shows up as a freeze. A writer gate that makes later reads wait inside their own
TIMEOUT shows up as cheap reads timing out (the behaviour #3685 fixed).

**Long-mode epochs** are drawn from the seed (`model.epoch_params`): which graphs are busy, 0–3
long-read streams per busy graph, read multiplier 0.5–10 (capped at 400 requests/s in all), write
multiplier 0.25–4, holdoff or Poisson, indexes (half the epochs), unique constraints (a third,
with deliberate violations), a write TIMEOUT of 0.5, 1.5 or 3 s (a third), BGSAVE (half of those
without constraints). Each server-hour (`model.hour_params`) has 2–4 graphs at scales 0.25–2.0
(4.0 in all) and THREAD_COUNT 4, 8, 16 or 32. A server is reused for the next epoch only if the
last one put its schema back: an epoch whose index or constraint commands are still running 60 s
after the load, or whose indexes and constraints could not all be dropped, ends the server-hour
there and fails `no_unexpected_errors`. The default seed comes from the candidate's digest,
so a re-run of the same candidate repeats it (`--seed` overrides).

## The checks

| check | holds when |
| --- | --- |
| `no_freeze` | no FalkorDB PING is answered more than 500 ms after it was due while the plain server answered within 100 ms (a reply from it within ±1 s is needed). A freeze seen counts however few PINGs came back; holding is unmeasured (null) when the prober got fewer than half the PINGs the epoch should have had from either server, or when a late PING had no plain-server reply near it. Only a PONG is a sample |
| `no_timeouts_while_waiting` | no cheap read times out. Cheap: its calibrated cost is under 5% of its TIMEOUT. A read that is not cheap is listed under `not_judged`, never counted. Unmeasured when no cheap read was answered or timed out |
| `writers_dont_starve` | every write finishes within 1 s of the end of the long reads it waited behind, counted from when it reached the head of its graph's write queue (writes queue in the order they reached the server) |
| `quiet_graphs_stay_fast` | on a graph with no long reads and no writes, every read finishes within 100 ms of its calibrated cost |
| `requests_drain` | every read finishes within 5 s, plus the time to work off the reads queued with it, of the end of the long reads it waited behind; nothing, a long read or a write sent with one, is still open 60 s after the load stops |
| `writes_add_up` | node counts of the upserted labels and edge counts of their pairs are within what the committed writes allow (between "committed" and "committed + outcome unknown"), every acknowledged new key has its node, with no extra edge and no key on two nodes |
| `upsert_edges_complete` | no pair type has fewer edges than the committed upserts created, and every acknowledged new pair has its edge (an engine that drops the upsert's final MATCH rows loses them, #3082) |
| `no_error_replies` | no request gets an error reply or loses its connection (an engine that rejects or drops a query breaks it) |
| `server_stays_up` | the server neither exits nor stops answering: no PING unanswered for 60 s during the load, and a PONG to one of 3 PINGs (10 s each) after it and after any failed step (`completed` 0 if it does). A container Docker could not start, or a Docker command that timed out, is not a crash: that run is unmeasured |

Long mode adds `timed_out_writes_leave_nothing`, `constraints_hold`, `index_matches_scan`,
`no_unexpected_errors` and the repros (`repros.py`). Memory growth over a server-hour
(`memory_steady` in an hour's last `epoch.json`) is recorded but not judged: a one-epoch replay
could never confirm it. The repros are
a write, a constraint command and an index build while a long read runs (no freeze), writers
behind a flood of reads at THREAD_COUNT 10, and cheap reads with TIMEOUT behind a waiting writer.
An error reply fails a repro (the engine refused the request); a request that got no reply at all
makes it a crash (the server died or hung) or unmeasured. A freeze repro whose scenario did not
form is unmeasured too: the command reached the engine after the long read ended, the index was
ready before the read started, or the PING probe stopped.

## How a check becomes a finding

Every check is a finding with metric `invariant` (1 held, 0 broke) named `control <check>`
(`long <check>`, `long repro <name>` in long mode); `server_stays_up` uses metric `completed`.

* **Timing checks** (`no_freeze`, `no_timeouts_while_waiting`, `writers_dont_starve`,
  `quiet_graphs_stay_fast`, `requests_drain`, `no_error_replies`) depend on timing, so a break
  needs a repeat. They are judged on the first four runs. When the candidate broke in either of its
  two runs and the baseline held in both, one more pair runs (baseline, candidate), and one rule
  (`verdict.judge_repeat`, the same for long-mode seeds and repros) decides: the candidate breaks
  again and the baseline holds → regression (0 vs 1); the baseline breaks on the repeat → 0 vs 0,
  `info`; the candidate holds → 1 vs 1, `info`; the repeat did not run or measure the check → the
  candidate is null. A break seen only in the extra pair makes a finding `info`, never a
  regression or an improvement. A timing break against a baseline that was not measured is not
  judged (both null). A request that waited more than 50 ms in the driver for a free
  connection is driver-limited and left out of `writers_dont_starve`, `quiet_graphs_stay_fast`
  and `requests_drain`; `quiet_graphs_stay_fast`, which times a read from its schedule, also
  counts the time it waited to be sent. Each count is in `epoch.json`, and a check with every
  sample driver-limited, or a quiet graph with no read answered in time to judge, is unmeasured.
  `no_timeouts_while_waiting` counts every TIMEOUT reply: the engine's TIMEOUT starts when the
  query reaches it, so time in the driver cannot cause one.
* **Data checks** (`writes_add_up`, `upsert_edges_complete`) and a crash count at once.
* Contract v1.4 sets the status: 0 on the candidate only is `regression`, 0 on both sides is `info`
  ("broken on both sides (not new)"), 0 on the baseline only is `improvement`.
* **Graph-build timing**, metric `p50_ms` (the median over a side's first runs, the values that
  decide the row's status; the extra pair's medians are in the summary):
  `build edge phase` and `build node phase` (total wall time of the phase), and
  `build edge query 1 row` and `build edge query 100 rows`. These run the edge-wiring shape on
  its own after the build, 5 times, median. A regression needs ratio > 1.5 and a difference over
  500 ms (a phase) or 2 ms (a query), in the first four runs and again in an extra pair.
  The wiring query's plan is in each build run's `build.json`.
* **Long mode** pairs epochs by seed. A timing check (the ones above and `no_unexpected_errors`)
  counts against the candidate only on a seed where it broke on the candidate, held on the
  baseline, and did the same again when that seed was replayed on both sides. Up to two seeds are
  replayed; a candidate-only break whose replay did not measure the check leaves it unmeasured.
  A data check (`writes_add_up`, `upsert_edges_complete`, `timed_out_writes_leave_nothing`,
  `constraints_hold`, `index_matches_scan`) or a crash counts at once, in any epoch, replays
  included, and also on a seed (or replay) the baseline never ran or did not measure. A seed
  where the candidate broke and the baseline held is a regression however the other seeds went. A
  repro is a timing check unless the server crashed in it, in its first run or its repeat. A server
  that dies during a server-hour is a crash on the epoch it did not finish.

The verdict is `regression` if any finding is a regression. Otherwise it is `error` if something
was not measured, with the reason in `error`: a run whose harness or Docker failed, a check that a
run could not judge, a build phase with no measured run on one side (a baseline that did not start,
or crashed in every build run), a long-mode seed that ran on one side only, a run skipped for the
time limit. Otherwise it is `pass`. `result.json` lists the regressions and improvements, then up
to 10 `info` rows; every finding, `ok` ones included, is in `raw/rows.csv` (phase "finding (…)").

### Reading a result

* A check that breaks on both sides is reported as `info`: it was there before the candidate, so
  the candidate did not introduce it. The summary counts these.
* A baseline that freezes holds `no_timeouts_while_waiting`. Its reads wait while the server is
  blocked, before their TIMEOUT starts, so they don't time out. Against such a baseline, a
  candidate that stops freezing but makes reads wait inside their TIMEOUT shows `no_freeze` as an
  improvement and `no_timeouts_while_waiting` as a regression in the same result. Both are real.

## Artifacts

`raw/rows.csv` holds every check of every run (1/0, empty when not measured), each phase's build
time and the per-op p99s. `raw/runs.json` summarises every run. `raw/<phase>-<n>-<side>/` holds a
run's `hour.json`, `server-tail.log`, `pings.csv`, and per epoch `epoch.json` (every check with its
numbers) and `raw.json.gz` (every request). `raw/images.json` names the server binary and module
found in each image.

## Engines

The check runs C images (Redis-based) and Rust images (Valkey- or Redis-based) alike. It needs
only the module at `/var/lib/falkordb/bin/falkordb.so` (or anywhere in the image) and the server its
`FALKORDB_SERVER` names (default `redis`): an image without that binary is an error, never a switch
to the other one. An engine that rejects a long-mode feature (for example
a constraint command) fails `no_unexpected_errors` on that epoch. That is `info` when both sides
reject it.

## Development

`--epoch-s`, `--drain-s`, `--control-scales`, `--build-scale`, `--long-hours`, `--seed`, `--net`,
`--max-conns`, `--skip-control`, `--long-scale-range`, `--repros` change the run for development. The nightly never passes them. Without
`PERF_SERVER_CPUS` / `PERF_CLIENT_CPUS` / `PERF_MACHINE_JSON` (a laptop) the servers run unpinned
and `machine.runner` is `local`. A quick local smoke:

```
uv run --project bench --frozen --group fuzz bench fuzz --candidate … --baseline … --out /tmp/fuzz --mode nightly \
    --epoch-s 40 --drain-s 20 --control-scales 0.05,0.02 --build-scale 0.02
```

Tests (no server, no Docker), in `bench/tests/test_fuzz.py`. `perf-contract.yml` runs them with the
rest of `bench/tests`:

```
uv run --project bench --frozen --group dev --group fuzz \
    python -I -B -m pytest bench/tests/test_fuzz.py -v -p no:cacheprovider
```

They validate every result shape with the contract's validator (`falkorbench.contract`).

## Limits

* No run starts that would end after 80 min (`queue`), 85 min (`nightly`) or 4.5 h (`long`) from the
  start; a run left out that way is unmeasured. A SIGTERM (the workflow's timeout) stops the check,
  removes its containers and still writes `result.json` (`error`).
* A timing break that does not repeat is reported (`info`) but does not fail the night. A
  regression that shows in only one run of three is missed.
* The control epoch uses fixed rates and one thread count (32), so a regression that needs a
  small worker pool or index and constraint traffic shows only in long mode.
* Build timing measures two phases of one graph shape at one scale (0.25). A cost that grows with
  graph size shows more at the control's scale 1.0, whose build time is in `raw/rows.csv`
  (`control-build`) but is not judged.
