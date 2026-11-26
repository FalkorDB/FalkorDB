# Release soak harness

Twelve production-shaped scenarios that run against a FalkorDB module (Rust or C) for
20 minutes each, check correctness against Python ground truth, and record latency,
memory and stability. `score.py` turns the results into a release-confidence score
(correctness 40, stability 25, resources 15, performance 20) and `gen_final.py`
renders the HTML report.

| Scenario | What it models |
|---|---|
| s01_social_feed | follow graph, feed reads under writes |
| s02_fraud_realtime | streaming MERGE ingest of payments plus fraud traversals |
| s03_graphrag_vector | docs/chunks/entities with a vector index (kNN + traversal) |
| s04_recommendation | collaborative-filtering reads on a large bipartite graph |
| s05_multitenant_churn | thousands of graphs created, queried and deleted |
| s06_large_analytics | var-length paths, shortest paths, `algo.*`, big aggregations |
| s07_mvcc_consistency | snapshot-isolation invariants under concurrent writes |
| s08_durability_recovery | AOF/RDB persistence, kill and restart, data checks |
| s09_replication | primary plus replica, replica-side reads and lag |
| s10_schema_under_load | index/constraint create and drop while traffic runs |
| s11_client_chaos | misbehaving clients next to a steady victim workload |
| s12_iot_retention | time-series ingest with rolling retention deletes |

## Requirements

- Redis 8.x `redis-server` on `PATH`, or set `SOAK_REDIS` (module mode).
- Or, on Linux, Docker and a `falkordb-server` image (image mode, below).
- bench's environment with its soak dependency group (`numpy`, `psutil`): from the
  repository root, `uv run --project bench --group soak python -I …` (the commands below
  are run from this directory, with `PY="uv run --project ../../.. --group soak python"`).

## Running

One scenario, against a module path or (Linux) an image:

```bash
$PY -I scenarios/s01_social_feed.py --module <path/to/module> --port 19000 \
  --out full/s01_social_feed-rust --duration 1200 --scale 1.0
$PY -I scenarios/s01_social_feed.py --image <falkordb-server image@sha256:...> --port 19000 \
  --out full/s01_social_feed-img --duration 300 --scale 0.25
```

With `--image`, `soakkit.py` runs the image's own server binary
(`${FALKORDB_SERVER:-redis}-server`, as its `run.sh` picks it: `valkey-server` on 6.2+
Rust images, `redis-server` on C and older ones) and its module
(`/var/lib/falkordb/bin/falkordb.so`) in a container with host networking, as the
calling user, with the scenario's work directory bind-mounted at the same path, so
RDB/AOF files, logs, kill/restart (s08) and a replica (s09) behave as in module mode.
It starts, probes and removes the container with bench's shared check kit
(`falkorbench.kit`: `start`, `host_pid`, `wait_ping`, `remove`).
It passes the scenario's own module arguments and nothing else: the image's `run.sh`
and its default `FALKORDB_ARGS` (`MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE
10000` on the falkordb-server images) are deliberately not applied, so both sides of a
comparison run with identical server arguments (s06's analytics would otherwise hit
the 1 s default timeout). `Server.pid` is the server's host PID, so the 1 Hz monitor
(psutil) works as before. Image mode needs Linux (host networking and host PIDs).

All Rust scenarios, one at a time, only when the machine is quiet (1-minute load
under 8 for 5 minutes; a run that sees load above 20 is discarded and retried):

```bash
./run_rust_main.sh <path/to/libfalkordb.dylib> <commit-label>
./run_c_rerun.sh <path/to/falkordb.so> <c-label>      # C baseline
$PY -I score.py full && $PY -I gen_final.py          # score and report
```

Results go to `full/<scenario>-<engine>/` (ignored by git); a previous run of the
same scenario is archived as `<dir>@<old-label>`. `findings.json` is the ledger of
bugs found so far, which the report reads.

Check the load average (`/proc/loadavg`, or `sysctl vm.loadavg` on macOS) before
trusting any numbers: outside load produces bogus p50/p99 regressions.

## Nightly check (`bench soak`, `check.py`)

`bench soak` is the soak as a perf check in the nightly contract (`bench/README.md`,
results schema 1). From the repository root:

```bash
uv run --project bench --group soak bench soak --candidate <image@sha256:...> \
  --baseline <image@sha256:...> --out out --mode nightly --check soak-s07 \
  --scenarios s07
```

The flags, the input checks, result.json and the stop signals are bench's shared check
kit (`falkorbench.kit`); the scheduling, the comparison and the gates below are the soak's.

It pulls both images and runs each scenario once per side on this machine, one side
after the other, flipping the order every other scenario. Servers are pinned to
`$PERF_SERVER_CPUS` when set (the VM workflow sets it). It always writes
`out/result.json`; `out/raw/` holds `rows.csv` (every comparison), `runs.csv` (every run:
exit code, wall time, setup time), `servers.json` (server binary and
`--version` per side) and each run's scenario output (`summary.json`, `checks.json`,
logs, latency series; RDB/AOF files are deleted).

Findings, named `sNN ...`:

| finding | metric | rule |
| --- | --- | --- |
| `sNN completed` | completed | 0 when the scenario raised, outlived its cap or a server log shows a crash; candidate-only is a regression at once |
| `sNN <check>` | invariant | each check the scenario records (1 held, 0 broken). A correctness check broken on the candidate only is a regression at once; broken on both sides it is info (not new). A latency or resource check (`TIMING_RE`: p99, SLO, schedule kept, RSS/used_memory plateaus, drift, replication lag...) counts as broken only when every pair of runs, first pass and re-run, has it broken on the candidate and held on the baseline. Generous hang bounds (recovery < 180 s, replica sync ≤ 900 s, an index reaching OPERATIONAL) are correctness: missing one is a run that did not finish |
| `sNN <op>` | p50_ms, throughput | each worker op with at least 200 successful samples on the baseline (an op the candidate nearly stopped doing still compares, at its throughput); crosses at p50 > 2.0x and > 2 ms slower (worst A/A swing 1.81x), or throughput < 0.7x (worst A/A 0.82x) |
| `sNN <op> (p99 shift, not gated)` | p99_ms | info only: a p99 shift of > 2x and > 10 ms on an op with ≥ 1,000 baseline samples is reported, never gated (worst A/A swing 56x) |
| `sNN setup wall time` | p50_ms | the scenario's ingest phase; crosses at > 1.5x and > 5 s slower (worst A/A 1.03x) |

A perf crossing (and a timing check broken on the candidate only) is a regression only
when a re-run of that scenario, both sides with the order swapped, crosses again; the
re-runs go worst first while the deadline allows. One that was not repeated is
reported as info and named so; one left without a re-run for lack of time makes the
verdict `error` (it was not measured to the end). Improvements are reported too.

The verdict is `error`, never `pass`, when something was not measured: a scenario the
deadline left out or cut short (a side starts only when its expected time still fits,
and a run stopped by the deadline does not count as a crash), a scenario where neither
side recorded a single check against a live server (the harness failed, not the
engine), or one whose baseline did not get that far. A candidate whose server never
starts is a `completed` regression. Servers listen on 127.0.0.1 only (host networking,
no password). The vs-C score (`score.py`) is not part of the verdict.

Where the gates come from: main@4c1088949 against itself on an n4-standard-16 (queue
mode for all 12 scenarios; nightly soak-b, s07–s12, with re-runs). Each gate sits above
the worst same-image swing those runs showed, in either direction: p50 1.81x (s10
`live_delete`, 3.67 vs 6.65 ms), throughput 1.225x (s07 `transfer_peak`), setup wall time
1.03x. p50 of 153 op comparisons stayed within 1.22x except that one. p95/p99 of contended
ops swung up to 125x/56x when one run caught a stall episode (s07 `account_delete`,
p99 9.9 vs 550 ms), and the same op's p99 "repeated" 1.88x (13.0 vs 6.9 ms) on its
re-run by chance, so tail latency is reported, not gated: a tail-only regression shows
as an info row (and in `raw/rows.csv`), never as a red verdict. Severe tails are the fuzz
check's (`no_freeze`, `no_timeouts_while_waiting`) and Pokec's PING-freeze check.
The p50 gate sits only 1.1x above its worst A/A swing, so it is re-derived from each
A/A evidence set (the PR's acceptance runs); a wider swing there moves the gate first.

**Known flaky checks** (`KNOWN_FLAKY` in `check.py`, contract v1.9, keyed by the finding's
name, so an entry holds whichever check runs its scenario). These are
correctness checks that an open bug breaks intermittently. An entry needs two things: an
open issue, and same-image evidence that the break is intermittent. It is removed when the
issue closes, and a unit test fails if an entry names no issue. A listed check is not gated
for that break:

- the summary names it, its issue and the side(s) that broke it;
- `raw/rows.csv` keeps it with status `known`.

It is still a regression when the candidate breaks it every time it is checked (the
scenario's "k of n runs failed" with k = n ≥ 10) in every run of the pass, so a known
flake cannot hide a total breakage.

| check | issue | evidence |
| --- | --- | --- |
| `s10 Live (under writes) index == scan during build/after` | #3095 | One index entry missing during or after a build under writes (`idx=1242 scan=1243`). main against itself: broke on both sides in a nightly run, on neither in a queue run. 6.2.0 vs 6.2.1 (run 45401220445): broke on 6.2.0 only. |
| `s10 Live: index-path count == scan-path count (same query)` | #3095 | The same loss within one query (`idx=4056 scan=4057`). 6.2.1: broke in run 45477499408, held in run 45401220445. |

**Known breaks on both sides** (reported as `info`; neither image introduced them):

| check(s) | issue | what breaks |
| --- | --- | --- |
| s10 `after build person_age` / `person_city: indexed result == ground truth`, `during build person_age` / `person_city: result == ground truth`, `before restart: Person.age indexed results == ground truth` | #3512 (root cause) | A range-index lookup of value 0 returns no rows (`eq 0: got 0 want 1945`) once a fulltext index on the same label has been dropped (a range index existed first). The cause is RediSearch's union-aliasing skip of a numeric `+0.0`; the fix, FalkorDB/RediSearch#31, is not in the submodule yet. #3512 describes only the reload trigger. |
| s03 `range index finds chunks by cid (0, 1, mid, last)`, `range index finds every initial chunk` | #3512 (root cause) | `cid` 0 is not found through the range index on `:Chunk`, a label that also has fulltext and vector indexes. |
| s03 `in-load: updated_ann_miss_top10 == 0` | #3886 | A chunk updated under load is missing from its own top-10 vector neighbours (HNSW leaves nodes unreachable, Rust and C). |
| s05 `steady state: used_memory plateau`, `used_memory returns to baseline after deleting all` | #3078 | On Rust, about 15–18 KB of `used_memory` stays behind for every GRAPH.COPY, even after the copies are deleted. s05 makes about 3,200 copies at scale 0.25, leaving +52.5 MB; C leaves none. |
| s11 `deeply nested expressions are answered within 60 s` (C images) | #650, #2400 | C spins on a query with thousands of nested parentheses (#650, fixed in Rust only) and recurses without bound on deep expressions (#2400). The probe gives up after 60 s instead of holding the scenario. |

A side that runs longer than its hang cap did not finish (a `completed` regression on
the candidate, even when the deadline then leaves no time for the baseline). The cap is
6x the load time (at least load + 15 min) until one side of the scenario has run, then
3x that side's time (at least + 10 min), so one hang costs about three normal runs.

`--mode` sets only the length:

| mode | seconds per scenario and side | scale | perf | runs |
| --- | --- | --- | --- | --- |
| queue | 60 | 0.05 | info only, no re-runs | correctness checks |
| nightly | 300 | 0.25 | gated, re-runs | one check per scenario, `soak-s01` … `soak-s12` |
| long | 1200 | 1.0 | gated, re-runs | one check per scenario, with timeouts of its own (unmeasured) |

The nightly registers one entry per scenario in `bench/checks.yaml`, `soak-s01` …
`soak-s12`, each on its own n4-standard-16 VM, so soak takes as long as its slowest
scenario instead of six in a row. An entry's `timeout_minutes` is 2.75x the slowest pair
(both sides of the first pass) seen on that machine type, plus 0.5 min to pull and 10 min
of margin, rounded up to 5 min: 40 to 80 min. The slowest pairs and the runs they come
from are `SLOWEST_PAIR` in `bench/tests/test_soak.py`, where a test fails when a timeout
no longer fits a pair 25% slower plus its swapped re-run. In the workflows the deadline
follows that timeout (`PERF_TIMEOUT_SECONDS`, set by `tests/perf/vm/run-check.sh`): the
check stops starting runs 10 min before it and cuts a run 5 min before it. Run by hand,
the deadline is the mode's (`MODES` in `check.py`) or `--deadline-minutes`. The long mode
needs entries with longer timeouts: the nightly ones leave its 1200 s sides no room.

Trends start fresh under these names. Nights before them ran soak as two checks,
`soak-a` (s01–s06) and `soak-b` (s07–s12); the dashboard keeps those nights under the
old names and does not carry their rows over to the new checks. Its runs index shows
every soak check, old and new, in one `soak` column: the worst verdict, with each
check's verdict in the cell's title. A run's page and the trends keep one entry per
check. Outside the VM
the check runs unpinned; nothing waits for a quiet machine, so read local numbers with
that in mind. Unit tests (no server or Docker needed):
`uv run --project bench --group dev --group soak python -I -B -m pytest bench/tests/test_soak.py`.
