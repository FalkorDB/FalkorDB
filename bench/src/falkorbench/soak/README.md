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
  --baseline <image@sha256:...> --out out --mode nightly --check soak-b \
  --scenarios s07,s08,s09,s10,s11,s12
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
| `sNN completed` | completed | 0 when the scenario raised, outlived its cap or a server log shows a crash; 0 on the candidate is a regression at once, whether or not the baseline failed too |
| `sNN <check>` | invariant | each check the scenario records (1 held, 0 broken). A correctness check broken on the candidate is a regression at once, whether or not the baseline broke it too and whether or not an issue tracks it (#4065). A latency or resource check (`TIMING_RE`: p99, SLO, schedule kept, RSS/used_memory plateaus, drift, CPU idle, replication lag...) broken on the candidate asks for the scenario's re-run, and counts as broken only when every pair of runs, first pass and re-run, has it broken on the candidate. Held on the candidate where the baseline broke it (a timing check: in every pair) is an improvement, "fixed". Generous hang bounds (recovery < 180 s, replica sync ≤ 900 s, an index reaching OPERATIONAL) are correctness: missing one is a run that did not finish |
| `sNN <op>` | p50_ms, throughput | each worker op with at least 200 successful samples on the baseline (an op the candidate nearly stopped doing still compares, at its throughput); crosses at p50 > 2.0x and > 2 ms slower (worst A/A swing 1.81x), or throughput < 0.7x (worst A/A 0.82x) |
| `sNN <op> (p99 shift, not gated)` | p99_ms | info only: a p99 shift of > 2x and > 10 ms on an op with ≥ 1,000 baseline samples is reported, never gated (worst A/A swing 56x) |
| `sNN setup wall time` | p50_ms | the scenario's ingest phase; crosses at > 1.5x and > 5 s slower (worst A/A 1.03x) |

A perf crossing (and a timing check broken on the candidate) is a regression only
when a re-run of that scenario, both sides with the order swapped, crosses again; the
re-runs go worst first while the mode's deadline allows. One that was not repeated is
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

**Which breaks the release gate counts.** A night, a manual run and `/perf` judge every
pass/fail check on the candidate alone (above). The release gate (the merge queue of a
version branch) keeps the older rule and blocks only what the candidate introduced:
`result.json`'s `broken` list (contract v1.14) says, for each pass/fail check broken on the
candidate, whether the baseline broke it too, its issue, and `introduced`, which is false
when the baseline broke it too (a timing check: in any pair) or when it is a known flaky
check below.

**Known flaky checks** (`KNOWN_FLAKY` in `check.py`, contract v1.9). These are
correctness checks that an open bug breaks intermittently. An entry needs two things: an
open issue, and same-image evidence that the break is intermittent. It is removed when the
issue closes, and a unit test fails if an entry names no issue. A listed check broken on the
candidate is a regression like any other; only the release gate leaves it out (`introduced`
false), unless the candidate breaks it every time it is checked (the scenario's "k of n runs
failed" with k = n ≥ 10) in every run of the pass, so a known flake cannot hide a total
breakage. Broken on the baseline only, it is info (a flake, not a fix). `raw/rows.csv`
marks it with kind `known`.

| check | issue | evidence |
| --- | --- | --- |
| `s10 Live (under writes) index == scan during build/after` | #3095 | One index entry missing during or after a build under writes (`idx=1242 scan=1243`). main against itself: broke on both sides in a nightly run, on neither in a queue run. 6.2.0 vs 6.2.1 (run 45401220445): broke on 6.2.0 only. |
| `s10 Live: index-path count == scan-path count (same query)` | #3095 | The same loss within one query (`idx=4056 scan=4057`). 6.2.1: broke in run 45477499408, held in run 45401220445. |

**Known breaks and their issues** (reporting only). This is not an exemption list: each
check here is a regression whenever the candidate breaks it. The report names the issue next
to the broken check (`ISSUES` in `check.py`; a unit test keeps that map equal to this
table). A check not listed has no issue known.

| finding | issue | what breaks |
| --- | --- | --- |
| `s10 after build person_age: indexed result == ground truth` | #3512 | A range-index lookup of value 0 returns no rows (`eq 0: got 0 want 1945`) once a fulltext index on the same label has been dropped (a range index existed first). The cause is RediSearch's union-aliasing skip of a numeric `+0.0` (#3512 is the root cause; it describes only the reload trigger); the fix, FalkorDB/RediSearch#31, is not in the submodule yet. |
| `s10 after build person_city: indexed result == ground truth` | #3512 | The same. |
| `s10 during build person_age: result == ground truth` | #3512 | The same. |
| `s10 during build person_city: result == ground truth` | #3512 | The same. |
| `s10 before restart: Person.age indexed results == ground truth` | #3512 | The same. |
| `s10 Live (under writes) index == scan during build/after` | #3095 | Known flaky, below. |
| `s10 Live: index-path count == scan-path count (same query)` | #3095 | Known flaky, below. |
| `s03 range index finds chunks by cid (0, 1, mid, last)` | #3512 | `cid` 0 is not found through the range index on `:Chunk`, a label that also has fulltext and vector indexes (findings.json F3). |
| `s03 range index finds every initial chunk` | #3512 | The same. |
| `s03 in-load: updated_ann_miss_top10 == 0` | #3886 | A chunk updated under load is missing from its own top-10 vector neighbours (HNSW leaves nodes unreachable, Rust and C; F10). |
| `s05 used_memory returns to baseline after deleting all (residual <50MB)` | #4050 | On Rust, `used_memory` stays well above the baseline after every graph is deleted (F7); C returns to it. #3078 (about 15–18 KB left behind per GRAPH.COPY) may contribute. |
| `s11 CLIENT KILLed query is cancelled (CPU idle within 5s of the kill)` | #4051 | A killed client's query keeps running until its TIMEOUT, on Rust and C (M1). |
| `s11 deeply nested expressions are answered within 60 s` | #2400 | C images: C recurses without bound on deep expressions (#2400; #650, fixed in Rust only, was the parentheses case). The probe gives up after 60 s instead of holding the scenario. |

A side that runs longer than its hang cap did not finish (a `completed` regression on
the candidate, even when the deadline then leaves no time for the baseline). The cap is
6x the load time (at least load + 15 min) until one side of the scenario has run, then
3x that side's time (at least + 10 min), so one hang costs about three normal runs.

`--mode` sets only the length:

| mode | seconds per scenario and side | scale | perf | runs |
| --- | --- | --- | --- | --- |
| queue | 60 | 0.05 | info only, no re-runs | correctness checks |
| nightly | 300 | 0.25 | gated, re-runs | 2 shards (`soak-a`, `soak-b`) |
| long | 1200 | 1.0 | gated, re-runs | must be sharded further (about 8 h for all 12) |

The nightly registers two entries in `bench/checks.yaml`, `soak-a` (s01–s06) and
`soak-b` (s07–s12), each on its own n4-standard-16 VM with a 120 min timeout (the
nightly deadline stops starting runs at 110 min and cuts a run at 115). Outside the VM
the check runs unpinned; nothing waits for a quiet machine, so read local numbers with
that in mind. Unit tests (no server or Docker needed):
`uv run --project bench --group dev --group soak python -I -B -m pytest bench/tests/test_soak.py`.
