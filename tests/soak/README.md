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

- Redis 8.x `redis-server` on `PATH`, or set `SOAK_REDIS`.
- The repo venv (`tests/requirements.txt`) plus `numpy` and `psutil`.

## Running

One scenario:

```bash
python -I scenarios/s01_social_feed.py --module <path/to/module> --port 19000 \
  --out full/s01_social_feed-rust --duration 1200 --scale 1.0
```

All Rust scenarios, one at a time, only when the machine is quiet (1-minute load
under 8 for 5 minutes; a run that sees load above 20 is discarded and retried):

```bash
./run_rust_main.sh <path/to/libfalkordb.dylib> <commit-label>
./run_c_rerun.sh <path/to/falkordb.so> <c-label>      # C baseline
python -I score.py full && python -I gen_final.py    # score and report
```

Results go to `full/<scenario>-<engine>/` (ignored by git); a previous run of the
same scenario is archived as `<dir>@<old-label>`. `findings.json` is the ledger of
bugs found so far, which the report reads.

Check `sysctl vm.loadavg` before trusting any numbers: outside load produces bogus
p50/p99 regressions.
