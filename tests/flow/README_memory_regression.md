# Memory regression tests

`test_memory_regression.py` provides **regression** tests for memory
consumption — distinct from the correctness tests in `test_memory_usage.py`.

For every entry in `memory_baselines.json` the test re-builds a deterministic
graph workload (e.g. *10K labeled nodes with one integer attribute*), reads
`total_graph_sz_mb` from `GRAPH.MEMORY USAGE`, and asserts that the current
value does not exceed the baseline by more than a configurable tolerance.
If the current value is significantly larger, CI fails and blocks the merge.

## Workloads currently covered

| Workload                       | Description                                          |
| ------------------------------ | ---------------------------------------------------- |
| `nodes_only_10k`               | 10K unlabeled, attribute-less nodes                  |
| `nodes_labeled_10k`            | 10K nodes labelled `:A`                              |
| `nodes_with_int_attr_10k`      | 10K `:A` nodes each with an integer attribute        |
| `nodes_with_string_attr_10k`   | 10K `:A` nodes each with a short string attribute    |
| `edges_10k`                    | 10K `()-[:R]->()` edges (20K nodes, 10K edges)       |
| `edges_with_attr_10k`          | 10K edges, each carrying one integer attribute       |
| `indexed_nodes_10k`            | 10K `:A` nodes with a range index on `v`             |
| `create_delete_cycles_5k`      | 5 rounds of create-then-delete + a final populate    |

Add a new test method in `test_memory_regression.py` to extend this list.

## Regression criterion

A workload fails iff

```
measured > baseline * (1 + TOLERANCE_PCT/100) + TOLERANCE_MIN_MB
```

The two tolerance knobs default to **10 %** and **0.5 MB** and can be tuned
through environment variables — see below. The absolute floor exists to
suppress false positives when the reported size is very small.

## Bootstrap

If a workload has no baseline yet (fresh checkout, newly added workload), the
test records the measured value, writes it to `memory_baselines.json`,
prints a notice, and passes. This keeps the false-positive rate at zero on
the very first run; subsequent runs perform real regression checks against
the committed baseline.

## Environment variables

| Variable                            | Default | Effect                                                                         |
| ----------------------------------- | ------- | ------------------------------------------------------------------------------ |
| `MEMORY_REGRESSION_UPDATE`          | `0`     | When `1`, re-measure every workload and overwrite `memory_baselines.json`.     |
| `MEMORY_REGRESSION_TOLERANCE_PCT`   | `10`    | Allowed relative growth, in percent, above the baseline before flagging.       |
| `MEMORY_REGRESSION_TOLERANCE_MIN_MB`| `0.5`   | Absolute MB floor added to the tolerance (false-positive guard for small sizes).|

## Updating baselines after an intentional change

1. Re-measure and overwrite the baselines locally:

   ```bash
   MEMORY_REGRESSION_UPDATE=1 \
       ./tests/flow/tests.sh TEST=test_memory_regression
   ```

2. Inspect the resulting diff of `tests/flow/memory_baselines.json` and make
   sure the change is expected.

3. Commit `tests/flow/memory_baselines.json` together with the code change
   that justifies the new values, so future CI runs use the refreshed
   baseline.

## How this integrates with CI

The file lives in `tests/flow/` and follows the standard `test_*.py` /
`testFooBar` naming convention picked up by RLTest, so the flow-test runner
(`tests/flow/tests.sh`, invoked by `make flow-tests` in CI) discovers and
executes it automatically. No workflow changes are required for regressions
to surface in CI and block merges.
