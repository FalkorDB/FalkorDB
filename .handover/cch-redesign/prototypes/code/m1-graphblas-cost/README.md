# m1-graphblas-cost: the first GraphBLAS coarsening bench

**Backs:** `../../m1-graphblas-cost.md`.

**Came from:** the scratch git worktree `m1gb` (the bench) and `m1gbtools` (a copy of the bench, the logs and `analyze.py`). The two copies of the bench were byte-identical.

## What it measures

The cost of the coarsening of design §3, written as GraphBLAS calls, on SNAP roadNet-CA, PA and TX. Every call is timed as wall time plus `GrB_wait(MATERIALIZE)` on its output.

For each ND depth 0–14 it runs the whole hierarchy:
- matching rounds;
- the 2-hop pass;
- contraction `select(OFFDIAG, C·A·Cᵀ)`;
- projection, and one `mxv` standing in for the boundary step;
- for depth d ≥ 1, the split.

The cells at depth d ≥ 1 are 2^d contiguous chunks of feral's order: a proxy, not real cells. Feral's whole ordering is timed in the same process.

The checks, and the breaks that fire them, are in the report.

## Files

| Path | What it is |
|---|---|
| `m1gb.patch` | Against base `7df8e588727f47619d9567d111c7fc26a2208caf`. Adds `graph/src/graph/graphblas/m1_coarsen_bench.rs` and registers it in `mod.rs` under `#[cfg(all(test, feature = "index-falkordb"))]`. |
| `scripts/analyze.py` | `python3 scripts/analyze.py LOG` prints the per-level call table for depth 0, the per-depth totals, and the total against feral. |
| `results/raw-logs.tar.xz` | The eight logs the report used, listed below. Extract with `tar -xJf`. |

The logs record no environment, so the settings in this table are inferred from their contents. The `JIT: RUN` line marks `M1_JIT=run`. `c_build` at depth 0 is 162 ms with `C` and 9 ms with `Cᵀ` (`M1_CT=1`), the figures the report gives.

| Log | Graph | Settings | Report column |
|---|---|---|---|
| `full_ca.log` | CA | defaults: no JIT, builds `C`; depths 0–14; 1 and 4 threads; 3 runs; `M1_FERAL=3` | "test" |
| `best_ca.log`, `best_PA.log`, `best_TX.log` | CA, PA, TX | `M1_CT=1 M1_JIT=run`; depths 0–14; 1 and 4 threads; 3 runs | "best" |
| `ct_ca.log` | CA | `M1_CT=1`, no JIT, `M1_DEPTHS=0` | the `Cᵀ` variant |
| `jitrun_ca.log` | CA | `M1_JIT=run`, builds `C`, `M1_DEPTHS=0` | the JIT effect |
| `burble.log`, `burble_jitrun.log` | CA | `M1_BURBLE=1`, 1 thread, 1 run, `M1_DEPTHS=0`; without and with `M1_JIT=run` | the generic-kernel share |

## Rebuild

From a clone of FalkorDB:

```bash
git worktree add --detach ../m1gb 7df8e588727f47619d9567d111c7fc26a2208caf
cd ../m1gb
git apply /path/to/prototypes/code/m1-graphblas-cost/m1gb.patch
git submodule update --init --recursive   # unless the native-deps cache already has these pins
export CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++   # macOS
export CARGO_TARGET_DIR=$PWD/target-m1
cargo test --release -p graph --features index-falkordb --lib --no-run
```

`--list --ignored` on the binary shows `graph::graphblas::m1_coarsen_bench::m1_coarsen: test`. The patch applies cleanly to the base (checked on 2026-11-13). This one was not rebuilt for the handoff.

## Data

`M1_GRAPH` is a decompressed SNAP edge list; see `../README.md`. The loader is the same as m1-optimize's: CA n = 1,965,206 with 5,533,214 entries.

## Run

```bash
env M1_CT=1 M1_JIT=run M1_GRAPH=/data/roadNet-CA.txt M1_THREADS=1,4 M1_RUNS=3 M1_DEPTHS=14 M1_FERAL=3 \
    "$BIN" --ignored --exact graph::graphblas::m1_coarsen_bench::m1_coarsen --nocapture > best_ca.log
python3 scripts/analyze.py best_ca.log
```

| Variable | Meaning |
|---|---|
| `M1_GRAPH` | Required. The SNAP text file. Use an absolute path. |
| `M1_THREADS` | GraphBLAS thread counts. Default `1,4`. |
| `M1_RUNS` | Runs per thread count. Default 3. Every run is timed; the report takes the median. |
| `M1_DEPTHS` | Deepest ND depth. Default 14. `0` runs depth 0 only. |
| `M1_FERAL` | Feral timing runs. Default 3. Feral runs at least once whenever depths ≥ 1 are run, because its order gives the cells. |
| `M1_CT=1` | Build `Cᵀ` from row-sorted tuples instead of `C`. |
| `M1_JIT=run` | `GxB_JIT_RUN`, as the shipped module runs. |
| `M1_BURBLE=1` | GraphBLAS burble. |
| `M1_VERIFY=1` | Extra per-level checks: `mate` is an involution, every pair is within the cap, and round pairs are adjacent. |
| `M1_BREAK` | `nomutual`, `nooffdiag`, `split` or `salt`: the report's breaks. Each must fail its check. |

## Expected output

The test prints:
- `FERAL run=… sec=…`;
- one `LVL threads=… run=… depth=… level=… n=… nnz=… rounds=… twohop=… <call>=<seconds> …` line per level;
- one `DEPTH … levels=… wall=… split=…` line per depth;
- `SPLIT` lines.

From the report, on the M5: CA depth 0 takes 902 ms "best" at 1 thread and 343 ms at 4 threads. All depths take 13.17 s at 1 thread and 4.79 s at 4 threads, against 6.97 s for feral.

## Left out

- The break runs' logs. They were not kept in the scratch folders; the report gives their results.
- The test binary and target directory (deleted at the time).
- The decompressed graphs (`m1gbdata`).
