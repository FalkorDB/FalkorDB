# m1-reuse-cost: the GraphBLAS cost of reusing depth 0's hierarchy (D5 option b)

**Backs:** `../../m1-reuse-cost.md`. Harness A gives the main body; harness B gives the "Addendum: capstop, measured". `../../d5-b-verdict.md` cites both. Harness B is also the starting point of `../m1-optimize`: its sources are m1opt's `orig/`.

**Came from:** the scratch folders `m1rcA` and `m1rcB`. Each `wt/` is an rsync of the `m1gb` worktree at the base, without `.git` or `target`. The first folder, `m1rc`, was abandoned after another writer broke its sweep; the report discards its runs, so it is not here.

## What it measures

For every ND depth ≥ 1, with real cells from a small untimed host ND driver, the GraphBLAS cost of:
- **R**: re-coarsen from scratch, the parent bench's `hierarchy`;
- **B1**: reuse the depth-0 matching and re-contract every reused level, splitting the straddlers;
- **B2**: the same coarse graphs, built with one masked split of the parent's level plus re-contraction of the straddler rows only;
- **CS** (capstop, harness B only): B2 while a cell's heaviest coarse vertex stays within R's cap, then fresh coarsening of the stopped cells.

Feral's whole ordering is timed in the same process. All runs use `M1_CT=1` under `GxB_JIT_RUN`.

## Files

| Path | What it is |
|---|---|
| `m1rcA.patch` | Harness A, against base `7df8e588727f47619d9567d111c7fc26a2208caf`. Adds `m1_coarsen_bench.rs` (the m1gb bench plus 23 lines) and `m1_reuse_bench.rs`, and the `mod.rs` line. |
| `m1rcB.patch` | Harness B, against the same base. Same two files, different code: capstop, `M1R_CELLS`, and a label pass with a separator at level 0. |
| `harness-A/scripts/` | `run.sh LOG VAR=VAL…` (one run, 12 GB RSS watchdog), `sweep.sh` (the main sweep), `analyze.py LOG [GRAPH] [--depths]`. |
| `harness-A/results/` | `tables.txt` (the parsed tables the report quotes), `sweep_status.txt`, `breaks_summary.txt`, and `raw-logs.tar.xz` with `main_{CA,PA,TX}_{fine,all}.log` and `breaks/`. |
| `harness-B/scripts/` | `run.sh` (adds `M1_JIT=run M1_CT=1`), `sweep.sh`, `analyze.py LOG [CA\|PA\|TX] [--depths] [--calls]`, `cmp_lvl.py NEW OLD` (R's per-level shape against m1gb's log). |
| `harness-B/results/` | `sweep_status.txt`, `breaks_summary.txt`, and `raw-logs.tar.xz` with `main_{CA,PA,TX}.log`, `unchanged_pa_chunks.log` and `breaks/`. |

## Rebuild

Both harnesses build the same way. Each needs its own worktree, because both patches create the same files:

```bash
git worktree add --detach ../m1rcB 7df8e588727f47619d9567d111c7fc26a2208caf
cd ../m1rcB
git apply /path/to/prototypes/code/m1-reuse-cost/m1rcB.patch     # or m1rcA.patch
git submodule update --init --recursive   # unless the native-deps cache already has these pins
export CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++   # macOS
export CARGO_TARGET_DIR=$PWD/target-rc
cargo test --release -p graph --features index-falkordb --lib --no-run
```

`--list --ignored` shows `graph::graphblas::m1_coarsen_bench::reuse::m1_reuse: test`. Both patches apply cleanly to the base, and applying them reproduces the scratch files byte for byte (checked on 2026-11-13). They were not rebuilt for the handoff.

## Data

Decompressed SNAP text files through `M1_GRAPH`; see `../README.md`.

## Run

The test is `graph::graphblas::m1_coarsen_bench::reuse::m1_reuse`. It takes m1gb's variables (`M1_GRAPH`, `M1_THREADS`, `M1_RUNS`, `M1_DEPTHS`, `M1_JIT`, `M1_CT`, `M1_FERAL`, `M1_BREAK`, `M1_VERIFY`; see `../m1-graphblas-cost/README.md`), plus the ones below.

**Harness A.**
- **Variables:**
  - `M1_REFINE=fine|all` (default `fine`): refine cells at the fine level only, or at every level;
  - `M1_CELLS=real|proxy` (default `real`);
  - `M1_DEPTHS` defaults to 40 here: depths run until every cell is a leaf.
- **Main sweep** (`sweep.sh`): CA, PA and TX, each with `fine` and `all`, `M1_THREADS=1,4 M1_RUNS=3 M1_FERAL=3 M1_CT=1 M1_JIT=run`.
- **Breaks:** PA, `M1_THREADS=1 M1_RUNS=2 M1_DEPTHS=4 M1_FERAL=0 M1_CT=1 M1_JIT=run M1_VERIFY=1 M1_BREAK=<b>`, where `<b>` is one of `b2mask`, `bweight`, `bcons`, `b1offdiag`, `basym`, `bperm`, `bmatch`, `lstar`, `sepcut`, `split`, `salt`, `nooffdiag` or `nomutual`. The control is `M1_BREAK=` (empty); `breaks/none.log` is its log.
- **Expected:** the control exits 0, and every break exits 101 with the message in `breaks_summary.txt`.

**Harness B.**
- **Variables:**
  - `M1R_CELLS=real|chunks` (default `real`; `chunks` is the earlier proxy of contiguous chunks of feral's order);
  - `M1R_CHECK=all`: full checks on every run, not only run 0 of each thread count;
  - `M1_DEPTHS` defaults to 14.
- **Main sweep:** CA, PA and TX, `M1_THREADS=1,4 M1_RUNS=3 M1_FERAL=3`.
- **Breaks:** PA, `M1_DEPTHS=4 M1_RUNS=2 M1_THREADS=1 M1_FERAL=1 M1_BREAK=<b>`, where `<b>` is one of `nofill`, `b1diag`, `sepclean`, `mergeparts`, `rematch`, `cap2`, `gfdrop`, `nosep`, `split`, `salt` or `nooffdiag`. `breaks/control.log` is the control.
- **R unchanged:** `M1R_CELLS=chunks M1_RUNS=1 M1_THREADS=1 M1_FERAL=1` on PA (`unchanged_pa_chunks.log`), then `python3 cmp_lvl.py unchanged_pa_chunks.log <m1-graphblas-cost's best_PA.log>`. That must report 177 levels with 0 differing. With `M1_BREAK=salt` added (`breaks/unchanged_salt_chunks.log`) most levels must differ.

**Analysis:** `python3 -I analyze.py LOG GRAPH`. Both `analyze.py` files read the quality stage's straddler shares from a hard-coded path, `SJ = …/m1reuse/straddlers.json`. That file is in `../m1-matching/stage2-results/raw-logs.tar.xz`. Edit `SJ` to point at your copy.

## Expected output

Each harness prints `FERAL run=… sec=…` and `LOAD` lines, plus a checks summary at the end. The other lines differ:
- **harness A:** `RLVL` (R per level), `BLVL` (B1/B2 per level), `DEPTHX` (per depth and method), `SHARE` (straddler shares), `LABELS`;
- **harness B:** `LVL`, `VAR` (per depth and path), `STRAD`, `PREP`, `LABELS`, `ROOT`.

The report's ratios to feral, at 1 thread over CA, PA and TX:
- harness A, fine cells: B2 0.92–0.94×, B1 1.17–1.21×, R 2.0–2.1×. At 4 threads: 0.44–0.46×, 0.53–0.55× and 0.80–0.81×.
- harness B: capstop 0.98–1.10× at 1 thread and 0.45–0.58× at 4 threads.

## Paths in the scripts

The scripts are verbatim and hard-code `/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad`:
- `A` and `B` are the harness folders, each with its frozen `graph-test` binary;
- `D` is `m1gbdata`, the decompressed graphs;
- `SJ` is the straddler file.

## Left out

- **Binaries:** `m1rcA/graph-test` and `m1rcB/graph-test`.
- **Discarded and smoke logs:**
  - `m1rcA/logs/aborted_main_CA_fine_runsh_overwritten.log` (4.1 MB): the sweep another writer broke; the report discards it;
  - `m1rcA/logs/smoke*.log`, `m1rcB/logs/smoke_pa.log`, `m1rcB/build.log`.
- **Snapshots:** `m1rcA/snap/`, which holds the final sources (the same as the patch), an intermediate copy, and `m1_coarsen_bench.foreign-0149.rs`, the other writer's 01:49 edit. Also `m1rcA/orig/`, which is m1gb's bench.
- **Never run:** `m1reuse-cost-instance2-unrun/`, a second instance's copy.
