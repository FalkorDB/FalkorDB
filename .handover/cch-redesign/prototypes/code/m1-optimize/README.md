# m1-optimize: the optimized GraphBLAS coarsening bench (capstop)

**Backs:** `../../m1-optimize.md`, including its Verification section (the `verify/` part here). Read that report's corrections note before using its numbers.

**Came from:** the scratch git worktree `m1opt` (the bench, scripts and results) and the verifier's worktree `m1verify` plus its work folder `m1verify-work`. Both worktrees were at the base commit below and committed nothing.

## What it measures

For ND depths 0–14 on SNAP roadNet-CA, PA and TX, the bench times the coarsening of design §3 on three paths:
- **R**: re-coarsen every depth from scratch;
- **B2**: reuse the parent depth's levels;
- **CS** (capstop): B2 per cell until the cell's heaviest coarse vertex would break R's cap, then coarsen fresh.

Feral's whole ordering runs in the same process. C METIS runs as its own process (`../cmetis`).

Every optimization sits behind one switch in `M1O`. The baseline is the same binary with `M1O=` empty. Equality is checked by hashes: the baseline writes 57 per graph (depth 0, plus split, R, B2 and capstop at depths 1–14), and every optimized run compares against them, at 1 and at 4 threads.

## Files

| Path | What it is |
|---|---|
| `m1opt.patch` | The bench, against base `7df8e588727f47619d9567d111c7fc26a2208caf`. It adds `graph/src/graph/graphblas/m1_coarsen_bench.rs` and `m1_reuse_bench.rs`, and registers the first in `mod.rs` under `#[cfg(all(test, feature = "index-falkordb"))]`. `m1_reuse_bench.rs` is pulled in by `m1_coarsen_bench.rs` as `mod reuse` through `#[path]`. |
| `verifier-on-m1opt.patch` | The verifier's two switches, `M1V_BREAK=hpar` and `M1V_FLOOR30=1`. Apply it **after** `m1opt.patch`. Both switches are off unless set. |
| `scripts/` | m1opt's scripts, unchanged: `build.sh`, `run.sh` (one run with a 12 GB RSS watchdog), `measure.sh` (one graph at 1 and 4 threads, with the hash check), `session.sh` (the final session), `analyze.py`, `split_calls.py`, `call_table.py`, `make_attempts.py`, `show_attempts.py`. |
| `results/baseline.csv`, `steps.csv`, `attempts.csv`, `final.csv` | Per-run phase totals. `attempts.csv` is the report's per-step table (`make_attempts.py` builds it from `steps.csv`; `show_attempts.py` prints it). `final.csv` is the final session. |
| `results/hashes/` | The baseline hash files: `ca_base.hash`, `pa_base.hash`, `tx_base.hash` (depths 0–14, 57 hashes each), `pa8_base.hash` (PA depths 0–3, 13 hashes, used by the breaks) and `smoke_pa.hash`. |
| `results/breaks_final.txt` | The break table: final binary, PA, depths 1–3. |
| `results/breaks_summary_dev.txt` | Breaks run while developing, on intermediate binaries. |
| `results/session_driver.log` | The final session's driver output (hash counts per process). |
| `results/raw-logs.tar.xz` | Raw logs: `session/{CA,PA,TX}_{base,final}.log` and `session/cmetis.log` (the final tables), `brk/` (the break runs and controls), `rule_pa.log`, `rule_pa_final.log`, `rule_pa_cap2.log` (the stop-rule replay) and `ca_base.log` (the step-0 baseline). Extract with `tar -xJf`. |
| `verify/scripts/` | The verifier's `build.sh`, `run.sh`, `session.sh`, `followup.sh` and `tot.py`. |
| `verify/results/` | The verifier's driver logs, its own baseline hashes (`ca_base_mine.hash` is byte-identical to `results/hashes/ca_base.hash`), the `*.struct` work-count summaries, and `raw-logs.tar.xz` with `CA_{base1,final1,final2,base2}.log`, `cmetis.log` and `fu/` (hpar break, split probe, FLOOR30 runs). |

## Rebuild

From a clone of FalkorDB:

```bash
git worktree add --detach ../m1opt 7df8e588727f47619d9567d111c7fc26a2208caf
cd ../m1opt
git apply /path/to/prototypes/code/m1-optimize/m1opt.patch
# optional, the verifier's switches:
git apply /path/to/prototypes/code/m1-optimize/verifier-on-m1opt.patch

# GraphBLAS, LAGraph and RediSearch come from submodules, built by graph/build.rs.
# Needed unless the native-deps cache already holds these pins (see CLAUDE.md):
git submodule update --init --recursive

# macOS: Homebrew LLVM, because Apple clang has no OpenMP
export CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++
export CARGO_TARGET_DIR=$PWD/target-m1opt
export CARGO_PROFILE_RELEASE_DEBUG=line-tables-only
cargo test --release -p graph --features index-falkordb --lib --no-run
```

- Cargo prints the binary as `Executable unittests src/lib.rs (…/release/deps/graph-<hash>)`. The base's release profile has `lto = true` and `codegen-units = 1`.
- `"$BIN" --list --ignored | grep m1_` must print `graph::graphblas::m1_coarsen_bench::reuse::m1_reuse: test`.
- `scripts/build.sh TAG` does the same and copies the binary to `bin/graph-test-TAG`. The original binaries were v1 (steps 0–5), v2 (steps 6–7), v3 (steps 8–13) and v4 (step 14 on, and the final session). Each new binary only added switches, so this patch, which is v4's source, can run every step.

**Checked on 2026-11-13** for this copy:
- on a fresh worktree at the base with `m1opt.patch` applied, the build above took 114 s with a warm native-deps cache, and `--list` showed the one test;
- that binary, run on PA at depths 0–3 with the final configuration against `results/hashes/pa8_base.hash`, matched 13/13 hashes at 1 and 4 threads, with 0 `HASHDIFF`.

## Data

`M1_GRAPH` is a decompressed SNAP edge list (`roadNet-CA.txt`, …); see `../README.md` for the URLs and checksums. The loader drops self-loops, makes the graph symmetric, removes duplicates and isolated nodes, and renumbers. That gives:
- CA: n = 1,965,206, 5,533,214 entries;
- PA: 1,088,092 and 3,083,796;
- TX: 1,379,917 and 3,843,320.

C METIS reads the int64 CSR from `../cmetis/to_nd_bin.py`.

## Run

One process, as `scripts/run.sh` and `measure.sh` start it:

```bash
F=hb2,hsplit,cpack,hcon,pk,book,hkey,hpar,gath,usel,zmatch,npar   # the final configuration
env M1_JIT=run M1_CT=1 \
    M1_GRAPH=/data/roadNet-CA.txt M1_THREADS=1,4 M1_RUNS=3 M1_DEPTHS=14 \
    M1_FERAL=0 M1_PERM_CACHE=/work/cache/CA.perm \
    M1_HASH=check:/work/cache/ca_base.hash \
    M1O=$F \
    "$BIN" --ignored --exact graph::graphblas::m1_coarsen_bench::reuse::m1_reuse --nocapture
```

| Variable | Meaning |
|---|---|
| `M1_GRAPH` | Required. The SNAP text file. |
| `M1_THREADS` | GraphBLAS thread counts, comma list. Default `1,4`. The host passes use the same count. |
| `M1_RUNS` | Timed runs per thread count. Default 3. Run 0 is an extra, untimed check run (hashes, all checks). |
| `M1_DEPTHS` | Deepest ND depth. Default 14. |
| `M1_FERAL` | Feral timing runs. Default 3. With `0` the order is read from `M1_PERM_CACHE`; if that file is missing or the wrong size, feral runs once. |
| `M1_PERM_CACHE` | File holding feral's order (int64, little-endian). Written when missing. |
| `M1_HASH` | `write:FILE` writes the per-(path, depth) hashes; `check:FILE` compares against them. |
| `M1_RULECHECK=1` | Replays the capstop stop rule on the host for every cell and compares stop levels. |
| `M1_NOCHECK=1` | Skips run 0. |
| `M1_JIT=run` | Sets `GxB_JIT_RUN`, as every reported run did. |
| `M1_CT=1` | Required; the test asserts it (build `Cᵀ` directly). |
| `M1_VERIFY=1`, `M1_BURBLE=1` | Symmetry and involution checks; GraphBLAS burble. |
| `M1O` | Comma list of optimization switches. Empty means the baseline. |
| `M1O_BREAK=<switch>` | Perturbs that switch's output, so the equality check must fail. |
| `M1O_CHUNK` | `GxB_CHUNK` when `chunk` is in `M1O`. Default 1048576. |
| `M1O_CALLWAIT=1` | Keeps the per-call waits even under `nowait`. |
| `M1_BREAK=<name>` | The baseline checks' breaks, kept from m1rcB: `split`, `cap2`, `salt`, `sepclean`, `mergeparts`, `gfdrop`, `nosep`, `rematch`, `b1diag`, `nomutual`, `nooffdiag`, `nofill`. |
| `M1V_BREAK=hpar`, `M1V_FLOOR30=1` | Only with `verifier-on-m1opt.patch`. `M1V_FLOOR30` stops a cell of 5,000 or more vertices from reusing at max(100, n/30). |

**The steps.** `M1O` for each row of the report's table, as the step logs recorded it. Each step ran with `scripts/measure.sh` on CA, at 1 and 4 threads, with `M1_HASH=check:` against the step-0 baseline's `ca_base.hash`.

| Step | Binary | `M1O` |
|---|---|---|
| s0 | v1 | (empty) |
| s1–s5 | v1 | `hb2`, then `+hsplit`, `+cpack`, `+hcon`, `+pk` |
| s6, s7 | v2 | `…,pk,book`, then `+caprc` |
| s8–s13 | v3 | `…,caprc,hkey`, then `+hpar`, `+gath`, `+usel`, `+zmatch`, `+nowait` |
| s14 | v4 | `hb2,hsplit,cpack,hcon,pk,book,caprc,hkey,hpar,gath,usel,zmatch,npar` (no `nowait`) |
| s15, s16 | v4 | s14 plus `chunk`, with `M1O_CHUNK=1048576`, then `262144` |
| final | v4 | `hb2,hsplit,cpack,hcon,pk,book,hkey,hpar,gath,usel,zmatch,npar` (no `caprc`: `hkey` replaced it) |

**The final session** is `scripts/session.sh`: for each of CA, PA and TX, a baseline process (`M1_FERAL=3`), a final process and C METIS three times. `measure.sh` checks against `cache/<g>_base.hash` if it exists and writes it otherwise. To compare against our hashes, copy `results/hashes/{ca,pa,tx}_base.hash` into the cache folder first.

**The breaks** (`results/breaks_final.txt`): PA, `M1_THREADS=1,4 M1_RUNS=0 M1_DEPTHS=3 M1_FERAL=0 M1_RULECHECK=1 M1_HASH=check:pa8_base.hash`, with `M1O` set to the final configuration and `M1O_BREAK=<switch>`. The controls are the same without `M1O_BREAK`, and the baseline with `M1O=` empty. Both controls pass 13/13. Every `M1O_BREAK` must fail.

The `base_*` lines rerun m1rcB's `M1_BREAK` breaks under the final configuration. All fail except `nooffdiag`: it skips an OFFDIAG call that `hcon` no longer makes. Without `hcon` it fails, as the last two lines show (`M1O=''` and `M1O='hb2'`).

**The rule check:** PA, `M1_RUNS=0 M1_DEPTHS=14 M1_RULECHECK=1`, baseline (`rule_pa.log`) and with `M1_BREAK=cap2` (`rule_pa_cap2.log`, which must fail).

**The verifier** (`verify/scripts/`):
- `session.sh` runs CA in the order baseline, final, C METIS ×3, final, baseline, with its own build `graph-test-v` (`m1opt.patch` only);
- `followup.sh` runs the split-break probe, the `hpar` break and its control on PA, and the FLOOR30 variant against its control on CA. It uses `graph-test-vb`, built with both patches.

## Expected output

The test prints:
- `PHASE threads=… run=… depth=… path=D0|ROOTMAP|SPLIT|R|B2|CS … wall=… pwall=… instr=… cyc=…` lines;
- `HASH` and, on a mismatch, `HASHDIFF` lines;
- a checks summary with `hash_compared: 57, hash_fail: 0`;
- `test result: ok`.

`measure.sh` prints `NAME exit=0 hash_compared: 57, hash_fail: 0 slow_borrow=0`.

Totals: `python3 scripts/analyze.py LOG` or `python3 verify/scripts/tot.py LOG`. The time breakdown: `python3 scripts/split_calls.py LOG`.

Reported medians on the Apple M5 (`../../m1-optimize.md`), for capstop on CA:
- baseline: 7.22 s and 107.5 G instructions at 1 thread, 3.36 s at 4 threads;
- final: 2.30 s and 37.4 G at 1 thread, 1.20 s at 4 threads.

Other machines will give other times. The hashes and the work counts on the `PHASE` lines should not change.

## Paths in the scripts

The scripts are verbatim. They hard-code the session's scratch folder, `/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad`, in `S`, `W`, `V` and `D`. Under it:
- `m1opt` was the worktree;
- `m1gbdata` held the decompressed graphs;
- `m1audit/cmetis/<G>.bin` held the C METIS inputs;
- `cmetis-driver/nd` was the C METIS driver.

Point these variables at your own folders before running.

## Left out

- **Binaries:** `bin/graph-test-v1…v4` (230 MB) and the verifier's `graph-test-v` and `graph-test-vb`.
- **Caches:** feral's orders and C METIS's permutations in `cache/` (68 MB). Both are regenerated.
- **Per-step logs:** `logs/CA_s1…s16_*.log`, 3.8 MB. Their numbers are in `steps.csv` and `attempts.csv`.
- **Development checks:** `logs/chk/`, 1.5 MB, run on intermediate binaries. The summary is kept as `breaks_summary_dev.txt`.
- **Build output and a smoke log:** `logs/smoke_pa.log` and `build-*.{err,exe,json,log}`.
- **`orig/`:** the pre-optimization sources. They are m1rcB's, in `../m1-reuse-cost/m1rcB.patch`.
