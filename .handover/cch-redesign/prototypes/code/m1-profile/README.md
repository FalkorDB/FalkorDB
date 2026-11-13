# m1-profile: profile of harness A's GraphBLAS bench

**Backs:** no report of its own. It was a profiling pass over harness A's bench (`../m1-reuse-cost`) on roadNet-CA, before m1-optimize. It covers R at depth 0, R at depths ≥ 1, B2 at depths ≥ 1, and the fine split.

It used:
- rusage per phase and per timed call: instructions, cycles, CPU time;
- samply profiles cut at the phase boundaries;
- GraphBLAS burble per phase.

It also wrote the C METIS inputs a second way (`M1_DUMP_ND`), byte-identical to `../cmetis/to_nd_bin.py`'s.

**Came from:** the scratch folder `m1prof`. Its `wt/` is an rsync of a worktree at the base with harness A's bench plus the profiling hooks. `NOTES-original.txt` is the folder's own README.

## Files

| Path | What it is |
|---|---|
| `m1prof.patch` | Against base `7df8e588727f47619d9567d111c7fc26a2208caf`. It holds the profiled `m1_coarsen_bench.rs` and `m1_reuse_bench.rs`, the new `m1_prof.rs` (rusage and phase markers), and the `mod.rs` line. It applies on its own. |
| `scripts/run.sh LOG {samply-out.json.gz\|-} VAR=VAL…` | One run, optionally under `samply record`, with a 12 GB RSS watchdog. |
| `scripts/ru_table.py`, `call_table.py`, `burble_sum.py` | Tables per phase, per timed call, and per burble message. |
| `scripts/analyze_prof.py`, `analyze2.py` | Cut a samply profile by the log's `PHASE` windows and attribute samples to the timed calls. `analyze2.py` needs the binary's dSYM for inline frames. |
| `scripts/derive_instr.py` | Instructions by kind of work: kernel, wait, convert, builder, host and so on (derived). |
| `scripts/struct_compare.py A.log B.log` | Checks that the profiled build's structural output equals harness A's logs. |
| `results/` | The parsed tables as plain text: `ru_CA.table.txt`, `callru_CA.table.txt`, `derived_instr_partition.txt`, `burble_CA_t{1,4}.summary.txt`, `CA_t{1,4}.analysis.txt` (the samply attributions), and `cmetis.log`. `raw-logs.tar.xz` holds the raw logs: `ru_{CA,PA,TX}.log`, `callru_CA.log`, `ru_CA_t4_bt{0,200}.log` (`KMP_BLOCKTIME`), `samply_CA_t{1,4}.log`, `burble_CA_t{1,4}.log`, `check_CA_fine{,.BROKEN}.log`, `dump_{CA,PA,TX}.log` and `build.log`. |

## Rebuild and run

The build is the same as `../m1-reuse-cost`, with `m1prof.patch`, plus the settings the notes give for symbolized profiles:

```bash
export CARGO_PROFILE_RELEASE_DEBUG=line-tables-only CARGO_PROFILE_RELEASE_STRIP=none CARGO_PROFILE_RELEASE_SPLIT_DEBUGINFO=packed
cargo test --release -p graph --features index-falkordb --lib --no-run
```

The test is `graph::graphblas::m1_coarsen_bench::reuse::m1_reuse`. All runs used `M1_GRAPH=…/roadNet-CA.txt M1_CT=1 M1_JIT=run M1_REFINE=fine`, and:

| Log | Extra settings |
|---|---|
| `ru_CA.log` (also PA, TX) | `M1_THREADS=1,4 M1_RUNS=3 M1_FERAL=3 M1_PROF=1` |
| `callru_CA.log` | `M1_THREADS=1,4 M1_RUNS=3 M1_FERAL=0 M1_PROF=1 M1_CALLRU=1` |
| `samply_CA_t1.log`, `_t4` | `M1_THREADS=1` (or 4) `M1_RUNS=1 M1_FERAL=0 M1_PROF=1`, under `run.sh … profiles/CA_t1.json.gz` |
| `burble_CA_t1.log`, `_t4` | `M1_THREADS=1` (or 4) `M1_RUNS=1 M1_PROF=1 M1_BURBLE_PHASES=r0,r,b2,split M1_BURBLE_DEPTHS=0,1,8,14` |
| `ru_CA_t4_bt0.log`, `bt200` | `M1_THREADS=4 M1_RUNS=3 M1_PROF=1 KMP_BLOCKTIME=0` (or 200) |
| `check_CA_fine.log` | `M1_THREADS=1,4 M1_RUNS=1`, no `M1_PROF`. `struct_compare.py` against harness A's `main_CA_fine.log` must report identical structure. The `.BROKEN` copy has one value altered, and the comparison must catch it. |
| `dump_CA.log` (also PA, TX) | `M1_DUMP_ND=<out.bin>`: writes the C METIS input and exits |

**The new switches:**
- `M1_PROF=1`: skips the untimed checks and B1 inside the phase windows;
- `M1_CALLRU=1`: rusage per timed call;
- `M1_BURBLE_PHASES` and `M1_BURBLE_DEPTHS`;
- `M1_DUMP_ND=path`.

The patch applies cleanly and reproduces the scratch files byte for byte (checked on 2026-11-13). It was not rebuilt for the handoff.

The scripts hard-code `/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1prof` as `P`.

## Expected output

`PHASE` lines with `wall`, `instr`, `cycles` and CPU per phase window, read by `ru_table.py`. With `M1_CALLRU=1` there are per-call rusage lines too.

On the M5 (`ru_CA.table.txt`), 1 thread:
- R at depth 0: 0.95 s, 11.4 G instructions;
- R at depths ≥ 1: 13.39 s, 168.0 G;
- B2 at depths ≥ 1: 4.90 s, 69.8 G;
- feral: 7.07 s, 65.3 G.

## Left out

- **Raw samply profiles:** `profiles/CA_t{1,4}.json.gz` with their `.syms.json` and `*.summary.json` (1.8 MB). The `.analysis.txt` files derived from them are kept.
- **`profiles/CA_t1.noinline.txt`.**
- **The binary and its dSYM:** `graph-test` and `graph-test.dSYM` (257 MB).
- **Data:** `ndin/*.bin` (129 MB), the same bytes as m1audit's C METIS inputs.
- **`orig/`:** harness A's final sources, which are `../m1-reuse-cost/m1rcA.patch`.
