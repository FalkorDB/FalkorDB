# m1-matching: plain-Rust ordering-quality harness (mutual-max matching, then reuse)

**Backs:**
- stage 1: `../../m1-matching-quality.md`, mutual-max matching against feral's SHEM;
- stage 2: `../../m1-reuse-quality.md`, reusing depth 0's hierarchy (D5 option b) and capstop.

`../../d5-b-verdict.md` recounts stage 2's logs (see `../d5-judge`).

**Came from:**
- `m1match`, the harness: a copy of FalkorDB's feral crates with the prototype's changes, plus the `m1bench` driver;
- `m1reuse`, stage 2's scripts, logs and untouched stage-1 copy.

Neither was a git repository, so this folder holds the sources themselves, not a patch.

## What it measures

Order quality on SNAP roadNet-CA, PA and TX: arcs, triangles and elimination-tree height of each order, and its ordering time, over 5 seeds.
- **Stage 1** compares stock feral (`shem`) with the design's mutual-max matching (`mm`) and its variants: without the weight cap, with a round cap, with feral's sequential 2-hop pass. It also reports C METIS on the same graphs, and the time to coarsen the largest component to ≤ 100.
- **Stage 2** adds reuse (`mm-reuse`), capstop (`mm-reuse-capstop`) and a variant that stops at ≥ (`mm-reuse-ge`). It records straddler statistics, and the checks that the stock and mutual-max paths are unchanged.

## Files

| Path | What it is |
|---|---|
| `harness/` | The stage-2 harness, buildable as is: workspace `Cargo.toml` and `Cargo.lock`, `bench/` (the `m1bench` binary), `crates/` (`feral-metis`, `feral-ordering-core`, `feral-amd`), `tools/parse.py` (stage-1 tables) and `sweep.sh` (stage-1 sweep). |
| `harness/base/feral-metis-base/` | The untouched stage-1 `feral-metis`, built as a second crate so `m1bench samebase` can compare the two. |
| `orig-stage1/` | The stage-1 versions of the changed files: `coarsen.rs`, `lib.rs` and `node_nd.rs` (identical to `harness/base/feral-metis-base/src/`), and `main.rs`, the stage-1 `m1bench`. |
| `diffs/stage1-feral-metis-vs-feral-f127189.diff` | What stage 1 changed in upstream `feral-metis`: `coarsen.rs` and `lib.rs` only. |
| `diffs/stage2-vs-stage1.diff` | What stage 2 changed: `coarsen.rs`, `lib.rs`, `node_nd.rs` and the bench's `main.rs`. |
| `stage1-results/` | `m1match/out/`: `main.txt`, `diag.txt`, `rounds.txt`, `breaks.txt`, `coarsen.txt`, `cmetis.txt`. |
| `stage2-scripts/` | `sweep_main.sh` (main comparison), `post.sh` (straddlers, breaks, same/samebase, roots, diagnostics), `watchdog.sh` (12 GB RSS), `parse_reuse.py`. |
| `stage2-results/` | `tables.txt` (the parsed tables the report quotes), `same.txt` and `watchdog.log` as plain text. `raw-logs.tar.xz` holds `out/{main,diag,breaks,strad,roots}.txt`, `out/post.err`, the PA smoke logs and `straddlers.json`, the quality stage's straddler shares, which `../m1-reuse-cost`'s `analyze.py` reads. |

**Provenance of the crates.** They are FalkorDB's feral fork, `https://github.com/FalkorDB/feral` at `f1271891d4719ff070e2eaf4a7641a7052604bcd`, the rev that `graph/Cargo.toml` pins. Checked against that checkout on 2026-11-13:
- `feral-ordering-core` and `feral-amd` are byte-identical;
- `feral-metis` differs only in `src/coarsen.rs`, `src/lib.rs` and `src/node_nd.rs`.

**One change from the scratch copy.** Two `Cargo.toml` files pointed at absolute scratch paths. They now point inside `harness/`:
- `bench/Cargo.toml`: `feral_base`'s path was `…/scratchpad/m1reuse/base/feral-metis-base`;
- `base/feral-metis-base/Cargo.toml`: the paths of `feral-ordering-core` and `feral-amd` were `…/scratchpad/m1match/crates/…`.

`Cargo.lock` is unchanged.

## Rebuild

```bash
cd harness
cargo build --release          # target/release/m1bench; the release profile has debug = 1
```

It fetches `FalkorDB/feral` at `f127189` for `feral_orig`, used by `m1bench same`, and `flate2`.

**Checked on 2026-11-13:** the build took 2.4 s with the crates already compiled. On PA, `m1bench same` printed `stock-copy == original feral f127189: true`, and `m1bench samebase … mm 1` printed `identical=true`.

**Stage 1's exact code.** Copy `harness/`, then:
- copy `orig-stage1/{coarsen,lib,node_nd}.rs` over `crates/feral-metis/src/`;
- copy `orig-stage1/main.rs` over `bench/src/main.rs`.

That is not needed to re-run stage 1's variants, though. Stage 2's report shows the stage-2 crate reproduces stage 1's SHEM and mutual-max orders: `samebase` identical in 12/12 runs, and arcs, triangles and height equal in 30/30.

**The frozen stage-2 binaries.** The scratch folder had two:
- `m1bench-diag` (00:43), built from these final sources. `post.sh` used it.
- `m1bench-main` (00:37), built about 6 minutes before the last edits to `node_nd.rs` and `main.rs`. `sweep_main.sh` used it. It lacks `M1_NOCHECK` and `M1_REP_BREAK`.

Only the final sources were kept. So the main sweep's binary cannot be rebuilt exactly; the final sources are a superset of it.

## Data

`m1bench` reads the SNAP `.txt.gz` files directly; see `../README.md`. It drops self-loops, makes the graph symmetric, and ranks isolated nodes first, as the stack's `nd.rs` does.

## Run

```
m1bench order <graph.txt.gz> <variant> <seed> <reps>
m1bench coarsen <graph.txt.gz> <variant> <seed>      largest component, coarsen to <= 100
m1bench same <graph.txt.gz>                          stock copy against the original feral crate
m1bench samebase <graph.txt.gz> shem|mm <seed>       this crate against harness/base
m1bench checkbreak <graph.txt.gz>                    the closure checker must reject a broken row
m1bench strad <graph.txt.gz> mm-reuse <seed>         straddler statistics (stage 2)
m1bench export <graph.txt.gz> <out.bin>              C METIS input (int64 n, nnz, xadj, adjncy)
m1bench evalperm <graph.txt.gz> <perm.bin>           quality of a C METIS permutation
```

**Variants:**
- `shem`, `shem-cap`, `mm`, `mm-nocap`;
- `mm-r<K>`: round cap K;
- `mm-inv`: break, min-weight proposals;
- `mm-seq2hop`: feral's sequential 2-hop pass;
- `swap<K>`: break, feral's order plus K random swaps;
- `mm-reuse`, `mm-reuse-capstop`, `mm-reuse-ge`;
- `mm-reuse-b1`…`b5`: reuse breaks.

**Environment:**
- `M1_NOCHECK=1` skips the untimed check rep of reuse runs;
- `M1_REP_BREAK=1`, `M1_SAME_BREAK=1` and `M1_SAMEBASE_BREAK=1` are the breaks of the repeat, `same` and `samebase` checks.

**Stage 1.** `harness/sweep.sh`. Edit `W` (the harness) and `D` (the folder of `.txt.gz` files).
- It writes `out/main.txt`: `shem` and `mm`, seeds 1–5, 3 reps.
- It also writes `diag.txt`, `rounds.txt`, `breaks.txt` and `coarsen.txt`.
- Then `python3 tools/parse.py out/main.txt` prints the tables.

C METIS for stage 1 (`stage1-results/cmetis.txt`):

```bash
m1bench export roadNet-CA.txt.gz CA.bin
../cmetis/nd CA.bin CA.perm            # build nd with ../cmetis/build.sh
m1bench evalperm roadNet-CA.txt.gz CA.perm   # prints RESULT … variant=cmetis …
```

**Stage 2.** Edit `S`, `B` and `D` in the scripts; `B` was the frozen binary. Then:
1. `stage2-scripts/sweep_main.sh` (main comparison into `out/main.txt`; run `watchdog.sh <pid>` beside it);
2. `stage2-scripts/post.sh`;
3. `python3 stage2-scripts/parse_reuse.py out/main.txt out/strad.txt out/diag.txt <stage1-results/cmetis.txt> straddlers.json`. This prints the tables and writes `straddlers.json`.

## Expected output

One `RESULT graph=… variant=… seed=… arcs=… triangles=… closure_violations=0 cover_ok=true … etree_height=… order_s_median=…` line per order.

From the reports, CA medians over 5 seeds:

| Variant | Arcs | Triangles | Time |
|---|---|---|---|
| feral SHEM | 12,631,332 | 219.0M | 7.07 s (stage 1), 7.20 s (stage 2) |
| C METIS (`cmetis.txt`) | 11,864,871 | 194.9M | |

`checkbreak`, the `mm-inv` and `swap1000` breaks, and the `*_BREAK` variables must each show a failure or a worse order. `stage1-results/breaks.txt` and `stage2-results/tables.txt` record what they gave.

## Left out

- **Binaries:** `m1reuse/bin/m1bench-main` and `m1bench-diag`.
- **`m1reuse/reuse_block.rs`:** an earlier draft of the reuse block that went into `node_nd.rs`; 795 of its 817 lines are in the final file.
- **Data:** the `.txt.gz` files.
