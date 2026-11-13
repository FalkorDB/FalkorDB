# H2: `book` works only with `pk` and `hb2` on (`npar` by reading the code, not run)

**Finding** (from `proofs/cch_coarsen_review`, finding H2; Lean theorems `book_needs_pk` and `book_needs_hb2` in `CchCoarsenReview/Book.lean`). In the m1-optimize bench, `number_fast` (`book`) and `number_par` (`npar`) leave four sets of arrays empty or cleared:
- `ct_i/ct_j` and `cl_q/cl_c` are always empty (`m1_reuse_bench.rs:513-516` and `:717-720`, m1opt.patch @7df8e5887);
- `ps_i/ps_j` are filled only for the stats line, and `number_patch` clears them (`:796`).

The switch-off paths of `pk` (`build_level`, `:946-949`) and `hb2` (`Mode::B2`, `:979-1010`) read those arrays. So with `book` on and either `pk` or `hb2` off, the reused levels come out wrong.

No reported number is affected. Every measured step that had `book` (step 6) or `npar` (step 14) also had `hb2` (step 1) and `pk` (step 5). The switches are not independent, though. That matters if the module keeps a switch-off path.

## Result

Rebuilt on 2026-11-13 from `../../prototypes/code/m1-optimize/m1opt.patch` at `7df8e588727f47619d9567d111c7fc26a2208caf`.

Every run uses the same setup:
- SNAP roadNet-PA;
- ND depths 0–3, 1 thread;
- the check run only (`M1_RUNS=0`);
- feral's order computed once and cached;
- the comparison against the saved baseline hashes `../../prototypes/code/m1-optimize/results/hashes/pa8_base.hash` (13 hashes: depth 0, plus split, R, B2 and capstop at depths 1–3).

| `M1O` | exit | hashes compared | hashes differing | first failed check |
|---|---|---|---|---|
| final configuration (`hb2,hsplit,cpack,hcon,pk,book,hkey,hpar,gath,usel,zmatch,npar`) | 0 | 13 | **0** | none |
| `hb2,pk,book` (control) | 0 | 13 | **0** | none |
| `book` | 101 | 0 (panics first) | — | `CHECK FAILED: projection lost vertices` (`m1_reuse_bench.rs:1140`), at depth 1 |
| `hb2,book` (no `pk`) | 101 | 0 (panics first) | — | `CHECK FAILED: projection lost vertices` (`:1140`), at depth 1 |
| `pk,book` (no `hb2`) | 101 | 13 | **10** | `CHECK FAILED: B1 != B2` (`:2344`); 65 of 65 levels differ |

How to read the three failures:
- **No `pk`:** the B2 level's Cᵀ is built from the empty `ct_i/ct_j`. The projection back through it loses every vertex, so the run stops at depth 1's B2 path.
- **No `hb2`:** the B2 split is masked by an `M` built from the empty `cl_q/cl_c`. With `ps_i` also empty, the level skips the straddler rows. The B2 levels differ from B1 everywhere, and B2 and capstop differ at depths 1–3. Split and R differ at depths 2–3 because the next depth's labels are derived from the broken B2 hierarchy.
- **Matches the earlier run:** this reproduces the first run, which used the m1opt v4 binary and was lost in the reboot. The outcomes are the same.

## Files

- `run.sh`: the whole procedure.
  - `setup`: temporary worktree, patch, submodules, download, sha256 checks.
  - `build`: as in the m1-optimize README.
  - `runs`: the five runs.
  - `trim`: copies the results here.
  - `clean`: removes the worktree and the temporary root.
  - The temporary root is `$H`, default `<repo>/../h2-tmp`, beside the checkout and outside `/tmp`. The repo is `$REPO`, by default the checkout that holds this script.
- `results/summary.txt`: the table above, as `run.sh runs` printed it.
- `results/<name>.log`: each run's lines that carry the result, from 1–3 KB each:
  - the run header (binary md5, `M1O`);
  - the per-path `HASH` lines, `HASHMISMATCH` lines and `CHECKS` structs;
  - the panic and the test result.

  The dropped lines are the loader line and the `STRAD` statistics. The full logs were 14–21 KB each.
- `results/build.txt`: the build's `Finished`/`Executable` lines from the script's own build step. That step was a no-op rebuild, because the first build had already run.

## What ran, exactly

1. `git worktree add --detach $H/wt 7df8e588727f47619d9567d111c7fc26a2208caf`, then `git apply m1opt.patch` (`git apply --check` passed first), then `git submodule update --init --recursive` (67 s).
2. Data: `curl -O https://snap.stanford.edu/data/roadNet-PA.txt.gz`, then `gunzip -k`. Both files match the sha256 in `../../prototypes/code/README.md`:
   - `.gz`: `450b8733…cdf549`;
   - `.txt`: `e7e9abdd…0579b1`.
3. Build with Homebrew clang 23.1.2 as `CC`/`CXX`, rustc and cargo 1.96.1, `CARGO_TARGET_DIR=$H/target` and `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`:

   `cargo test --release -p graph --features index-falkordb --lib --no-run`

   - It took 1 min 56 s with a warm native-deps cache.
   - `--list --ignored` shows `graph::graphblas::m1_coarsen_bench::reuse::m1_reuse`.
   - The binary's md5 was `7b2155e335d72a6feac296e35eb0e231`.
4. Each run:

   ```
   env M1_JIT=run M1_CT=1 M1_GRAPH=$H/data/roadNet-PA.txt M1_THREADS=1 M1_RUNS=0 M1_DEPTHS=3 \
       M1_FERAL=0 M1_PERM_CACHE=$H/cache/PA.perm M1_HASH=check:<pa8_base.hash> M1O=<switches> \
       $H/graph-test --ignored --exact graph::graphblas::m1_coarsen_bench::reuse::m1_reuse --nocapture
   ```

   The first run (the final configuration) has no cached order, so it runs feral once and writes `PA.perm`. Its 13/13 match shows the order is the one the baseline used.
5. `run.sh clean` deleted the worktree, the target dir, the data and the binary afterwards.

To re-run from scratch: `bash run.sh all`, then `bash run.sh clean`. It needs:
- network access, for SNAP and the submodules;
- Homebrew LLVM on macOS;
- about 10 GB free for the target dir.
