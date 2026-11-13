# M1: how fast the GraphBLAS coarsening prototype can go

**Corrections from the verification (main session, 2026-11-13).** An independent re-run (the Verification section at the end) reproduced the CA timings, instruction counts and 57/57 hashes within 1 %. Read the body with four corrections:
- **The stop rule is not the quality harness's rule for cells of 5,000 or more vertices.** The host replay checks this bench's own rule: a floor of 100 and R's cap. For those cells the harness reuses only down to max(100, n/30) coarse vertices, uses each cell's own cap, and runs five fresh coarsenings. With the harness's floor alone, capstop takes 2.44 s at 1 thread and 1.32 s at 4 on CA (+6 % and +9 %; about 0.36× and 0.19× feral). The harness's cap and its five fresh runs were not measured, and both add cost. The quality result in `m1-reuse-quality.md` holds for the harness's rule, not for this one.
- **The 4-thread ratios divide by serial feral and serial C METIS.** Neither was run with threads.
- **GraphBLAS-only rewrites saved 0.10 s in the final set** (`usel`, `zmatch`), not 0.23 s. `caprc` was replaced by the host pass `hkey`.
- **Two break claims were wrong.** `hpar` had no break of its own; the verifier added one, and it fails as it should. Once `hcon` is on, the `split` break fails inside a GraphBLAS call, not at the split check, because host passes trust their input. A module version needs debug assertions there.

**Verdict: capstop's GraphBLAS part is now a third of feral's whole ordering at 1 thread and under a fifth at 4 threads.** It keeps the same output: every matrix, weight, map and label is identical to the unoptimized code.

| Graph | capstop, 1 thread | / feral | / C METIS | capstop, 4 threads | / feral | / C METIS |
|---|---|---|---|---|---|---|
| CA | 7.22 → **2.30 s** | **0.33** | 0.45 | 3.36 → **1.20 s** | **0.18** | 0.24 |
| PA | 3.85 → **1.23 s** | **0.35** | 0.47 | 1.83 → **0.68 s** | **0.19** | 0.26 |
| TX | 4.63 → **1.48 s** | **0.33** | 0.44 | 2.23 → **0.81 s** | **0.18** | 0.24 |

**Time left for the host steps** (initial separators, FM, MMD, tree), derived as the reference time minus capstop's time:

| Graph | to match feral, 1 thr | 4 thr | to match C METIS, 1 thr | 4 thr |
|---|---|---|---|---|
| CA | 4.58 s | 5.67 s | 2.76 s | 3.85 s |
| PA | 2.31 s | 2.86 s | 1.42 s | 1.97 s |
| TX | 3.05 s | 3.72 s | 1.89 s | 2.56 s |

Three things to know before using these numbers:

1. **Most of the gain comes from host passes, not from GraphBLAS kernels.** The big wins read a matrix's CSR arrays in place (an O(1) unload), do one pass on the host, and hand the result back as a new matrix (an O(1) load). The matrices stay GraphBLAS matrices, but the work is plain Rust. Of the final 2.30 s on CA at 1 thread:
   - 1.23 s are host passes;
   - 0.52 s are GraphBLAS kernels, almost all of them the matching rounds;
   - 0.55 s are the uncoarsening stand-ins, the projection and boundary `mxv` the earlier benches also timed.

   GraphBLAS-only reformulations saved 0.23 s in total (`caprc`, `usel`, `zmatch`).
2. **In instructions, capstop still costs more than C METIS's whole ordering.** On CA it runs 37.4 G instructions against C METIS's 32.5 G and feral's 65.2 G. Its wall time is lower because the host passes run at about 3.8 instructions per cycle (37.4 G in 9.9 G cycles), against 1.5 for C METIS (32.5 G in 21.8 G cycles).
3. **Without the stand-ins**, the coarsening alone takes 1.75 s on CA at 1 thread and 1.00 s at 4 threads (derived: capstop minus `project` and `bnd`).

The B2 path (reuse without the cap stop, which fails on quality) went from 6.01 to 1.79 s on CA at 1 thread. R (re-coarsen every depth) went from 13.62 to 7.15 s.

## Every change, in order (CA, capstop total, median of 3)

Each change was measured on top of the ones before it, one at a time, at 1 and 4 threads. Its equality was checked against the baseline in the same process. Rows marked "no" were dropped. The full table, with depth 0, B2 and R columns, is `m1opt/attempts.csv`.

| # | Switch | What it does | Kind | 1 thr wall | 1 thr instr | 4 thr wall | Kept |
|---|---|---|---|---|---|---|---|
| 0 | — | baseline | | 7.20 s | 107.2 G | 3.33 s | |
| 1 | `hb2` | B2 level in one pass over the parent level's CSR, splicing the straddler rows and columns in | host | **−2.45 s (−34 %)** | −34.6 G | −0.79 s | yes |
| 2 | `hsplit` | fine split per depth in one pass over A0's CSR | host | −0.57 s (−12 %) | −9.0 G | −0.09 s | yes |
| 3 | `cpack` | leader numbering in one pass over `mate`; C and Cᵀ packed; C·A without a transpose; w_c summed on the host | host | −0.13 s (−3 %) | −2.5 G | +0.02 s | yes |
| 4 | `hcon` | contraction C·A·Cᵀ and OFFDIAG in one pass over A's CSR | host | −0.52 s (−13 %) | −4.7 G | −0.12 s | yes |
| 5 | `pk` | Cᵀ of a reused level packed from the map; the level's weight vector not built (only checks read it) | host | −0.35 s (−10 %) | −9.0 G | −0.23 s | yes |
| 6 | `book` | numbering in fewer passes; capstop patches the numbering instead of renumbering when a cell stops | host | −0.29 s (−9 %) | −1.7 G | −0.28 s | yes |
| 7 | `caprc` | cap mask by a row scale and a column scale | GraphBLAS | −0.13 s (−5 %) | −1.8 G | −0.04 s | yes, then superseded by 8 |
| 8 | `hkey` | keys and the cap test in one pass over A's CSR | host | −0.24 s (−9 %) | −3.3 G | −0.03 s | yes |
| 9 | `hpar` | host passes split across the threads | threads | 0.00 s | +0.2 G | **−0.39 s (−22 %)** | yes |
| 10 | `gath` | capstop's gather by appending CSR rows | host | −0.12 s (−5 %) | −2.2 G | −0.04 s | yes |
| 11 | `usel` | unmatched set by `select(m == false)` instead of a generic masker | GraphBLAS | −0.06 s (−3 %) | −0.8 G | −0.03 s | yes |
| 12 | `zmatch` | matches carried in the gather vector; no per-round `mate` assign | GraphBLAS | −0.04 s (−1.5 %) | −0.1 G | −0.01 s | yes |
| 13 | `nowait` | no per-call waits; one wait per level output | GraphBLAS | +0.01 s | 0.0 G | −0.01 s | no: not measurably better |
| 14 | `npar` | numbering and Cᵀ packing split across the threads | threads | +0.02 s | 0.0 G | −0.10 s (−8 %) | yes |
| 15 | `chunk` 1M | GraphBLAS chunk 1,048,576, so small calls run on one thread | GraphBLAS | −0.02 s | 0.0 G | +0.39 s | no: worse |
| 16 | `chunk` 256K | the same at 262,144 | GraphBLAS | −0.02 s | −0.1 G | +0.09 s | no: worse |
| | | **final** (`hb2,hsplit,cpack,hcon,pk,book,hkey,hpar,gath,usel,zmatch,npar`) | | **2.30 s** | **37.4 G** | **1.20 s** | |

- **Spreads** across the three runs (`m1opt/steps.csv`):
  - instructions: under 3 % on every row;
  - wall time: under 2 % on 84 % of the total rows. The widest are 4-thread rows: 7 % at step 3 and 9 % in the dropped 256K chunk run.

  The load average at run starts was 1.5 to 4.4.
- `caprc` stays in the table but not in the final set. With `hkey` the cap mask is no longer built.
- **Where 4 threads gained:** `hpar` and `npar`. The host passes were serial until then, so steps 1 to 8 helped 4 threads less than 1 thread.
- **Why `nowait` did nothing:** the waits that cost something were the unjumble after C·A and the zombie removals after the masked products. `hcon`, `hb2` and `hsplit` had already removed those calls. The waits left are on vectors and finish no pending work.
- **Why the chunk setting hurt:** it also makes the 2M-entry vector calls of the matching rounds single-threaded.

## Copies, scans and waits found and removed

CA, capstop, 1 thread, call times summed over all depths (median of 3). Baseline `m1opt/logs/ca_base.log`, final `m1opt/logs/session/CA_final.log`.

| Found | Baseline calls | Replaced by | Final |
|---|---|---|---|
| **B2 rebuilt each level three times to drop a few percent of it.** It built M; ran a masked dot3, whose wait rewrites the whole matrix to drop its zombies; ran an `eWiseMult` to copy the values back; resized (sparse to hyper and back); ran three products for the straddler rows plus a transpose fill; and merged with `eWiseAdd`. | 2.74 s, 39.8 G (`split_mxm`, `split_mult`, `union`, `strad`, `m_build`, `resize`) | `hb2`: one pass over the parent level, splicing the straddler rows and columns in | 0.35 s, 6.2 G |
| **The fine split ran a masked dot3 over all 5.5M entries every depth**, plus a zombie-removal wait and a single-threaded build of M, to drop the rows of the vertices that left. | 0.74 s, 11.5 G | `hsplit`: one filter pass | 0.14 s, 2.6 G |
| **The contraction ran a bucket transpose of Cᵀ inside every C·A**, then an unjumble wait, a second product, and an OFFDIAG select that copies the result again. | D0 0.38 s + CS 0.34 s | `cpack` then `hcon`: one pass, rows come out sorted and without the diagonal | D0 0.11 s + CS 0.10 s |
| **Leader numbering took three calls and a host round trip.** It did an `extractTuples` of a bitmap vector (a bitmap-to-sparse conversion), a build, a generic bitmap gather and a single-threaded build of Cᵀ. C·w used a transposed C (bitmap saxpy, bitmap to full). | D0 0.07 s + CS 0.06 s | `cpack`: one pass over `mate`, which is unloaded in O(1) | D0 0.02 s + CS 0.02 s |
| **The cap mask built three temporaries the size of A in four passes**, and the key apply under it ran the generic masker. | CS cap 0.24 s + key 0.16 s | `caprc`, then `hkey`: keys and cap test in one pass | CS key 0.05 s |
| **The unmatched set was a complemented-mask copy through the generic masker** every round, and **`mate` was rewritten every round** through a generic bitmap assign, although it is read only after the loop. | D0 0.13 s (`m_unext`, `m_assign`, `m_full`) | `usel` and `zmatch` | D0 0.07 s (`m_unext`, `z_add`, `z_final`) |
| **Cᵀ of each reused level went through `GxB_Matrix_build`**, which checks the order and converts hyper to sparse, although the numbering already holds it in row order. | 0.20 s | `pk`: packed | 0.06 s |
| **A weight vector was built for every reused level, though only the checks read it.** | 0.11 s | `pk`: not built | 0 |
| **capstop numbered a level twice whenever a cell stopped**, and rescanned every level to count each cell's coarse vertices. | `number` 0.77 s | `book`: one numbering plus a patch; counts carried | 0.31 s |
| **capstop's gather extracted each stopped block, copied it out with `extractTuples`, then rebuilt it with a sorting build.** | 0.16 s | `gath`: rows appended | 0.03 s |
| **The depth-0 maps were copied out of Cᵀ and w with `extractTuples` for the reuse paths.** | 0.014 s | `cpack` keeps them on the host | 0 |

The remaining serial host work at 4 threads is `number` (0.24 s) and the straddler part of `b2_splice`. The deep levels are below the 100,000-entry threshold for splitting a pass across threads.

## Final totals, all three graphs, one session

One process per graph and configuration. Each ran a check run (not timed), then 3 timed runs at 1 thread and 3 at 4 threads. Feral ran 3 times inside the baseline process. C METIS `METIS_NodeND` ran 3 times as its own process. The 1-minute load at run starts was 1.8–4.4. Logs: `m1opt/logs/session/{CA,PA,TX}_{base,final}.log` and `m1opt/logs/session/cmetis.log`. Medians; per-run values are in `m1opt/final.csv`.

| Graph | Thr | R base → final | B2 base → final | capstop base → final | feral | C METIS |
|---|---|---|---|---|---|---|
| CA | 1 | 13.62 → 7.15 s | 6.01 → 1.79 s | **7.22 → 2.30 s** | 6.87 s | 5.05 s |
| CA | 4 | 5.09 → 2.97 s | 2.78 → 0.89 s | **3.36 → 1.20 s** | | |
| PA | 1 | 7.13 → 3.68 s | 3.16 → 0.95 s | **3.85 → 1.23 s** | 3.54 s | 2.65 s |
| PA | 4 | 2.70 → 1.62 s | 1.48 → 0.49 s | **1.83 → 0.68 s** | | |
| TX | 1 | 8.97 → 4.66 s | 4.01 → 1.20 s | **4.63 → 1.48 s** | 4.53 s | 3.37 s |
| TX | 4 | 3.49 → 2.03 s | 1.88 → 0.62 s | **2.23 → 0.81 s** | | |

**Instructions** (median, G):

| Graph | capstop base → final, 1 thr | 4 thr | feral | C METIS |
|---|---|---|---|---|
| CA | 107.5 → 37.4 | 117.8 → 41.3 | 65.2 | 32.5 |
| PA | 57.5 → 20.3 | 63.1 → 22.2 | 35.0 | 17.3 |
| TX | 70.0 → 24.7 | 76.5 → 27.2 | 43.8 | 22.0 |

- **Speed-up on capstop** is 3.1× at 1 thread and 2.7–2.8× at 4 threads, on all three graphs.
- **At 4 threads the code executes more instructions than at 1 thread**: 41.3 against 37.4 G on CA. The extra comes from the chunked host passes, the atomics in `npar` and GraphBLAS's 4-thread methods.
- **Spreads** across the three runs: instructions under 1 % on every row. Wall time on the totals is under 4 %, except CA's baseline R at 4 threads (5.3 %).
- **Peak RSS** was 2.03 GB for the baseline process and 1.88 GB for the final one on CA. Each figure is the whole process: the text graph, host arrays, feral and the check run.

### Where the final capstop's time goes (CA)

`python3 m1opt/split_calls.py m1opt/logs/session/CA_final.log`; calls summed over depth 0, the split and depths 1–14.

| Group | 1 thread | 4 threads | Largest calls (1 thread) |
|---|---|---|---|
| host passes | 1.23 s, 21.2 G | 0.77 s, 24.9 G | `b2_splice` 0.35, `number` 0.31, `hcon` 0.21, `split_host` 0.14, `key` 0.09 |
| GraphBLAS kernels | 0.52 s, 6.9 G | 0.24 s, 6.9 G | matching `mxv` 0.17, gather `m_extract` 0.07, `m_unkey` 0.06, `m_eq` 0.05 |
| stand-ins (projection, boundary `mxv`) | 0.55 s, 9.4 G | 0.20 s, 9.6 G | `bnd` 0.39, `project` 0.16 |
| **total** | **2.30 s** | **1.21 s** | |

Depth 0 alone is 0.53 s at 1 thread and 0.23 s at 4. Of that, the matching rounds are about 0.32 s at 1 thread (derived from the calls above). Depths 1–9 cost 0.09–0.12 s each at 1 thread, against 0.33–0.41 s in the baseline.

## capstop: the rule and its check against the quality harness

**The rule.** This copy implements the quality harness's `reuse_capstop` rule (`m1match`, `reuse_levels`) in one stage with a floor of 100 per cell, the bench's floor.
- A cell reuses the parent's next level by B2 while two things hold: it has more than 100 coarse vertices, and the next level keeps every one of its coarse vertices within the cap.
- A cell at 100 or fewer is done; it gets no fresh coarsening.
- A cell that would break the cap is gathered with the other stopped cells and coarsened fresh with R's matching.
- So is a cell whose parent has no next level. B2 needs the parent's level matrix; the quality harness composes levels from the depth-0 maps instead.

This differs from m1rcB's capstop, which had no per-cell floor and kept reusing small cells until the cap broke. So the baseline here (7.22 s on CA) is not m1rcB's 7.35 s.

**The check** (`M1_RULECHECK=1`). The quality harness's walk is replayed on the host for every cell at every depth: compose the depth-0 maps over the cell's live vertices, then stop at the floor, at the last level, or before the first level whose heaviest coarse vertex exceeds the cap. The GraphBLAS capstop's stop level and reason must match the replay. The one allowed exception is a cell limited by its parent, which can only stop earlier.

| Graph | Cells checked | Same stop level and reason | Parent-limited (B2 needs the parent's level) | Other differences |
|---|---|---|---|---|
| PA (`m1opt/logs/rule_pa.log`) | 32,766 | 32,762 | 4 | **0** |
| PA, final config (`m1opt/logs/rule_pa_final.log`) | 32,766 | 32,762 | 4 | **0** |
| CA (`m1opt/logs/ca_base.log`) | 65,532 | 65,462 | 70 | **0** |

**Break.** `M1_BREAK=cap2` lets capstop reuse up to twice the cap. On PA, 8,190 cells then differ (for example depth 1, cell 0: capstop stops at level 22 for the floor, while the rule stops at level 14 for the cap), and the run fails (`m1opt/logs/rule_pa_cap2.log`).

**One remaining difference from the quality harness: the cap.** The quality harness uses each cell's own cap, 1.5 × cell weight / floor. This bench uses R's cap for the depth, 1.5 × (n / 2^d) / 100, so that capstop's fresh part and R use the same matching. Replayed with each cell's own cap, the stop level changes for 1,858 of PA's 32,766 cells and 1,862 of CA's 65,532. These are mostly at depths 9–12, where cell sizes spread the most. The order quality of this bench's capstop is not measured.

## Checks and deliberate breaks

**Equality with the baseline.** Every check run hashes every level of every path at every depth: the matrix tuples, Cᵀ, the weights, cells, maps, the projected labels and capstop's stop records. That is 57 hashes per graph: depth 0, and the split, R, B2 and capstop at depths 1–14.
- The baseline wrote them; each optimized run compares at 1 and at 4 threads in the same process.
- Results: 57/57 equal for every kept step on CA, and for the final configuration on CA, PA and TX.
- 1-thread and 4-thread hashes also agree within each process.

**Checks kept from m1rcB**, all passing in every run:
- B1 = B2 on every level;
- B1 valid: no diagonal, symmetric, weight conserved, host weights = Cᵀ·w;
- the accounting identity;
- no matching in B1 or B2;
- the fine split's pattern against a host count;
- separators separate;
- capstop's reused levels within the cap, and its weight conserved;
- fingerprints identical across runs and thread counts.

**Breaks.** Final binary, PA, depths 1–3, 1 and 4 threads (`m1opt/logs/breaks_final.txt`). Both controls pass: the final configuration and the baseline (13/13 hashes equal).

| Break (`M1O_BREAK`) | Perturbation | Caught by |
|---|---|---|
| `hb2` | drops row 0's kept entries | B1 ≠ B2; 6/13 hashes differ |
| `hsplit` | drops row 0 of the split | split pattern; 6/13 |
| `cpack` | swaps two coarse ids in Cᵀ | 7/13 hashes |
| `hcon` | +1 on one contracted value | B1 ≠ B2; 13/13 |
| `pk` | swaps two entries of a packed Cᵀ | B1 ≠ B2; 6/13 |
| `book` | +1 on one patched weight | 1/13 hashes |
| `hkey` | raises one edge's key priority | 13/13 hashes |
| `gath` | +1 on one gathered value | 3/13 hashes |
| `usel` | selects the matched proposers instead | 13/13 hashes |
| `zmatch` | skips the stale-proposal reset | weight not conserved |
| `npar` | +1 on one weight in the parallel numbering | B1 level invalid; 6/13 |
| `caprc` | column scale with FIRST instead of PLUS | 13/13 hashes |
| `nowait` | +1 on one stored value of A_c | B1 ≠ B2; 13/13 |

The m1rcB breaks were run again under the final configuration:
- caught: `split`, `cap2`, `salt`, `sepclean`, `mergeparts`, `gfdrop`, `nosep`, `rematch`, `b1diag`, `nomutual`;
- not caught: `nooffdiag` skips an OFFDIAG call that `hcon` no longer makes, so it changes nothing. Without `hcon` (baseline, and `hb2` alone) it is caught: "A_c has a diagonal".

## Setup

- **Worktree:** `m1opt`, at 7df8e5887. The full path is `/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1opt`; every `m1opt/` path in this report is under it. Nothing was committed.
- **Sources:**
  - Code: `graph/src/graph/graphblas/m1_coarsen_bench.rs` and `m1_reuse_bench.rs`, registered in `mod.rs` as in m1gb.
  - Starting point: m1rcB's files (in `m1opt/orig/`). They already had capstop, which was changed to the per-cell rule above.
  - Every switch is in `M1O`. Breaks are in `M1O_BREAK`; the baseline checks' breaks are in `M1_BREAK`.
- **Build:** `m1opt/build.sh`.
  - `cargo test --release -p graph --features index-falkordb --lib`; the release profile has `lto = true` and `codegen-units = 1`, plus line tables.
  - Homebrew clang. Own `CARGO_TARGET_DIR` (deleted).
  - Binaries in `m1opt/bin/`, with each run's md5 and `--list` in its log:
    - v1 for steps 0–5;
    - v2 for steps 6–7;
    - v3 for steps 8–13;
    - v4 for step 14 on, and for the final session.

    A new binary only adds switches.
- **Run:** `m1opt/run.sh`, with a 12 GB RSS watchdog that never fired. Settings: `M1_JIT=run`, `M1_CT=1`, depths 0–14.
- **Machine:** Apple M5, 10 cores.
- **Feral:** its order comes from a cache (`m1opt/cache/*.perm`) except in the session's baseline processes, which also time it.
- **Measures:**
  - Wall time is the sum of each path's timed calls, as in the earlier reports.
  - Instructions and cycles come from `proc_pid_rusage` (`RUSAGE_INFO_V4`), read around every timed call.
  - The check run (run 0) is excluded from the medians.
  - The uncoarsening stand-ins (`project`, `bnd`) are timed as in m1rcB.
- **Changes to the timing itself:** the vector copies in `coarsen_level` are now timed. They were untimed before; the cost is about 1 ms per large level.
- **Host passes and memory ownership:**
  - Matrices are read through `GxB_unload_Matrix_into_Container` plus `GxB_Vector_unload`, then put back. The arrays come out as 32-bit offsets and indices.
  - New matrices are loaded as 32-bit CSR with `GxB_Vector_load` and handed to GraphBLAS.
  - In this test binary, Rust's allocator and GraphBLAS's are both libc malloc, so Rust `Vec` buffers are handed over without a copy.
  - Matrices in a non-CSR format were converted on the fly: this happened 9 times in the runs before `hcon`, and 0 times after.
- **Parallel host passes** split rows into one chunk per thread, only above 100,000 entries, with `std::thread::scope`.
- **Analysis scripts:**
  - `m1opt/analyze.py` (phase totals and CSV);
  - `m1opt/make_attempts.py` (attempts table);
  - `m1opt/split_calls.py` and `m1opt/call_table.py` (call breakdowns);
  - `m1opt/session.sh` (the final session).

## Not measured

- **The module's allocator.** In the module, GraphBLAS allocates with RedisAlloc. The packed arrays would have to come from it, so the zero-copy hand-over of Rust `Vec`s does not carry over as written.
- **Linux, and other thread counts.** 16 threads were not run, and the M5's mix of 4 performance and 6 efficiency cores may affect the 4-thread numbers.
- **The quality of this capstop's orders.** Its stop rule matches the quality harness except for the cap definition; the order itself was not measured.
- **Real host steps.** The projection and boundary step are still stand-ins (0.55 s on CA at 1 thread); a real FM replaces them. Initial separators, MMD, tree and peeling are not included.
- **A host matching.** The matching rounds are the last large GraphBLAS cost (about 0.32 s of depth 0 on CA at 1 thread, derived). The plain-Rust mutual-max coarsening in `m1-matching-quality.md` took CA's largest component to 100 vertices in 0.35 s. Moving the rounds to the host is the next lever, and it would leave almost no GraphBLAS kernel in the coarsening.
- **Candidates not tried, each estimated under 2 %:**
  - named user ops for the runtime JIT: `m1_unkey` is 0.06 s at 1 thread. A PreJIT harvest needs Linux;
  - matching rounds restricted to the unmatched vertices: the audit estimated about 27 ms per depth 0;
  - a parallel straddler part in `b2_splice`.
- **Memory of the GraphBLAS part alone.**
- **Best-of-5 fresh stages** in capstop and R.

## Verification

An independent re-run and code review, by a separate agent, on 2026-11-13. Nothing in `m1opt` was changed.

**Verdict: the timings, the instruction counts and the equality hold on CA. Three claims need correcting.**
1. **capstop does not follow the quality harness's stop rule for cells of 5,000 or more vertices.** The host replay it is checked against is this bench's own version of the rule. With the harness's floor for those cells, capstop costs 6 % more at 1 thread and 9 % more at 4 (measured below). The headline still holds.
2. **The 4-thread ratios compare against serial feral and serial C METIS.** The tables do not say so.
3. **Two break claims are wrong.** `hpar` had no break of its own, and the `split` break is not caught by the split check under the final configuration. I added an `hpar` break; it fails as it should.

### Setup

- **Worktree:** my own, `m1verify`, at 7df8e5887. I copied the two bench files and `mod.rs` from `m1opt` and checked them byte for byte with `cmp` before building.
- **Build:** my own target dir, with the same command as `m1opt/build.sh`: release, `lto = true`, `codegen-units = 1`, Homebrew clang. `--list` shows the one test, `m1_reuse`. The target dir was deleted at the end.
- **Files:** in `m1verify-work/`, full path `/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1verify-work`. Every `m1verify-work/` path below is under it.
- **Session:** one session on CA, in this order: baseline, final, C METIS 3 times, final, baseline (`m1verify-work/session.sh`).
  - Each bench process did a check run, then 3 timed runs at 1 thread and 3 at 4 threads.
  - Feral ran 3 times inside each baseline process.
  - The 1-minute load was 1.9–3.8 at the timed run starts and 3.7–3.9 at the C METIS runs.

### Re-run on CA

Medians of 3 runs per process. Logs: `m1verify-work/logs/CA_{base1,final1,final2,base2}.log` and `m1verify-work/logs/cmetis.log`. Totals by `m1verify-work/tot.py`, which uses the same path sums as `m1opt/analyze.py`.

| | Report | Verifier, 1st process | Verifier, 2nd process |
|---|---|---|---|
| capstop baseline, 1 thr | 7.22 s, 107.5 G | 7.19 s, 107.4 G | 7.24 s, 107.4 G |
| capstop final, 1 thr | 2.30 s, 37.4 G | 2.30 s, 37.5 G | 2.32 s, 37.5 G |
| capstop baseline, 4 thr | 3.36 s, 117.8 G | 3.34 s, 117.8 G | 3.36 s, 117.9 G |
| capstop final, 4 thr | 1.20 s, 41.3 G | 1.19 s, 41.2 G | 1.22 s, 41.3 G |
| R baseline → final, 1 thr | 13.62 → 7.15 s | 13.56 → 7.13 s | 13.64 → 7.15 s |
| R baseline → final, 4 thr | 5.09 → 2.97 s | 4.98 → 2.97 s | 5.21 → 3.00 s |
| B2 baseline → final, 1 thr | 6.01 → 1.79 s | 5.99 → 1.79 s | 6.02 → 1.80 s |
| B2 baseline → final, 4 thr | 2.78 → 0.89 s | 2.75 → 0.89 s | 2.77 → 0.89 s |
| feral, whole ordering, serial | 6.87 s, 65.2 G | 6.75 s, 65.2 G | 6.94 s, 65.3 G |
| C METIS, whole process, serial | 5.05 s, 32.5 G | 5.06 s, 32.5 G (3 runs, 5.059–5.067 s) | |

- **Noise.** Within a process, instructions vary by under 1 %. capstop's wall time varies by under 2 %; R's baseline at 4 threads varies by up to 5 %, as in the report.
- **Against the report,** capstop is within 1 %, R within 2.4 % and feral within 1.7 %. That is run-to-run noise.
- **Ratios from my numbers:**
  - capstop final / feral: 0.33–0.34 at 1 thread, 0.17–0.18 at 4 threads;
  - capstop final / C METIS: 0.45–0.46 and 0.24.

  These are the report's ratios.
- **Instructions per cycle:** capstop final runs 37.5 G instructions in 9.9 G cycles (3.8). C METIS runs 32.5 G in 21.8 G cycles (1.5). Confirmed.
- **Breakdowns.** The report's groups reproduce within 0.01 s on my `CA_final1.log` (`m1opt/split_calls.py`): host passes 1.23 s, GraphBLAS kernels 0.52 s, stand-ins 0.55 s at 1 thread. So does depth 0: 0.52 s, of which the matching rounds are 0.30 s, or 0.32 s with the 2-hop pass.

### Equality

- **Same code.** My baseline build wrote 57 hashes that are byte-identical to `m1opt/cache/ca_base.hash`.
- **Final configuration.** Both final processes matched 57/57 at 1 and 4 threads. The second baseline process also matched 57/57.
- **Timed runs.** The timed runs are not hashed. They run the same code, because the switches are global, and their per-level fingerprints (n, nnz, sum) agreed: 301 comparisons per process, 0 failures.
- **Same work.** On every PHASE line, the work counts are identical between baseline and final at both thread counts: levels, reused and fresh levels, coarsened vertices and stop reasons. No level is skipped and no cell is dropped.
- **What the hashes cover.** From reading `hierarchy_keep`, `hash_lvl` and `build_capstop`:
  - every level's full tuple lists of A_c and C' and its weights;
  - for reused levels, also the cells, parent ids and maps;
  - capstop's gathered graph, its weights and its fresh hierarchy;
  - the labels after projection, and every cell's stop record.

  Each kept switch changes at least one of these, so a different output would show.

### Breaks

- **The report's breaks.** I read the 13 break logs and both controls in `m1opt/logs/brk/`. Each break perturbs only its own switch's output, and each run fails on a hash or on a check that reads that output.
  - Two fail on a check rather than a hash. `zmatch` makes `mate` stop being an involution, which fails "vertex weight not conserved" at the first level. `npar` fails "B1 level invalid", with 6 of 13 hashes different.
  - Both are failures for the right reason.
- **`hpar` had no break of its own.** I added one in my copy (`M1V_BREAK=hpar`): in the threaded `emit_csr`, chunk 1's first value takes its second value's place.
  - Setup: PA, depths 1–3, final configuration.
  - The control passes 13/13.
  - The break fails with 10 HASHDIFF lines, all at 4 threads, plus 12 HASHMISMATCH lines and "B1 != B2". The mismatches at 1 thread come from the prep pass, which runs at 4 threads and derives the cells.
  - Logs: `m1verify-work/logs/fu/hpar_control.log` and `hpar_break.log`.
- **The `split` break is not caught by the split check under the final configuration.** The report lists it as caught. The run does fail, but inside a GraphBLAS call in R at depth 1.
  - Probe on PA, depth 1: with `M1O=hsplit,cpack` the run fails at "split pattern". Adding `hcon` makes it fail at `GxB_Vector_extract_Vector` instead. Logs: `m1verify-work/logs/fu/split_cpack.log` and `split_cpack_hcon.log`.
  - The broken split leaves entries to vertices that are not live. The GraphBLAS contraction drops those entries, because C has no row for those vertices. `hcon` maps them to coarse id `NONE` (u32::MAX), writes that as a column, and loads the CSR without a check.
  - With a valid split this cannot happen, so the equality result stands. But the host passes trust their input, so a bad input becomes an invalid matrix instead of a check failure. A module version needs a debug assertion there.

### Fairness

- **JIT:** both sides print `JIT: RUN`.
- **Build:** baseline and final run from the same binary in both sessions.
- **Graph:** the same file on both sides. C METIS's input, `m1audit/cmetis/CA.bin`, has n = 1,965,206 and 5,533,214 entries, as the bench's load does. Its converter, `m1audit/to_nd_bin.py`, follows the same rules.
- **Waits:** I diffed the baseline path (`M1O` empty) against m1rcB's originals in `m1opt/orig/`. Every per-call wait is still there: `w8m` and `w8v` wait unless `nowait` is on, and `nowait` is not in the final set.
- **Timer scope:** the only change on the baseline path is that two vector copies per level, `mate` and `u`, are now inside the `idx` timer. The whole `idx` timer is 0.011 s of the baseline capstop, so this is negligible.
- **Untimed work did not grow.** The phase wall clock minus the sum of timed calls is:
  - capstop at 1 thread: 0.149 s in the baseline, 0.116 s in the final;
  - R: 0.845 s, then 0.808 s.

  Nothing moved out of the timers. The gap is mostly the per-level checks and fingerprints, which run on both sides.
- **Correction: the 4-thread ratios compare 4-thread capstop against serial feral and serial C METIS.** Neither reference was run with threads. "Under a fifth at 4 threads" means under a fifth of serial feral.
- **Correction: GraphBLAS-only rewrites saved 0.10 s in the final set, not 0.23 s.** The 0.23 s includes `caprc` (0.13 s), and `hkey` replaced that path. What remains is `usel` and `zmatch`.

### The stop rule

- **What the bench does is confirmed.** On CA, final configuration, at 1 and 4 threads, capstop matches the host replay in 65,462 checks and is parent-limited in 70. There are 0 other differences.
  - The report's 65,532 counts CA's 32,766 cells twice, once per thread count.
- **Correction: the replay is not the harness's rule.** `rule_replay` (`m1_reuse_bench.rs`, line 1505) is this bench's own version: a floor of 100 for every cell, and R's cap for the depth. The harness's `reuse_levels` under `reuse_capstop` (`m1match/crates/feral-metis/src/node_nd.rs`, lines 769–777 and 832–842) works differently:
  - **Cells of 5,000 or more vertices:** they reuse only the top stage, down to max(100, n/30) coarse vertices, with cap 1.5 × W / that floor (about 45 for unit weights). Then `best_of_bisections_top` runs five fresh coarsenings down to 120.
  - **Smaller cells:** floor 120 (`coarsen_floor`) and cap 1.5 × W / 120.
- **What this changes.** On CA, the cells of depths 1–8 average 7,700 vertices or more. For them, the bench's capstop reuses far deeper than the harness's.
  - The bench's capstop coarsens fresh 0.2 % (depth 1) to 3.9 % (depth 6) of what R coarsens, and 7.6 % over depths 1–14 (`CA_final1.log`).
  - The harness's capstop coarsened 13–16 % at depths 1–6 (`m1-reuse-quality.md`).
  - The quality result was measured for the harness's rule, not for this one.
- **The cost of the harness's floor, measured.** I added `M1V_FLOOR30=1` in my copy. With it, a cell of 5,000 or more vertices stops reusing at max(100, n/30) and coarsens fresh from there, in one fresh run, with the bench's cap.
  - Setup: same binary and session, run in the order control, variant, variant, control; 3 timed runs each, no check run. Logs: `m1verify-work/logs/fu/CA_{ctl1,f30_1,f30_2,ctl2}.log`.

  | capstop, final configuration | 1 thread | 4 threads | instructions, 1 thread |
  |---|---|---|---|
  | bench's rule (control, 2 processes) | 2.31 / 2.31 s | 1.22 / 1.21 s | 37.5 G |
  | harness's floor (2 processes) | 2.44 / 2.45 s | 1.31 / 1.33 s | 39.0 G |

  - The harness's floor adds 0.14 s (+6 %) at 1 thread and 0.11 s (+9 %) at 4 threads.
  - Fresh coarsening at depths 1–6 rises to 3.4–5.2 % of R's.
  - capstop / feral becomes about 0.36 at 1 thread and 0.19 at 4 threads (derived, with feral at 6.85 s, the median of my 6 runs). The verdict holds.
  - This variant still differs from the harness in two ways, neither measured. It keeps the bench's cap: the harness's cap of about 45 would break earlier on this bench's one-stage depth-0 levels and add fresh work. And it runs one fresh coarsening, not five.

### Not verified

- PA and TX: neither their timings nor their 57/57 hashes were re-run.
- The per-step table: only the baseline and final configurations were re-measured. The step deltas depend on order and were not re-run.
- samply and burble: not repeated.
- The rule on PA: I read the logs but did not re-run them. `rule_pa_final.log` ran with `caprc` and without `npar`, not the final set. At 1 thread `npar` does nothing, so the run is equivalent.

### Files

- **Worktree `m1verify`:** scratch, at 7df8e5887, nothing committed. Full path `/private/tmp/claude-501/-Users-dvirdukhan-Code-worktrees-FalkorDB-pr-reviews-FalkorDB/f9e19c4b-505c-4a27-80e5-7848348679f5/scratchpad/m1verify`.
  - Its two bench files are m1opt's plus two verifier switches, `M1V_BREAK=hpar` and `M1V_FLOOR30`. Both are read from the environment and off by default.
- **`m1verify-work/`:**
  - scripts: `session.sh`, `followup.sh`, `tot.py`;
  - logs: `logs/` and `logs/fu/`;
  - binaries: `bin/graph-test-v`, built from m1opt's sources unchanged, and `bin/graph-test-vb`, with the two switches.

  The target dir was deleted.
