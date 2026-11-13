# M1: what the GraphBLAS coarsening of design §3 costs on road graphs

**Verdict: it fails the ordering-time budget at the default of 1 thread.** Each ND depth costs about as much as depth 0, because the first two or three levels of every depth carry almost all the work, and every depth starts from the whole graph. For all depths on roadNet-CA, the GraphBLAS calls alone take **13.2 s at 1 thread**. Feral's *entire* ordering takes **6.97 s** on the same graph. At 4 threads they take **4.8 s**. PA and TX show the same ratio. Per-call fixed overhead is not the problem: deep levels cost < 0.5 ms. The cost is per entry at the large levels.

## Setup

- **Code.** A scratch ignored test, `graph/src/graph/graphblas/m1_coarsen_bench.rs`, in a detached worktree at origin/main 7df8e5887 (`scratchpad/m1gb`, feature `index-falkordb`). It calls the GraphBLAS FFI directly. A copy and the logs are in `scratchpad/m1gbtools/`. Nothing was committed.
- **Machine.** Apple M5, 10 cores, 32 GB. Built with `cargo test --release`. Load average was 1.7–2.7 from other sessions.
- **Timing.** Each call is timed as wall time plus `GrB_wait(MATERIALIZE)` on its output. Every number is the median of 3 runs, at 1 and at 4 GraphBLAS threads. The untimed checks add about 5 % (the "wall" column in the logs).
- **Graph.** The SNAP skeleton: symmetric, self-loops dropped, isolated nodes removed. CA has n = 1,965,206 and 5,533,214 entries, with edge and vertex weights of 1. This is **not** the stack's 1.54M-node California, so the design's 3.62 s and 3.07 s do not apply here.
- **Per level, as written in §3:**
  - **(1) Matching.**
    - Cap mask, computed only when `2·max(w) > cap`.
    - Keys `K = apply(A, user IndexUnaryOp (w≪32)|fmix32(j^salt))`.
    - Rounds until a round matches nothing. Each round is 7 calls:
      - `y⟨u,s,r⟩ = K MAX_FIRST_UINT64 u`;
      - unkey (user op);
      - `pf = eWiseAdd(SECOND, idx, p)`;
      - `q = GxB_Vector_extract_Vector(pf, pf)`;
      - `m⟨p,s⟩ = (q == idx)`;
      - `mate⟨m⟩ = p`;
      - `u = p⟨¬m⟩`.
    - The GraphBLAS 2-hop pass (anchor mxv, `GxB_Matrix_build_Vector`, `GxB_Matrix_sort`, even/odd selects, two range extracts, two eWiseMults) when a level shrinks by less than 15 %.
    - Then `C`, by leader = min(i, mate) plus a host prefix numbering.
  - **(2) Contraction.** `select(OFFDIAG, C·A·Cᵀ)` with PLUS_TIMES_INT32, and `C·w`.
  - **(3) Projection.** `mxv(Cᵀ, labels)` from a random 2-way label on the coarsest level. One extra `mxv(A, labels)` stands in for the boundary step (`bnd`).
  - **(4) Split.** `A⟨A,s⟩ = M ANY_PAIR_INT32 Mᵀ`.
- **Stop rule.** ≤ 100·2^d vertices or < 5 % shrink, with `cap = 1.5·(n/2^d)/100`.
- **Depths d ≥ 1.** The cells are 2^d contiguous chunks of feral's permutation, built with the split call. Separators are *not* removed. This is a proxy for real cells; a real per-cell run is unmeasured.
- **Feral.** `feral_metis::metis_order` on the same CSR, in the same process, serial, median of 3.

## Correctness checks and deliberate breaks

Checks:
- `mate` is an involution;
- round pairs are adjacent;
- every pair is within the cap;
- `A_c` has no diagonal;
- Σw is conserved;
- C has exactly n entries;
- split nnz equals a host count;
- per-level (n, nnz, Σ) is identical across 3 runs × {1, 4} threads.

Each check fires when broken on purpose (1 run each):

| Break | Caught by |
|---|---|
| skip the mutual test | "mate is not an involution" |
| skip OFFDIAG | "A_c has a diagonal" |
| shift the split labels by one | "split pattern" |
| per-call salt | "hierarchy differs across runs/threads" |

The control run passes. The timers also respond to kernel changes. `m_extract` drops from 105 to 48 ms when PreJIT is allowed, and `c_build` drops from 162 to 9 ms with the variant below.

## Depth 0 on CA, summed over its 20 levels (ms, median of 3)

The "test" column is the configuration the task asked for: JIT off (the test default), building `C`. The "best" columns build `Cᵀ` from row-sorted tuples and run under `GxB_JIT_RUN`, as the shipped module does.

| Call | 1 thr (test) | 1 thr (best) | 4 thr (best) | Kernel |
|---|---|---|---|---|
| keys (user op) | 49.6 | 49.9 | 15.4 | **generic** apply, `jit: undefined` |
| matching mxv MAX_FIRST | 114.8 | 115.0 | 37.6 | factory dot2 |
| unkey (user op) | 38.6 | 38.5 | 13.1 | **generic** |
| pf eWiseAdd / q==idx | 10.0 / 32.1 | 10.1 / 32.9 | 6.2 / 11.6 | factory |
| extract_Vector | 104.8 | 48.4 | 18.1 | **generic** subref; PreJIT hit under RUN |
| `mate⟨m⟩ = p` | 51.0 | 50.9 | 17.4 | **generic** assign |
| `u = p⟨¬m⟩` | 69.6 | 71.7 | 29.5 | **generic** masker |
| 2-hop (11 levels) | 14.8 | 15.0 | 14.3 | user selects and unkey **generic** |
| leader + numbering + extract | 35.6 | 36.8 | 16.8 | extract **generic** |
| build C (C: 162.3, Cᵀ: 9.0) | 162.3 | 9.0 | 3.3 | iso build, single-threaded |
| C·A | 153.6 | 163.5 | 66.5 | factory saxpy, plus unjumble |
| ·Cᵀ | 155.4 | 143.4 | 55.0 | factory saxpy (bucket transpose under T1) |
| OFFDIAG | 59.6 | 60.3 | 17.0 | factory |
| C·w / project / bnd | 5.1 / 13.2 / 25.1 | 24.4 / 2.5 / 25.2 | 10.1 / 1.0 / 7.3 | factory |
| **total** | **1,102** | **902** | **343** | |

**Level 0 alone** (n = 1.97M, 9 rounds) is 406 / 325 / 109 ms. Levels shrink by about 0.58 each at first. The 2-hop pass fires at levels 9–11 and 13–19, and rounds per level run 5–9. Coarsening stopped at 2,806 vertices with 1,098 entries, on the < 5 % rule: at least 1,708 coarse vertices have no edge, which points to many small components (derived). A per-cell stop would keep coarsening the large component for a few more levels. Each of those levels costs < 0.4 ms (derived from levels 17–19).

**Generic share.** Under JIT_RUN, the calls that still run a generic kernel take 241 ms of the 902 ms (27 %). The burble log shows the user operators are `jit: undefined`, because `GrB_IndexUnaryOp_new` carries no name or C definition. So **neither JIT nor a PreJIT harvest can ever specialize them as written.** PR 1 must create them with `GxB_IndexUnaryOp_new(…, name, defn)`. Built-in types also hit generic paths: the valued-mask assign, the complemented-mask masker and the bitmap subref. A harvest covers only the variants it sees.

## All depths, and the comparison with feral

Times are seconds: the timed calls plus the splits, median of 3.

| Graph | n | entries | feral (whole ordering) | GB 1 thr (test) | GB 1 thr (best) | GB 4 thr (best) |
|---|---|---|---|---|---|---|
| CA | 1,965,206 | 5,533,214 | 6.90 / 6.97 (1 run) | 15.97 | **13.17** | **4.79** (test: 5.70) |
| PA | 1,088,092 | 3,083,796 | 3.57 | unmeasured | 6.87 | 2.65 |
| TX | 1,379,917 | 3,843,320 | 4.51 | unmeasured | 8.69 | 3.28 |

**Per depth on CA (best).**
- At 1 thread, depths 0–10 each cost 0.88–0.91 s, depth 12 costs 0.82 s, depth 13 costs 0.62 s and depth 14 costs 0.30 s.
- At 4 threads, depths 0–8 cost 0.34 s each.
- Each split costs 43–60 ms at 1 thread and 12–17 ms at 4 threads.
- A depth costs the same as depth 0 because the levels it drops are the cheap ones at the bottom.

**What this means for §3 and the M1 gate (derived).**
- The GraphBLAS part alone is **1.9× feral's whole ordering at 1 thread** on all three graphs. It still has no FM, initial partition, MMD, peeling or supergraph.
- If every generic kernel cost nothing, the 27 % share would leave ≈ 9.6 s on CA, still above 6.97 s.
- The plain-Rust mutual-max coarsening in `m1-matching-quality.md` takes the largest CA component to ≤ 100 in 0.35 s, serial. The GraphBLAS depth 0 takes 0.90 s, about 2.6× slower. That comparison is across harnesses: different stop rules and allocation.
- At 4 threads, the calls fit under feral (4.8 s against 7.0 s) but leave ≈ 2.2 s for all the host work. D5 (1 thread) and this formulation are therefore in conflict.
- §3's fallback, "cells below a floor skip coarsening", targets per-depth fixed cost. That is the wrong lever here.
- Levers that would act on per-entry cost, all unmeasured:
  - reuse depth d's hierarchy for depth d+1 by splitting each level's matrix with the projected labels instead of re-coarsening;
  - fewer matching rounds (each costs ≈ 30 ms at level 0, 1 thread);
  - folding `C·A·Cᵀ` into one masked call;
  - building `Cᵀ` (already measured: −153 ms per depth).

## Not measured

- Peak memory of the GraphBLAS part. The process peak was 1.26 GB on CA, but that includes the text file, the host edge lists and feral.
- Real per-cell runs with separators removed.
- Hanging-tree peeling before coarsening.
- 16 threads.
- Ordering quality; see `m1-matching-quality.md`.
- Linux, and a PreJIT harvest of the user operators.

`scratchpad/m1gb/target-m1` was deleted. The worktree, the logs and the decompressed graphs (`scratchpad/m1gbdata`) remain under the session's scratchpad.
