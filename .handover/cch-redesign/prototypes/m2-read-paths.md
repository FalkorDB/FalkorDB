# M2 prototype: read paths for the CCH topology

**Verdict.** Reading `Up` through GxB row iterators **fails both gates**. Its query p99 is 1.7–2.4× today's, and its customization is 2.2–2.8× today's. Read-only loads of Rust-owned arrays **work in GraphBLAS v10.5.0 and copy nothing**. The measurements support D3 option (b): Rust-owned arrays, loaded read-only for the algebra and read as slices.

## Setup

- **Machine.** Apple M5 (10 cores, 32 GB), macOS (Darwin 25.6.0). Other sessions were running, with load average 1.4–2.3.
- **Code.** Scratch worktree at `origin/cch/08c-build-catch-up` (31ca1b802): `scratchpad/m2/wt`. The bench is `graph/src/index/falkordb/cch/m2_bench.rs`, with a copy at `scratchpad/m2/m2_bench.rs`. It is never committed.
- **Build.** `cargo test --profile bench --features index-falkordb`: opt-level 3, codegen-units 1, **no LTO**. The shipped build has LTO, so these are not shipped-binary numbers. All three tests appeared in `--list`.
- **Data.** SNAP roadNet-CA: 1,971,281 nodes and 5,533,214 directed entries. Phase 1 uses the stack's own `Topology::build`, which is feral ND plus the game, and takes 7.0 s and 7.6 s in the two runs. It gives 12,672,804 arcs and 226,766,191 triangles. Weights are integers 1..100, hashed per direction (the earlier harness's metric).
- **Which graph each gate comes from.** The 526 µs gate (#3324) was measured on this same SNAP graph: here, today's code gives a best-of-3 p99 of 524–534 µs. The 219 ms gate (#3323) comes from the stack's "California", which has 1.54M nodes and 5.84M arcs (u1 §1). That graph is not reproducible here, and the M1 report notes the same mismatch. On SNAP roadNet-CA, today's `customize()` takes **721–753 ms**. So customization is compared to today's code as a ratio.

**Variants.** All four use the same node maps, metric arrays and algorithm. Only the reads of `Up` differ.

| | Reads `up(r)` from | Arc lookup (`arc_slot`) |
|---|---|---|
| today | the stack's `customize()` and `Cch::query`, unchanged | binary search |
| (a) flat | a copy of today's code behind a row-cursor trait, over today's arrays | binary search |
| (b1) iterators, wrapper | a `GrB_Matrix` UINT64 in rank space, value = arc id. Read with matrix.rs's `Iter<Uint64Extract>`, one reused iterator per role, re-seeked per row | `GrB_Matrix_extractElement_UINT64` |
| (b2) iterators, lean | the same matrix, read with raw `seekRow`, then `getColIndex` + `get_UINT64` + `nextCol` per entry: the cheapest iterator use | same as (b1) |

Customization buffers x's row and streams y's row. It seeks a y row only when the row has a z above y, which comes to 12,681,517 seeks: 1,971,281 x rows and 10,710,236 y rows. GraphBLAS ran on 1 thread in part 1. Relax and the queries are single-threaded.

## 1. Customization and queries

**Correctness.**
- The metrics of (a), (b1) and (b2) all have **0 differing slots** out of 12,672,804 (weights and middles, bit for bit), compared with today's `customize()`.
- All 10,000 query answers have bit-identical distances and identical paths across the four variants. 9,860 of the pairs are reachable.
- **Break.** Swapping two arc ids in every 1,000th row of the matrix (1,583 rows) gives 3,166 differing slots, and 224 distance and 231 path mismatches out of 2,000 queries. The checks can fail.

**Customization.** Times are in ms, the median of 3 reps within each process. There were 2 processes, and the table shows run 1 / run 2.

| | total | validate | seed | relax | instructions |
|---|---|---|---|---|---|
| today `customize()` | 721 / 753 | – | – | – | 14.4G |
| (a) flat | 584 / 600 | 44 | 43 | 497 / 506 | 9.48G |
| (b1) wrapper iterator | 1,731 / 1,629 | 89 | 101–129 | 1,528 / 1,416 | 29.1G |
| (b2) lean iterator | 1,331 / 1,323 | 89 | 106–122 | 1,119 / 1,125 | 23.9G |

- Today's total also includes the well-formedness check and `into_metric`'s chunking, which (a) leaves out. That is why (a) is lower.
- Relax on iterators is **2.2× (b2) to 2.8–3.1× (b1)** the relax on flat arrays: +620 to +1,030 ms on this graph.
- Validate and seed double, because each `extractElement` is an FFI binary search.
- **Against the 219 ms gate.** Applying the measured ratios to 219 ms gives about 490 ms for (b2) and about 610–670 ms for (b1). That is derived, not measured on that graph. The design estimated 0.6–0.9 s, and (b) misses the gate either way. Variant (a) costs the same as today.

**Queries.** 10,000 random pairs (xorshift with a fixed seed). Each query is timed alone after 300 warm-up queries, 3 reps in each of 2 processes. "Best-of-3" takes each query's minimum over the 3 reps.

| µs, best-of-3, run 1 / run 2 | p50 | p95 | p99 | max |
|---|---|---|---|---|
| today | 378 / 385 | 483 / 491 | **524 / 534** | 589 / 611 |
| (a) flat | 377 / 383 | 482 / 490 | **524 / 531** | 606 / 623 |
| (b1) wrapper iterator | 877 / 847 | 1,152 / 1,101 | **1,274 / 1,197** | 1,524 / 1,453 |
| (b2) lean iterator | 643 / 638 | 837 / 827 | **904 / 897** | 1,053 / 1,029 |

- The p99 of single passes (no best-of): today 531–548 µs, with one rep at 842 µs under load; (a) 530–672 µs; (b1) 1,250–1,362 µs; (b2) 937–1,255 µs.
- **Against the 526 µs gate:**
  - today and (a) sit on the line (524–534 µs);
  - (b2) is **+371 µs (1.7×)**;
  - (b1) is **+671 to +748 µs (2.3–2.4×)**.
- **Controls.** (a) is within 1% of today, which shows the trait copy adds nothing. (b1) and (b2) differ by 25–35%, so the measurement can tell a cheaper iterator use from a dearer one.
- **Why.** The extra cost is per entry, not per seek: (b2) adds about 14G instructions over (a) in customization. How a query's extra time splits between iterator steps and `extractElement` lookups during unpacking is **unmeasured**.

## 2. Read-only load of Rust-owned arrays

**Supported in v10.5.0, for each array separately.**
1. Call `GxB_Vector_load(V, &X, type, n, bytes, GxB_IS_READONLY (7078) [+ arena], NULL)`. In `Source/container/GxB_Vector_load.c` the vector becomes "shallow": GraphBLAS never frees `X`, and ownership stays with the caller.
2. Put the three vectors into a `GxB_Container` (`p`, `i`, `x`, with `format = GxB_SPARSE`, `orientation = GrB_ROWMAJOR`, `nrows`, `ncols` and `nvals` set).
3. Call `GxB_load_Matrix_from_Container`.

Any operation that would write the matrix returns `GxB_OUTPUT_IS_READONLY` (`GB_RETURN_IF_OUTPUT_IS_READONLY` → `GB_is_shallow`). That includes `GB_wait` when the matrix has pending work, but a loaded matrix has none.

**Measured on roadNet-CA's `Up`** (n × n, 12,672,804 arcs, value = arc id):

| | u64 Ap/Aj/Ax (218.5 MB) | u32 Ap/Aj/Ax (109.3 MB) |
|---|---|---|
| load | 16–34 µs | 6–15 µs |
| `GxB_IS_READONLY` / `memoryUsage(A)` | 1 / 240 B (header only) | 1 / 240 B |
| index bits (offset/row/col) | 64/32/64, kept as given | 32/32/32 |
| `setElement` on A; in-place `GrB_transpose(A, …, A)` | `GxB_OUTPUT_IS_READONLY`; same | same |
| `GrB_transpose` → Down | 10.0–12.6 ms; nvals and sum of ids (80,299,974,274,806) correct | 10.0–11.4 ms; correct |
| masked `C⟨struct A⟩ = A PLUS_PAIR A` | 220–227 ms; sum = 226,766,191 = triangle count | 189–195 ms; same |
| `memoryUsage(A)` after both ops | 240 B | 240 B |
| arrays (FNV checksum) after the ops | unchanged | unchanged |
| after `GrB_Matrix_free`, then same-size allocations | `malloc_size` unchanged, checksums unchanged; Rust frees them later with no double free | same |

**Copies, from a run started on a dump of the u32 arrays**, so that the peak footprint is the arrays' own and not Phase 1's. Footprint is `phys_footprint` from `proc_pid_rusage`.
- Before: 214 MB.
- After the transpose: 327 MB. The +113 MB is the output, Down at 109.3 MB.
- After the mxm: peak and current both 482 MB. That is the 136.4 MB `C` plus about 19 MB of transient workspace.
- **A is never copied.** Burble reports a "10-thread atomic bucket transpose" and a "saxpy C<M>=A*B … (use mask) … hash". Neither one rebuilds A.

**Break.** The same u32 arrays loaded with `GrB_DEFAULT` handling (ownership moves to GraphBLAS) show `IS_READONLY` 0 and `memoryUsage(A)` 109.3 MB. After the free, `malloc_size` of Aj or Ax falls to 0: the "freed by GraphBLAS" check fires. That copy was leaked from Rust afterwards.

**Caveats.**
- Part 2 ran on GraphBLAS's default 10 threads. At D5's single thread, transpose and mxm times are **unmeasured**.
- Concurrent readers of one read-only matrix are untested (design risk 4).
- All of this is macOS only; Linux is unmeasured.

## Gates (design M2)

| Gate | (a) flat / read-only arrays | (b) GxB row iterators |
|---|---|---|
| query p99 ≤ 526 µs (SNAP roadNet-CA) | 524–534 µs best-of-3: at the bar, the same as today | **fail**: 897–904 (lean), 1,197–1,274 (wrapper) |
| customize ≤ 219 ms (stack's California) | same as today (ratio 1.0; 721–753 ms on SNAP CA) | **fail**: 1.8–2.4× today's total, 2.2–3.1× its relax |
| query p99 ≤ `:edge-c` | unmeasured | unmeasured |

**Unmeasured:**
- `:edge-c` on this graph;
- scoped µs per arc;
- the full-seed pass time and peak;
- valued against iso `Down`;
- customization on the stack's 1.54M-node graph;
- Linux, and LTO builds.

**Cleanup.** `CARGO_TARGET_DIR` and the dump were deleted. The scratch worktree and logs (`scratchpad/m2/run1.log`, `run2.log`) and the decompressed roadNet-CA are kept under `scratchpad/m2/`.
