# CCH ordering, supergraph and incremental upkeep on GraphBLAS: the synthesis

Base: design A (both judges' winner), with the judges' fatal-flaw fixes and grafts applied. Tags: [B] research brief `../research/00-research-brief.md`; [U1]–[U4], [R1]–[R3] the source reports; [FOLD] `../../cch-background-fold/discussion.md`. Stack code was read at `origin/cch/08c-build-catch-up` ("@08c", the stack tip, which contains #3773 `partial.rs`, #3774 `WeightArr`, #3363 `maint.rs` and #3772's `absorb`); PR facts come from `gh pr view` on #3363, #3772, #3773 and #3774. "Derived" means reasoning in this document with no outside source. "Unmeasured" means nobody has measured it. Nothing here was measured for this design.

## 1. Summary

1. **Ordering.** Our own multilevel nested dissection (ND). Cells of 5,000 vertices or more are processed *one GraphBLAS call per depth for all of them* (matching by masked `mxv`, contraction by `C·A·Cᵀ`, label projection, one `select` to split). Initial bisection, boundary-seeded node FM and leaf minimum degree are host loops over private, Container-unloaded level matrices. METIS's settings, not feral's.
2. **Supergraph.** Liu's elimination tree and a postorder relabel, Gilbert–Ng–Peyton column counts (exact arc count *before* anything is built, used as the size guard), then today's elimination game (G1) reading `triu(S(π,π),1)` from GraphBLAS and writing two rank-space `GrB_Matrix<u32>`: `Up`, and `Down = Upᵀ`, both carrying **arc ids as values**.
3. **Arc identity.** An arc's id is the stored value; ids are dense row-major on a fresh build (so #3774's metric layout is unchanged) and append-only within a lineage.
4. **Batched build.** Phases ReadSkeleton → Order → Supergraph+Seed → Relax → Seal → (Repair). Ordering batches may read different snapshots; topology covers one snapshot plus a log of written *pairs*; the metric is a push sweep seeded from one snapshot and never patched mid-sweep.
5. **Incremental upkeep.** The order stays fixed. Deletes reseed to +∞ (S1); unjoined inserts get a rank-ordered closure repair (S2, Davis–Hager); new nodes take reserved bottom ranks; decay starts a new lineage (S6) or generation (S7). Subtree re-dissection (S5) on elimination subtrees is a later PR.
6. **Recompute (goal 3).** Every change becomes: topology first, then one rank-ordered scoped recustomization (#3773's engine, generalized to ids and new arcs born at +∞), bit-identical to a full customization of the new topology. Bounded at 10,000 recomputed arcs on any path that holds the host lock; past that, a builder Repair batch. Readers swap one `Arc<Cch>` = (topology, metric).
7. **Structural decisions never read a budget**; budgets only decide *where* work runs, so writer, builder and replica reach the same hierarchy from the same start.

**What it gives up:** METIS-class order quality only (flow refinement is a later, gated PR; IFC orders query 2.2× faster on Cal/Nev [R1 §2]); no local re-dissection in the first ship; new versions answer with the Dijkstra fallback while a hand-off or rebuild is pending; replicas promise equal distances, not equal paths; topology stays at 16 B/arc (C parity) rather than B's 8 B/arc.

### 1b. What changed from design A, and the judges' disagreements resolved

| Point | Decision | Why |
|---|---|---|
| Bare edge-id logs (fatal in B, latent in A and C) | The build log holds **node pairs**. Ids are turned into pairs per hop, where both versions are at hand (§5) | A deleted id reused before absorb would otherwise leave the old pair's arc at its stale weight: a wrong shortest path |
| Batched customization: A's pull-form cursor vs C's push sweep + Seal | **Push sweep from one seed snapshot; every later write is applied at the Seal** (C) | Pull costs ~1.3 µs/arc against 0.09 µs/arc for push (`maint.rs:82-94` @08c) [U1 §5]; push needs no cursor invariant |
| Writes during ordering: A joins them into U0, C defers all to the Seal | **A's**: log pairs join U0 at Supergraph; only writes after seeding go to the Seal | Cheaper: no S2 fill repair for edges known before the supergraph exists |
| Master writer budget (A left open; B "budget only decides where"; C "no full customization under the lock") | **One budget, R_gil = 10,000 arcs, on every path holding the host lock, master writer included**; decisions are budget-free (B) | A 0.47–0.5 s customization on Redis's main thread (#3772, #3363 "Open") stalls every client; the cost is a Dijkstra-fallback window, measured by M4/M6 |
| Fresh arcs: B "always changed" vs A/C "+∞ preset" | **+∞ preset** with the bitwise change test | Equivalent results (Lemma A, §6b) and a smaller cone |
| Re-dissection unit: B's ND cells with intruder rows vs C's elimination subtrees | **Elimination subtrees, addressed by their rank set**, not an assumed contiguous range; input includes T's outward edges | Closed under arcs [B C4]; Davis–Hager parent changes break postorder contiguity (judge 2) |
| Arc ids: B's `arc_base + pstart` vs stored values | **Stored values** | B's scheme reads `GxB_Iterator` internals, not a stable API (judge 1, fatal) |
| Holes before compaction: A 12.5 %, C 25 % | **12.5 %** | Memory parity with C |
| Size guard: separator count (A 4,096; B max(2,000, 1 %n); C 2,048) | **Predicted arc count from column counts** is the primary guard; top separator > 4,096 is an early cheap check | Arcs are what costs memory; counts are exact and come before the game [R3 §1] |
| Naming: A's FixUp vs C's Stale | **`Building{Repair}`**, with the explicit invariant "a version whose entry is Building never serves a hierarchy" | Same mechanism; the invariant is what matters |

## 2. Data structures

`n_r = node_cap + reserve`, `reserve = max(1,024, node_cap/64)` (derived). Rank-space matrices are `n_r × n_r`, sparse, by row, 32-bit indices, waited before publish.

| Structure | Type | Versioned how | Bytes |
|---|---|---|---|
| `Up` | `VersionedMatrix<u32>`, value = arc id | base `m` shared by `Arc` across the lineage; additions in `dp` only (fill only grows within a lineage, so `dm` stays empty [R3 §4, U4 §5]); folded into `m` on the builder only | 4 col + 4 id per arc; 4 per rank |
| `Down` | `VersionedMatrix<u32>` = `Upᵀ`, same ids | same | 8 per arc; 4 per rank |
| `rank_of`, `node_of` | `ChunkedArr<u32>` (#3774's `WeightArr` generalized, plus append) | chunk copy-on-write (`customize.rs:182-189` @08c uses `Arc::make_mut`, so a unique chunk is written in place) | 8 per node |
| Metric (4 arrays) | indexed by arc id, length `arc_capacity`; **format out of scope** | #3774 chunks + append | 32 per arc today |
| `LineageStats` | arcs at build, deletions, dead ids, nodes past reserve, generation | copied | O(1) |
| Build state | `Arc<BuildState>` in the `Building` entry (§5); the metric under construction is builder-owned scratch | `Arc` per part | transient |

`Rank` becomes `NonZeroU32` (4 B; the `Option` niche is kept; mechanical change everywhere, [B §1]).

**Arc identity (the I3 slot problem).** Option (a) of [U1 §4]: the id is the value of `Up(lo,hi)` and of `Down(hi,lo)`.
- A fresh build assigns ids in row-major order, so `id = up_range(r).start + i` as today and #3774's chunking and locality are unchanged.
- S2 appends ids `arc_capacity, arc_capacity+1, …`; the metric arrays append chunks (the last chunk and the chunk table are copied on write), and new entries are preset to (+∞, +∞, none, none).
- Ids are never reused within a lineage (a lineage = one topology build and the incremental versions descended from it). An older reader's arrays are shorter and its `Up` does not contain the new ids, so it never reads them.
- S6/S7 start a new lineage with dense ids; S5 (later) appends ids for T's rebuilt rows and leaves holes.
- `Down` carrying ids removes the two `arc_slot` binary searches per apex (`partial.rs:202-203` @08c) and the one per propagated arc (`:247-249`). An iso `Down` (12 B/arc topology) is the fallback if M2 says memory decides.

**Contract changes, and every reader they touch.** I3 becomes I3′: "`up_ids(r)` yields `(hi, id)` ascending; arrays are sized by `arc_capacity`; dead ids hold (+∞, none) and nothing reads them". Readers: `relax`'s `x_slots_start + i` (`customize.rs:497-513` @08c), `query.rs:113` (`up_range.zip(up)`), `partial.rs:151,170` (slot → `hi`; the heap now carries `(lo, hi, id)`), `Cch::joins` (now an `Up` lookup). I1 changes only in size (`n_r`); unused ids still take the lowest ranks (`nd.rs:61-64`). I7 is strengthened (ids equal). I2, I4, I5, I6, I8, I9 are unchanged [U1 §2]. Every changed function gets a line in `proofs/*/COVERAGE.tsv`.

**Against C.** C spends 48 B/arc (16 topology + 32 metric) and 64 B plus row headers per node [U1 §1]. This design: 8 + 8 + 32 = **48 B/arc** (44 with iso `Down`), and 8 (maps) + 8 (two 32-bit row-pointer arrays) = **16 B/node**, against Rust's 32 B/node today. It holds while GraphBLAS keeps 32-bit indices at these sizes (`csr.rs:277-278` [R3 §5]; gate M3). No per-arc counters (CHOLMOD rejected multiplicities as too costly [R2 §2]).

## 3. Ordering algorithm

**Input skeleton**, read in node-id row slabs (§5): per indexed type, `P = pattern(m) ∪ pattern(dp)` of the slab rows, **ignoring `dm`**; union across types; at phase end `S = select(OFFDIAG, eWiseAdd(ONEB, P, Pᵀ))` (as `input.rs:112-113` @05, C `cch.c:181-185` [U3 §3]). Ignoring `dm` gives a superset of the live pattern, which is valid for any order and for I4 (a superset supergraph only carries extra +∞ arcs [R2 §1]); it avoids both `effective`'s ~400 MB / 76 ms merge at 9M edges [FOLD] and any question of what `dm` means for multi-edge pairs (derived).

| Step | Operation | GraphBLAS or host, and why |
|---|---|---|
| 0 Degrees | `d = reduce_rows(PLUS, S)`; `d = 0` and unused ids take the lowest ranks in id order (`nd.rs:61-64`) | GraphBLAS, one call |
| 1 Hanging trees | Rounds of degree-1 peeling: masked `mxv` (PLUS_PAIR) over a frontier of the peeled vertices' neighbours; ranks in (round, id) order. After 64 rounds, the rest of the 1-shell (from `LAGraph_KCore`) is finished by a host pass. Zero fill: each peeled vertex has one higher neighbour [U4 §1, R1 §1] | GraphBLAS, host for long chains (per-call overhead × rounds) |
| 2 Components | `LAGr_ConnectedComponents` on a private copy (it unloads `G->A`, `FastSV7.c:42-46` [U4 §1]) | GraphBLAS |
| 3 Coarsen all large cells at depth d | Every cell ≥ 5,000 vertices shares one block-diagonal matrix `A_d`. **Match** (Manne–Bisseling [R1 §3]): key `K = apply(A_d, positional op: (w ≪ 32) \| scramble(j))`; `y⟨unmatched⟩ = K MAX_FIRST unmatched`; mutual pairs by `GxB_Vector_extract_Vector(partner, partner)` compared with the row index; rounds to a fixpoint; 2-hop masked `A·A` over unmatched vertices when a level shrinks < 15 % (feral's rule [U3 §2]). **Contract:** `A_c = select(OFFDIAG, C·A_d·Cᵀ)` PLUS_TIMES_INT32, vertex weights `C·w` (`Coarsen_Matching.c:483-506` [U3 §3]). A part stops at ≤ 100 vertices or < 5 % shrink (METIS `ometis.c:399-403`) and is carried by an identity map | GraphBLAS: one call per step per depth for all parts (graft from C, judge 2) |
| 4 Initial separator per part | 7 greedy-growing trials + edge FM (METIS `initpart.c:82-89`); separator = lighter of METIS's boundary + two-/one-sided FM (`separator.c:36-56`) and a König cover by host matching | Host, parallel over parts: per-call overhead dominates at ~100 vertices [U3 §3] |
| 5 Uncoarsen | Labels projected by one `mxv(Cᵀ, labels)` per level for all parts; boundary from one `mxv(A, side)`; node FM seeded **from the boundary only** (feral's full rescan, `fm_refine.rs:350-378`, is the bug to avoid [B §5.2]) over the private level's `Ap/Aj/Ax` through a Container unload (legal on a private copy [B C5]) | Host FM, parallel over parts: serial FM avoids mt-Metis's +6.1 % separator loss [R1 §1] |
| 6 Split | One `select` with an IndexBinaryOp reading the label vector drops every cross-part and separator edge at once → `A_{d+1}`; separator vertices take the top ranks of their cell's range (`ometis.c:199-204`) | GraphBLAS, one call per depth (B's idea) |
| 7 Cells < 5,000 | Whole multilevel ND on the host over unloaded arrays | Host: avoids per-call overhead × ~26k cells on California (derived) |
| 8 Leaves ≤ 120 (METIS `MMDSWITCH`, `defs.h:52`) | Exact minimum degree on 128-bit neighbour bitsets, ties → lowest id | Host |
| 9 Tree, counts, relabel | Liu's elimination tree (O(nnz·α), C `cch.c:312-375`), host postorder q, π′ = π∘q (fill unchanged, survey p.8), Gilbert–Ng–Peyton row/column counts [R3 §1] | Host over a private extract `B = S(π,π)` |

Best of 5 bisections for cells ≥ 5,000 (`ometis.c:352-387`) and vertex compression (`ometis.c:97-113`) are flags measured by M1 [U3 §4]. Flow refinement (O2: one `LAGr_MaxFlow` per depth over the disjoint union of node-split bands, plus a `LAGraph_MinCut` binding [R1 §1, U4 §1]) is a later PR behind M1's data.

**Expected quality.** Same family as METIS and feral: between C METIS (5,527,199 arcs, 48.6M triangles) and feral (5,838,003 arcs, 53.9M) on California (FALKORDB.md:29-30 [U3 §4]). Evidence: METIS's settings that feral lacks (MMD at 120, coarsening to 40–100, boundary separator + FM both ways) are what separates the two, and best-of-5 + the `mindiff` tie-break alone moved feral from 59.6M to 53.9M triangles [U3 §4]. Batched matching against SHEM is unmeasured [U3 §3]; M1 gates it.

**Determinism.** Seeded integer keys; only MIN/MAX/integer PLUS; no ANY, no BFS parents [U4 §1]. The order is a function of the skeleton snapshot; independence from thread count is gate M1.

**Guards for large separators.** (a) After the top bisection, a separator over 4,096 vertices ends the build `Failed("separator too large")`: by CCH Lemma 2 that clique sits in ≥ n/3 search spaces [R1 §2]; California's treewidth bound is 252, social/web graphs reach 0.5–27 % of n [R1 §2]. (b) After step 9, column counts give the exact arc count; above `max_arcs` (default 8 per skeleton entry, configurable; California is 1.76 — derived from 5.84M/3.31M [U1 §1]) the build fails *before* the game or the metric is allocated. This catches the 1500×1500 grid (~11 arcs/entry extrapolated, §8).

## 4. Supergraph construction

1. `B′ = GxB_Matrix_extract_Vector(S, node_of′, node_of′)` (private), `U0 = select(TRIU, B′, 1)`.
2. Unload `U0` into a Container (private, zero-copy) and run today's child-absorption game (`order.rs:381-437`) in increasing rank: row(r) = U0(r,:) ∪ ⋃children(row(c) ∖ {r}), sort, dedup. Column counts preallocate `Aj` exactly.
3. Ids in row-major order; `(Ap, Aj, Ax = id)` loaded with `GxB_load`/`GrB_Matrix_build` as `Up.m` (copy vs zero-copy settled in PR 3); `Down.m = GrB_transpose(Up.m)`; `GrB_wait` both.

Cost: O(|U0| + arcs) plus row sorts; the game is 97 ms on California over the CSR today [U1 §5]; expect ≤ 200 ms with the extract, transpose and load (unmeasured, M2). It meets I2, I4, I5/I6 (child-absorption fixpoint [R3 §1]), I7 and I3′. Because the order is now a postorder, every elimination subtree is a contiguous rank range *at build time*, which the later S5 and G2 use. G2 (tagged frontier with `GxB_MAX_SECOND_UINT32`, fixing the ANY collision [B C1]) is a later PR behind a gate (P2: ≤ 97 ms and equal to G1 on the test kit); G1 stays the fallback.

## 5. Batching on the native index builder

**Principle (derived).** Any order is valid; only the supergraph must cover the published graph (I4), and only the metric must match it. So ordering batches may read slabs from different snapshots, the topology covers **one** snapshot V_sg plus logged pairs, and the metric covers **one** seed snapshot V_c plus logged pairs.

**The log, and the id-reuse fix.** Writers never apply into build state; `place(id)` always answers "not covered", and a `Building` entry records written ids as today (`maint.rs:297-314` @08c). At each batch's publish, the framework hands the catch-up the ids written between the batch's fork F and the published version P (`since_last_indexer_step`, `versioned.rs:436-443` [U2 §2]). The CCH catch-up turns each id into pairs **using F and P, both at hand**: its pair in F (deleted or reused since) and its pair in P (created or reweighted since), exactly as #3772's `absorb` does (`maint.rs:598-631` @08c). Pairs, not ids, are appended to `BuildState.log` (an `FxHashSet<(u64,u64)>`, O(written) under the lock). A pair's state is a function of the current graph only, so the log is later resolved against the current version alone, no matter how many hops it spans; an id deleted, reused and absorbed later can no longer hide its old pair (the judges' fatal flaw). Past 1 M pairs the log keeps only "many", and the Seal finds structural changes by one GraphBLAS pass instead: `triu(S_now(π,π),1)⟨¬struct(Up)⟩` non-empty = new adjacencies, followed by a full push customization (derived).

| Phase | Batch unit | Partial state | Snapshot rule |
|---|---|---|---|
| ReadSkeleton | node-id slab ≈ 2M edges | `Vec<Arc<Matrix>>` of immutable slabs; at the end `concat` + ONEB/OFFDIAG | any |
| Order | one depth (or one coarsening level of depth 0 on huge graphs); host cells in groups ≈ 4M edges | `Arc` level stack, label/rank vectors, cell queue; a batch replaces only what it consumed | none (private skeleton) |
| Supergraph + Seed | §4 over `S ∪ log pairs` resolved at V_sg; then seeds of every arc from the same fork (V_c = V_sg) into flat arrays | `Arc<Topology>`; flat seed arrays (builder scratch) | V_sg = V_c; log cleared |
| Relax | push sweep over apex ranks [a, b), sized by Σ C(\|up(x)\|, 2) from the column counts (≈ 50M triangles ≈ 225 ms, derived from 219 ms for 48.6M [#3323, U3 §4]) | flat metric buffers (builder-owned; only the builder reads them) | none: seeds are frozen at V_c |
| Seal | absorb `log` (pairs since V_c) against its fork V_s: §6 topology then §6b scoped recompute, unbounded, off the lock; chunk the metric | the finished `Cch` | V_s |
| Repair | same as Seal, from a held `base: Arc<Cch>` at V_m | `base`, `log` since V_m | fork |

California needs one Supergraph+Seed batch and about one Relax batch; the split matters above ~100M triangles. The Relax buffers stay builder-owned and unversioned: a pre-Seal batch can be Discarded only when its generation is dead, because its catch-up is an O(written) log append that never refuses (derived; framework detail checked in PR 8).

**Publish outcomes.** Same/Swap: the new `Arc<BuildState>`. CatchUp (all phases): the hop's ids become log pairs. **Adoption** (Seal or Repair): the hop's pairs are absorbed under the lock by the §6/§6b writer path within R_gil → `Built`. Past R_gil: publish `Building{Repair, base = the batch's hierarchy at V_s, log = the hop's pairs}` (B's resumable absorb), so the Seal's work is kept and the next Repair absorbs only the writes made *during* the previous batch. Batch k's work is proportional to the writes during batch k−1, so the backlog shrinks geometrically whenever absorb throughput beats write throughput (derived; at 0.22 ms per SET [#3363], up to ~4,500 SET/s). The `MAX_REDOS = 5` backstop (`cch/mod.rs:396` @08c) now counts only rounds whose backlog did not shrink.

**Readers while Building.** New versions use the Dijkstra fallback (222 ms median on California, `fallback.rs:28-31` [U2]); older versions keep their `Arc<Cch>`.

## 6. Incremental maintenance (built index, fixed order)

One flush per `Commit` segment on the private version (`maint.rs:467-536` @08c). The order of work is always **record pairs → classify → topology → resolve front → recompute → wait → publish**. Every structural decision below is a pure function of (held hierarchy, commit); budgets only abort an inline attempt and move the same function to a Repair batch.

| Change | Rule | Correctness | Cost | Where |
|---|---|---|---|---|
| Weight SET/REMOVE; insert on a joined pair; edge delete | Metric only (S1: a pair's last edge deleted seeds +∞) | G′ ⊆ H unchanged [R2 S1, R3 §4] | 0.22 ms p50 SET, 0.35 ms delete (#3363) | writer |
| Node delete | its edges cascade as deletes (`edges_deleted`, `maint.rs:238-258`); rank and arcs stay; `deletions += deg` | as S1; all its arcs reach +∞ through the cone (§6b) | O(deg) + cones | writer |
| Edge insert (u, v), unjoined, u < v | **S2 closure repair** (Davis–Hager): add v to up(u); pop rows by rank; row j with additions D: if p(j) is unchanged push D ∖ {p} into up(p); if p(j) changed (new entry below the old parent), push up(j) ∖ {p_new} into up(p_new). Stop when nothing is new. One pass serves every insert of the commit (DH01 Thm 4.2 [R2 §2]). New arcs get appended ids, metric preset +∞ | Only adds arcs and stops when U(j) ∖ {p(j)} ⊆ U(p(j)) everywhere: the validity rule [U4 §5, R2 §5], so I2, I4–I6 hold; nothing needs contiguity | ancestor case: one path segment, median 6–22.5 new arcs on grids; across the top separator ~0.9·n, 4–7 % of arcs [R3 §3]; road graphs unmeasured (M4) | writer if ≤ N_max = 4,096 new arcs, else Repair |
| Node created within `n_r` | nothing: it already holds a bottom rank; its edges are S2 inserts (degree 1: one arc, no fill) [B C9, R2 S3] | as S2 | small | writer |
| Node id past `n_r` | if an arc-less vertex holds a low rank, swap ranks with it (both arc-less: no arc, value or middle changes, B's rule); else a Repair batch resizes `Up`/`Down` and the maps by a block at the **top**; edges become S2 inserts (root-path unions: 46/94/190 arcs on grids [R3 §3]); `nodes_past_reserve += 1` | any rank is valid | O(n) row pointers once per block | swap: writer; resize: Repair |
| Reused node id | inherits its old rank and (now +∞) arcs; edges classified as above | — | — | writer |
| Relationship types | fixed by the definition; other types ignored (`spanning`, `maint.rs:317-325`); a definition change is DROP/CREATE | — | — | — |

**Hand-off.** S2 runs on the private copy and aborts at N_max; the recompute aborts at R_gil. On abort the private result is dropped and the entry becomes `Building{Repair, base: last consistent Arc<Cch>, log: this commit's pairs}` under the same lineage; the Repair batch runs the identical function unbounded. This replaces the two rebuild branches, `update → (None, Rebuild)` and `absorb → None` (`maint.rs:552, 576, 582, 621` @08c [B §2]).

**Decay** (thresholds derived, M6 calibrates; C's valve `cch_index.c:468-496` [U3 §1]):

| Signal | Action |
|---|---|
| deletions ≥ max(1,024, n/8) (C parity) or dead ids ≥ 12.5 % | S6: rebuild `Up` from the same order, new lineage, full customization (~97 + 219 ms on California [U1 §5]) |
| arcs ≥ 1.25 × arcs at build, a single commit adding > arcs/8, or reserve exhausted | S7: re-order as a new generation |
| `dp` > max(64k, √\|m\|) entries | builder fold of `Up`/`Down` (ids unchanged, metric untouched; `versioned_matrix.rs:151-171`) |

First ship: S6/S7 go through `Building` (fallback while building). Later: a shadow generation keeps the old lineage serving (framework change).

## 6b. Recomputing modified edges (goal 3)

**What an arc's value reads.** Arc a = (lo, hi), lo < hi, holds `min(seed(a), min over x ∈ down(lo) ∩ down(hi) of dn(x,lo) + up(x,hi))` (and the mirror for the other direction), the middle being the first x to win under strict `<` in increasing x (`partial.rs:176-219` @08c). So a's value or middle can change only when (i) its seed changes, (ii) its apex set changes (an arc (x,lo) or (x,hi) is new), (iii) a bottom arc's value changes bitwise, or (iv) its apexes' ranks change (middles are ranks, I9).

**Propagation rule.** When a changed arc (lo, hi) changes bitwise, enqueue (min(hi,w), max(hi,w)) for every w ∈ up(lo) ∖ {hi} (`partial.rs:241-254` @08c). An arc is a bottom arc only of triangles whose apex is its own lower end, so this closure reaches every arc that reads it, and all of them have a higher lower end, so popping in increasing lower rank finalizes each arc after its inputs (I8; `partial.rs:8-15`).

**Lemma A (new arcs born at +∞).** Let H ⊆ H′ be valid topologies, N = H′ ∖ H, and extend the exact metric of H with (+∞, +∞, none, none) on N. (1) A new arc (j, v) has no old apex: an apex x with (x,j), (x,v) ∈ H puts j, v in up(x), and I5 would already have joined them. (2) A triangle new to an *old* top arc has a new bottom arc (by (1) applied to its bottoms). (3) A new bottom arc that computes +∞ equals its preset, does not propagate, and cannot lower or win a strict-`<` tie anywhere; one that computes finite changed bitwise and propagates to its tops. Hence the front `N ∪ {arcs of changed seeds}` reaches every arc whose value or middle differs, and since every popped arc is recomputed from scratch, the result is **bit-identical to a full customization of H′** (derived; B's "fresh arcs always changed" gives the same result with a larger cone).

**Lemma B (re-dissecting an elimination subtree T, the later S5).** Re-rank T within **its own rank set** R_T (not an assumed range; Davis–Hager parent changes break contiguity). (1) No arc enters T from below outside it: (o, t) with o < t makes t an ancestor of o, so o ∈ T [B C4]. (2) Every arc with an apex in T has both ends in up(x) ∖ T ⊆ up(root_old), a clique (I5), so S5 adds no arc outside T and the boundary clique K_T = {(y,z): y,z ∈ up(x) ∖ T for some x ∈ T} keeps its pattern. (3) Every vertex of R_T stays below every y in K_T, so K_T's values are mathematically unchanged; but their middles name T ranks and the floating-point association of sums can move, so K_T is recomputed and the bitwise test decides propagation. (4) Arcs outside T ∪ K_T read only unchanged arcs unless the cone reaches them. Front = all of T's arcs (new ids, +∞ preset) ∪ K_T. S5's input is A(T, :) — T's edges *including its outward ones*, which rebuild up(t) beyond T (judge 1's fix).

**The recompute set per change type.** The front is recorded as **node pairs** (and fresh-arc flags) and resolved to ids only after every topology change of the commit or log segment, so S2 and S5 relabels need no remapping (B's graft).

| Change that reaches the index | Front (then closed by the propagation rule) | Why complete |
|---|---|---|
| Weight SET/REMOVE; insert on a joined pair; edge delete (+∞ if the last of its pair) | the pair's arc (its MIN-reduced seed in both directions) | (i) only |
| Node delete | the arcs of its deg(v) pairs; its fill arcs reach +∞ through the cone (every lower path from v starts with one of its edges) | (i) + (iii) |
| S2 insert (new fill arcs) | the inserted pairs' arcs ∪ every new arc, preset +∞ | Lemma A |
| Node past reserve (top block) | as S2 | Lemma A |
| Rank swap of two arc-less vertices | ∅ | no arc reads either rank |
| Re-dissected subtree T (S5, later) | T's arcs ∪ K_T | Lemma B |
| New lineage or generation (S6, S7) | everything: a full push customization in Relax batches | new ids |
| Build log at Seal/Repair | the union of the above over all logged pairs, after all their topology | the metric is a pure function of (topology, seeds) |

**Algorithm.** `recustomize_seeded` (`partial.rs:98-258` @08c) with four changes: (1) the front takes pairs resolved to `(lo, hi, id)` after topology; (2) the heap key is `(lo, id)`; (3) apexes come from merging `Down(lo)` and `Down(hi)` (base + `dp` iterators, reused), which yields both bottom ids with no `arc_slot` search; (4) an arc budget aborts the pass with nothing published. Seeds still come pair by pair from the graph (`PairWeights`, #3363), so no weight matrix is built. Two valves stay off the lock only: front > n/8 distinct arcs (`partial.rs:131`) and **cone > 385k popped arcs**, both switching to a full push sweep. The cone valve is the break-even of 1.3 µs/arc scoped against ~0.5 s for the full path on California (`maint.rs:82-94` @08c; derived: 0.5 s / 1.3 µs).

**Cost on California** (1.25–1.54M nodes, 5.9M arcs; numbers cited):

| Case | Cost |
|---|---|
| per recomputed arc, scoped | ~1.3 µs, seeds included (`maint.rs:82-94`); M2 must keep it ≤ 1.3 µs with GraphBLAS rows |
| single SET | 0.22 ms p50 → ~170 arcs (#3363; derived) |
| k random pairs | 0.41 s at 512, 0.51 s at 1,024, 0.95 s at 4,096; full path ~0.5 s (`maint.rs:82-94`) |
| 200×200 grid, one edge | 10.7 ms scoped vs 464 ms full (#3773) |
| ancestor insert | (6–22 new arcs + their cone) × 1.3 µs ≈ 0.25–0.5 ms (derived from [R3 §3]) |
| cross-top-separator insert | 4–7 % of arcs ≈ 290k arcs ≈ 0.4 s if roads behave like grids → Repair batch (derived) |
| full push sweep | 219 ms (#3323), 0.09 µs/arc |
| today's full customization under the lock | 0.47 s at a build commit (#3772), ~0.5 s in `GRAPH.EFFECT` (#3363 "Open") — both removed |

**Interleaving with a batched build.**

| When the edge changed | Topology | Metric |
|---|---|---|
| before its ReadSkeleton slab | in S | — (no metric yet) |
| after its slab, before V_sg | log pair joins U0 at Supergraph | seeded at V_c = V_sg |
| after V_sg/V_c, during Relax | frozen; applied at the Seal by S2 (new arcs +∞) | **never patched mid-sweep** (a raised seed cannot be unwound from a partially relaxed arc); the Seal's scoped pass recomputes from scratch, front = logged pairs' arcs ∪ new arcs |
| during the Seal or a Repair batch | hop pairs, absorbed at adoption under the lock if ≤ R_gil | same |
| after adoption | ordinary writer path (§6) | same |

So "modified before the batch that covered it" means "read by the phase's one snapshot", and "modified after" means "in the log, applied at the Seal/Repair after topology". A subtree rebuilt while its ancestors' metric is final is Lemma B's K_T front, run on a private version and published with the topology. The order is always **topology (S2/S5, new ids, `dp`, metric arrays extended) → recompute → `GrB_wait` → publish**; a recompute never runs on a topology missing an arc (`partial.rs` would answer `NeedsRebuild`, and the relax panics on a broken closure, `customize.rs:519`).

Worked example (derived): a 4.6 s California build under 100 SET/s logs ~460 pairs; the Seal absorbs them off the lock in ~0.4 s (512-pair figure); the ~40 SETs during the Seal (~7k arcs, ~9 ms) fit R_gil at adoption. At 1,000 SET/s: Seal 0.11 s → Repair 24 ms → adoption, three rounds.

**Where it runs, and the bound.** Under the host lock (master writer flush, replica `GRAPH.EFFECT`, adoption catch-up, writer rebase): S2 ≤ N_max = 4,096 new arcs and recompute ≤ R_gil = 10,000 arcs (~13 ms at 1.3 µs/arc, the same as a column's 10,000-change catch-up cap, `online_build.rs:126-131` [U2]). Past either: `Building{Repair}`, same function unbounded off the lock. A front larger than the valves runs a full push sweep, only off the lock. S5, S6, S7 run only in builder batches.

**MVCC.** All of it writes a private version: `WeightArr::set` copies only its chunk (`customize.rs:182-189`), `Up.dp`/`Down.dp` are duplicated per changing version (O(|dp|), bounded by the fold rule). `Cch` holds `Arc<TopologyVersion>` and the metric together and is swapped as one `Arc` at publish, so a reader holds a consistent (topology, metric) pair. An aborted inline attempt publishes a version whose entry is `Building` — it serves no hierarchy, so no reader ever sees a half-recomputed metric or a topology missing an arc; older versions keep theirs.

## 7. MVCC and replication

- **Each version holds** `IndexEntry<CchIndex>` → `Built(Arc<Cch>{Arc<TopologyVersion>{Up, Down, maps, stats}, Metric})` or `Building(Arc<BuildState>)`. `Up.m`/`Down.m` are shared across a lineage; `dp` and metric chunks until written.
- **Publish rules kept** [U2 §3]: private-only writes (`cow.rs:48-59`); every GraphBLAS object waited (`mvcc_graph.rs:332-347`); replaced versions dropped outside the lock (`online_build.rs:348-360`); the data version kept (`mvcc_graph.rs:247-248`). Folding `dp` happens only on a builder version; it changes no id, so the metric is untouched.
- **Determinism: same distances, not same paths.** Within a lineage, writer, Repair batch and replica that apply the same commits produce bit-identical hierarchies, because decisions are budget-free and the recompute is bit-identical to a full customization. Across nodes, lineages differ: a build's order depends on when slabs were read, and an RDB load rebuilds (`cch/mod.rs:375-377` @tip; C `cch_index.c:161-163` [U3 §1]). Distances are exact on every node; equal-cost paths may differ (tie rules follow rank, I9). Caveat (derived): with non-integer weights, different hierarchies sum in different orders and can differ in the last bit; integer-valued weights are exact. C is already history-dependent through stale arcs [B §5.7]. Same paths would need the rank vector and topology events on the wire, a new effects format, for a property C lacks.
- **RDB.** Definition only, as today; the load rebuilds in the background. Persisting the rank vector (4 B/node, ~6 MB on California) to skip the 3.62 s ordering [U1 §5] is open question 4.

## 8. Memory and time budget

California: node capacity 1.54M [U1 §1] (the task's 1.9M scales per node); 5.84M arcs (feral) / 5.53M (METIS).

| Item | Value | Basis |
|---|---|---|
| Topology, steady | 16 B × 5.84M + 16 B × 1.54M ≈ **118 MB** (iso `Down` 95 MB), against 142.7 MB today | derived; today measured (#3321) |
| Metric | 187 MB, +10 % chunk headers (396–412 vs 365 MB held, #3774); ≤ +12.5 % holes before S6 | computed [U1]; measured (#3774) |
| Total vs C | ~305 MB vs C's ~364 MB (48 B/arc + 64 B/node) | derived |
| Build peak over input | skeleton ~20 MB + levels ≤ 2× 2-core (~60 MB) + topology 118 MB + seeds/metric flat then chunked 2 × 187 MB ≈ **0.6 GB**; feral alone peaks 447 MiB, METIS 363 MiB (#3321) | derived, unmeasured (M1) |
| Ordering time | target ≤ feral 3.62 s; C METIS 3.07 s | measured baselines (#3321) |
| Supergraph / customize | ≤ 200 ms / 219 ms | 97 ms today (#3321); customize (#3323) |
| Per update | SET 0.22 ms; delete 0.35 ms; ancestor insert ~0.25–0.5 ms; cross insert → Repair ~0.4 s | measured (#3363); derived |
| Lock hold per event | ≤ ~13 ms (R_gil), against 0.47 s today | derived bound; today cited (#3772) |
| Per writing version | changed chunks (median 0.23 MB + 181 KB table, #3774) + `dp` copy | measured / derived |

**9M edges.** Road-like (4.5M pairs at California's ~3.5 arcs per pair): ~15.8M arcs, ~4.2M nodes, topology ~320 MB, metric ~506 MB (derived); slab reads avoid the 400 MB / 76 ms `effective` transient [FOLD]. The 1500×1500 grid (8,994,000 edges [FOLD]): R3's grid arcs per node (9.6 → 14.7 → 20.8 per ×4 in n [R3 §3]) extrapolate to ~45 arcs/node, ~100M arcs, ~4.8 GB — refused by guard (b) before allocation (unmeasured extrapolation).

## 9. Implementation plan

**Measurements first** (each with a deliberate break to show it can fail):

| Id | Measure | Pass |
|---|---|---|
| M1 | GraphBLAS ND prototype on California and a 9M-edge road graph: arcs, triangles, tree height, time, peak, order at 1/4/16 threads | arcs ≤ 5.84M, triangles ≤ 53.9M; time ≤ 4.5 s; peak ≤ 447 MiB over input; identical order across thread counts |
| M2 | `Up`/`Down` over `VersionedMatrix<u32>` with feral's order: query p99 vs `:edge-c`, full customization, scoped µs/arc, valued vs iso `Down` | p99 ≤ 526 µs on roadNet-CA (#3324) and ≤ C; customize ≤ 1.2 × 219 ms; scoped ≤ 1.3 µs/arc |
| M3 | integer width of `Up`/`Down` at California size | 32-bit |
| M4 | 10k sampled inserts on California (local ≤ 3 hops, random): new arcs, cone size | p95 local ≤ N_max and ≤ R_gil |
| M5 | Randomized property test: S2 + generalized scoped pass (+ Lemma B once S5 lands) vs full customization (bitwise) and Dijkstra, including id reuse across hops | 0 mismatches over 10k graphs |
| M6 | write-mix replay: arcs, height, query p99 over time | sets §6 thresholds and R_gil |
| M7 | batched build under concurrent writes: lock hold per publish, column-build delay | p99 ≤ 13 ms; delay ≤ largest batch |
| M8 | Seal/Repair convergence at 100 / 1,000 / 5,000 SET/s | adopt within 4 rounds at ≤ 1,000 SET/s; lock ≤ 13 ms |

**PRs** (each small, closing its own issue, listing new functions in `proofs/*/COVERAGE.tsv`):
1. GraphBLAS wrappers (masked `mxv`/`mxm` over semirings, `extract_Vector`, TRIU/IndexBinaryOp selects, reduce-to-vector, `Matrix<u32>`, private Container unload) + Linux PreJIT harvest. Proves kernels at PreJIT speed (generic kernels cost +6–87 % [U4 §4]).
2. `ChunkedArr<T>` with append; metric append preset +∞.
3. Topology over GraphBLAS with I3′, still fed by feral; readers changed. Proves bit-identical metric vs today; gates M2, M3.
4. Pair log with per-hop resolution, replacing the id log; generalized recompute engine (pairs → ids after topology, `(lo, id)` heap, `Down` ids, +∞ preset, budget abort, cone valve). Proves M5 (weights, id reuse).
5. S2 closure repair, `Building{Repair}` hand-off, R_gil on every lock path, node reserve and swap. Replaces both rebuild branches. Proves M4, M5.
6. GraphBLAS ND (level-synchronous large cells, host small cells), Liu + postorder + column counts, guards, behind `CCH_ORDERING` (default feral). Proves M1.
7. Flip the default; delete feral and the `CchInput` CSR. Closes goal 1.
8. Phased batched build (ReadSkeleton → Order → Supergraph+Seed → Relax → Seal → Repair). Proves M7, M8.
9. Decay signals, S6, S7, `dp` fold.
10. Later, each behind data: S5 on elimination subtrees (Lemma B), shadow generation, G2, O2 flow refinement.

## 10. Risks and open questions

1. **Fallback windows.** With R_gil on the master writer, a 512-SET commit (0.41 s inline today) gives new versions Dijkstra (222 ms p50) until Repair publishes. M4/M6 decide whether the master's writer may take a larger budget (a config) than replica `GRAPH.EFFECT`.
2. **Query latency over GraphBLAS rows.** Iterator attach costs 1,654 instructions, re-seek 298 (`degree_bench.rs:16-19` [U1]); merging `m` + `dp` per row may break 526 µs p99. Mitigation: eager builder folds, reused iterators (M2).
3. **Order quality** of batched matching vs SHEM is unmeasured [U3 §3]; if M1 fails, keep feral behind PR 3's API (the CSR still leaves the supergraph and customization paths).
4. **RDB order persistence**, and whether replicas should load it.
5. **Cross-separator inserts on road data** (unmeasured); if frequent, S5 moves forward.
6. **Two hierarchies alive** during S6/S7 with readers: ~2 × 0.3 GB on California.
7. **PreJIT coverage and thread-count determinism** of every new operator [U3, U4].
8. **Proof sketches** (Lemmas A and B, stale ordering snapshots, the ignored-`dm` superset, pair-log completeness) rest on M5 until it runs.
9. **Builder-owned Relax buffers** assume a pre-Seal batch is never Discarded while its generation lives; PR 8 must check this against `versioned.rs`.
10. **Floating-point last-bit differences** across replicas for non-integer weights (§7) need documenting for `algo.SPpaths` users.
