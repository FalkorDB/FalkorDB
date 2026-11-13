# Design C: CCH ordering and supergraph, quality and parallelism first

Angle: level-synchronous multilevel nested dissection (ND) with max-flow refinement (O2), a tagged-frontier supergraph (G2 with the MAX-on-tag fix), and incremental upkeep by fixed-order insertion (S2) plus subtree re-dissection (S5) driven by decay signals. It includes the team's added goal 3 (§6b): every change that reaches the index leaves every affected arc's metric recomputed and correct.

Tags: [B] the research brief `00-research-brief.md`; [U1]–[U4], [R1]–[R3] the source reports; [FOLD] `.handover/cch-background-fold/discussion.md`. PR numbers refer to PR bodies read with `gh pr view` (#3363, #3772, #3773, #3774). "Derived" means my own reasoning with no outside source. "Unmeasured" means nobody has measured it.

## 1. Summary

1. **Ordering.** We write our own ND over the read-only relationship matrices. Its skeleton is built as GraphBLAS pattern operations straight from each type's base matrix and its two delta matrices (`m`, `dp`, `dm`), so no merged copy (`effective`) and no CSR is ever built.
2. **One call per step for all parts.** Every part at a recursion depth lives in one matrix with the cross-part edges dropped. Matching, contraction, label projection, band BFS and max flow each run as one GraphBLAS/LAGraph call per depth. The per-part host loops (initial bisection, boundary FM, leaf minimum degree) run in parallel across parts.
3. **Flow refinement.** At each depth, one `LAGr_MaxFlow` call over the disjoint union of every part's node-split band network, with a super source and super sink, improves every separator at once (O2). The König/FM separator is the fallback for each part.
4. **Supergraph.** The order is relabelled into a postorder of the elimination tree. The rows U are then built by the tagged frontier, using `GxB_MAX_SECOND_UINT32` so the entry that continues wins any collision (contradiction C1). The host elimination game (G1) is the fallback.
5. **Storage.** U and its transpose are rank-space `GrB_Matrix<u32>`s whose value is a **stable, append-only arc id**. That id is the handle a metric attaches to, which replaces I3.
6. **Building.** The build runs as native-index batches: one depth of ordering per batch, then frontier rounds, then customization by rank range, then a Seal batch. All topology comes from one private skeleton snapshot. Edges written during the build are inserted at the Seal by S2.
7. **Maintenance.** Edges are inserted by S2: on the writer when a work budget allows, otherwise in a builder repair batch. A cross-cell insert re-dissects the subtree at its endpoints' lowest common ancestor (S5). A delete only reseeds its arc with +∞ (S1). Decay counters trigger compaction (S6) or a new-generation re-order (S7). While a re-order builds, the old hierarchy keeps serving.
8. **Recompute.** Every change becomes a rank-ordered scoped recustomization. Its starting set ("front") is the changed seeds, plus every new arc (born at +∞), plus the boundary clique of any re-dissected subtree. Topology always changes first, and the recompute runs on a private version.

**What it gives up:**
- the flat slot layout: after inserts, new arcs lose locality in the metric arrays;
- bit-identical paths across replicas: it guarantees the same distances only;
- a small, simple first PR: ND, flow refinement and G2 are all prototypes first;
- minimal fill after deletes: dead arcs stay until S6 cleans them up.

## 2. Data structures

### Per hierarchy version: `Arc<Cch>` = (`Arc<TopologyVersion>`, `MetricVersion`)

| Structure | Space, type | Versioning | Bytes |
|---|---|---|---|
| `node_to_rank`, `rank_to_node` | node and rank space, `u32` | chunked copy-on-write arrays: #3774's `WeightArr` generalised to `ChunkedArray<T>`, with append | 4 B/node each |
| `upward` U | rank × rank, strictly upper, row-major, value = arc id `u32`, 32-bit indices | `base: Arc<Matrix<u32>>`, fully waited and immutable, plus `added: Matrix<u32>` (hypersparse, owned by the version, deep-copied on its first write in a new version, O(\|added\|)) | 4 B column index + 4 B value per arc; 4 B/row of row pointers |
| `downward` Uᵀ | same, stored by row of the transpose, value = arc id | same split | 8 B/arc, 4 B/row |
| elimination parent | implicit: parent(r) = min of U row r over base ∪ added (I6) | — | 0 |
| `arc_id_end`, decay counters | scalars | copied | — |
| metric (out of scope) | four arrays indexed by **arc id**, length `arc_id_end` | #3774 chunks, plus an append that clones the chunk table | 32 B/arc today |
| reserve ranks | the bottom `R_bottom` and top `R_top` ranks are pre-placed empty rows, default max(1,024, n/64) each (derived) | — | 16 B/reserved rank |

Writers and the Seal only ever **add** to U. A removal (when S5 rewrites a subtree, or at compaction) happens only in a builder batch, which writes a new `base` with `GxB_Matrix_concat` of row tiles. So no removal delta exists. Once `added` passes a square-root share of `base`, it is folded into `base`, on the builder only. That is the `VersionedMatrix` rule (`versioned_matrix.rs:151-171` [U2 §4]), but not the GIL-time `fold_oversized_deltas` [FOLD]. Every matrix is waited before it is published (`mvcc_graph.rs:332-347` [U2 §3b]).

**Bytes against C and today, per arc:**

| | today | C | this design |
|---|---|---|---|
| Topology | 16 B (`RankRows` ×2, 8-byte `Rank` [U1 §1]) | 16 B (`up`, `down`, 8 B each [U1 §1]) | 16 B: U and Uᵀ with ids. Iso Uᵀ costs 12 B, but adds a binary search for each apex in scoped recompute. |
| Metric | 32 B | 32 B | 32 B. Out of scope; 24 B if middles became `u32`. |
| **Total** | **48 B** | **48 B** | **48 B (parity)** |

Per node it costs 16 B, against 32 B today and C's 64 B plus row headers [U1 §1]. 32-bit indices hold while dimensions are at most 2³¹ (`GrB_get_set_integers.tex:7-20` [U4 §4]). That is a gate in §9 (P3), because R3 lists "whether U actually gets 32-bit indices" as open [R3, Open].

### Arc identity (the I3 slot problem)

Today's contract I3 says `up(r)[i]` is slot `up_range(r).start + i` (`order.rs:123-125` [U1 §2]). That is replaced by **I3′**: `up(r)` yields `(rank, arc_id)` pairs, ascending by rank. An arc id never changes while its topology generation lives.

- **At a full build**, ids are given out in row-major order, so the metric's locality matches today's flat layout. The ids are filled into the values of a private U by unloading a Container (`iota`), which is allowed on a private copy only (C5 [B §6]).
- **S2 appends** a new id (`arc_id_end += 1`) for each new arc.
- **S5 also appends**: it gives new ids to the subtree's rows, and the old ids become holes.
- **Holes are never reused.** Reusing them would let a reader on an older version see a recycled id. They are reclaimed only by S6 and S7, which start a new id space, and therefore a new metric built from scratch.

This is option (a) of [U1 §4], and it avoids option (b)'s conflict with #3774's chunked arrays [B §4b].

**Readers that change:**
- the `seed` and `relax` slot arithmetic (`customize.rs:270-276`);
- the query's `up_range(...).zip(up(...))` (`query.rs:113`);
- the `up(lo)[slot - up_range(lo).start]` read in `partial.rs:158`, whose heap now carries `(lo, hi, id)`;
- the arrays sized by `arc_count` (`customize.rs:208-214`), now sized by `arc_id_end`;
- `Cch::joins`, now a lookup in U [U1 §2, U2 §2].

I2 and I4–I9 are unchanged. I7 is now "Uᵀ = transpose(U), ids equal".

### Builder-private state (`Building` only, `Arc<BuildState>`)

The build state holds the skeleton snapshot `S0` (an iso-valued bool matrix), the part-label and rank vectors (`GrB_Vector<u32>`), the coarse-level stack of the current depth, the frontier `F` and its emitted parts, the metric under construction, and `deferred` (a roaring set of written edge ids).

A batch never writes into the `Arc` it read: it builds a new one. A writer's copy of the entry is a single `Arc` bump.

## 3. Ordering algorithm

The skeleton S is the input. A batch computes `S0 = offdiag(ONEB(P, Pᵀ))`, where P = (pattern(m) masked by ¬dm) ∪ pattern(dp), per type, unioned across types. That is two or three pattern calls per type. It is cheaper than `effective`'s f64 merge, which costs about 400 MB and 76 ms at 9M edges [FOLD]. The cost of the pattern route is unmeasured.

1. **Isolated nodes, reserves, components.**
   - Degree is a `reduce` to a vector; degree-0 nodes and the bottom reserve take the lowest ranks, as `nd.rs:61-64` does today [B C9].
   - Components come from `LAGr_ConnectedComponents` on a private dup, because it unloads `G->A` (`FastSV7.c:42-46` [U4 §1]).
   - Each component becomes a part at depth 0.
2. **Peel trees.** Up to 64 rounds of degree-1 peeling. Each round is a masked `mxv` (`PLUS_PAIR`), followed by an assign of the peeled vertices into the next low ranks, in round order. That adds no fill (derived, [U4 §1 notes]). The round cap bounds the cost on long hanging chains (derived).
3. **Depth loop.** At depth d, all parts share one matrix, `A_d⟨A_d, structure⟩ = Lᵀ ANY_PAIR L`. Here L (part × node) is built from the label vector with `GxB_Matrix_build_Scalar_Vector` (`PartitionQuality.c:118-122` [U4 §1]). The product keeps only intra-part edges, in one masked dot-product call [U4 §2].
4. **Coarsen all parts together.** Coarse vertex ids are grouped by part, so each part's rows stay contiguous.
   - **Matching** (Manne–Bisseling, a locally dominant ½-approximation [R1 §3A]):
     - Each edge gets a key, (edge weight ≪ 32) | scramble(j), from a positional `apply`, where scramble is a seeded bijection.
     - `y⟨unmatched⟩ = K MAX_FIRST x_unmatched` picks each vertex's heaviest unmatched neighbour.
     - Extract by the choice vector, then compare with the row index, to keep the pairs that chose each other.
     - This takes O(log n) rounds, needs no incidence matrix (unlike LAGraph's 4 entries per edge [U3 §3]), and freezes any part that is already at or below its floor of 40–100 vertices (METIS's value [U3 §1]).
     - If a level shrinks by less than 15%, a 2-hop pass runs: a masked `A·A` over unmatched vertices, capped by degree, as feral does [U3 §2].
   - **Contraction:** `offdiag(C·A·Cᵀ)` with `PLUS_TIMES` and vertex weights `C·w`, where C is the coarsening map (`Coarsen_Matching.c:483-506` [U3 §3]).
5. **Initial separator per part** (host loop, run in parallel across parts). A row iterator, reused across rows (1,654 instructions to attach against 298 to re-seek [B §3]), reads each part's ≤ 100-vertex block of rows. Each part then runs:
   - 7 trials of greedy graph growing, scored by edge cut;
   - a König vertex cover;
   - node FM.

   This is METIS's L1 [U3 §1]. A host loop beats GraphBLAS here, because per-call overhead dominates at about 100 vertices [U3 §3].
6. **Uncoarsen all parts together.** Labels are projected with one `mxv` by `Cᵀ` per level, using 3-way labels A/B/S as METIS does. At each level:
   - (a) gains for every part come from one `mxv` of A against the side indicator, followed by one greedy batched move round;
   - (b) host boundary-seeded one-sided node FM runs per part, in parallel. Seeding from the boundary only fixes feral's full rescan (`fm_refine.rs:350-378` [B §5.2]).

   Batched moves alone measured +6.1% separator size; adding serial FM recovers it (mt-Metis [R1 §1]).
7. **Flow refinement at the finest level (O2).**
   - **Band.** Take the vertices within distance 3 of each separator: a multi-source BFS from all separators at once, three masked `vxm` calls. PT-Scotch's width 3 loses no quality [R1 §1].
   - **Network.** Each band vertex v becomes v_in → v_out with capacity w(v). Edges become v_out → u_in with capacity 2⁴⁰. Each part's A-side outside the band contracts to a terminal s_p, and its B-side to t_p. A super source σ links to every s_p and every t_p links to a super sink τ.
   - **Cut.** Because the parts are disjoint, one `LAGr_MaxFlow(σ, τ)` (`LAGraphX.h:1635`) gives every part its own min cut. A BFS on the residual graph from σ, with `LAGraph_MinCut` (`:1649`, still to bind [U4 §1]), gives the reachable set R. The new separator is {v : v_in ∈ R, v_out ∉ R}.
   - **Acceptance.** The old separator lies inside the band and separates the two terminals, so the min cut is never larger than it (derived). A part keeps its old separator if the new sides break the 1.2× balance bound (METIS `ufactor=200` [U3 §1]).
8. **Assign ranks.**
   - Each part's range [lo, hi) is split as A = [lo, lo+|A|), B = next, and S = the top |S| ranks.
   - The prefix sums are a host pass over the parts (2^d of them at depth d).
   - New labels come from an `apply`.
   - Separator vertices are ordered by node id, which is deterministic. Step 4 of §4 relabels them into a postorder anyway.
9. **Leaves.** Parts of ≤ 120 vertices (METIS's `MMDSWITCH` [U3 §1]) get a host exact minimum-degree ordering on 128-bit neighbour bitsets, in parallel. That is O(k²) per leaf, roughly 0.2 G operations on California (derived). Leaf thresholds 0, 120 and 200 are measured, as [U3 §4] asks.
10. **Size guard.** After the coarsest depth-0 separator is projected, its weight bounds |S0|.
    - If |S0| exceeds `CCH_MAX_TOP_SEPARATOR` (default 2,048, derived: about 2.1M arcs in one clique), the build ends `Failed("no small separator")` and queries use the Dijkstra fallback (proc `cch/fallback.rs` [U2 §2]).
    - The basis: CCH Lemma 2 puts a clique the size of |S0| into at least n/3 search spaces, and heuristic treewidth on social and web graphs is 0.5–27% of n [R1 §2].
    - A second check runs after the order: if the column counts (Gilbert–Ng–Peyton [R3 §1]) predict more than 32 arcs per skeleton entry, refuse.

**Determinism.**
- Matching keys come from a seeded scramble, and every reduction is MIN, MAX or integer PLUS.
- The min cut is the source-side reachable set in the residual graph, which is the same for every maximum flow (derived, standard), and FP64 sums of integer capacities are exact.
- BFS parents are never used (they are non-deterministic [U4 §1]); only levels are.
- Whether the result is independent of the thread count is a prototype gate (P6).

**Expected quality.**
- The baseline is C METIS on California: 5,527,199 arcs and 48.6M triangles. Feral sits +5.6% / +10.9% above it [U3 §4].
- Steps 3–6 match METIS's recipe: best of 7 trials, boundary FM, 2-hop coarsening, MMD-like leaves [U3 §4]. Expect roughly METIS.
- Step 7 is the KaHIP/PT-Scotch recipe. KaHIP's separators are 10% smaller than METIS's [R1 §1]. On Cal/Nev, flow-based orders (IFC8) give 217.7 triangle units against METIS's 364.1, and 21.7 µs queries against 47.9 µs (IFC Table 2 [R1 §2]).
- My target is 10% fewer triangles than METIS on California, which is far short of IFC. It is unmeasured; P1 is its gate.

## 4. Supergraph construction

1. `B = GxB_Matrix_extract_Vector(S0, π, π)`.
2. Build the elimination tree with Liu's host algorithm, O(nnz·α), over B's rows (C's `cch.c:312-375` [R3 §1]).
3. Take a host postorder q, set π′ = π∘q, and re-extract B′ = S0(π′, π′). The fill does not change (survey p.8 [R3 §1]). Every subtree is now a contiguous rank range.
4. `U0 = select(B′, TRIU, 1)`. The tag of U0 entry (r, s) is the next row in column s, or s for the last one. A host pass over a row iterator on `U0ᵀ` computes it in O(nnz).
5. Frontier loop. `Pᵀ` holds one entry per child, at (parent(c), c). Each round:
   - `F = mxm(Pᵀ, F, GxB_MAX_SECOND_UINT32)`;
   - `F = select(F, i < value)` with a user `IndexUnaryOp`;
   - push F onto the list of parts.

   Under MAX, the entry that continues has the larger tag, so it survives the collision at the lowest common ancestor that an ANY semiring would lose (C1 [B §6]). Each arc is produced exactly once (R3, checked at n ≤ 50 [R3 §1]). The loop runs at most tree-height rounds, two calls each. The work is O(|G+|).
6. Assemble once with `GrB_Matrix_build` from the parts, avoiding the pending-tuple cost of reassembling every round (C3, `GrB_examples.tex:170-176` [R3 §1]). Give out ids with a Container `iota` on the private U. `Uᵀ = transpose(U)`. Wait on both.

**Subtree batches (used by S5 and on huge graphs).** Run steps 4–5 on the U0 rows of a rank range [a, b] that is a union of complete subtrees. Tags are computed among that range's own entries, and entries that leave the range are dropped.

- **The rows inside the range come out exact** (derived). An arc (r, s) with r in the range comes from a U0 entry at a descendant of r, which is also in the range. And no path from outside the range enters it, because paths only climb.
- **An ancestor row outside every batch** is seeded with its children's finished rows minus itself, the child-absorption rule (DH99 Alg. 1 [R2 §2]).

**Cost.** O(|G+|) work, plus about 2H GraphBLAS calls, where H is the tree height: at most 1,232 on Europe, unmeasured on California [R3 §1]. Today's host game takes 97 ms on California [U1 §5]; matching it is gate P2.

**Fallbacks.**
- G1 is the host game over a row iterator on B′, with `GrB_Matrix_build` for output [B §4b].
- G4, the repeat-until-unchanged fixpoint, is kept as a test oracle only.

**Contract met.** I2, I4–I9, with I3 replaced by I3′ (§2).

## 5. Batching on the native index builder

| Phase (cursor) | One batch | State it leaves (in `Arc<BuildState>`) |
|---|---|---|
| `Skeleton` | builds S0 from its fork; isolated nodes, components, peeling | S0, ranks of the low vertices |
| `Order{depth, level}` | one depth, or one coarsening level of a huge depth 0; target ≤ 250 ms (derived) | labels, partial ranks, the coarse-level stack |
| `Supergraph{round}` | Liu + postorder in the first batch, then a slice of frontier rounds; usually a single batch on California | π′, F, parts |
| `Customize{apex_rank}` | the first batch seeds every arc from its own snapshot S_c; later batches relax apexes over a rank range | metric chunks, apex cursor, S_c's version |
| `Seal` | applies `deferred` (§6b.3), then runs the scoped recompute | the finished `Cch` |

- **Cursor and `place(id)`.**
  - All topology comes from S0, and seeds come from S_c, so nothing a writer does is "below the cursor".
  - `place` answers "not covered" until the Seal, so writers only record edge ids, as a `Building` CCH entry does today (`maint.rs:297-314` [U2 §3]).
  - Each batch moves `since_last_indexer_step` into `deferred` and clears it. A writer's copy therefore costs O(writes since the last batch), not O(writes since the build began) [U2 §4].
- **Progress.** `index_build_read` reports work done across the phases, weighted by phase cost. It returns 0 today (`versioned.rs:858-861` [U2 §3]).
- **Publishing.**
  - Before the Seal, a batch publishes as Same or Swap. A CatchUp only merges written ids into `deferred`: no Refresh is needed, because nothing derived from the graph has to be re-read.
  - The Seal's catch-up runs the §6b writer path under the GIL, within the same budget as a writer. Over budget, it answers `Absorbed::Restart`, which re-runs only the Seal phase, never the order (#3772's restart counter, which marks the build `Failed` after 5 [U2 §2]).
  - A Discarded batch repeats from the published cursor.
- **What readers see.**
  - On a first build, the version has no hierarchy, so queries use the Dijkstra fallback (222 ms median on California [U2 §2]).
  - During an S7 re-order, the old generation stays `Built`, keeps being maintained (§6), and keeps serving. The new one builds as a shadow job and is adopted at its Seal. This changes the framework: an entry is `Built` while it also has a build job (§10).

## 6. Incremental maintenance

These run on the private version in `maint::flush` and `absorb`. They replace the two "structural write means rebuild" branches (`maint.rs:551-582, 621-623` [B §2]). The writer budget per flush is W = 4,096 new arcs plus A = 16,384 recomputed arcs, about 21 ms at 1.3 µs per arc (`maint.rs:85-94` [U1 §5]). The budget values are derived; P4 tunes them. Past the budget, nothing is written: the version gets `Stale{last consistent Arc<Cch>, backlog}`, queries on it use the fallback, and a builder **repair batch** (S2 or S5, then §6b) publishes the result through the Seal path.

| Change | Rule and correctness | Cost | Where |
|---|---|---|---|
| Weight SET/REMOVE; insert where the pair already has an arc (original, fill or stale) | Metric only (§6b) | the cone | writer, within A |
| Insert (u, v), u < v, where v is an ancestor of u | S2 walk: add v to `up(j)` from j = u up the parent chain, until v ∈ up(j) or j = v. The tree is unchanged, because v is at least parent(u). The stop rule is I5 (R3, checked). New ids are appended to `added`. | at most one path segment: median 6–22.5 arcs on grids [R3 §3] | writer, within W |
| Insert across sibling subtrees, below the root | Davis–Hager update along P(u) ∪ P(v) merged by rank (DH99 Thm 4.1: only the new path's rows change [R2 §2]); or S5 at T = subtree(LCA(u, v)) | 4–7% of all arcs per edge at the top separator of a grid [R3 §3]; DH cost is proportional to the changed entries | writer if within W; otherwise a repair batch, which picks S5 when \|T\| ≤ n/4 and the predicted S2 fill is over 5% of T's arcs (derived) |
| Insert across the top separator or between trees | same rules; on grids about 0.9·n arcs [R3 §3] | | repair batch; counts toward S7 |
| Edge delete; node delete (its cascade of edges) | S1: the arc stays and is reseeded (+∞ if it was the pair's last edge). G − e ⊆ G ⊆ G⁺ [R1 §3] | metric only | writer, within A |
| Node created within capacity, or within the bottom reserve | Nothing: it holds an isolated bottom rank (`nd.rs:61-64` [B C9]). Its first edge is an ancestor insert with no fill. A second edge to x2 adjacent to x1 (splitting a road edge) is also an ancestor insert, so 0 fill (derived). | O(1) | writer |
| Node past the bottom reserve | Takes the next top-reserve rank, above every built rank. Each edge (w, x) walks x's path to its root and adds w; the root's parent becomes w. | \|P(x)\| arcs: 46 / 94 / 190 on 256 / 1,024 / 4,096-node grids [R3 §3] | writer, within W; once the top reserve runs out, S7 |
| A reused node id | Inherits its old rank; its edges are classified as above | — | — |
| Relationship types | Types are fixed by the definition; edge types are immutable. A named type created later only brings ordinary inserts. A definition change is DROP/CREATE. | — | — |

**S5 correctness.**
- An elimination subtree is closed: no lower-ranked vertex outside T has an arc into T (C4 [B §6]).
- Re-ordering T inside its own range [a, b] leaves every U row outside the range unchanged [R3 §3].
- The new outward entries of T, up(root_new) ∖ T = N(T) ∖ T, are a subset of the old up(root_old), and that set is a clique (I5). So no arc outside T is added.
- The Uᵀ rows of up(root_old) lose T's old entries and gain the new ones.
- S5 runs our ND on the induced subgraph A(T, T), read from the repair batch's fork with a masked extract, then G2 on T's rows (§4).

**Decay signals and their actions** (all thresholds derived and unmeasured; DH99 §7.3 is the only analogue: fill grew from 381k to 7.4M under a fixed order [R2 §2]):

| Signal | Action |
|---|---|
| A subtree's fill added by S2 exceeds 10% of its arcs | S5 on that subtree, as a builder batch |
| Dead ids reach 25% of `arc_id_end` | S6: rebuild U from π with a fresh snapshot, new ids, full customization |
| Deletions reach max(1,024, n/8) since the build | S6 (C's valve, `cch_index.c:468-496` [U3 §1]) |
| Arcs exceed 1.15 × arcs at build time, height exceeds 1.25 × height at build time, or the top reserve is exhausted | S7, a new generation |

S6 and S7 are the only steps that compact the id space.

## 6b. Recomputing modified edges

**The quantity.** Arc (lo, hi) holds the shortest lo–hi path through strictly lower ranks (`customize.rs` module docs). Its value reads only its seed and its lower triangles (x, lo), (x, hi), for x in down(lo) ∩ down(hi) (`partial.rs:8-15`). So an arc must be recomputed exactly when:
- its seed changed, or
- one of its lower-triangle arcs changed value, or
- its set of lower triangles changed, or
- the rank of its middle changed (I9).

**Lemma A (new arcs are born at +∞, derived).**
- Let M be the exact customization of a valid topology H. Let H′ ⊇ H be valid, with new arcs N. Extend M with (+∞, +∞, no middle, no middle) on N.
- **Old triangles are unaffected.** A triangle of H′ whose lower arcs are both old has an old top arc, by I5 in H. So a triangle new to an old arc contains an arc from N as a lower arc. That arc's candidate is +∞, which never wins a strict `<`. Every old arc's value and middle therefore already equal H′'s customization, unless a seed changed or a member of N becomes finite.
- **New arcs have only new apexes.** A new arc (j, v) has no apex in old down(v) ∩ down(j): such an apex would have put j and v together in one up-row, so the arc would already exist by I5.
- So `recustomize_seeded` (#3773's pull and push, with a heap ordered by lower rank) is exact on H′ when its front is N ∪ C, where C is the set of changed seeds. It is also bit-identical to a full customization, because its tie rules are the same (`partial.rs:16-20`).

**Lemma B (re-dissecting T, derived).**
- Every arc outside T keeps its **value**. T still occupies [a, b], so the set of vertices ranked below each outside arc is the same as before.
- An arc with an apex x in T has both ends in up(x) ∖ T, a subset of up(root_old). So only the boundary clique K_T, the arcs among up(root_old), can change its **middle**.
- Front = every arc with its lower end in T (all new, so a forward sweep over T's apexes) ∪ K_T (forced to a full pull). With no graph change, values are equal, so nothing propagates. With the triggering edge inside T, values can fall and propagate as usual.

**Recompute set per change** (each followed by the push closure: when an arc's value changes, its top arcs (hi, w), w ∈ up(lo), are recomputed):

| Change | Front |
|---|---|
| Weight SET/REMOVE, delete, or insert on a pair that has an arc | its arc, if the pair's MIN-reduced seed in either direction changed [U1 §1] |
| Node delete | the arcs of its incident edges (deg(v) pairs) |
| S2 insert | every new arc N (born at +∞), plus the inserted pair's arc |
| Node past the reserve | the new arcs on its root paths |
| S5 of T | T's arcs ∪ K_T (Lemma B) |
| S6 or S7 (new generation) | a full customization, as builder batches |

**Algorithm and cost.**
1. Topology goes first, for the whole unit: every S2 and S5 change of the unit is applied as one batched update (DH01 Thm 4.2 [R2 §2]).
2. Then one call: `recustomize_seeded(front, seed_fn)`, with the heap keyed by (lower rank, arc id).
   - Each apex is a merge of Uᵀ(lo) and Uᵀ(hi), whose ids give (x, lo) and (x, hi) with no `arc_slot` search.
   - Each push is one U lookup.
   - Seeds come from `PairWeights`, which needs no weight matrix (#3363).
3. Measured costs on California:
   - about 1.3 µs per recomputed arc (`maint.rs:85-94`);
   - a single SET is 0.22 ms p50 (#3363);
   - 512 random pairs take 0.41 s (`maint.rs`);
   - the full path is about 0.5 s;
   - on a 200×200 grid, scoped takes 10.7 ms against 464 ms full (#3773).
4. Derived estimates:
   - An ancestor insert adds about 6–22 arcs plus the cone of the new edge, which is a weight-decrease cone: about 1–30 ms.
   - A top-separator insert touches about 25% of California's arcs if roads behave like grids. That is effectively a full customization: a repair batch, and an S7 signal.
   - S5 costs ND(T) + G2(T) + sweep(T) at about 0.09 µs per arc [U1 §5] + |K_T| × 1.3 µs. With |up(root)| = 100–300, K_T is 5k–45k arcs, so 6–60 ms (derived).

**Interleaving with a batched build** (§5).
- **(i) Edges changed before the batch that covered their region.** Topology comes from S0. Every edge created before S0 is in U. Every weight change before S_c is in the seeds.
- **(ii) Edges changed after their region was covered.** Created after S0: Seal S2 from `deferred`, born at +∞, in the front. Reweighted or deleted after S_c: the Seal front. Changed after the Seal's fork: the publish catch-up, by the same writer rule within budget, else a Seal restart. Deletions never restart [U2 §2].
- **(iii) Writes between batches.** These are only recorded ids. No partial metric is ever patched mid-sweep: an arc in the sweep that receives a raised seed cannot unwind the old value, so the Seal's pull recomputes it from scratch (derived).
- **(iv) A subtree rebuilt while its ancestors' metrics are final.** Lemma B: recompute K_T, and propagate only when a value changes.
- **(v) Order.** Topology change, then I2–I9 hold, then the recompute. `partial.rs` panics on a broken closure (`partial.rs:221-223` [U1 §2]), so the order is enforced.

**Where it runs, its bound, the hand-off.**
- **On the writer, under the GIL:** classification is O(written × log row length). S2 runs up to W, and the recompute counts arcs up to A. Past either, the private result is dropped and the version is marked `Stale`. No full customization ever runs under the GIL. Today one does, about 0.47 s (#3772) and inside `GRAPH.EFFECT` on a replica (#3363).
- **In a repair batch:** S2/S5 and the recompute run off the GIL. A front larger than n/8 arcs switches to a full sweep, which is C's valve [U3 §1].

**MVCC.**
- Recomputation writes only into the private `MetricVersion` (copy-on-write chunks [#3774]) and the private `added` matrices. Both are published together in one new `Arc<Cch>`, so a reader holds a consistent (topology, metric) pair or the older one.
- A `Stale` version never serves its hierarchy.
- Readers on older snapshots keep their `Arc`.
- Nothing becomes visible part-way: a flush either completes or is dropped with the private version (`maint.rs` "all updates computed before any entry changes").

## 7. MVCC and replication

- **What a version holds.** Each version holds `Arc<Cch>`: the shared base matrices, its own small `added` matrices, the metric chunks it wrote, and the decay counters. A writer's version costs O(|added| + touched chunks). A table clone costs 181 KB and a median version owns 0.23 MB [U2 §4].
- **Folding.** The builder folds `added` into a new base, which is published by Swap or CatchUp.
- **Dropping.** Replaced versions are dropped outside the GIL [U2 §3c].
- **Determinism decision: the same distances only.**
  - The topology depends on history: which inserts were S2 and which S5, when batches ran, the deferred sets.
  - C is already history-dependent through its stale arcs [B §5.7].
  - Distances are a function of the graph, and every hierarchy computes them exactly.
  - Paths agree whenever the shortest path is unique. Among equal-cost paths, middles follow rank (I9) and may differ.
  - Making paths identical would mean a single rebuild point for the whole cluster. Rejected.
- **RDB.** The definition is saved, and the hierarchy is rebuilt on load (C parity, `cch_index.c:161-163` [U3 §1]).
  - Optionally, save π (4 B per node) in a later RDB version. The rebuild would then skip ND, the cost that dominates: 3.62 s of 4.63 s [U1 §5].
  - That is open, because it changes the wire format.

## 8. Memory and time budget

**California.** 1.54M-node skeleton, 3.31M entries, 5.84M arcs with feral, 5.53M with METIS [U1 §1, U3 §4].

| Item | Estimate | Basis |
|---|---|---|
| Steady topology | 16 B × 5.84M + 16 B × 1.54M = **118 MB**, against 142.7 MB today | derived from the measured #3321 sizes |
| Metric | 186.8 MB | computed [U1 §1] |
| Total | about 305 MB, against C's about 364 MB (48 B/arc + 64 B/node) | derived |
| Holes | at most 25% of the metric before S6 | derived |
| Build peak: ND | S0 about 20 MB, plus the intra-part matrix, coarse levels (about 2×), the matching key matrix (26 MB) and about 10 n-vectors: **about 150–200 MB** above the graph. feral peaks at 447 MiB and METIS at 363 MiB [U1 §5]. | derived |
| Build peak: G2 | parts (about 12 B × 5.84M = 70 MB) + U + Uᵀ (94 MB) + U0: **about 200 MB**, against 185 MiB today [U1 §5] | derived |
| Time | ND unmeasured: target ≤ C METIS's 3.07 s [U1 §5]. G2 target ≤ 97 ms. Customization 0.22–0.47 s [U1 §5, #3772]. | — |
| Per update | SET 0.22 ms p50 (#3363), plus 4 × 16 KB chunk copies when ids are appended. Ancestor insert about 1–30 ms. Node past the reserve: \|P(x)\| arcs. | cited / derived |

**9M edges.** For the 1500×1500 grid in [FOLD]:
- S0 is about 72 MB for 18M symmetric entries, built by the pattern route instead of the 400 MB merge [FOLD].
- The grid's fill grows about 6 arcs per node for each ×4 in n (9.6, 14.7 and 20.8 arcs per node at n = 256, 1,024 and 4,096 [R3 §3]). Extrapolated, that is about 48 arcs per node, so about 108M arcs: topology about 1.7 GB, metric about 3.5 GB (derived; a grid is the worst case).
- A road graph of the same size at California's 3.5 arcs per undirected pair would have about 32M arcs: topology about 0.5 GB, metric about 1 GB (derived).
- During an S7, the peak is two hierarchies (the old one serving), the same as C's synchronous rebuild holding the old and the new.

## 9. Implementation plan

**Prototypes first.** Each has a pass/fail gate and a fallback.

| | Measures | Pass | On fail |
|---|---|---|---|
| P1 | ND quality (O1, then O2) on California and roadNet-CA against C METIS (`:edge-c`); arcs, triangles, height | O1 arcs ≤ METIS +2% and triangles ≤ +5%; O2 triangles ≤ METIS −10%; ordering time ≤ 3.07 s | drop flow (O1 only). If O1 also fails, keep feral behind the new API. |
| P2 | G2 time, rounds H, exact equality with the host game | ≤ 97 ms; equal on every graph in the test kit | G1 |
| P3 | query p99 with U/Uᵀ read by row iterator plus the `added` merge; 32-bit indices asserted | p99 ≤ 526 µs on roadNet-CA (#3324) and ≤ C | fold `added` before publish; or an iso Uᵀ |
| P4 | S2 + Lemma A recompute: 10k random inserts per class on California | ancestor inserts p99 ≤ 5 ms; equal to a full customization of H′; sets W and A | the S2 path is hand-off only |
| P5 | `LAGr_MaxFlow` on the bands, per depth | ≤ 30% of ND time | flow only on depths 0 to 3 |
| P6 | the order at 1, 4 and 16 threads | bit-identical | pin the GraphBLAS context threads for ND |

**PRs.** Each is small and closes its own issue.
1. GraphBLAS wrappers: masked `mxm` and `mxv` over any semiring, `extract_Vector`, `build_Scalar_Vector`, positional `apply`, user `IndexUnaryOp`, reduce-to-vector, `concat`, a `LAGraph_MinCut` binding, and a Linux PreJIT harvest. Generic kernels cost +6% to +87% [B §3].
2. `ChunkedArray<T>` with append (generalising #3774).
3. Topology on GraphBLAS with I3′, still fed by feral. Proves the contract change and query p99 (P3).
4. G2 + Liu + postorder (P2).
5. Our ND, O1 (P1, P6), with the size guard.
6. Flow refinement, O2 (P1, P5).
7. Build phases, cursor, `deferred`, Seal, progress.
8. S2 on the writer with budgets, Lemma A fronts, `Stale` + repair batch. Replaces #3363's rebuild branch (P4).
9. Decay counters, S6, S7 shadow generation.
10. S5 with the K_T recompute (Lemma B). Its test checks the result against a full customization.

Each PR lists its new functions in `proofs/*/COVERAGE.tsv` (the Lean gate since #3172).

## 10. Risks and open questions

- **ND quality is unproven in GraphBLAS form.** Batched matching may coarsen worse than SHEM (unmeasured [U3 §3]). Fallbacks: P1's.
- **Level-synchronous imbalance.** Parts frozen at different levels waste calls. Per-call overhead × depth × levels is unmeasured [B §5.10].
- **MaxFlow on bands.** FP64 only, no warm start [R1 §1], and an experimental LAGraph routine with open TODOs [B §3].
- **G2 is unverified** beyond Python at n ≤ 50, and needs a user op plus PreJIT.
- **Query cost of the GraphBLAS row iterator** (P3). Rust today pins every matrix to sparse or hypersparse, row-major [U4 §3].
- **Framework change.** A `Built` entry that also has a shadow build job (S7), and a `Stale` state. Both need review of `versioned.rs`.
- **Fallback while Stale.** Cross-separator inserts send the version to the Dijkstra fallback until the repair batch publishes. That is shorter than today's full rebuild, but under sustained cross-cell writes it can be frequent. Thresholds are unmeasured on road graphs [B §5.4].
- **Budgets W and A,** and the decay thresholds, are all derived. Defaults come from P4.
- **Order labels across S5.** S5 renumbers T's ranks, so a reader's version mid-query must hold its own rank maps. They are per version by construction; P4 tests it.
- **Open questions for the team:**
  - Persist π in the RDB?
  - Is "same distances only" acceptable for `algo.SPpaths` users?
  - Is the shadow generation's 2× peak memory acceptable on 9M-edge graphs?
