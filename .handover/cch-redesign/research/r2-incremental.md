# R2: Incremental, MVCC-friendly CCH topology (literature)

Scope: web research on keeping a CCH elimination order and its chordal supergraph current under node and edge inserts and deletes. Source keys such as [DH99] are listed at the end. "Derived" marks reasoning that is mine and not taken from a paper.

## Bottom line

- No published "dynamic CCH" handles topology changes. The CCH papers assume a fixed topology. Every CH or tree-decomposition paper that does handle edge inserts and deletes keeps the vertex order fixed and patches the shortcut set ([CH-TS] §5.1, [DCH] §4.3, [ZY22] §7).
- The closest mathematical model is sparse Cholesky symbolic update and downdate [DH99, DH01, DH05]. An insert changes only the rows on one elimination-tree path. A delete can be absorbed by keeping a superset of the needed arcs. CHOLMOD itself never removes entries on a downdate and prunes them later with a full symbolic pass [CHOLMOD-UG].
- Ordering costs far more than building the supergraph. On Europe the order takes 341.2 s and the supergraph (chordal completion) takes 1.6 s [Survey, Table 1, p.23]. The order is the thing to keep. Rebuilding the supergraph from a kept order is cheap enough to serve as the cleanup step.
- No source measures how fast a CCH order loses quality under topology updates. The one quantitative analog is an LP sequence in [DH99] §7.3: with a fixed order, the fill grew from 381k entries to a 7.4M peak.

## 1. CH and CCH under topology changes

- **CCH treats topology as fixed.** The motivating game scenario says "the basic topology of the map is fixed", and impassable fields are handled as metric changes [CCH, pp.1–2]. Partial updates rely on "we do not have to insert or remove arcs" [CCH, §7.7, p.15]. The experiments section says "if the graph topology is fixed, as we expect it to be" [CCH, §9.6, p.30]. The 2025 survey explains the three phases by saying "topological updates happen much less frequently than metric updates" [Survey, p.1]. RoutingKit's guidance is to "store the order and rebuild the CCH as needed" [RoutingKit].
- **Correctness needs only a chordal supergraph.** The supergraph must be chordal with the rank order as a perfect elimination order [Survey, p.3 footnote 1; CCH, p.4]. Customization sets non-input arcs to ∞ ("respecting") [Survey, p.5]. So a supergraph does not have to be minimal.
- **Classic CH (Geisberger et al.).** "Because we only deal with a small amount of edge changes, we keep the original node order." Existing shortcuts are updated. Only the nodes reachable in the search graph from "seed" nodes are recontracted, and min{u,v} of a new edge is a seed. Shortcuts that are no longer needed are kept: "they do not invalidate the correctness ... although they can somewhat slow down the query" [CH-TS, §5.1–5.3, PDF pp.14–16].
- **DCH [DCH, §4.3, p.609].**
  - A deletion is a weight increase to +∞.
  - Inserting an edge that is already a shortcut is a weight decrease.
  - Otherwise DCH adds the shortcut (u,v). For each upward neighbour w of u it adds or relaxes (v,w), and it recurses.
  - The order stays fixed. Vertex operations are "a sequence of edge deletions/insertion". The paper does not say where a new vertex is ranked.
- **H2H [ZY22, §7].** Same treatment. After an insert, "the tree decomposition T is reorganized", and the labels of vertices whose parent or shortcuts changed are recomputed top-down.
- **Theory.** [DynTW] maintains a tree decomposition of width ≤ 6k+5 under edge updates in amortized O_k(2^{√(log n)·log log n}) time, but only "under the promise that the treewidth ... never grows above k". The O_k "hides factors depending on k". I found no practical evaluation.
- **Current state.**
  - Rust (#3363, #3772): a deletion is reseeded with +inf. An edge between nodes the hierarchy does not join sends the index back to `Building`, which is a full rebuild including the order.
  - C (civil-kitten `origin/master` a61fbb67a, `src/index/cch_index.c`):
    - a new adjacency sets `pending_rebuild` (:581);
    - each deletion increments `stale` (:596);
    - either one escalates to `CCHIndex_Build` (:493–494);
    - the stale valve is max(n/8, 1024) (:468–476).

## 2. Sparse Cholesky symbolic updates

How the two fit together: supergraph H = the pattern of L; the upward row N⁺(u) = column L_u; elimination-tree parent = min L_j\{j} [DH99, p.607–608], which is the "first entry of its finished upward row" in our `order.rs:20–22` (main 7df8e5887). An edge insert is a symbolic update, an edge delete is a symbolic downdate, and a vertex insert or delete is a row add or row delete.

- **Fill path.** (i,j) is in the filled graph iff G has an i–j path whose intermediate vertices are all ranked below min{i,j} [DH99, p.610, citing Rose–Tarjan–Lueker].
- **Building the pattern.** L_j = {j} ∪ ⋃_{children c} (L_c\{c}) ∪ A_j [DH99, Alg. 1, p.608]. This is the CCH survey's contraction step N[v] ← N[v] ∪ N[u]\v, where v = parent(u) [Survey, Obs. 3 / Alg. 3.1, p.10].
- **What an update touches.**
  - After an update, only the columns on the *new* path P̄(k) change, where k is the lowest-ranked touched index. Also P(k) ⊆ P̄(k), so search spaces only grow.
  - A downdate touches only the *old* path [DH99, Thm 4.1, p.611].
  - Cost is O(Σ_{j∈P̄(k)} |L̄_j|) [DH99, Lemma 4.2, p.614].
  - For a batch, the changed columns form a subtree, the union of the paths, and are processed in one pass [DH01, Thm 4.2].
- **Exact shrinking on delete needs multiplicities.**
  - The multiplicity of an entry counts the children that contributed it plus the input entry [DH99, §3, p.609]. That is "one additional integer value for each nonzero" [DH05, §3.2, p.630].
  - CHOLMOD stopped doing this. Its guide says "I actually did this for a while but it's costly", and "a downdate (cholmod_updown) does not remove any entries from L". `cholmod_resymbol` prunes them later [CHOLMOD-UG, lines 2350–2353 and 3550–3553].
  - [DH05] lets the entries become numerically zero and prunes them "by a complete symbolic factorization, taking O(|L|) time ... done rarely" (p.630). Row add needs no multiplicities (p.628).
- **Vertex delete and insert.**
  - `cholmod_rowdel` sets row/column k "to the kth row and column of the identity matrix". The slot stays, so the dimension never changes [rowdel header].
  - The pattern of a new row k is the union of elimination-tree paths walked up from each of its neighbours (`cs_ereach`) [ereach]. A vertex ranked at the top therefore costs one arc per ancestor of each of its neighbours.
- **Decay with a fixed order** [DH99, §7.3, pp.624–625]:
  - Order PA, computed on the starting matrix: 381k → a peak of 7.4M; it passes PB after 267 steps.
  - Order PB, computed on the *superset* of all columns that can ever appear: 831k → a peak of 1.49M.
  - The authors suggest reordering when the factor grows "too large".
  - The superset order is the matrix-world version of the CCH game scenario: order the union of every possible edge, and close edges with ∞.

## 3. Dynamic partitioning and nested dissection

- **ParMETIS** has an adaptive routine only for k-way partitioning (`AdaptiveRepart`). The fill-reducing orderings, `NodeND` and `V32_NodeND`, have no adaptive variant [ParMETIS, §5.4 and Fig. 1].
  - AdaptiveRepart coarsens locally, balances the coarsest graph with both remapping and diffusion, and keeps whichever scores better on the ITR factor.
  - ITR is the ratio of communication time to redistribution time [ParMETIS, §3.3, p.7].
- **Repartitioning trade-off** [GPSurvey, §8.3]:
  - Partitioning from scratch and then remapping gives good quality but high migration.
  - Diffusion gives low migration but poorer quality.
  - Adding "virtual nodes" whose edge weights are the migration costs optimizes both together.
  - For CCH, "migration" means re-ranking a vertex, which invalidates its rows and their weights.
- **Locality.** With a separator S, each cell's order "can be optimized independently", and "the set of inserted edges in C_i is independent of all other cells" [Survey, p.7]. The separator tree can be rebuilt from the elimination tree [Survey, Lemma 2, p.9], so it does not need to be stored.
- **Derived (from the fill path rule).** Reordering a cell whose ranks are contiguous changes only arcs that have an endpoint in that cell. So local re-dissection means:
  1. reorder the cell;
  2. rebuild its rows;
  3. push the change from the cell's top row upward with S2 below.

  This holds only if no lower-ranked vertex outside the cell has an arc into it. Earlier cross-separator inserts can break that, and then the cell to redo is the common ancestor cell.
- **Costs.**
  - The ordering dominates: 341.2 s versus 1.6 s on Europe [Survey, Table 1]. InertialFlowCutter needs 242 s on 16 cores [IFC, p.11].
  - feral-metis ordering time and quality on our graphs: unmeasured here.
- **When to rebuild.** No CCH source gives a rule. Analogs:
  - "too large" fill [DH99, §7.3];
  - the ITR cost balance [ParMETIS];
  - C's stale valve.

  Signals to track (derived):
  - arc count |H| against its value at build time;
  - elimination-tree height, which equals the maximum search space [CCH, p.4];
  - the count of deleted edges and vertices since the last build.

## 4. Persistent structures for MVCC

- **Path copying** copies the changed node and its ancestors, so it costs the number of ancestors. Node copying gives "O(1) cost per update for any pointer-based structure with any constant indegree" [MIT; DSST].
- **Fit (derived from [DH99, Thm 4.1]).** One structural update changes the rows on one elimination-tree path. A new version can therefore be new copies of those rows, with every other row shared.
  - Today `Topology` is "one flat pair of arrays" (`order.rs:20`). Inserting an arc shifts every later offset, so each version is a full copy.
  - A rank-keyed copy-on-write row store would avoid that. The engine already has `data_structures/cow_btree/` and chunked copy-on-write weight arrays (#3774).
- **Arc ids must be stable.** The metric is "the four per-arc arrays, indexed by arc id" (`mod.rs:256`). An incremental topology needs append-only arc ids, otherwise every insert invalidates the metric.
- **Rank labels.** An order-maintenance structure [OM; Dietz–Sleator 1987] gives a new vertex a rank between existing ones without renumbering. That suits matrices kept in node-id space, as the redesign wants: the rank becomes a label vector.
- **GraphBLAS.** Pending tuples and zombies are lazy inserts and deletes (`Doc/UserGuide/GrB_print.tex:499–502`, v10.5.0). Version sharing has to come from the engine's versioned layer. Its cost is unmeasured here.

## 5. Candidate strategies

**Invariant** (derived; [Survey, Obs. 1 and p.3 footnote]):

- (I1) every skeleton edge is an arc of H;
- (I2) for every u, N⁺(u)\{p(u)} ⊆ N⁺(p(u)), where p(u) is u's lowest upward neighbour.

(I2) is the same as "every upward neighbourhood is a clique": induct down from the top rank. Any H that satisfies I1 and I2 gives correct customization and correct elimination-tree queries, even if it is larger than the minimal fill.

| # | Strategy | Why it stays correct | Structural cost per update | Effect on quality | Source |
|---|---|---|---|---|---|
| S1 | Delete = ∞ weight; keep arcs and the vertex slot | G' ⊆ G ⊆ H, and H is unchanged, so I1 and I2 still hold | O(1) | H stops shrinking; search spaces stay the same | [DCH §4.3], [CHOLMOD-UG], [rowdel], #3363 |
| S2 | Insert (u,v), rank(u) < rank(v): add v to N⁺(u), then walk up rank by rank applying the I2 merge until nothing changes; a batch takes one pass over the union of paths | Only adds arcs that lie in fill(H+uv), and stops when I2 holds everywhere | O(Σ \|N⁺(j)\| over the new path from u) | Paths only grow [DH99 Thm 4.1]. Edges that cross a separator merge two subtrees: fill ≈ \|path_A\| × \|boundary\| (derived) | [DH99/DH01], [CH-TS §5.3], [DCH §4.3] |
| S2-GrB | The same closure as a fixpoint: U ← U ∨ Pᵀ·(U minus the parent arcs), with P = arg-min upward neighbour, repeated (derived) | Every step stays inside the minimal fill, and a fixpoint satisfies I2, so the result is exactly the minimal fill | One masked mxm per round; round count unproven and unmeasured | as S2 | derived |
| S3 | New vertex: give it a rank label below its neighbours, inside their cell (order maintenance) | Each edge it gains is applied with S2 | Degree 1: one arc, no fill. Degree d: a d-clique on its neighbours, then S2 | Local if its neighbours are local. Ranking it at the top instead adds 1 to every search space below it [ereach] | [OM], derived |
| S4 | Edge across separator S: move one endpoint up into S, then restore I2 in the new order (derived) | Old arcs stay as a superset; the closure is redone locally | Grows linearly with path lengths, unlike S2's product (unmeasured) | Separator grows by 1 | derived |
| S5 | Re-dissect the smallest cell that contains the change, rebuild its rows, then apply S2 at its top (derived) | Only the cell's rows depend on its internal order (fill path rule) | ND on the cell plus the cell's fill | Restores local quality | [Survey p.7, Lemma 2] |
| S6 | Resymbol: rebuild H from the same order, in rank-range batches on the index builder | Recomputes the exact fill | O(\|H\|); 1.6 s on Europe | Removes dead arcs and the excess fill from S2 | [CHOLMOD-UG], [DH05 §3.2], [Survey Table 1] |
| S7 | Full reorder when a decay signal crosses its threshold; a warm start keeps old separators unless the gain is worth the re-ranking | Full rebuild | Ordering cost (341 s on Europe with IFC; feral-metis unmeasured) | Restores quality | [ParMETIS], [GPSurvey §8.3] |

Suggested composition, all unmeasured:

1. **On writes and build catch-up:** S1 + S2 + S3. That replaces the "structural write → rebuild or `Restart`" path in #3363 and #3772.
2. **When dead arcs accumulate:** S6 on the background builder.
3. **On a decay threshold:** S7.
4. **Later:** S4 and S5, only if measurements show cross-separator inserts dominate.
5. **No multiplicity counters.** They cost one integer per arc, against the memory rule.

## Sources

- [CCH] Dibbelt, Strasser, Wagner, *Customizable Contraction Hierarchies*, arXiv:1402.0402v5 (JEA 2016): https://arxiv.org/abs/1402.0402
- [Survey] Bläsius, Buchhold, Wagner, Zeitz, Zündorf, *CCH – A Survey*, arXiv:2502.10519: https://arxiv.org/abs/2502.10519
- [CH-TS] Geisberger, Sanders, Schultes, Vetter, *Exact Routing in Large Road Networks using CH* (Transportation Science 2012): https://publikationen.bibliothek.kit.edu/1000028701/142973925
- [DCH] Ouyang et al., PVLDB 13(5) 2020: https://www.vldb.org/pvldb/vol13/p602-ouyang.pdf
- [ZY22] Zhang, Yu, *Relative Subboundedness of CH and H2H* (SIGMOD 2022, full version): https://www1.se.cuhk.edu.hk/~ykzhang/mod22-full.pdf
- [DynTW] Korhonen et al., *Dynamic treewidth*, arXiv:2304.01744: https://arxiv.org/abs/2304.01744
- [DH99] Davis, Hager, SIMAX 20(3) 1999: https://people.clas.ufl.edu/hager/files/update.pdf
- [DH01] Davis, Hager, SIMAX 22(4) 2001: https://people.clas.ufl.edu/hager/files/multirank.pdf
- [DH05] Davis, Hager, SIMAX 26(3) 2005: https://people.clas.ufl.edu/hager/files/rowmod.pdf
- [CHOLMOD-UG] SuiteSparse v7.8.3: https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.8.3/CHOLMOD/Doc/CHOLMOD_UserGuide.tex
- [rowdel] https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.8.3/CHOLMOD/Modify/cholmod_rowdel.c
- [ereach] https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.8.3/CSparse/Source/cs_ereach.c
- [ParMETIS] ParMETIS 4.0 manual: https://karypis.github.io/glaros/files/sw/parmetis/manual.pdf
- [GPSurvey] Buluç et al., arXiv:1311.3144: https://arxiv.org/abs/1311.3144
- [IFC] Gottesbüren et al., arXiv:1906.11811: https://arxiv.org/abs/1906.11811
- [RoutingKit] https://github.com/RoutingKit/RoutingKit/blob/master/doc/CustomizableContractionHierarchy.md
- [MIT] https://courses.csail.mit.edu/6.854/20/Notes/n02-persistent.html
- [DSST] Driscoll, Sarnak, Sleator, Tarjan, JCSS 38(1) 1989: https://dblp.uni-trier.de/rec/journals/jcss/DriscollSST89.html
- [OM] Bender et al., ESA 2002: https://www.doi.org/10.1007/3-540-45749-6_17
