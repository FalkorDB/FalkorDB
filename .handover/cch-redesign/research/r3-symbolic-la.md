# R3: building the supergraph as symbolic Cholesky in GraphBLAS, and updating it incrementally

Revisions read: Rust `origin/main` 7df8e5887, C `origin/master` a61fbb67a, LAGraph d01064de, GraphBLAS v10.5.0.
"Checked here" means a randomized Python check, run on ~6,000 random graphs (n ≤ 50) per claim, of statements I derived rather than found in a paper. The scripts are in `check/` beside this file (copied from the session scratchpad, which was temporary).

## Answers in brief

1. Building the supergraph for a fixed order is the same problem as **symbolic Cholesky factorization** of A in rank space. Our `up(r)` row is column r of L, and `down(r)` is row r of L. Neither LAGraph nor GraphBLAS has a routine for it: LAGraph at d01064de has no elimination tree, symbolic factorization or nested dissection (a grep of `src/`, `experimental/` and `include/` finds nothing).
2. One GraphBLAS formulation does O(|G+|) work and produces each arc exactly once. It needs no mask against a growing result, and it costs one `mxm` plus one `select` per elimination-tree level. Its precondition is that ranks are a postorder of the elimination tree, which is the rank order the CCH survey already recommends. Europe's tree height is at most 1,232 (DSW Table 5), so a full build of Europe would take about 1.2k GraphBLAS call pairs.
3. Fill never crosses between sibling subtrees. In a postorder, a subtree is a contiguous range of ranks, and reordering the ranks inside one range changes only that range's rows (checked here). The supergraph can therefore be built, and repaired, one subtree batch at a time.
4. Incremental updates under a fixed order are a solved problem (Davis & Hager 1999): an inserted edge changes only the rows on one elimination-tree path. Each kind of change costs differently:
   - **Deletes change no topology.**
   - **An edge to an ancestor** adds at most one path segment of arcs.
   - **An edge across sibling cells** can add about n·(a few percent) arcs: on ND-ordered grids it added 4–7% of all arcs (measured below). Such an edge should mark the subtree at its endpoints' LCA for a local re-dissection instead.

## 1. The elimination game as matrix operations

Notation: B = A(π, π) (A in rank space), U0 = triu(B, 1) (original upward arcs), and U = the upward rows of the filled graph G+. `up(r)` is row r of U, and `down(r)` is column r of U.

- **Fill-path theorem.** (i, j) is in G+ iff G has a path from i to j whose interior vertices all rank below min(i, j). Source: Rose, Tarjan & Lueker 1976; also stated in the proof of Davis & Hager 1999, Prop. 3.2.
- **Elimination tree.** parent(r) = min up(r). Liu's algorithm computes it from A alone, without fill, by path compression in O(nnz·α(n)) (Davis 2006 §4.1). C does this (`cch.c:312-375`). Rust skips the pass and reads the parent off the finished row instead (`order.rs:20-24`, `order.rs:426-428`).
- **The tree as a spanning forest (checked here).** The subtree at j is the connected component of j in G restricted to ranks ≤ j. Give each edge the weight max(rank i, rank j) and take any minimum spanning forest (MSF). A union-find over the n−1 forest edges then rebuilds the elimination tree exactly; ties between equal weights do not matter. LAGraph has a Borůvka MSF, `LAGraph_msf` (`experimental/algorithm/LAGraph_msf.c:20`, marked "not ready for src"), so this is a parallel GraphBLAS route to the tree. Liu's sequential pass is already near-linear and needs 2n words of memory, so the MSF route only adds parallelism.
- **Rows by child absorption.** up(r) = U0(r,:) ∪ ⋃ over children c of (up(c) \ {r}). Both engines build rows this way today (`order.rs:362-437`, `cch.c:419-497`). The same recurrence is Algorithm 1 in Davis & Hager 1999 §3, and the CCH survey's Algorithm 3.1 (Bläsius et al. 2025, p.10).
- **Columns by row subtree.** down(s) is the union of the elimination-tree paths from each r with U0(r, s) up to s. Ribizel & Anzt 2023 (Ginkgo) compute it row-parallel: the tree and its postorder on the CPU, then for each row a count pass, an allocation, and a fill pass. Each row splits into the disjoint path segments [j_k, LCA(j_k, j_{k+1})).
- **Counts before building.** Gilbert, Ng & Peyton 1994 give all |up(r)| and |down(r)| in O(nnz·α) without building G+. The counts allow exact preallocation and sizing batches by arc count.

### Formulations considered

| Formulation | Work | GraphBLAS calls | Verdict |
|---|---|---|---|
| **A. Tagged frontier** (below) | O(\|G+\|), each arc made once | ≤ tree height, `mxm`+`select` each | **use** |
| B. Child absorption by tree level: rows of height h = select_offdiag(Πₕ ⊕.⊗ U(children,:)) ∪ U0 | O(\|G+\|) | tree height | works, but reads rows of a matrix it is still writing |
| C. Eliminate independent sets, F = H(:,I)·H(I,:) | Σ\|up(x)\|² = number of triangles | few | rejected: Europe has 578M triangles vs 73.9M arcs (DSW Table 4), 7.8× the work |
| D. (min, max) bottleneck closure | dense n² | — | rejected |

Formulation A is the Ribizel–Anzt segment decomposition rewritten as a frontier loop (checked here: every arc is produced exactly once, with no duplicates).

**Pitfall (B, and any "assign into U each level" scheme).** Inserting into a matrix queues pending tuples, and the next operation that reads that matrix assembles them all (GraphBLAS User Guide, `GrB_examples.tex:170-176`). Reading U on every level could therefore cost O(H·|G+|). Collect each level's output separately and assemble once, with `GrB_Matrix_build` or `GxB_Matrix_concat` (`GraphBLAS.h:4393`).

## 2. Pseudocode: from π and A to U

```
B   = GrB_extract(A, rows = rank_to_node, cols = rank_to_node)    // rank space
U0  = GrB_select(B, GrB_TRIU, 1)                                   // r < s
parent = elimination_tree(U0)          // Liu on columns of U0 (host, O(nnz α)), or the MSF route
post   = postorder(parent)             // relabel the ranks; the fill is the same (survey p.8)
U0  = relabel(U0, post); parent = relabel(parent, post)
Pt  = GrB_build({ (parent[c], c) })    // n-1 tuples: moves an entry from rank c to its parent
// tag: each U0 entry (r, s) carries the next larger row r' in column s, or s if there is none
F   = U0 valued with tags              // one host pass over sorted columns, O(nnz)
parts = [U0]
while nvals(F) > 0:                                    // ≤ tree height iterations
    F = GrB_mxm(Pt, F, semiring = ANY_SECOND)          // (c, s, tag) -> (parent[c], s, tag)
    F = GrB_select(F, row_below_value)                 // user IndexUnaryOp: keep i < tag
    parts.push(F)                                      // disjoint from all earlier parts
U   = build_or_concat(parts)                           // one assembly, no dedup needed
Ut  = GrB_transpose(U)                                 // the downward rows, if kept
```

Inserting edge (u, v) with u < v is the update for adding pattern w = e_u + e_v (Davis & Hager 1999, Thm 4.1). Only rows on the new tree path from u change, and the cost is proportional to the entries that change.

```
if v in up(u): return                                   // already an arc
if v is a tree ancestor of u:                           // common case: edge within one cell or to an ancestor separator
    j = u
    while j != v and v not in up(j): up(j).insert(v); j = parent[j]
    // the tree is unchanged; this stop rule holds because up(c)\{parent} ⊆ up(parent) (checked here)
else:                                                   // the endpoints are in sibling subtrees
    // new path = P(u) ∪ P(v) merged by rank; rows at or above old LCA(u, v) are unchanged (checked here)
    // run Davis–Hager Alg. 3 up that path, or mark subtree(LCA) dirty for local re-dissection (§3)
```

## 3. Nested-dissection tree structure

- Every arc joins a tree ancestor to its descendant, so two sibling subtrees share no arc (survey Lemma 2, p.9). The arcs inside a cell do not depend on any other cell (survey p.7).
- In a DFS postorder, every subtree occupies a contiguous rank range (survey p.8, Fig. 4).
- **Batches (a).** The rows of a subtree depend only on that subtree's edges, A(T[a], :). Independent subtrees can be built in parallel on the background builder, and the separator rows come last. In formulation A, a batch is "start the frontier from the U0 rows in range". Its outputs land on unique positions, some of them in ancestor rows, so batches merge without coordination.
- **Local repair (b).** Reordering ranks within one postorder subtree range leaves every row outside the range bit-identical (checked here). Re-dissecting the subtree at an inserted edge's LCA is therefore a rewrite of one row block. If the LCA is the root, the whole graph is rebuilt.

Fill added by one edge on 2-D grids, measured here with a geometric ND in Python:

| grid n | arcs | tree height | cross-top-separator edge (median / max) | edge to an ancestor (median / max) | new node at the top, degree 3 |
|---|---|---|---|---|---|
| 256 | 2,446 | 46 | 178 / 238 (7.3%) | 6 / 18 | 46 |
| 1,024 | 15,048 | 94 | 766 / 1,012 (5.1%) | 15 / 41 | 94 |
| 4,096 | 85,014 | 190 | 3,722 / 4,529 (4.4%) | 22.5 / 81 | 190 |

An edge across the top separator adds roughly 0.9·n arcs, because both endpoints' separator chains become fully connected to each other. An edge to an ancestor adds at most one path segment. A new node placed at the top adds exactly one arc per vertex on the union of its neighbors' root paths. Road networks and other graphs: unmeasured.

## 4. Incremental and MVCC behaviour

- **Edge or node delete: no change to the topology.** Old G+ is still chordal, still has π as a perfect elimination order, and still contains the new G. The removed arc's weight becomes +∞, which is how the code already seeds a closed edge (`customize.rs:51-56`). Shrinking the supergraph would need a multiplicity count on each arc (Davis & Hager Alg. 2), which is a per-arc memory cost. Recommendation: never shrink in place; reclaim the arcs at the next rebuild.
- **Edge insert.** Use the ancestor walk or the dirty-subtree rule above. Today an edge with no arc returns `NeedsRebuild` (`customize.rs:227-229`).
- **Node insert.** A dense rank space has no free slot inside a range. Possible placements:
  - Append the node at the top as a new root. Cost per the table: one arc per vertex on the union of its neighbors' root paths.
  - Reuse a deleted node's rank when its id is reused.
  - Use sparse rank keys with gaps in a hypersparse matrix. Cost: unmeasured.
- **MVCC.** Fill only ever adds arcs, which maps onto `VersionedMatrix` delta-plus (`versioned_matrix.rs:8-22`). Delta-minus stays empty for U. Replacing a subtree's row block is a copy-on-write of the base matrix, O(nnz U) per publish, unless U is kept as row tiles (`GxB_Matrix_split`/`concat`). Cost: unmeasured.

## 5. Customization as masked `mxm`, and memory

**Customization formula.** Take a batch X of apex ranks that all have the same tree height:

```
Wup⟨struct(U), accum=MIN⟩   = Wdown(X,:)ᵀ  min.plus  Wup(X,:)     // y→z via x: y→x + x→z
Wdown⟨struct(U), accum=MIN⟩ = Wup(X,:)ᵀ    min.plus  Wdown(X,:)   // z→y via x: z→x + x→y
```

**Why one batch per height level is correct.** The arc (x, y) is improved only by apexes below both ends, and those are descendants of x, which have smaller height. So processing batches in increasing height order matches the module's bottom-up invariant (`customize.rs:77-85`). The MIN accumulator also removes the race on same-arc writes that blocks parallelising today (`customize.rs:251-259`).

**Cost.** Work equals the number of triangles, and there are as many sequential steps as the tree is tall. Counting triangles per arc is `plus.pair` under the mask U, as in LAGraph's TriangleCount.

**Tie rule.** The rule (an original edge beats a tie, otherwise the lowest apex wins; `customize.rs:66-75`) needs a lexicographic (weight, apex) MIN. That means either a user-defined type and monoid (JIT) or a second positional pass. Both are unmeasured.

**Memory, derived from sizes rather than measured.** Today's topology costs 16 B per arc (8-byte `Rank` in both `upward` and `downward`), plus 2·8(n+1) B of offsets, plus 16 B per node for the two order maps (`order.rs:44-46`, `order.rs:129-148`). U plus Uᵀ as iso-valued boolean GraphBLAS matrices with 32-bit indices would cost 8 B per arc plus 2·4(n+1) B. GraphBLAS v10 uses 32-bit indices "when possible" (ChangeLog, v10.0.0), and a test on main asserts 32-bit offsets and column indices on a default matrix (`csr.rs:277-278`). For Europe's 73.9M arcs this is about 1.18 GB today versus about 0.59 GB as GraphBLAS matrices.

**Caveat for weights (out of scope).** Each GraphBLAS matrix stores its own copy of the pattern. Four weight matrices would repeat it four times, where today's flat slot arrays share one pattern. A single matrix with a 24-byte struct value type keeps one pattern but gives up the built-in kernels.

## Open and unmeasured

- GraphBLAS call overhead with up to 1.2k levels per batch, multiplied by the number of batches.
- Whether U actually gets 32-bit indices at production sizes.
- How large the fill from cross-cell inserts gets on real graphs, and the dirty fraction at which a full rebuild becomes cheaper.
- Whether the user-defined tie-break monoid is fast enough.

## Sources

- Rose, Tarjan & Lueker, "Algorithmic aspects of vertex elimination on graphs", SIAM J. Comput. 5(2):266–283, 1976.
- Liu, "The role of elimination trees in sparse factorization", SIAM J. Matrix Anal. Appl. 11(1):134–172, 1990. https://epubs.siam.org/doi/abs/10.1137/0611010
- Gilbert, Ng & Peyton, "An efficient algorithm to compute row and column counts for sparse Cholesky factorization", SIAM J. Matrix Anal. Appl. 15(4):1075–1091, 1994. https://www.osti.gov/biblio/10177954
- Davis & Hager, "Modifying a sparse Cholesky factorization", SIAM J. Matrix Anal. Appl. 20(3):606–627, 1999. https://people.clas.ufl.edu/hager/files/update.pdf
- Davis, *Direct Methods for Sparse Linear Systems*, SIAM 2006, Ch. 4 (§4.1 elimination tree, §4.4–4.5 counts).
- Lipton, Rose & Tarjan, "Generalized nested dissection", SIAM J. Numer. Anal. 16(2):346–358, 1979 (O(n log n) fill and O(n^{3/2}) work for planar graphs).
- Ribizel & Anzt, "Parallel Symbolic Cholesky Factorization", SC-W 2023. https://www.csm.ornl.gov/srt/conferences/Scala/2023/paper_3.pdf
- Dibbelt, Strasser & Wagner, "Customizable Contraction Hierarchies", ACM JEA 21, 2016. https://arxiv.org/abs/1402.0402 (Tables 4–5)
- Bläsius, Buchhold, Wagner, Zeitz & Zündorf, "Customizable Contraction Hierarchies – A Survey", 2025. https://arxiv.org/abs/2502.10519
- SuiteSparse:GraphBLAS v10.5.0 `Doc/ChangeLog`, `Doc/UserGuide/GrB_examples.tex`, `Include/GraphBLAS.h`; LAGraph d01064de `experimental/algorithm/LAGraph_msf.c`.
