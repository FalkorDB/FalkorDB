# U4: GraphBLAS v10.5.0 and LAGraph d01064de building blocks for CCH ordering, supergraph, and incremental upkeep

Revisions read: Rust `origin/main` 7df8e5887. LAGraph d01064de and GraphBLAS v10.5.0 (ea79e354), the gitlinks named in `deps/native-deps.lock`. C reference `origin/master` in civil-kitten. "LG" means LAGraph, "GB" means the GraphBLAS source tree, and "derived" marks reasoning of my own with no outside source.

## Bottom line

- **Neither library has nested dissection, a vertex separator, an elimination tree or symbolic factorization.** A grep of both trees for nested dissection, elimination tree, etree, separator, METIS and symbolic Cholesky found only the Fiedler-vector code. Every piece below would have to be put together by us.
- **For multilevel ND, every step except refinement has a GraphBLAS/LAGraph form:**
  - matching and coarsening
  - contraction (`S·A·Sᵀ`)
  - BFS and connected components (the only stable ones)
  - a spectral split
  - max-flow/min-cut
  - bipartite maximum matching, which the König edge-to-vertex separator step needs

  Nothing in LAGraph does FM/KL refinement, which moves one vertex at a time.
- **There is no work-efficient linear-algebra form of the elimination game.** The usable GraphBLAS form is one masked `mxm` + `select` per elimination-tree level. Restricted to "dirty" rows, the same kernel performs incremental edge insertion. Deletions need no change to the topology.
- **The Rust wrappers are narrow:** boolean `ANY_PAIR` products, eWise ops and a few selects. Most of what this design needs is new: general semirings, extract by index vector, positional apply, reduce-to-vector, sort, and diag.

## 1. LAGraph algorithms

"Exp" means `experimental/algorithm/`. "Bound" means the function is declared in `graph/src/graph/graphblas/lagraph{,x}_bindings.rs`. `liblagraphx.a` is built and linked (`native-deps/src/recipes/lagraph.rs:56-59`, `graph/build.rs:153`). No CCH code calls any of these.

| Algorithm | Status | What it computes, and its inputs and outputs | Cost and limits | Determinism | Rust |
|---|---|---|---|---|---|
| `LAGraph_MaximalMatching` | exp | Luby-style maximal matching on the line graph. It takes the incidence matrix `E` (n×e) and `Eᵀ`, and returns a bool vector over edges. Kinds: random, heavy (score × weight) or light (MaximalMatching.c:15-43, 176-186). | O(e) space (:42). Retries up to 50 times on a stall (:78, 243-249). It needs `LAGraph_Incidence_Matrix`, which asserts the graph is undirected with no self-edges and makes three e-length arrays (LG experimental/utility/LAGraph_Incidence_Matrix.c:89-95, 140-142). | Seeded; uses MIN/MAX semirings plus pattern-only `any_one_bool`. | bound :334, unused |
| `LAGraph_Coarsen_Matching` | exp | One coarsening level. Outputs: the coarse adjacency, the parent vector, and the new/old labels (Coarsen_Matching.c:15-56). The contraction is `S·A` and then `(S·A)·Sᵀ` with `PLUS_TIMES`, which sums weights (:489, :501), followed by `DeleteSelfEdges`. | O(n+e) (:58). Asserts the graph is undirected (:337-366) with no self-edges (:329). Carries no vertex weights; we would add `S·w`. Its TODO says it still uses pack/unpack (:12-13). Has a custom IndexUnaryOp (:92-101, 173). | Seeded; `MIN_SECONDI`. | bound :343, unused |
| `LAGraph_MaximalIndependentSet` | exp | Luby MIS. The optional `ignore_node` vector restricts it to a subset (MaximalIndependentSet.c:64-76). | Expected O(log n) rounds, each one masked `vxm`/`mxv` (:210-292). Needs a symmetric matrix, cached `out_degree` and no self-edges (:99-114). Stall retries ≤ 32 (:304). | Seeded; MAX. | bound :297, unused |
| `LAGraph_Hdip_Fiedler` (+ `LAGraph_Laplacian` in the same file :250) | exp | Fiedler vector by Householder deflation and inverse power iteration with PCG (Wu, Song, Zhang 2014; Hdip_Fiedler.c:538-570). Inputs: the Laplacian L, its ∞-norm, `kmax=[20,50]`, tolerances (:624-637). | FP32 with dense n-vectors (:660-684). The method is for a connected graph. | FP32 sums; whether results vary with thread count is unverified. | bound :448/:428, unused |
| `LAGr_MaxFlow` | exp | Synchronous parallel push-relabel (Baumstark et al. 2015; Peries & Davis HPEC'25). Outputs: the flow value, and optionally the flow and residual matrices (MaxFlow.c:18-52). | Needs GraphBLAS v10 (:59). Built on user-defined types (:224-599, 776-930). The flow matrix is "very expensive" (:35-37). | FP64 | bound :610, **used** (algo_procedures.rs:3293) |
| `LAGraph_MinCut` | exp | From MaxFlow's residual R: one BFS from the source gives S and S̄, and two `mxm` with diagonal matrices give the cut edges (MinCut.c:104-119). | This is an **edge** cut. Getting a vertex separator needs a node-split network or a König step. | follows BFS | **not bound** |
| `LAGr_MaximumMatching` | exp | Maximum bipartite matching (Azad–Buluç; MaximumMatching.c:22-37). This is the König step that feral-metis does with Kuhn's O(V·E) (feral-metis separator.rs:10-18). | Takes a raw A and Aᵀ. Pack/unpack TODO (:19). | — | bound :550, unused |
| `LAGr_PartitionQuality` | exp | Coverage and performance of a labeling. Builds the cluster×node matrix straight from the label vector with `GxB_Matrix_build_Scalar_Vector` + `GxB_USE_INDICES` (PartitionQuality.c:118-122), then `C⟨C⟩ = C·A` (:147). | one `mxm` | exact counts | bound :525, unused |
| `LAGraph_MultiSourceBFS` | exp | Level and parent matrices for several sources at once, using one masked `mxm` per level (MultiSourceBFS.c:172-207). | Bitmap/full hint gives **nsrc×n memory** (:102, :150). | Parent uses `ANY_SECONDI` (:93-96), so it is not deterministic. | bound :378, unused |
| `LAGr_BreadthFirstSearch` | **src** | Push/pull BFS (Beamer, Yang et al.). Gives level and/or parent (template :18-34, 184-186). | Bitmap/full dense outputs (:149, :175). | Level is exact. Parent uses `ANY_SECONDI` (:140-143), and ANY "picks x or y arbitrarily" (GB Doc/UserGuide/GrB_objects_BinaryOp.tex:32). | bound :512 (the Extended variant is used, algo_procedures.rs:1162) |
| `LAGr_ConnectedComponents` (FastSV7) | **src** | Component id = smallest node id in the component (LAGr_ConnectedComponents.c:25-41). | Samples rows when `nvals > 8n` (FastSV7.c:337-338). **It unloads `G->A` while it runs, so G must not be shared** (:42-46, 479-491). | MIN | bound :522, **used** (algo_procedures.rs:910) |
| `LAGraph_KCore` / `KCore_All` / `KCore_Decompose` | exp | Peeling, one `vxm` per round (KCore.c:126-145; AllKCore.c:126-155). | Symmetric, no self-edges. | exact | bound :305-319, unused |
| `LAGr_TriangleCount` | **src** | Six masked-product methods. The default is Sandia_LUT, `sum((L·Uᵀ).*L)` (TriangleCount.c:31-37, 173-176). | Optionally permutes `A(P,P)` first (:279-292). | exact | bound :605 |

Notes on what these mean for ND (derived):

- An **elimination order whose first ranks have no fill** can come from KCore peeling. A vertex removed while it has ≤1 remaining neighbour creates no fill, so the trees hanging off the 2-core can take the lowest ranks, in peel-round order.
- The **coarsening heuristics differ.** LAGraph's "heavy" matching multiplies a random score by the edge weight (MaximalMatching.c:181-183). METIS/feral visit vertices in degree order and pick the heaviest neighbour (feral coarsen.rs:3-13). How their quality compares is unmeasured.
- **Refinement has no counterpart.** Move gains can be computed in bulk as `A·side`, but FM's one-move-at-a-time loop is sequential.

## 2. GraphBLAS operations

- **Permute into rank order.** Use `GrB_extract(T, -, -, A, P, n, P, n)`, as LAGr_TriangleCount.c:285-288 does. The v10 form takes index lists as `GrB_Vector`: `GxB_Matrix_extract_Vector` (GB Include/GraphBLAS.h:5482-5491). For a large, non-contiguous I it builds an inverse map (GB Source/extract/GB_I_inverse.c:10-15), and the result may come back jumbled (GB_subref.c:279). An equivalent form builds P with `GxB_Matrix_build_Scalar_Vector` and computes `P·A·Pᵀ` as two `mxm` calls.
- **Quotient (coarse) graph.**
  - Build `S` (part×node) from a label vector (PartitionQuality.c:118-122).
  - The coarse graph is `S·A·Sᵀ` with `PLUS_TIMES` (Coarsen_Matching.c:489, 501), then `select OFFDIAG`. Coarse vertex weights are `S·w`.
  - Boundary vertices are the columns of `S·A` that hold another part's entry.
  - Derived: one call keeps the intra-part edges for **every** part at a recursion level. Compute `C⟨A,struct⟩ = Sᵀ ANY_PAIR S` as a masked dot product, where each column of S has one entry. That lets ND run one GraphBLAS call per level instead of one per part.
- **Select by index.** Built-in positional selects: `GrB_TRIL/TRIU/OFFDIAG/ROWLE/ROWGT/COLLE/COLGT` and the `GrB_ROWINDEX/COLINDEX` apply ops (GraphBLAS.h:1398-1417).
  - To keep a set of rows without renumbering, multiply by `GrB_Matrix_diag(v)`, as MinCut.c:114-119 does.
  - To keep them and renumber, use extract.
  - Arbitrary predicates need `GxB_IndexBinaryOp_new` (:3336).
- **Elimination game / symbolic Cholesky.** I found no work-efficient GraphBLAS formulation. The options are below. The rows are U(v) = ranks above v joined to v, and the parent is p(v) = min U(v).
  1. *Sequential tree union.* `U(v) = A⁺(v) ∪ ⋃_children (U(c)∖{v})` in rank order. This is what C does (cch.c:312-376 builds the elimination tree with union-find in O(nnz·α); cch.c:419 builds the supergraph) and what Rust does (order.rs:381-430). GraphBLAS's only role would be the row iterator in and `build` out. No CSR copy is needed.
  2. *One round per tree level* (derived). Group rows by height in the elimination tree. Let T_h map each child to its parent at height h. Then `U[h] = TRIU(A⁺[h] ∪ T_h·U, 1)`, which is one `mxm` + one `eWiseAdd` + one `select`. The number of rounds equals the tree height, which is unmeasured on our graphs. The parent vector comes from `apply(COLINDEX)` followed by a MIN reduce over rows. Depths come from pointer jumping, the same trick FastSV uses for grandparents (FastSV7.c:67-80).
  3. *Repeat until nothing changes* (derived). `F ← F ∪ TRIU(Fᵀ·F, 1)`, masked with `¬F`. Its smallest fixed point is the filled graph, by the fill-path characterization (Rose–Tarjan–Lueker 1976). The round count reaches the tree height on a chain whose ranks only increase, and each round can cost the whole lower-triangle count. Use it only as a test oracle.
  4. *Parallel literature.* Gilbert & Hafsteinsson (1990) take O(log² n) time on m\* processors, building the elimination tree first and the fill second ([record](https://iris.landsbokasafn.is/en/publications/parallel-symbolic-factorization-of-sparse-linear-systems/)). That is a PRAM algorithm, not GraphBLAS.
- **Lower triangles (customization).** The lower triangles of an arc (u,w) are the v with U(v,u) and U(v,w), so `C⟨U⟩ = Uᵀ⊕.⊗U`. This is the masked dot-product kernel (GB Source/mxm/GB_AxB_dot3.c), the same family as Sandia_LUT. `PLUS_PAIR` gives a triangle count per arc, i.e. the customization work. `MIN_PLUS` gives one relaxation pass, but basic customization needs a bottom-up order. That means one round per level, or one task per separator; the CCH survey parallelizes by separator-tree tasks (arXiv 2502.10519 §3.2.3). The weight arrays are out of scope here.

## 3. What the Rust wrappers expose

**Already there:**
- `Matrix<bool|u64|f64>` with new/set/get/build, `transpose`, `element_wise_add/multiply` and `set_pattern` (matrix.rs:1062-1135)
- `lmxm/rmxm`, which are unmasked `ANY_PAIR_BOOL` only (:1140-1179)
- `count_in_rows`, a `PLUS_PAIR` mxv (:852)
- `drop_diagonal`, a select with OFFDIAG (:1019)
- row iterators with seek (:2080-2202)
- `Effective`/`effective_fwd`, which gives `(m∖dm)∪dp` in one call and borrows the base when the deltas are empty (versioned_matrix.rs:98-135, tensor.rs:1060)
- `CsrExport` (csr.rs)
- minimal `Vector<bool|u64|f64>`
- raw LAGraph FFI, plus a private `LAGraph_Graph` helper that handles borrowed matrices (algo_procedures.rs:358-395)

Rust also pins every matrix to sparse or hypersparse, row-major (matrix.rs:457-482).

**New:**
- masked `mxm/mxv/vxm` over any semiring (MIN_PLUS_FP64, PLUS_TIMES, ANY/MIN_SECONDI) with descriptors
- `GxB_Matrix_extract_Vector`, `GxB_Matrix_build_Scalar_Vector` + `GxB_USE_INDICES`
- general `select` (TRIU/TRIL/ROW*/COL*/VALUE*)
- positional apply
- reduce-to-vector with MIN/PLUS
- `GxB_Vector_sort` (GraphBLAS.h:7388)
- `GrB_Matrix_diag`
- vector extract/assign for pointer jumping
- a shared `LAGraph_Graph` owner type
- bindings for `LAGraph_MinCut`

## 4. Performance

- **JIT.** Production runs `GxB_JIT_RUN`. Only PreJIT kernels run fast, and anything else falls back to generic kernels (matrix.rs:172-198). An arm64 A/B against `JIT_OFF` measured +6% to +87% (matrix.rs:183-185). Main vendors 224 PreJIT kernels, including the MaxFlow/MSF user-type kernels (`build/graphblas/PreJIT/`). Any new CCH semirings and custom index ops (e.g. Coarsen_Matching.c:92-101) would need a Linux `native-deps prejit` harvest (CLAUDE.md).
- **Threads.** GraphBLAS is OpenMP-parallel inside each call. Rust sets the thread count globally (matrix.rs:227-238). A builder thread could cap itself by engaging a `GxB_Context` (GB Doc/UserGuide/GrB_context.tex:39-46). Only tests do that today (csr.rs:111-145).
- **Concurrent readers** may share a matrix only if it has no pending work (GrB_init.tex:254-259). Every published matrix must be waited. FastSV7 must get a private dup.
- **Memory.** With dimensions ≤ 2³¹ and nnz ≤ 2³², GraphBLAS uses 32-bit indices (GrB_get_set_integers.tex:7-20). An iso-bool upward pattern costs about 4 B/arc. Today's `Topology` holds 8-byte `Rank`s in both directions (order.rs:46, 129-160): about 16 B/arc against about 8 B/arc for U+Uᵀ. That figure is derived, unmeasured. Measured analogue: 6.44 B/id against 11.69 B/id for 32-bit against 64-bit columns (tensor.rs:159-162).
- **Hypersparse.** Space is O(|A|), and n = 2⁶⁰ works (GrB_concepts.tex:54-58). Making delta matrices hypersparse measured 18× faster than sparse (matrix.rs:624-629). LAGraph's bitmap/full temporaries cost O(n), or O(nsrc·n), at node-capacity dimensions.
- **Scale.** The survey says computing the order takes about two orders of magnitude longer than contraction, and about 6 minutes for a continental network (§5.1.1). Ordering, not the supergraph, is the cost to beat. GraphBLAS per-call overhead × the number of levels is unmeasured.

## 5. Keeping it current under MVCC

Validity invariant (derived). Rows U(v) give a valid CCH topology when all of these hold:

- U(v) ⊆ {ranks > v}
- A⁺(v) ⊆ U(v)
- U(v)∖{p(v)} ⊆ U(p(v))

Together these make every U(v) a clique. The filled graph is the smallest family that satisfies them.

What each change needs:

- **Edge or node delete.** The invariant still holds, so the topology needs no work and the weight becomes ∞. The topology drifts toward a superset of what the graph needs.
- **Edge insert (u,w).** Add w to U(u) and push U(u)∖{p} up the parent chain, which may have a new parent, until a row gains nothing. This is kernel (2) masked to the dirty rows and with `¬U`, so a full build is just "insert everything into an empty U". Davis & Hager report symbolic updates in time proportional to the entries that change ([SIMAX 1999](https://people.clas.ufl.edu/hager/files/update.pdf), per its abstract; I did not check the body).
- **Node insert.** Either choice is correct; neither has been measured:
  - At the top rank, the new node joins every ancestor row of each of its neighbours.
  - At a leaf rank, its neighbours must become a clique, which costs about deg² inserts.

  Rank keys with gaps in a hypersparse rank space would avoid renumbering, at the price of 64-bit indices.
- **Publishing.** Keep U as a `VersionedMatrix<bool>` that only adds, i.e. base plus `dp` (versioned_matrix.rs:710-760). `Matrix::clone` is shallow (versioned_matrix.rs:90-96), so a batch must dup or add a delta, and wait before publishing.
- **ND order.** I found no incremental method. The survey does not cover topology updates (§1). The practical route is a placement rule plus a periodic full rebuild as a new generation (online_build.rs:4-8).
