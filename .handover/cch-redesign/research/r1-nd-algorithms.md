# R1: Nested dissection options for a GraphBLAS-native, incremental CCH order

Scope: the literature on node-separator / nested-dissection (ND) algorithms, how each one maps onto GraphBLAS and LAGraph, and how order quality changes CCH cost. I read the arXiv PDFs as extracted text, LAGraph at `d01064de` (`deps/LAGraph` in the parking-lot-pin worktree), the GraphBLAS v10 header, and feral-metis at `f127189`. **I measured nothing on FalkorDB.** Every number below comes from the cited paper.

## Bottom line

- **Our current algorithm is the fastest option and gives the worst orders.** The ND we run today is feral-metis, a METIS-style multilevel ND: SHEM matching with a 2-hop fallback, FM refinement, and a separator taken by minimum vertex cover (`feral-metis/src/lib.rs:14-20`). That family is the fastest in the CCH literature, and its orders are the worst. On continental road graphs, METIS orders give 2.9x the triangles, 2.2–2.4x the customization time and 3.2–4.0x the query time of InertialFlowCutter. On a 0.4M-node graph the gap shrinks to about 1.4x ([IFC, Table 2, p.10](https://arxiv.org/pdf/1906.11811v2)).
- **No published GraphBLAS or linear-algebra ND exists.** I searched and found no nested-dissection or node-separator implementation of that kind. LAGraph has the building blocks, all under `experimental/` with open TODOs: matching-based coarsening, a Fiedler-vector solver, push-relabel max flow, min cut, bipartite maximum matching, label propagation and k-core.
- **Three candidates are realistic on GraphBLAS** (section 3):
  - **(A) Multilevel ND built from mxv/mxm.** Expect about METIS quality, the same as today.
  - **(B) A plus max-flow refinement on a band around each separator.** Expect quality between METIS and KaHIP. Unmeasured.
  - **(C) BFS level structure plus max flow**, a "coordinate-free Inertial Flow". This is the simplest. Its quality on graphs without coordinates is an open question in the literature.
  - **Not recommended:** pure spectral and pure BFS-level separators. Both are measurably worse than multilevel.
- **On non-road graphs the choice of order algorithm matters much less than the graph's shape.** For any order, the supergraph contains a clique the size of the minimum balanced separator, and that clique lies inside at least n/3 search spaces ([CCH, Lemma 2, p.6](https://arxiv.org/pdf/1402.0402v5)). Measured treewidth upper bounds are 0.5–27% of n for social, web and knowledge graphs, against under 0.1% for road graphs ([Maniu et al.](https://arxiv.org/abs/1901.06862)). The redesign needs a size guard, not only a better separator.
- **There is no incremental ND for CCH in the literature I found.** The nearest precedents:
  - a fixed-order symbolic Cholesky update, which changes only the entries along one elimination-tree path;
  - local repartitioning (diffusion, band refinement).

## 1. Algorithm families and how they fit GraphBLAS

| Family | Core kernel | GraphBLAS / LAGraph fit | Quality evidence |
|---|---|---|---|
| BFS level structure (George 1973; [George & Liu 1978](https://epubs.siam.org/doi/10.1137/0715069), Sparspak) | BFS from a pseudo-peripheral node; pick a middle level | Direct: `LAGr_BreadthFirstSearch` gives levels. | Spectral separators are about **half the size** of Sparspak level-structure separators on the Boeing-Harwell set ([Pothen, Simon & Liou 1990, PDF p.15](https://www.cs.purdue.edu/homes/apothen/Papers/spectral-partition.pdf)). |
| Spectral bisection / spectral ND | Fiedler vector via Lanczos/LOBPCG; each step is one SpMV, O(e) per Lanczos step (same paper, PDF p.12) | Direct: SpMV is `GrB_mxv`. LAGraph has `LAGraph_Hdip_Fiedler` (Householder deflation plus inverse power iteration with PCG), still marked "TODO: rename this method and add to src" (`LAGraph_Hdip_Fiedler.c:4-6, 546-550`). | Spectral ND needs **37% more** factorization operations than multilevel ND over 21 matrices (1.68 vs 1.23 Tflop), and is worse on all 21. Multilevel spectral bisection is 10–35x slower than METIS, with cuts up to 70% larger ([Karypis & Kumar 1998, PDF pp.3, 21, 25-26](https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf)). On irregular (social/web) graphs, Sphynx's LOBPCG cuts are about **3.3x** ParMETIS's; on meshes about 1.23x ([Sphynx](https://arxiv.org/abs/2105.00578)). |
| Multilevel (METIS, Scotch, KaHIP; parallel: ParMETIS, PT-Scotch, mt-Metis) | Coarsen by matching, bisect the coarse graph, refine (KL/FM) while uncoarsening, convert edge cut to node separator by König's theorem | Coarsening fits: the coarse graph is `S·A·Sᵀ` (`LAGraph_Coarsen_Matching.c:489,501`). LAGraph's matching needs an n×e incidence matrix plus its transpose (`LAGraph_MaximalMatching.c:30,42`; `Coarsen_Matching.c:383-415`), which is O(e) extra memory. FM refinement is sequential (one move at a time, rollback). The batched version fits: gains come from one `mxv` against a side-indicator vector. König: `LAGr_MaximumMatching` (Azad–Buluç, `LAGr_MaximumMatching.c:22-24`). | Batched "Greedy" refinement gives separators **6.1% larger** than serial FM. Greedy+SFM matches FM ([mt-Metis, pp.8-9](https://dlasalle.github.io/publications/mtmetis2015nd.pdf)). mt-Metis is within 1.0% fill of serial ND-Metis at up to 10x speedup on 16 cores (same, p.2). Matching coarsening stalls on star-like / power-law graphs; the fixes are 2-hop matching or size-constrained label propagation ([Buluç et al. survey, PDF p.18](https://arxiv.org/pdf/1311.3144)). feral-metis already uses the 2-hop fix (`coarsen.rs:9-13`). |
| Multilevel + flow refinement (KaHIP node separators; PT-Scotch band graphs) | Max flow on a band around the separator | `LAGr_MaxFlow` (GraphBLAS push-relabel; Peries & Davis, HPEC'25; FP64 only, `LAGr_MaxFlow.c:18-27, 42-43`) plus `LAGraph_MinCut` (one BFS on the residual graph, `LAGraph_MinCut.c:33-34`). The band comes from a multi-source BFS. | KaHIP separators are **10% smaller than Metis** and 62% smaller than Scotch ([Sanders & Schulz](https://arxiv.org/abs/1509.01190)). PT-Scotch refines only a band of width 3 around the separator with no loss of quality ([Chevalier & Pellegrini](https://arxiv.org/abs/0907.1375)). |
| Inertial Flow ([Schild & Sommer 2015](https://aschild.github.io/papers/roadseparator.pdf)) | Sort nodes along a coordinate axis, contract the first and last quarter into terminals, take a min cut ([CCH survey, p.8](https://arxiv.org/pdf/2502.10519)) | Fits: `GxB_Vector_sort` (`GraphBLAS.h:7384-7388`), `LAGr_MaxFlow`, `LAGraph_MinCut`. **Needs coordinates.** | On Europe: 0.45x METIS's triangles and 0.43x its query time, against 0.35x and 0.32x for IFC8. Ordering takes 450 s vs 125 s for METIS (IFC, Table 2). |
| FlowCutter / InertialFlowCutter ([Hamann & Strasser](https://arxiv.org/abs/1504.03812); [Gottesbüren et al.](https://arxiv.org/abs/1906.11811)) | A sequence of incremental unit-capacity max-flow problems; terminals only grow, so the flow is reused. "Piercing" picks the next terminal ([IFC, pp.3-4](https://arxiv.org/pdf/1906.11811v2)). O(c·m) for a largest cut of size c ([FlowCutter, p.1](https://arxiv.org/pdf/1504.03812v3)) | Weak fit. The augmenting-path flow is reused across iterations, and each augmentation is one BFS on the residual graph. That means c sequential BFS passes per cut per source/target pair. Push-relabel (`LAGr_MaxFlow`) recomputes from scratch, and whether it can be warm-started is unknown. | **The best CCH orders** (section 2). Its coordinate-free mode (F3/F20/F100 = 3/20/100 random source/target pairs) won PACE 2016 for tree decompositions ([FlowCutter, p.4](https://arxiv.org/pdf/1504.03812v3)). Also the slowest: F20 took 16,130 s on Europe vs 125 s for METIS (IFC, Table 2). |

Two cheap reductions apply whichever family we pick:

1. **Peel trees and split degree-2 chains, once.** Remove tree parts (the 2-core, `LAGraph_KCore`) and split degree-2 chains before running ND. IFC does this together with a biconnected-component split, once up front ([IFC, p.6](https://arxiv.org/pdf/1906.11811v2)). Exhaustive reductions before ND come out about 6x faster than Metis on road networks, with less fill ([Ost, Schulz & Strash](https://arxiv.org/abs/2004.11315)).
2. **Use the separator hierarchy as the work unit.** Multilevel partition hierarchies and rooted tree decompositions correspond one-to-one ([Hamann & Strasser 2019](https://www.mdpi.com/1999-4893/12/9/198)), so the separator hierarchy is the natural unit for both parallel and incremental work. Cells are independent, which gives task parallelism ([IFC §2.8, p.7](https://arxiv.org/pdf/1906.11811v2); [mt-Metis](https://dlasalle.github.io/publications/mtmetis2015nd.pdf)).

## 2. How order quality turns into CCH cost

Customization time tracks the triangle count, and query time tracks search-space size ([FlowCutter, p.18](https://arxiv.org/pdf/1504.03812v3)). The rows below are from [IFC Table 2 (p.10)](https://arxiv.org/pdf/1906.11811v2). Order times are sequential.

The arc and triangle columns are copied exactly as IFC prints them. **Compare them as ratios only**: the header says ×10⁶, but every value is 10x the same quantity in FlowCutter Table 1 and CCH Table 7 (for example, METIS on Europe is 69.9 there vs 697.4 here). The search-space and treewidth columns agree across the three papers.

| Graph (n) | Order | CCH arcs (IFC units) | Triangles (IFC units) | Customization (ms) | Query (µs) | Order time (s) |
|---|---|---|---|---|---|---|
| Colorado (0.44M) | METIS | 13.7 | 63.9 | 58.8 | 21.1 | 1.8 |
| | IFC8 | 12.5 | 43.9 | 43.4 | 14.9 | 12.9 |
| Cal/Nev (1.9M) | METIS | 65.0 | 364.1 | 310.1 | 47.9 | 9.8 |
| | IFC8 | 58.0 | 217.7 | 225.1 | 21.7 | 79.0 |
| Europe (18M) | METIS | 697.4 | 13,238 | 8,302 | 645.3 | 124.6 |
| | KaHIP 1.00 | 683.3 | 5,745 | 4,170 | 223.7 | 242,681 |
| | Inertial Flow | 674.0 | 5,897 | 4,177 | 280.0 | 450.3 |
| | FlowCutter F20 | 588.3 | 4,624 | 3,527 | 211.4 | 16,131 |
| | IFC8 | 588.6 | 4,587 | 3,509 | 203.8 | 2,448 |
| USA (24M) | METIS | 861.7 | 12,739 | 7,804 | 491.4 | 171.9 |
| | IFC8 | 743.0 | 4,424 | 3,607 | 121.7 | 2,023 |

- **METIS's penalty grows with graph size.** Query time is 1.4x IFC8's on Colorado, 2.2x on Cal/Nev, 3.2x on Europe and 4.0x on USA. Arc counts stay within about 1.2x, so the cost is in the triangles and the search spaces, not in the storage. IFC's verdict: "Metis is ranked last by a large margin" ([p.9](https://arxiv.org/pdf/1906.11811v2)).
- **IFC8 on 16 cores orders Europe in 242 s** ([p.11](https://arxiv.org/pdf/1906.11811v2)), about 2x METIS's sequential time.
- **Grid-like non-road graph (game map):** METIS and KaHIP "seem to be very close… nearly indistinguishable" with basic customization ([CCH, p.30](https://arxiv.org/pdf/1402.0402v5)). METIS even produced fewer triangles there: 602M against 864M for KaHIP. On Europe the same table shows METIS at 1,409M against KaHIP's 578M (CCH, Table 7, p.23). The authors trace this to grid regions with Θ(√n) separators ([p.20](https://arxiv.org/pdf/1402.0402v5)).
- **Social, web and knowledge graphs:** I found no CCH measurement. The theory says these graphs will be expensive under any order:
  - **ND orders are close to optimal.** For graphs with Θ(n^α) minimum balanced separators, an ND order gives an O(1)-approximation of the optimal metric-independent search space (CCH, Theorem 1, p.6). Lemma 2 makes the lower bound unavoidable for any order.
  - **Road graphs have small treewidth.** Roads of about 10⁷ nodes have treewidth around 10² ([CCH, p.2](https://arxiv.org/pdf/1402.0402v5)).
  - **Other graph types do not.** Heuristic treewidth upper bounds ([Maniu, Senellart & Jog, ICDT 2019](https://arxiv.org/abs/1901.06862)):
    - road networks: under 0.1% of n (CA 252, Paris 521);
    - Stack-Math: 11,100 of 1.1M nodes;
    - WikiTalk: 12,843 of 2.4M;
    - YAGO: 79,059 of 2.6M;
    - LiveJournal: 919,532 of 4.0M.
  - **The lower bounds are small, so the true values are unknown.** The orders we can compute in practice give supergraphs of the upper-bound size. Unmeasured for CCH.

## 3. Recommendation: three candidates on GraphBLAS primitives

**A. Multilevel ND in GraphBLAS: the baseline, about METIS quality.**

1. **Peel trees.** Take the 2-core with `LAGraph_KCore`; tree parts are eliminated first and add no fill.
2. **Match.** Each unmatched vertex picks its heaviest unmatched neighbor with a masked `mxv`, plus a second `mxv` to recover the index of that maximum. Pairs that pick each other are matched: the locally-dominant ½-approximation, which needs a logarithmic number of rounds ([Manne & Bisseling](https://webspace.science.uu.nl/~bisse101/Articles/CP75.pdf)). This avoids LAGraph's incidence matrix.
3. **Handle power-law stalls.** Fall back to 2-hop matching or size-constrained label propagation (`LAGraph_cdlp` as a base).
4. **Contract.** The coarse graph is `S·A·Sᵀ`.
5. **Bisect the coarsest graph** sequentially by greedy graph growing ([Karypis & Kumar](https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf)).
6. **Refine while uncoarsening.** Compute gains with `mxv` and move vertices in batches. Expect about +6% separator size over serial FM ([mt-Metis](https://dlasalle.github.io/publications/mtmetis2015nd.pdf)); serial FM on the boundary alone would recover that.
7. **Turn the edge cut into a node separator** by König's theorem with `LAGr_MaximumMatching`. Matching gives separators about 11% smaller than taking the smaller endpoint set ([Pothen, Simon & Liou, PDF p.14](https://www.cs.purdue.edu/homes/apothen/Papers/spectral-partition.pdf)).
8. **Recurse** per cell, using masks or `GrB_extract`.

Expected quality: the same as feral-metis today. That is fine for small and grid-like graphs, and 2–4x slower queries on large road-like graphs (section 2).

**B. A plus a flow-based separator improvement: the quality upgrade.**

1. **Cut out a band.** After each separator from A, take the nodes within BFS distance d of it (multi-source BFS; PT-Scotch uses d=3).
2. **Collapse the rest.** Contract everything outside the band into a source and a sink.
3. **Cut.** Run `LAGr_MaxFlow` and `LAGraph_MinCut`, then convert the cut to a node separator.

This is the KaHIP / PT-Scotch recipe. KaHIP's separators are 10% smaller than METIS's, and its Europe CCH queries are 2.9x faster (224 vs 645 µs). Expected: between METIS and KaHIP. **Unmeasured** on our graphs, and the cost of `LAGr_MaxFlow` on bands is unmeasured too. Node capacities need node splitting, a temporary graph of 2× the band, and `LAGr_MaxFlow` works only in FP64.

**C. Level structure plus max flow: experimental, the cheapest to write.**

1. **Find terminals.** Find a pseudo-peripheral pair s, t with a few BFS passes (as Sparspak's FNDSEP does).
2. **Sort.** Key each node by d(s,v) − d(t,v), the same key FlowCutter uses to break piercing ties ([IFC, p.4](https://arxiv.org/pdf/1906.11811v2)), and sort with `GxB_Vector_sort`.
3. **Contract the ends.** Contract the first and last 25% into terminals.
4. **Cut.** Take a min cut with `LAGr_MaxFlow`, then convert it to a node separator.

When nodes carry point properties, use real Inertial Flow; on Europe that is about 2.2x better than METIS on triangles (table). Without coordinates the quality is unknown: IFC calls transferring its ideas to graphs without geography "an open question" ([p.15](https://arxiv.org/pdf/1906.11811v2)).

**What each candidate offers for incremental updates.** No paper covers incremental ND for CCH. These follow from the sources cited:

- **Deletions.** A deleted edge or node needs no topology change: G − e ⊆ G ⊆ G⁺, so the old supergraph is still valid and only the weights change. Weights are out of scope.
- **Insertions with the order fixed.** An inserted edge that is not already an arc of G⁺ changes the supergraph only along elimination-tree paths. The Cholesky analogue is Davis & Hager's symbolic update, where "the only sets L_j that change are those associated with P(k)", in time proportional to the entries that change ([§4.1, PDF p.7](https://people.clas.ufl.edu/hager/files/update.pdf)).
- **Quality decay.** Order quality decays when new edges cross separators. The repair unit is the smallest separator-hierarchy cell that holds both endpoints: rerun A, B or C on that cell only.
- **Precedents for local repair.** Multilevel diffusion repartitioning (Schloegel, Karypis & Kumar, JPDC 1997) takes less time than partitioning from scratch for a comparable cut. PT-Scotch's band refinement is another.
- **Spectral warm starts** (reusing the last Fiedler vector) are also unmeasured.

**Size guard (all candidates).** Before building the supergraph, estimate the top-level separator size. A separator of size |S| puts a |S|-clique into at least n/3 search spaces (CCH, Lemma 2). If |S| is in the thousands, as on the social and knowledge graphs above, refuse to build or fall back to plain search.

## Sources
- Dibbelt, Strasser, Wagner, *Customizable Contraction Hierarchies*, https://arxiv.org/pdf/1402.0402v5
- Hamann, Strasser, *Graph Bisection with Pareto Optimization* (FlowCutter), https://arxiv.org/pdf/1504.03812v3
- Gottesbüren, Hamann, Uhl, Wagner, *Faster and Better Nested Dissection Orders for CCH* (InertialFlowCutter), https://arxiv.org/pdf/1906.11811v2
- Bläsius et al., *CCH – A Survey*, https://arxiv.org/pdf/2502.10519
- Karypis, Kumar, SISC 1998 (METIS), https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf
- Pothen, Simon, Liou, SIMAX 1990, https://www.cs.purdue.edu/homes/apothen/Papers/spectral-partition.pdf
- George, Liu 1978, https://epubs.siam.org/doi/10.1137/0715069
- LaSalle, Karypis, *Efficient ND for Multicore* (mt-Metis), https://dlasalle.github.io/publications/mtmetis2015nd.pdf
- Chevalier, Pellegrini, PT-Scotch, https://arxiv.org/abs/0907.1375
- Sanders, Schulz, *Advanced Multilevel Node Separator Algorithms*, https://arxiv.org/abs/1509.01190
- Schild, Sommer, *On Balanced Separators in Road Networks*, https://aschild.github.io/papers/roadseparator.pdf
- Acer et al., Sphynx, https://arxiv.org/abs/2105.00578
- Buluç et al., *Recent Advances in Graph Partitioning*, https://arxiv.org/pdf/1311.3144
- Ost, Schulz, Strash, *Engineering Data Reduction for ND*, https://arxiv.org/abs/2004.11315
- Hamann, Strasser, *Correspondence between Multilevel Graph Partitions and Tree Decompositions*, https://www.mdpi.com/1999-4893/12/9/198
- Maniu, Senellart, Jog, treewidth of real-world graphs, https://arxiv.org/abs/1901.06862
- Manne, Bisseling, PPAM 2007, https://webspace.science.uu.nl/~bisse101/Articles/CP75.pdf
- Davis, Hager, *Modifying a Sparse Cholesky Factorization*, https://people.clas.ufl.edu/hager/files/update.pdf
- Peries, Davis, HPEC'25 push-relabel in LAGraph, https://doi.org/10.1109/HPEC67600.2025.11196375
- Schloegel, Karypis, Kumar, *Multilevel Diffusion Schemes for Repartitioning of Adaptive Meshes*, JPDC 47(2) 1997 (abstract only; full text not read)
- Local: LAGraph `d01064de` `experimental/algorithm/{LAGraph_Coarsen_Matching,LAGraph_MaximalMatching,LAGraph_Hdip_Fiedler,LAGr_MaxFlow,LAGraph_MinCut,LAGr_MaximumMatching}.c`; GraphBLAS v10 `Include/GraphBLAS.h`; feral-metis `f127189` `src/{lib,coarsen}.rs`
