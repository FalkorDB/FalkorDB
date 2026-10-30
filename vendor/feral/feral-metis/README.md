# feral-metis

Multilevel nested-dissection fill-reducing ordering for sparse
symmetric matrices (Karypis & Kumar 1998), implemented in pure Rust,
clean-room from the published METIS papers.

- **License:** MIT
- **Dependencies:** `feral-ordering-core`, `feral-amd` (used as the
  small-leaf base case).
- **MSRV:** stable Rust, edition 2021. No `unsafe`.

## What METIS does

METIS computes a fill-reducing permutation by recursively bisecting
the adjacency graph of `A`. At each level it (1) coarsens the graph
by heavy-edge matching, (2) computes an initial bisection on the
coarsest graph, (3) uncoarsens with Fiduccia-Mattheyses (FM)
boundary refinement at every level, and (4) extracts a vertex
separator from the resulting edge bisection. The two halves recurse
independently; small subgraphs fall through to AMD as the leaf
algorithm. The resulting permutation tends to outperform AMD on
large 2D / 3D mesh-shaped problems and underperform AMD on small or
arrow-structured matrices — picking the right backend is a
problem-dependent choice that downstream solvers (and FERAL itself)
make at the analysis boundary.

## Reference

- Karypis, G., and Kumar, V. (1998). *A Fast and Highly Quality
  Multilevel Scheme for Partitioning Irregular Graphs.* SIAM Journal
  on Scientific Computing, 20(1), 359–392. The original METIS paper
  describing multilevel coarsening, initial partitioning, and FM
  refinement.
- Karypis, G., and Kumar, V. (1999). *A Fast and Highly Quality
  Multilevel Scheme for Partitioning Irregular Graphs.* Companion
  paper covering the nested-dissection driver.

Full BibTeX in `dev/references.bib` of the parent repository.

## Contract

`feral-metis` conforms to the FERAL ordering-crate contract defined
by [`feral-ordering-core`](https://crates.io/crates/feral-ordering-core).

```rust,ignore
use feral_metis::{metis_order, CscPattern};

let col_ptr = [0, 2, 5, 8, 10];
let row_idx = [0, 1,  0, 1, 2,  1, 2, 3,  2, 3];
let pattern = CscPattern::new(4, &col_ptr, &row_idx).expect("valid CSC");
let perm = metis_order(&pattern).expect("metis_order");
```

## Clean-room status

Implemented from the published METIS papers, not from the
GPL-licensed reference C codebase. Algorithmic decisions (matching
heuristic, initial-partition strategy, FM gain computation, leaf
threshold) are documented per-module in source comments and in
`dev/plans/ordering-metis.md` of the parent repository.

## License

MIT.
