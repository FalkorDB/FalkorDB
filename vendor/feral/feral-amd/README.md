# feral-amd

Approximate Minimum Degree (AMD) fill-reducing ordering for sparse
symmetric matrices, implemented in pure Rust using the in-place
quotient-graph algorithm of Amestoy, Davis and Duff (1996, 2004).

- **Status:** feature-complete (Slice A + Slice B of the development
  plan). Byte-for-byte match with the SuiteSparse AMD reference on
  the pinned oracle fixture suite.
- **License:** MIT
- **Dependencies:** none in the runtime library.
- **MSRV:** stable Rust, edition 2021. No `unsafe`.

## What AMD does

Given the sparsity pattern of a symmetric matrix `A`, AMD computes a
permutation `P` such that the Cholesky factorization `P A Pᵀ = L Lᵀ`
(or symmetric indefinite `P A Pᵀ = L D Lᵀ`) introduces far fewer
non-zeros than the natural ordering. "Fill" is the set of zero
entries in `A` that become non-zero in `L`; minimising fill is NP-hard
in general, so AMD is a greedy heuristic that in practice rivals
nested dissection on small-to-medium matrices and is much cheaper to
compute.

The canonical reference algorithm is Tim Davis's `amd_2.c` inside
SuiteSparse (BSD-3-Clause). `feral-amd` is a clean-room Rust
transliteration, cross-checked line-by-line against the in-tree
[faer-rs](https://github.com/sarah-quinones/faer-rs) port of the same
algorithm.

## Literature

- **Amestoy, P. R., Davis, T. A., and Duff, I. S. (1996).** *An
  Approximate Minimum Degree Ordering Algorithm.* SIAM Journal on
  Matrix Analysis and Applications, 17(4), 886–905. Introduces the
  approximate external degree bound that makes the algorithm O(|A|)
  per pivot instead of O(|A|·n).
- **Amestoy, P. R., Davis, T. A., and Duff, I. S. (2004).**
  *Algorithm 837: AMD, an Approximate Minimum Degree Ordering
  Algorithm.* ACM Transactions on Mathematical Software, 30(3),
  381–388. Describes the implementation details pinned down in
  SuiteSparse: dense-row handling, mass elimination, supervariable
  detection, the in-place quotient graph, and garbage collection.
- **George, A. (1973).** *Nested Dissection of a Regular Finite
  Element Mesh.* SIAM Journal on Numerical Analysis, 10(2), 345–363.
  The alternative ordering paradigm; AMD is complementary rather
  than competitive — they target different matrix regimes.
- **Davis, T. A., Rajamanickam, S., and Sid-Lakhdar, W. M. (2016).**
  *A Survey of Direct Methods for Sparse Linear Systems.* Acta
  Numerica, 25, 383–566. Places AMD in the broader ecosystem of
  direct methods.

Full BibTeX entries live in `../../dev/references.bib`. Curated
organisation-level notes are in `../../.crucible/wiki/concepts/approximate-minimum-degree.org`
and `../../.crucible/wiki/summaries/amestoy1996-amd.org` / `amestoy2004-amd-implementation.org`.

## Algorithm

AMD operates on the quotient graph `G/S` of `A`, where `S` is the set
of supervariables eliminated so far. At each step it:

1. **Selects** the lowest-degree supervariable `me` from degree
   buckets indexed by `mindeg` (linear scan, LIFO tie-break).
2. **Forms a new element** by merging `me`'s adjacency with every
   element already in `me`'s list. Supervariables that were in those
   elements are absorbed into the new element; the old elements are
   marked dead (*standard absorption*, `amd_2.c` around line 355).
3. **Updates the approximate external degree** of every surviving
   neighbour using the two-pass `w[e]` trick from the 1996 paper.
   Dead elements encountered during the degree computation are
   folded into `me` on the spot (*aggressive absorption*).
4. **Mass-eliminates** neighbours `i` whose only remaining element is
   `me` and whose outside variable list is empty — they pivot
   concurrently with `me`.
5. **Detects and merges supervariables** that are structurally
   indistinguishable: variables with the same adjacency hash, same
   element count and variable count, and literally the same neighbour
   set. Merged variables share a single pivot block in the final
   permutation.
6. **Re-inserts** the surviving neighbours into the degree buckets
   with their updated approximate degree.

### Dense-row handling

Variables whose initial degree exceeds `max(16, min(n, α·√n))` with
`α = 10` by default are classified as "dense" and deferred to the end
of the permutation. This is a structural device — it keeps a few hub
vertices (e.g. an arrow-matrix's central row) from dominating the
degree computation for every other variable. A negative `α` disables
the heuristic.

### Garbage collection

The algorithm maintains its entire quotient-graph state in a single
`Vec<i32>` of length `nzaat + nzaat/5 + n` (integer division). When
the workspace cursor `pfree` catches up with the end of that arena,
an inline compaction sweep slides every live adjacency list down to
close the holes left by absorbed elements. The number of compactions
fired is reported as `AmdStats::ncmpa`.

### Element post-order

After elimination the algorithm performs a postorder traversal of
the AMD-internal assembly tree (roots = final pivots without a
parent), using a *big-child-last* heuristic so the final permutation
groups siblings by size. Supervariables absorbed during step 5 are
expanded into their pivot block at this point, and dense-deferred
variables are appended.

## Usage

### Library

```rust
use feral_amd::{amd_order, amd_order_with_stats, AmdOptions, CscPattern};

// Full symmetric CSC pattern (both halves present, rows sorted).
// Example: 4x4 tridiagonal.
let col_ptr = [0, 2, 5, 8, 10];
let row_idx = [0, 1,  0, 1, 2,  1, 2, 3,  2, 3];
let pattern = CscPattern::new(4, &col_ptr, &row_idx).expect("valid CSC");

// Shortest form: just the permutation.
let perm = amd_order(&pattern).expect("amd_order");
assert_eq!(perm.len(), 4);

// Permutation + diagnostic counters (compactions, flops, ...).
let (perm, stats) = amd_order_with_stats(&pattern).expect("amd_order");
println!("ncmpa={} ndiv={} nms_ldl={}", stats.ncmpa, stats.ndiv, stats.nms_ldl);

// Custom options.
let opts = AmdOptions { aggressive: true, dense_alpha: 10.0 };
let _ = feral_amd::amd_order_opts(&pattern, &opts);
```

The input must be the full symmetric pattern — both the upper and
lower triangles. If your matrix is stored as upper-triangular only,
symmetrise with `A + Aᵀ - diag(A)` before handing it to `feral-amd`.

### CLI

```
$ cargo run --release --bin feral-amd -- path/to/triplet.txt
n: 5
nnz: 13
ncmpa: 0
n_dense_deferred: 0
ndiv: 4
nms_ldl: 4
nms_lu: 4
perm: 4 3 2 0 1
```

The triplet format is one `row col` pair per line, 0-indexed, with
`#` and `%` comments allowed. The CLI symmetrises the input.

### Benchmark skeleton

```
$ cargo run --release --bin feral-amd-bench
arrow(200)           n=   200 nnz=    598 ncmpa=  0 ndense=  1 ndiv=       199 time=0.005ms
band(100, 5)         n=   100 nnz=   1070 ncmpa=  1 ndense=  0 ndiv=       488 time=0.017ms
band(500, 10)        n=   500 nnz=  10390 ncmpa=  1 ndense=  0 ndiv=      4955 time=0.120ms
grid(30x30)          n=   900 nnz=   4380 ncmpa=  1 ndense=  0 ndiv=      9331 time=0.163ms
grid(60x60)          n=  3600 nnz=  17760 ncmpa=  1 ndense=  0 ndiv=     56165 time=0.513ms
```

## Evidence that it works

### External-oracle match

`tests/data/amd_oracle/` contains seven fixtures whose reference
output was pinned by a throwaway harness running the BSD-licensed
[`amd` 0.2.2](https://crates.io/crates/amd) Rust crate (a port of
Timothy Davis's SuiteSparse AMD). For each fixture we store the
permutation and flop counters that the SuiteSparse implementation
produced.

`tests/oracle_match.rs` rebuilds each pattern from the same
programmatic generator the harness used, runs
`amd_order_with_stats`, and asserts exact equality — not just on
the permutation, but on `ncmpa`, `ndiv`, `nms_ldl`, `nms_lu`, and
`n_dense_deferred`.

| fixture       | n   | description                          | perm + stats match? |
|---------------|-----|--------------------------------------|---------------------|
| `diag_4`      | 4   | pure diagonal                        | ✓                   |
| `tridiag_10`  | 10  | tridiagonal                          | ✓                   |
| `arrow_5`     | 5   | hub at 0 + diagonal                  | ✓                   |
| `arrow_200`   | 200 | hub at 0 + diagonal (tests dense)    | ✓                   |
| `band_20_3`   | 20  | banded, bandwidth 3 (tests GC)       | ✓                   |
| `grid_7x7`    | 49  | 2D 5-point stencil                   | ✓                   |
| `amd_demo_24` | 24  | 6×4 grid (canonical demo substitute) | ✓                   |

Two additional tests assert `n_mass_elim > 0` on patterns where
mass elimination must fire (tridiag_10, band_20_3, grid_7x7), and
two assert `n_supervar_merge > 0` on patterns where supervariable
detection must fire (band_20_3, grid_7x7). These guard against a
regression that silently disables either Slice B branch even if the
permutation still happens to line up.

### Unit tests

`src/` contains 36 `#[cfg(test)]` unit tests covering:

- Input validation in `CscPattern::new` (bad lengths, monotone
  col_ptr, out-of-range row indices).
- `AmdWorkspace::new` fast paths: empty pattern, pure diagonal,
  dense-deferred hub, zero-degree variable, `dense_alpha < 0`.
- `clear_flag` wrap behaviour at the `wbig` boundary.
- `flip` involution.
- Individual `run_elimination` fixtures (arrow, grid, band with GC).
- `finalize_permutation` bijection on every fixture.

Run everything with:

```
$ cargo test -p feral-amd
test result: ok. 36 passed; 0 failed; 0 ignored
test result: ok. 12 passed; 0 failed; 0 ignored
```

### Static checks

```
$ cargo clippy -p feral-amd --all-targets -- -D warnings   # clean
$ cargo fmt --check                                        # clean
```

The library uses `#![forbid(unsafe_code)]` and `#![deny(missing_docs)]`.

### Clean-room isolation

The `amd` crate appears **only** in the throwaway fixture-generation
harness at `tests/data/amd_oracle/harness/*.txt`, which is not built
by Cargo. `crates/feral-amd/Cargo.toml` has zero runtime dependencies
and no dev-dependencies on `amd`. A workspace grep guards the
invariant:

```
$ grep -r 'amd = "' crates/feral-amd/Cargo.toml | grep -v harness
# (no matches)
```

## Architecture

```
src/
├── lib.rs         Public surface: amd_order, amd_order_with_stats,
│                  amd_order_opts, AmdOptions.
├── pattern.rs     CscPattern — borrowed CSC sparsity pattern with
│                  validation.
├── error.rs       AmdError (IndexOverflow, NonSymmetric,
│                  MalformedInput).
├── stats.rs       AmdStats (ncmpa, n_mass_elim, n_supervar_merge,
│                  n_dense_deferred, ndiv, nms_ldl, nms_lu).
├── workspace.rs   AmdWorkspace::new — owns pe/iw/len/nv/elen/degree
│                  scratch arrays and runs initialization (dense-row
│                  classification, zero-degree fast path, degree-bucket
│                  construction).
├── algo.rs        run_elimination + finalize_permutation — the main
│                  loop and the post-order expansion.
└── bin/
    ├── feral-amd.rs       Triplet-file CLI.
    └── feral-amd-bench.rs Small in-crate bench skeleton.
```

Every function that ports a block of faer's `amd.rs` cites the faer
line range in its doc comment so the translation is auditable.

## Limitations and scope

- **Input type is deliberately minimal.** `CscPattern<'a>` borrows
  two `&[usize]` slices. The crate does not depend on any sparse
  matrix library; downstream solvers convert at the boundary.
- **Not yet integrated into `feral`.** Integration will come via
  `dev/plans/ordering-integration.md` once the sibling ordering
  crates (METIS, SCOTCH, KaHIP) also exist.
- **Matrices must be structurally symmetric.** AMD operates on
  `A + Aᵀ`; handing it an unsymmetric pattern produces meaningless
  orderings. Debug builds assert symmetry; release builds trust the
  caller for the sake of speed.
- **`i32` index space.** The scratch arrays use `i32` with a
  sentinel of `-1`, matching SuiteSparse. Matrices too large for
  `i32` workspaces return `AmdError::IndexOverflow` from
  `AmdWorkspace::new` rather than silently wrapping.

## Reproducing the oracle fixtures

The throwaway harness is preserved under `tests/data/amd_oracle/harness/`
as extensionless `.txt` files so Cargo does not build them:

```
mkdir -p /tmp/amd_oracle && cd /tmp/amd_oracle
cp .../tests/data/amd_oracle/harness/main.rs.txt   src/main.rs
cp .../tests/data/amd_oracle/harness/Cargo.toml.txt Cargo.toml
cargo run --release -- /tmp/amd_oracle_out
diff -ru /tmp/amd_oracle_out .../tests/data/amd_oracle/
```

Harness SHA-256s are listed in `tests/data/amd_oracle/README.md` so
re-generated fixtures can be audit-checked.

## Plan and history

- Full design and commit-by-commit plan:
  `../../dev/plans/ordering-amd-upgrade.md`
- Per-commit journal (decisions, tried-and-rejected, evidence):
  `../../dev/journal/2026-04-16-01.org`
- Session checkpoint: `../../dev/sessions/2026-04-16-01.md`
