# Edge storage at the data-structure boundary

Measures the two multi-edge storage designs **as structures**, not as engines:

- **(A) container per cell** — the C engine's tensor, a tagged `GrB_Vector`
  handle in the matrix cell. Measured here.
- **(C) inline-first with sentinel promotion** — the Rust engine's tensor.
  Measured by `graph/src/graph/graphblas/tensor_cost_bench.rs`, which builds the
  *same* fixture (`tensor_cost_c_comparable`) so the two are comparable.

Why the boundary and not whole queries: a query-level ratio is a ratio between
two engines, and the tensor is one component of each. Every time we checked, that
difference mattered — engine-level differencing overstated the multi-edge entry
charge by 1.6x for (C) and 5.1x for (A), and overstated the per-identifier slope
by 3.6x and 26x, which is enough to *invert* the ordering of the two slopes. See
`docs/papers/tensor.tex` §evalboundary and §evalfanout.

## Building

Two inputs live outside this repo's tree and are overridable:

| | |
| --- | --- |
| `SRC` | a checkout of `master` (the C engine's sources) |
| `BIN` | a C build tree, for `libfalkordb_static.a` and its GraphBLAS |

```sh
git worktree add /tmp/cmaster master          # SRC
make                                          # BIN, if not already built
SRC=/tmp/cmaster BIN=$PWD/bin/macos-arm64v8-release ./build.sh
OMP_NUM_THREADS=1 ./tensor_bench
```

The engine-level companion names its two modules the same way:

```sh
FALKORDB_ROOT=/path/to/checkout \
RUST_MODULE=/path/to/target/release/libfalkordb.dylib \
GAP1_WORK=/tmp/gap1work \
python3 engine_space_and_transition.py space
```

The Rust-side measurement tests must run **serially** —
`cargo test --release -p graph tensor_cost -- --ignored --nocapture
--test-threads=1`. Without `--test-threads=1` the eight of them run
concurrently, which both interleaves their output and inflates every
instruction count; a point read measured 8,156 instructions in parallel
against 566 serially.

`build.sh` compiles **only** `tensor_bench.c`. `tensor.c`, `tensor_iterator.c`
and `delta_matrix/*.c` come out of `libfalkordb_static.a`, so what is measured is
the shipped implementation rather than a recompilation of it — 323 of the
archive's 392 members link in, with no stubbed symbols.

`OMP_NUM_THREADS=1` matters: the C side is measured single-threaded, and the Rust
side uses GraphBLAS's default thread count. That asymmetry is why the comparison
is stated in *instructions* and why the bulk-build rows are the ones to trust
least.

## What each file is

| file | what it does |
| --- | --- |
| `tensor_bench.c` | the harness: point reads, iteration (forward and transposed), fan-out to `k = 16`, row/column degree, bulk insert/delete/clear, the working-set sweep, promote/demote, build paths, space |
| `calib.c` | measures the instruction-counter overhead the harness subtracts |
| `engine_space_and_transition.py` | the *engine-level* companion: `relation_matrices_sz_mb` across `k`, and the build-order gap that isolates a transition. Takes `FALKORDB_ROOT`, `C_MODULE`, `RUST_MODULE` and `GAP1_WORK` from the environment; a module that is missing is skipped, which is how one engine is measured alone |
| `build.sh` | compiles and links the above against `$BIN` |

## Caveats that belong with the numbers

- **Separate processes, separate allocators.** The C harness and the Rust
  measurement tests do not share a process, so absolute wall clock is not
  comparable across them; instructions are.
- **GraphBLAS versions are not aligned.** The shipped archive here is built
  against whatever `$BIN` was built against — 10.4.0 on the machine these were
  last run on, where the Rust side is on 10.5.0. The `k <= 2` rows reproduce the
  paper's table to within 1%, so the minor version is not what moves them, but
  any cross-version claim needs both sides rebuilt on one.
- **Warm and sequential, except where it is not.** Every read here walks pairs in
  ascending order with the data hot, which is what makes the instruction counts
  comparable. `bench_sweep` is the exception: it grows the working set from 10^4
  to 8x10^6 pairs and probes in a scrambled order, so above the cache the reads
  genuinely miss. Its instruction column is a *control* — it should be flat, and
  is, since a cache miss retires no extra instruction — and its nanoseconds are
  the result. The Rust counterpart is `tensor_cost_cold_cache`; run both to get
  the ratio, since the absolute times are not comparable across processes.
- **`bench_sweep`'s times are minima, not medians.** Every other measurement here
  reports three repetitions and the paper takes the median. The sweep takes the
  minimum of three, because on a machine that is not quiet every sample is an
  upper bound and the minimum is the least contaminated. Compare ratios between
  the two engines rather than magnitudes.
