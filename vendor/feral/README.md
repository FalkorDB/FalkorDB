# Vendored feral ordering crates

The CCH index (`graph/src/index/falkordb/cch`, cargo feature `index-falkordb`) orders its
graph by nested dissection with these three crates. They are copies of releases on
crates.io, by John Kitchin, from <https://github.com/jkitchin/feral>, under the MIT
licence; each directory keeps its `LICENSE` and upstream `README.md`.

| directory | crate and version | upstream commit (`.cargo_vcs_info.json`) |
|---|---|---|
| `feral-metis` | feral-metis 0.3.0 | 32c936b8d1f6a5d42ccf96b7dfb1c05f80e402da |
| `feral-amd` | feral-amd 0.2.1 | 5ab8074b3c87c96d223e56e8012c38b1cf591e0e |
| `feral-ordering-core` | feral-ordering-core 0.2.2 | 32c936b8d1f6a5d42ccf96b7dfb1c05f80e402da |

## Changes from the releases

- **64-bit indices.** `feral-ordering-core` defines `pub type Idx = i64`, and every `i32`
  index, count and weight in the three libraries is an `Idx` (upstream they are `i32`). A
  graph is then bounded by memory, not by 2^31 vertices or entries. The permutation is
  identical to the `i32` build's wherever both fit. Names such as `n_i32` are kept to
  keep the diff small. One unit test, which asserts an `i32` overflow bound, keeps `i32`.
- **Linear component split.** `feral-metis`'s `node_nd.rs` ordered each connected
  component after a scan of all `n` vertices, which is quadratic on a graph with many
  components (132 s on a road network with 294k isolated vertices). `split_and_recurse`
  groups the components with one counting sort and places a single-vertex component
  directly. The permutation is unchanged.
- **Recursion by value.** `recurse` takes its graph by value and drops it once both
  halves are built, so a level's parent no longer stays alive under its children
  (peak memory -38% on California). The permutation is unchanged.
- **Linear 2-hop matching.** `coarsen.rs`'s `two_hop_pass` rescanned a hub's whole
  adjacency for each of its neighbours, quadratic in the hub's degree (a star of 200k
  leaves: 11 s). Each vertex now keeps where its scan resumes, past the neighbours already
  matched (22 ms). The permutation is unchanged.
- **Ordering quality.** Two changes, each a separate commit, bring `feral-metis`'s node
  bisection closer to METIS 5's (`libmetis/ometis.c`, `sfm.c`). They change the
  permutation; on California they cut the chordal hierarchy from 5,932,138 arcs and 59.6M
  triangles to 5,838,003 and 53.9M (C METIS: 5,527,199 and 48.6M).
  - Node FM (`fm_refine.rs`, `refine_separator_fm`): a move that leaves the separator as
    light but the sides better balanced becomes the pass's best state (METIS's `mindiff`).
  - Bisection (`node_nd.rs`, `best_of_bisections`): a subgraph of 5,000 vertices or more
    is coarsened once to `max(100, n / 30)` vertices, bisected five times there, and the
    lightest separator is refined back up, as METIS's `MlevelNodeBisectionL2` does.
- Removed: the binaries, the integration tests and their data, and two extractors the
  changes above left unused.
- Packaging: `publish = false`, path dependencies, formatted with `rustfmt`'s defaults
  (`rustfmt.toml` here), as upstream is.
