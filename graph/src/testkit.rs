//! Graph test kit: weight matrices, generators for them and a shortest-path oracle.
//!
//! A test graph is a [`Matrix<f64>`] of weights: `n x n` for `n` nodes, one entry per directed
//! pair `(u, v)` that has an edge, holding its weight (`+inf` is a closed edge, which reads as
//! no edge). There are no parallel edges and no relationship types here: those belong to a
//! real `Graph`, and the CCH test kit materializes one from a matrix, with decoys that must
//! never win (heavier parallel edges, closed edges, unindexed types, self-loops). The kit has
//! deterministic generators (grids, paths, stars, random sparse graphs, zero-weight cycles,
//! isolated nodes, ...) and an oracle: [`dijkstra`] and [`all_pairs`], with
//! [`assert_path_valid`], [`assert_path_simple`] and [`check_index_against_oracle`] to check any
//! shortest-path answer against it. Nothing here knows about an index or a `Graph`.
//!
//! The oracle must not share a bug with the code it checks. It reads a matrix out once
//! (`extractTuples`) into an adjacency list of its own and runs plain Dijkstra over that, so
//! no GraphBLAS operation stands between a weight and its answer, while the index's input
//! reads the same matrices through GraphBLAS. The self-tests at the bottom check [`dijkstra`]
//! against hand-computed cases lifted from the C flow tests (`tests/flow/test_cch.py` on
//! `master`), and the checker against a separately written Dijkstra.
//!
//! Test-only (`#[cfg(test)]`): crate code reaches it as `crate::testkit`. Index-specific
//! test kits build on it and re-export it.

use crate::graph::graphblas::matrix::Matrix;
use crate::graph::graphblas::test_init;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;

/// An empty `n x n` weight matrix: `n` nodes, no edges.
#[must_use]
pub(crate) fn empty(n: usize) -> Matrix<f64> {
    test_init::ensure_init();
    Matrix::<f64>::new(n as u64, n as u64)
}

/// `n` nodes and one edge per entry of `edges`, `(u, v, weight)`.
///
/// One `GrB_Matrix_build`, with no `dup` operator: GraphBLAS itself refuses a repeated pair.
///
/// # Panics
/// If a pair `(u, v)` repeats (a matrix holds one weight per pair), or a node is out of range.
#[must_use]
pub(crate) fn weights(
    n: usize,
    edges: &[(u64, u64, f64)],
) -> Matrix<f64> {
    let mut g = empty(n);
    let rows: Vec<u64> = edges.iter().map(|&(u, _, _)| u).collect();
    let cols: Vec<u64> = edges.iter().map(|&(_, v, _)| v).collect();
    let vals: Vec<f64> = edges.iter().map(|&(_, _, w)| w).collect();
    g.build(&rows, &cols, &vals);
    g
}

/// The number of nodes of `g`.
#[must_use]
pub(crate) fn node_count(g: &Matrix<f64>) -> usize {
    g.nrows() as usize
}

/// Every edge of `g` as `(u, v, weight)`, row by row, columns ascending.
#[must_use]
pub(crate) fn entries(g: &Matrix<f64>) -> Vec<(u64, u64, f64)> {
    let (rows, cols, vals) = g.extract_tuples();
    rows.into_iter()
        .zip(cols)
        .zip(vals)
        .map(|((u, v), w)| (u, v, w))
        .collect()
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct HeapEntry {
    dist: f64,
    node: u64,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        // Reversed: `BinaryHeap` is a max-heap, Dijkstra wants the smallest
        // distance out first. NaN can't occur here -- see the assert in
        // `adjacency`.
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The oracle's own copy of `g`: the out-edges of each node, read out of the matrix once.
///
/// # Panics
/// If a weight is negative (`-inf` included) or NaN.
fn adjacency(g: &Matrix<f64>) -> Vec<Vec<(u64, f64)>> {
    let mut adj: Vec<Vec<(u64, f64)>> = vec![Vec::new(); node_count(g)];
    for (u, v, w) in entries(g) {
        assert!(
            w >= 0.0,
            "dijkstra oracle requires non-negative weights, got {w} on ({u}, {v})"
        );
        adj[u as usize].push((v, w));
    }
    adj
}

/// Plain Dijkstra from `src` over `g`'s edges. A self-loop (`(u, u)`, a diagonal entry) is
/// relaxed like any other edge and never improves a distance. `f64::INFINITY` for every node
/// `src` cannot reach.
///
/// # Panics
/// If `src` is out of range, or any edge weight is negative (`-inf` included) or NaN:
/// the oracle only promises correctness over the weight domain `CchInput` allows
/// (`>= 0`). A `+inf` edge (a closed edge) is relaxed like any other and never
/// improves a distance, so it reads as no edge.
///
/// Also if a node is left unreachable only because a path to it over open edges sums past
/// `f64::MAX` (`1e308 + 1e308` is `+inf`): its distance is not an `f64`, and `+inf` would
/// claim it is unreachable. An overflow on a node reached more cheaply another way is
/// harmless and allowed.
///
/// Not checked against FalkorDB's own Dijkstra (`dijkstra_single_path` behind
/// `algo.SPpaths`, `runtime/functions/algo_procedures.rs`), on purpose: that one is private
/// and needs a `Runtime` and a loaded `Graph`, so it is only reachable through Cypher, and
/// it silently skips edges whose weight is not finite: the same answer for a `+inf` edge,
/// but NaN and `-inf` are input this oracle must reject. It is the oracle of the flow tests
/// instead (`tests/flow/test_cch.py` compares CCH queries with `algo.SPpaths`). If a pure
/// shortest-path function over a weight matrix is ever added to the engine, cross-check this
/// one against it.
#[must_use]
pub(crate) fn dijkstra(
    g: &Matrix<f64>,
    src: u64,
) -> Vec<f64> {
    let n = node_count(g);
    assert!((src as usize) < n, "src {src} out of range for {n} nodes");
    dijkstra_over(&adjacency(g), src)
}

fn dijkstra_over(
    adj: &[Vec<(u64, f64)>],
    src: u64,
) -> Vec<f64> {
    let n = adj.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut settled = vec![false; n];
    // Nodes some relaxation over an open edge reached at a sum past `f64::MAX`.
    let mut overflowed = vec![false; n];
    dist[src as usize] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(HeapEntry {
        dist: 0.0,
        node: src,
    });

    while let Some(HeapEntry { dist: d, node: u }) = heap.pop() {
        // Settled the first time it is popped, at its final distance; a later entry for it
        // is stale. Each node is expanded once, so this terminates whatever the relaxation.
        if std::mem::replace(&mut settled[u as usize], true) {
            continue;
        }
        for &(v, w) in &adj[u as usize] {
            // `d` is finite (only finite distances are pushed), so `nd` is `+inf` either over
            // a closed edge or by overflow.
            let nd = d + w;
            if nd.is_infinite() && w.is_finite() {
                overflowed[v as usize] = true;
            }
            if nd < dist[v as usize] {
                // With weights >= 0 a settled distance is final. A non-strict `<=` above
                // would re-reach a settled node over a zero-weight edge: fail, not loop.
                assert!(!settled[v as usize], "dijkstra re-reached settled node {v}");
                dist[v as usize] = nd;
                heap.push(HeapEntry { dist: nd, node: v });
            }
        }
    }

    // Overflow cannot make a finite distance wrong: the lost sum is above every finite one.
    // It can only leave a reachable node at `+inf`.
    if let Some(v) = (0..n).find(|&v| overflowed[v] && dist[v].is_infinite()) {
        panic!(
            "dijkstra oracle: the distance from {src} to {v} overflows f64, \
             so it cannot tell it from unreachable"
        );
    }
    dist
}

/// `dijkstra` from every node, over one read of `g`. `result[s][t]` is the cheapest
/// `s -> t` weight, `f64::INFINITY` if unreachable.
#[must_use]
pub(crate) fn all_pairs(g: &Matrix<f64>) -> Vec<Vec<f64>> {
    let adj = adjacency(g);
    (0..adj.len() as u64)
        .map(|s| dijkstra_over(&adj, s))
        .collect()
}

/// The weight of every edge of `g` but self-loops, by pair.
fn edge_weights(g: &Matrix<f64>) -> HashMap<(u64, u64), f64> {
    entries(g)
        .into_iter()
        .filter(|&(u, v, _)| u != v)
        .map(|(u, v, w)| ((u, v), w))
        .collect()
}

/// Checks that `path` is a valid witness for an `src -> dst` query result:
/// consecutive nodes are joined by an edge of `g`, the path starts at
/// `src` and ends at `dst`, and the sum of the weights of its hops equals
/// `weight`.
///
/// `path` is a node sequence (as `Cch::query` returns), not an edge list: a pair holds
/// one weight here, the cheapest of its parallel edges in the real graph it stands for.
///
/// # Panics
/// If the path is empty, does not start/end at `src`/`dst`, skips a pair
/// with no edge between them, or its weight sum does not match `weight`
/// ([`weights_agree`]).
pub(crate) fn assert_path_valid(
    g: &Matrix<f64>,
    src: u64,
    dst: u64,
    weight: f64,
    path: &[u64],
) {
    assert_path_valid_over(&edge_weights(g), node_count(g), src, dst, weight, path);
}

fn assert_path_valid_over(
    edges: &HashMap<(u64, u64), f64>,
    n: usize,
    src: u64,
    dst: u64,
    weight: f64,
    path: &[u64],
) {
    assert!(!path.is_empty(), "path must not be empty");
    assert_eq!(path[0], src, "path does not start at src");
    assert_eq!(*path.last().unwrap(), dst, "path does not end at dst");
    if let Some(&node) = path.iter().find(|&&node| node as usize >= n) {
        panic!("path node {node} is not in the graph ({n} nodes)");
    }

    if path.len() == 1 {
        assert_eq!(src, dst, "single-node path must have src == dst");
        assert_eq!(weight, 0.0, "a trivial src == dst path must weigh 0");
        return;
    }

    let mut sum = 0.0;
    for hop in path.windows(2) {
        let (u, v) = (hop[0], hop[1]);
        let w = *edges
            .get(&(u, v))
            .unwrap_or_else(|| panic!("no edge ({u}, {v}) on claimed path"));
        sum += w;
    }

    assert!(
        weights_agree(sum, weight),
        "path sums to {sum}, claimed weight is {weight}"
    );
}

/// Two path weights agree: both finite and non-negative (no path in the weight domain
/// `>= 0` weighs less than zero, however it rounds), and within `1e-6` or a relative
/// `1e-9`, whichever is looser. Summing the same edges in another order (a
/// shortcut adds its two halves first) rounds differently, and at large
/// weights the difference exceeds any fixed absolute tolerance: `1e16 + 1 + 1`
/// is `1e16` hop by hop but `10000000000000002` with the units added first.
#[must_use]
fn weights_agree(
    a: f64,
    b: f64,
) -> bool {
    a.is_finite()
        && b.is_finite()
        && a >= 0.0
        && b >= 0.0
        && (a - b).abs() <= 1e-6_f64.max(1e-9 * a.abs().max(b.abs()))
}

/// A node id in `[0, n)` from the next draw. `n` must be `> 0`. Every generator below draws
/// from a [`SmallRng`] seeded with its `seed`, so a failing seed reproduces.
pub(crate) fn next_node(
    rng: &mut SmallRng,
    n: usize,
) -> u64 {
    rng.random_range(0..n as u64)
}

/// An integer weight drawn uniformly from `[1, wmax]`.
fn next_weight(
    rng: &mut SmallRng,
    wmax: u32,
) -> f64 {
    f64::from(rng.random_range(1..=wmax))
}

/// An `rows x cols` grid, 4-connected, unit weights. `rows, cols >= 1`.
#[must_use]
pub(crate) fn grid(
    rows: u64,
    cols: u64,
) -> Matrix<f64> {
    assert!(rows >= 1 && cols >= 1);
    let id = |r: u64, c: u64| r * cols + c;
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let u = id(r, c);
            if c + 1 < cols {
                let v = id(r, c + 1);
                edges.extend([(u, v, 1.0), (v, u, 1.0)]);
            }
            if r + 1 < rows {
                let v = id(r + 1, c);
                edges.extend([(u, v, 1.0), (v, u, 1.0)]);
            }
        }
    }
    weights((rows * cols) as usize, &edges)
}

/// `n` nodes, `m` random directed edges on distinct pairs, integer weights in `[1, wmax]`.
/// A self-loop or a pair already drawn is drawn again (no self-loops, matching `_rand_graph`
/// in the C flow tests, and one weight per pair); deterministic in `seed`.
///
/// # Panics
/// If `m` is more than the `n * (n - 1)` pairs there are.
#[must_use]
pub(crate) fn random_sparse(
    n: usize,
    m: usize,
    wmax: u32,
    seed: u64,
) -> Matrix<f64> {
    weights(n, &random_edges(n, m, wmax, seed))
}

/// The edges [`random_sparse`] draws, in draw order.
///
/// # Panics
/// If `m` is more than the `n * (n - 1)` pairs there are.
fn random_edges(
    n: usize,
    m: usize,
    wmax: u32,
    seed: u64,
) -> Vec<(u64, u64, f64)> {
    assert!(n >= 1 && wmax >= 1);
    assert!(
        m <= n * (n - 1),
        "{m} edges on distinct pairs need more than {n} nodes"
    );
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut edges = Vec::with_capacity(m);
    let mut drawn = HashSet::new();
    // Until all `m` are drawn: the caller's `m <= n * (n - 1)` means a free pair always
    // exists, so a nearly full graph only takes more draws, never fewer edges.
    while edges.len() < m {
        let u = next_node(&mut rng, n);
        let v = next_node(&mut rng, n);
        if u == v || !drawn.insert((u, v)) {
            continue;
        }
        edges.push((u, v, next_weight(&mut rng, wmax)));
    }
    edges
}

/// `count` disjoint components, each an independent `random_sparse` graph of
/// `nodes_per_component` nodes and `edges_per_component` edges, with no edge
/// crossing components.
#[must_use]
pub(crate) fn disconnected_components(
    count: usize,
    nodes_per_component: usize,
    edges_per_component: usize,
    wmax: u32,
    seed: u64,
) -> Matrix<f64> {
    assert!(count >= 1 && nodes_per_component >= 1);
    let edges: Vec<_> = (0..count)
        .flat_map(|c| {
            let offset = (c * nodes_per_component) as u64;
            random_edges(
                nodes_per_component,
                edges_per_component,
                wmax,
                seed ^ (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            )
            .into_iter()
            .map(move |(u, v, w)| (u + offset, v + offset, w))
        })
        .collect();
    weights(count * nodes_per_component, &edges)
}

/// A single node, no edges.
#[must_use]
pub(crate) fn single_node() -> Matrix<f64> {
    empty(1)
}

/// A star: node `0` is the hub, `leaves` outer nodes, edges both ways with
/// weight `w`.
#[must_use]
pub(crate) fn star(
    leaves: u64,
    w: f64,
) -> Matrix<f64> {
    assert!(leaves >= 1);
    let edges: Vec<_> = (1..=leaves)
        .flat_map(|leaf| [(0, leaf, w), (leaf, 0, w)])
        .collect();
    weights((leaves + 1) as usize, &edges)
}

/// A simple directed path `0 -> 1 -> ... -> n-1`, unit weights, plus the
/// reverse arcs so the underlying skeleton is connected both ways.
#[must_use]
pub(crate) fn path(n: usize) -> Matrix<f64> {
    assert!(n >= 1);
    let edges: Vec<_> = (0..n.saturating_sub(1) as u64)
        .flat_map(|u| [(u, u + 1, 1.0), (u + 1, u, 1.0)])
        .collect();
    weights(n, &edges)
}

/// `k` node-disjoint two-hop paths `0 -> mid -> dst`, each hop costing
/// `hop_w` -- every one of the `k` paths ties at `2 * hop_w`. A trap for a
/// checker that asserts path *identity* instead of weight: [`assert_path_valid`]
/// must accept whichever of the `k` middle nodes the caller passed, not a
/// single canonical one. `k >= 1`.
#[must_use]
pub(crate) fn tied_paths(
    k: u64,
    hop_w: f64,
) -> Matrix<f64> {
    assert!(k >= 1);
    let dst = k + 1;
    let edges: Vec<_> = (1..=k)
        .flat_map(|mid| {
            [
                (0, mid, hop_w),
                (mid, 0, hop_w),
                (mid, dst, hop_w),
                (dst, mid, hop_w),
            ]
        })
        .collect();
    weights((k + 2) as usize, &edges)
}

/// A directed cycle `0 -> 1 -> ... -> len-1 -> 0`, every edge weight `0.0`.
/// Dijkstra must not keep relaxing a zero-weight edge (the `nd < dist[v]`
/// check must be strict, not `<=`), and every node on the cycle is at
/// distance `0.0` from every other. `len >= 2`.
#[must_use]
pub(crate) fn zero_weight_cycle(len: u64) -> Matrix<f64> {
    assert!(len >= 2);
    let edges: Vec<_> = (0..len).map(|u| (u, (u + 1) % len, 0.0)).collect();
    weights(len as usize, &edges)
}

/// `base` with the edge `u -> v` at weight zero (added, or made free if it is there) -- a
/// free shortcut: distance to `v` can never exceed distance to `u`.
#[must_use]
pub(crate) fn with_zero_weight_edge(
    mut base: Matrix<f64>,
    u: u64,
    v: u64,
) -> Matrix<f64> {
    base.set(u, v, 0.0);
    base
}

/// `base` with `extra` additional isolated nodes appended after its existing
/// `n`. A real `Graph -> CchInput` can have `n` larger than any node id
/// that appears in an edge -- a node with no incident edge among the indexed
/// relationship types is still a node in the graph -- and no generator above
/// produces that shape on its own, since they only draw node ids that get
/// used.
#[must_use]
pub(crate) fn with_isolated_nodes(
    mut base: Matrix<f64>,
    extra: usize,
) -> Matrix<f64> {
    let n = (node_count(&base) + extra) as u64;
    base.resize(n, n);
    base
}

/// A returned path must be simple: no node appears twice. With zero-weight
/// edges a walk can revisit a node while keeping the right weight and valid
/// hops, so [`assert_path_valid`] alone does not catch it.
///
/// # Panics
/// If any node repeats.
pub(crate) fn assert_path_simple(path: &[u64]) {
    let mut seen = HashSet::new();
    for &node in path {
        assert!(seen.insert(node), "path repeats node {node}: {path:?}");
    }
}

/// Like [`random_sparse`], but about half the edges weigh exactly `0.0`
/// (drawn per edge, row by row), the rest an integer in `[1, wmax]`. Zero-weight edges
/// create zero-weight cycles and many equal-weight walks, which is where a
/// path can go wrong while its weight stays right.
#[must_use]
pub(crate) fn random_zero_heavy(
    n: usize,
    m: usize,
    wmax: u32,
    seed: u64,
) -> Matrix<f64> {
    let mut rng = SmallRng::seed_from_u64(seed ^ 0xA5A5_A5A5_A5A5_A5A5);
    let edges: Vec<_> = random_edges(n, m, wmax, seed)
        .into_iter()
        .map(|(u, v, w)| (u, v, if rng.random::<bool>() { 0.0 } else { w }))
        .collect();
    weights(n, &edges)
}

/// End-to-end differential check of an index against the oracle: for every
/// ordered pair `(s, t)`, `index_query(s, t)` must agree with [`all_pairs`]
/// on reachability and weight ([`weights_agree`]), and every returned path must
/// pass [`assert_path_valid`] (a witness check: weight only, never path
/// identity, so ties are fine) and [`assert_path_simple`] (no repeated node).
/// `g` is read once.
///
/// Convention for `s == t` (C's `test01_create_query_drop`): the index
/// returns weight `0` and the single-node path `[s]`; `None` there is a
/// failure.
///
/// # Panics
/// On the first disagreement, naming the pair.
pub(crate) fn check_index_against_oracle(
    index_query: impl Fn(u64, u64) -> Option<(f64, Vec<u64>)>,
    g: &Matrix<f64>,
) {
    let oracle = all_pairs(g);
    let edges = edge_weights(g);
    let n = node_count(g) as u64;
    for s in 0..n {
        for t in 0..n {
            let expected = oracle[s as usize][t as usize];
            let got = index_query(s, t);
            if s == t {
                let (w, path) =
                    got.unwrap_or_else(|| panic!("({s}, {t}): s == t must return weight 0"));
                assert_eq!(w, 0.0, "({s}, {t}): s == t must weigh 0, got {w}");
                assert_path_valid_over(&edges, n as usize, s, t, w, &path);
                assert_path_simple(&path);
            } else if expected.is_infinite() {
                assert!(
                    got.is_none(),
                    "({s}, {t}): unreachable in the oracle but the index returned {got:?}"
                );
            } else {
                let (w, path) = got.unwrap_or_else(|| {
                    panic!("({s}, {t}): oracle weight is {expected} but the index returned None")
                });
                assert!(
                    weights_agree(w, expected),
                    "({s}, {t}): index weight {w}, oracle weight {expected}"
                );
                assert_path_valid_over(&edges, n as usize, s, t, w, &path);
                assert_path_simple(&path);
            }
        }
    }
}

mod self_tests {
    use super::{
        all_pairs, assert_path_valid, dijkstra, disconnected_components, empty, entries, grid,
        node_count, path, random_sparse, single_node, star, tied_paths, weights,
        with_zero_weight_edge, zero_weight_cycle,
    };

    /// `test14_parallel_selfloop_unreachable`, but for the parallel edges (a matrix holds one
    /// weight per pair; the real-graph tests add parallel edges as decoys): a self-loop is a
    /// no-op, an isolated node is unreachable. The self-loop is on the source, so its weight
    /// must not reach `dist[src]`.
    #[test]
    fn selfloop_unreachable() {
        let g = weights(4, &[(0, 1, 2.0), (1, 2, 1.0), (0, 0, 0.5)]);

        let dist = dijkstra(&g, 0);
        assert_eq!(dist[0], 0.0);
        assert_eq!(dist[1], 2.0);
        assert_eq!(dist[2], 3.0);
        assert!(dist[3].is_infinite());
        assert_path_valid(&g, 0, 2, 3.0, &[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "requires non-negative")]
    fn dijkstra_rejects_a_negative_weight() {
        let _ = dijkstra(&weights(2, &[(0, 1, -1.0)]), 0);
    }

    #[test]
    #[should_panic(expected = "requires non-negative")]
    fn dijkstra_rejects_a_nan_weight() {
        let _ = dijkstra(&weights(2, &[(0, 1, f64::NAN)]), 0);
    }

    #[test]
    #[should_panic(expected = "requires non-negative")]
    fn dijkstra_rejects_a_negative_infinite_weight() {
        let _ = dijkstra(&weights(2, &[(0, 1, f64::NEG_INFINITY)]), 0);
    }

    /// `1e308 + 1e308` is `+inf`: node 2 is reachable over open edges, but its distance is no
    /// `f64`, so the oracle refuses rather than call it unreachable.
    #[test]
    #[should_panic(expected = "the distance from 0 to 2 overflows f64")]
    fn dijkstra_refuses_a_distance_that_overflows() {
        let _ = dijkstra(&weights(3, &[(0, 1, 1e308), (1, 2, 1e308)]), 0);
    }

    /// The same overflow, on a node a cheap edge reaches anyway: the answer is right, so it
    /// stands.
    #[test]
    fn an_overflow_on_a_node_reached_more_cheaply_is_harmless() {
        let g = weights(3, &[(0, 1, 1e308), (1, 2, 1e308), (0, 2, 1.0)]);
        assert_eq!(dijkstra(&g, 0), vec![0.0, 1e308, 1.0]);
    }

    /// A `+inf` weight is a closed edge, which reads as no edge: with no other route its
    /// target is unreachable, and with a finite detour the detour wins.
    #[test]
    fn a_plus_inf_edge_reads_as_no_edge() {
        let cut = weights(3, &[(0, 1, f64::INFINITY), (1, 2, 1.0)]);
        assert_eq!(dijkstra(&cut, 0), vec![0.0, f64::INFINITY, f64::INFINITY]);
        let detour = weights(3, &[(0, 2, f64::INFINITY), (0, 1, 1.0), (1, 2, 1.0)]);
        assert_eq!(dijkstra(&detour, 0), vec![0.0, 1.0, 2.0]);
        let open = weights(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
        assert_eq!(all_pairs(&detour), all_pairs(&open));
    }

    #[test]
    #[should_panic(expected = "GrB_Matrix_build_FP64")]
    fn weights_refuses_a_repeated_pair() {
        let _ = weights(2, &[(0, 1, 5.0), (0, 1, 2.0)]);
    }

    #[test]
    #[should_panic(expected = "need more than 1 nodes")]
    fn random_sparse_refuses_edges_on_a_single_node() {
        let _ = random_sparse(1, 3, 5, 0);
    }

    #[test]
    fn random_sparse_allows_no_edges_on_a_single_node() {
        assert_eq!(random_sparse(1, 0, 5, 0).nvals(), 0);
    }

    /// The weight draw must not overflow at the top of the range, and must stay in
    /// `[1, wmax]`, reaching both ends.
    #[test]
    fn weights_stay_in_range_even_for_wmax_u32_max() {
        let g = random_sparse(8, 40, u32::MAX, 7);
        assert_eq!(g.nvals(), 40);
        for (_, _, w) in entries(&g) {
            assert!((1.0..=f64::from(u32::MAX)).contains(&w), "weight {w}");
        }
        let drawn: Vec<f64> = entries(&random_sparse(8, 40, 3, 7))
            .into_iter()
            .map(|e| e.2)
            .collect();
        assert!(drawn.iter().all(|w| (1.0..=3.0).contains(w)));
        assert!(drawn.contains(&1.0) && drawn.contains(&3.0), "{drawn:?}");
    }

    // ---- generator sanity ------------------------------------------------

    /// `empty(n)` has no edges, so every node but the source is unreachable;
    /// `single_node()` is one node at distance 0 from itself.
    #[test]
    fn empty_and_single_node_have_no_edges() {
        let dist = dijkstra(&empty(4), 0);
        assert_eq!(dist[0], 0.0);
        assert!(dist[1..].iter().all(|d| d.is_infinite()));
        assert_eq!(dijkstra(&single_node(), 0), vec![0.0]);
    }

    /// `grid`, `path` and `star` hold exactly their arcs, each both ways at its weight.
    #[test]
    fn generators_have_their_shapes() {
        let both_ways = |pairs: &[(u64, u64)], w: f64| {
            let mut e: Vec<_> = pairs
                .iter()
                .flat_map(|&(u, v)| [(u, v, w), (v, u, w)])
                .collect();
            e.sort_by(|a, b| a.partial_cmp(b).unwrap());
            e
        };
        // 0 1 2
        // 3 4 5
        let g = grid(2, 3);
        assert_eq!(node_count(&g), 6);
        let grid_pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)];
        assert_eq!(entries(&g), both_ways(&grid_pairs, 1.0));
        let g = path(4);
        assert_eq!(node_count(&g), 4);
        assert_eq!(entries(&g), both_ways(&[(0, 1), (1, 2), (2, 3)], 1.0));
        let g = star(3, 2.0);
        assert_eq!(node_count(&g), 4);
        assert_eq!(entries(&g), both_ways(&[(0, 1), (0, 2), (0, 3)], 2.0));
    }

    /// `u == v` draws and pairs already drawn are drawn again: never a self-loop, and still
    /// exactly `m` edges. Three nodes and all six pairs, so a third of the draws are
    /// self-loops and the last pairs take many draws.
    #[test]
    fn random_sparse_draws_no_self_loops() {
        for seed in 0..32 {
            let g = random_sparse(3, 6, 5, seed);
            assert_eq!(g.nvals(), 6, "seed {seed}");
            assert!(entries(&g).iter().all(|e| e.0 != e.1), "seed {seed}");
        }
    }

    #[test]
    fn disconnected_components_cannot_reach_across() {
        let g = disconnected_components(3, 5, 6, 9, 1234);
        let dist = dijkstra(&g, 0); // component 0
        for (t, d) in dist.iter().enumerate().skip(5) {
            assert!(d.is_infinite(), "component 0 must not reach node {t}");
        }
    }

    /// The same seed gives the same graph, and another seed another graph (a generator
    /// that ignored its seed would pass the first half alone).
    #[test]
    fn random_sparse_is_deterministic_in_its_seed() {
        let a = entries(&random_sparse(20, 50, 9, 42));
        let b = entries(&random_sparse(20, 50, 9, 42));
        assert_eq!(a, b);
        assert_ne!(a, entries(&random_sparse(20, 50, 9, 43)));
    }

    // ---- ties: weight only, never path identity ---------------------------

    /// All `k` middle nodes are equally valid witnesses for the shortest
    /// `0 -> dst` path -- a checker that hardcodes "the path must be [0, 1,
    /// dst]" is wrong even though it would pass on `k == 1`.
    #[test]
    fn tied_paths_all_resolve_to_the_same_weight() {
        let k = 5;
        let hop_w = 1.5;
        let g = tied_paths(k, hop_w);
        let dst = k + 1;

        let dist = dijkstra(&g, 0);
        assert_eq!(dist[dst as usize], 2.0 * hop_w);

        for mid in 1..=k {
            assert_path_valid(&g, 0, dst, 2.0 * hop_w, &[0, mid, dst]);
        }
    }

    // ---- zero-weight edges and cycles --------------------------------------

    /// Every node on an all-zero-weight cycle is at distance `0.0` from
    /// every other, and `dijkstra` never re-reaches a node it settled (a
    /// non-strict relaxation would, over the zero-weight edge back to it).
    #[test]
    fn zero_weight_cycle_collapses_to_distance_zero() {
        let g = zero_weight_cycle(6);
        let dist = all_pairs(&g);
        for row in &dist {
            assert!(row.iter().all(|&d| d == 0.0));
        }
    }

    /// A zero-weight shortcut `u -> v` makes `v` exactly as cheap to reach
    /// as `u` -- it must not be ignored (a `<=` vs `<` relaxation bug would
    /// silently drop it) nor treated as negative (it must not be re-relaxed
    /// in a loop).
    #[test]
    fn zero_weight_edge_is_a_free_shortcut() {
        let base = grid(1, 3); // 0 -- 1 -- 2, unit weights
        let g = with_zero_weight_edge(base, 0, 2);
        let dist = dijkstra(&g, 0);
        assert_eq!(
            dist[2], 0.0,
            "the zero-weight shortcut must win over cost 2"
        );
    }
}

/// Self-tests of `check_index_against_oracle`.
mod checker_tests {
    use super::{
        Matrix, assert_path_valid, check_index_against_oracle, entries, grid, node_count, path,
        random_sparse, random_zero_heavy, tied_paths, weights, weights_agree, with_isolated_nodes,
        zero_weight_cycle,
    };

    /// A correct stand-in index: Dijkstra with predecessors, written
    /// separately from the kit's `dijkstra` so the checker is not tested
    /// against its own oracle's code path.
    fn reference_query(
        g: &Matrix<f64>,
        s: u64,
        t: u64,
    ) -> Option<(f64, Vec<u64>)> {
        let n = node_count(g);
        let edges = entries(g);
        let mut dist = vec![f64::INFINITY; n];
        let mut pred = vec![u64::MAX; n];
        let mut done = vec![false; n];
        dist[s as usize] = 0.0;
        for _ in 0..n {
            let u = (0..n)
                .filter(|&i| !done[i] && dist[i].is_finite())
                .min_by(|&a, &b| dist[a].total_cmp(&dist[b]))?;
            done[u] = true;
            for &(a, b, w) in &edges {
                if a as usize == u && dist[u] + w < dist[b as usize] {
                    dist[b as usize] = dist[u] + w;
                    pred[b as usize] = a;
                }
            }
            if u == t as usize {
                break;
            }
        }
        if !dist[t as usize].is_finite() {
            return None;
        }
        let mut p = vec![t];
        while *p.last().unwrap() != s {
            p.push(pred[*p.last().unwrap() as usize]);
        }
        p.reverse();
        Some((dist[t as usize], p))
    }

    #[test]
    fn checker_accepts_a_correct_index_on_varied_graphs() {
        let graphs = [
            grid(3, 3),
            path(5),
            tied_paths(3, 1.5),
            zero_weight_cycle(4),
            with_isolated_nodes(grid(2, 2), 2),
            random_sparse(8, 20, 9, 5),
        ];
        for g in &graphs {
            check_index_against_oracle(|s, t| reference_query(g, s, t), g);
        }
    }

    #[test]
    #[should_panic(expected = "index weight")]
    fn checker_catches_a_wrong_weight() {
        let g = path(4);
        check_index_against_oracle(
            |s, t| reference_query(&g, s, t).map(|(w, p)| (if w > 1.0 { w + 1.0 } else { w }, p)),
            &g,
        );
    }

    #[test]
    #[should_panic(expected = "no edge (0, 2)")]
    fn checker_catches_a_bogus_path() {
        // Only the nontrivial answers are corrupted: (0, 0) and (0, 1) stay
        // valid, so the panic has to come from the missing (0, 2) hop.
        let g = path(4);
        check_index_against_oracle(
            |s, t| {
                reference_query(&g, s, t)
                    .map(|(w, p)| if s == t { (w, p) } else { (w, vec![s, t]) })
            },
            &g,
        );
    }

    #[test]
    fn a_path_summed_in_another_order_is_valid() {
        let g = weights(4, &[(3, 2, 1e16), (2, 1, 1.0), (1, 0, 1.0)]);
        let shortcut_first = 1e16 + (1.0 + 1.0);
        assert_ne!(shortcut_first, 1e16 + 1.0 + 1.0);
        assert_path_valid(&g, 3, 0, shortcut_first, &[3, 2, 1, 0]);
        assert!(weights_agree(shortcut_first, 1e16));
        assert!(!weights_agree(f64::INFINITY, f64::INFINITY));
        assert!(!weights_agree(f64::NAN, 0.0));
        assert!(!weights_agree(3.0, 4.0));
        // Within the tolerance of 0, but a negative distance is never right.
        assert!(!weights_agree(-5e-7, 0.0));
    }

    /// A one-node path on a node the graph does not have: the early return for `src == dst`
    /// must not skip the bounds check.
    #[test]
    #[should_panic(expected = "path node 3 is not in the graph (3 nodes)")]
    fn assert_path_valid_rejects_a_node_outside_the_graph() {
        assert_path_valid(&path(3), 3, 3, 0.0, &[3]);
    }

    /// A real path whose weight is claimed wrong.
    #[test]
    #[should_panic(expected = "claimed weight is 3")]
    fn assert_path_valid_rejects_a_wrong_weight() {
        assert_path_valid(&path(3), 0, 2, 3.0, &[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "unreachable in the oracle")]
    fn checker_catches_a_path_to_an_unreachable_node() {
        let g = with_isolated_nodes(path(3), 1);
        check_index_against_oracle(
            |s, t| {
                if s == t {
                    Some((0.0, vec![s]))
                } else {
                    reference_query(&g, s, t).or(Some((1.0, vec![s, t])))
                }
            },
            &g,
        );
    }

    /// A closed (`+inf`) edge, with no other route and with a finite detour: a correct index
    /// passes, and one that answers over the closed edge fails, whatever weight it claims.
    #[test]
    fn checker_reads_a_plus_inf_edge_as_no_edge() {
        let cut = weights(3, &[(0, 1, f64::INFINITY), (1, 2, 1.0)]);
        let detour = weights(3, &[(0, 2, f64::INFINITY), (0, 1, 1.0), (1, 2, 1.0)]);
        for g in [&cut, &detour] {
            check_index_against_oracle(|s, t| reference_query(g, s, t), g);
        }
    }

    #[test]
    #[should_panic(expected = "(0, 1): unreachable in the oracle")]
    fn checker_catches_a_path_over_a_closed_edge_with_no_other_route() {
        let g = weights(3, &[(0, 1, f64::INFINITY), (1, 2, 1.0)]);
        check_index_against_oracle(
            |s, t| match (s, t) {
                (0, 1) => Some((f64::INFINITY, vec![0, 1])),
                _ => reference_query(&g, s, t),
            },
            &g,
        );
    }

    #[test]
    #[should_panic(expected = "path sums to inf, claimed weight is 2")]
    fn checker_catches_a_path_over_a_closed_edge_with_a_detour() {
        let g = weights(3, &[(0, 2, f64::INFINITY), (0, 1, 1.0), (1, 2, 1.0)]);
        check_index_against_oracle(
            |s, t| match (s, t) {
                (0, 2) => Some((2.0, vec![0, 2])),
                _ => reference_query(&g, s, t),
            },
            &g,
        );
    }

    #[test]
    #[should_panic(expected = "returned None")]
    fn checker_catches_a_missing_path() {
        let g = path(3);
        check_index_against_oracle(|s, t| (s == t).then(|| (0.0, vec![s])), &g);
    }

    /// Two zero-weight arcs between 0 and 1, then 1 -> 2 at cost 1.
    fn zero_loop() -> Matrix<f64> {
        weights(3, &[(0, 1, 0.0), (1, 0, 0.0), (1, 2, 1.0)])
    }

    /// The failure core found: right weight, every hop an edge, yet node 0
    /// and node 1 each appear twice. `assert_path_valid` accepts it; the
    /// checker's separate simplicity check is what refuses it (next test).
    #[test]
    fn a_repeating_path_passes_the_witness_check() {
        let g = zero_loop();
        assert_path_valid(&g, 0, 2, 1.0, &[0, 1, 0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "path repeats node")]
    fn checker_rejects_a_repeating_path_with_correct_weight_and_hops() {
        let g = zero_loop();
        check_index_against_oracle(
            |s, t| match (s, t) {
                (0, 2) => Some((1.0, vec![0, 1, 0, 1, 2])),
                _ => reference_query(&g, s, t),
            },
            &g,
        );
    }

    #[test]
    fn checker_accepts_a_correct_index_on_zero_heavy_graphs() {
        for seed in 0..40u64 {
            let g = random_zero_heavy(8, 24, 5, seed);
            assert!(entries(&g).iter().any(|e| e.2 == 0.0), "seed {seed}");
            check_index_against_oracle(|s, t| reference_query(&g, s, t), &g);
        }
    }
}
