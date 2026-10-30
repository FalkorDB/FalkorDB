//! Graph test kit: edge lists, generators and a shortest-path oracle.
//!
//! Plain descriptions of directed weighted graphs ([`EdgeList`], and [`TypedEdgeList`] with a
//! relationship type per edge), deterministic generators for them (grids, paths, stars,
//! random sparse graphs, zero-weight cycles, parallel edges, self-loops, isolated nodes,
//! several relationship types), and an oracle: [`dijkstra`] and [`all_pairs`], with
//! [`assert_path_valid`], [`assert_path_simple`] and [`check_index_against_oracle`] to check
//! any shortest-path answer against it. Nothing here knows about an index or a `Graph`.
//!
//! The oracle must not share a bug with the code it checks, so it is written without
//! reference to any engine code, and the self-tests at the bottom cross-check [`dijkstra`]
//! against an independent Floyd-Warshall and against hand-computed cases lifted from the C
//! flow tests (`tests/flow/test_cch.py` on `master`).
//!
//! Test-only (`#[cfg(test)]`): crate code reaches it as `crate::testkit`. Index-specific
//! test kits build on it and re-export it.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;

/// A plain directed edge list over nodes `0..n`. May contain parallel edges
/// and self-loops (`u == v`); `to_input` (the CCH test kit) is what resolves both into the
/// clean shape `CchInput` expects.
#[derive(Clone, Debug, Default)]
pub struct EdgeList {
    pub n: usize,
    pub edges: Vec<(u64, u64, f64)>,
}

impl EdgeList {
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
        }
    }

    pub fn push(
        &mut self,
        u: u64,
        v: u64,
        w: f64,
    ) {
        debug_assert!((u as usize) < self.n && (v as usize) < self.n);
        self.edges.push((u, v, w));
    }
}

/// Like [`EdgeList`], but each edge also carries a relationship-type id
/// (`0..rel_count`) -- the C graph underneath a CCH index has many
/// relationship types and an index definition selects a subset of them
/// (`CchIndexDef::rel_types`); [`TypedEdgeList::select`]
/// performs the same selection before parallel edges are resolved to the cheapest, and
/// [`random_multi_type`] generates one directly so tests of the graph input can
/// build a matching Cypher graph with `TypedEdgeList::to_cypher_create` (the CCH test kit).
#[derive(Clone, Debug, Default)]
pub struct TypedEdgeList {
    pub n: usize,
    pub edges: Vec<(u64, u64, f64, u16)>,
}

impl TypedEdgeList {
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            edges: Vec::new(),
        }
    }

    pub fn push(
        &mut self,
        u: u64,
        v: u64,
        w: f64,
        rel_type: u16,
    ) {
        debug_assert!((u as usize) < self.n && (v as usize) < self.n);
        self.edges.push((u, v, w, rel_type));
    }

    /// Projects down to a plain [`EdgeList`] holding only the edges whose
    /// `rel_type` is in `rel_types` -- the selection an index definition
    /// performs before `to_input` (the CCH test kit)'s resolution runs over the result.
    #[must_use]
    pub fn select(
        &self,
        rel_types: &[u16],
    ) -> EdgeList {
        let mut out = EdgeList::new(self.n);
        for &(u, v, w, t) in &self.edges {
            if rel_types.contains(&t) {
                out.push(u, v, w);
            }
        }
        out
    }
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
        // `dijkstra`.
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

/// Plain Dijkstra from `src` over `edges`, taken as given -- parallel edges
/// and self-loops are relaxed like any other edge rather than pre-resolved,
/// since this is the oracle and must not share the resolution step with
/// `to_input` (the CCH test kit). `f64::INFINITY` for every node `src` cannot reach.
///
/// # Panics
/// If `src` is out of range, or any edge weight is negative (`-inf` included) or NaN:
/// the oracle only promises correctness over the weight domain `CchInput` allows
/// (`>= 0`). A `+inf` edge (a closed edge) is relaxed like any other and never
/// improves a distance, so it reads as no edge.
///
/// Not checked against FalkorDB's own Dijkstra (`dijkstra_single_path` behind
/// `algo.SPpaths`, `runtime/functions/algo_procedures.rs`), on purpose: that one is private
/// and needs a `Runtime` and a loaded `Graph`, so it is only reachable through Cypher, and
/// it silently skips edges whose weight is not finite: the same answer for a `+inf` edge,
/// but NaN and `-inf` are input this oracle must reject. It is the oracle of the flow tests
/// instead (`tests/flow/test_cch.py` compares CCH queries with `algo.SPpaths`). If a pure
/// shortest-path function over an edge list is ever added to the engine, cross-check this
/// one against it.
#[must_use]
pub fn dijkstra(
    edges: &EdgeList,
    src: u64,
) -> Vec<f64> {
    let n = edges.n;
    assert!((src as usize) < n, "src {src} out of range for {n} nodes");

    let mut adj: Vec<Vec<(u64, f64)>> = vec![Vec::new(); n];
    for &(u, v, w) in &edges.edges {
        assert!(
            w >= 0.0,
            "dijkstra oracle requires non-negative weights, got {w} on ({u}, {v})"
        );
        adj[u as usize].push((v, w));
    }

    let mut dist = vec![f64::INFINITY; n];
    dist[src as usize] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(HeapEntry {
        dist: 0.0,
        node: src,
    });

    while let Some(HeapEntry { dist: d, node: u }) = heap.pop() {
        if d > dist[u as usize] {
            continue;
        }
        for &(v, w) in &adj[u as usize] {
            let nd = d + w;
            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                heap.push(HeapEntry { dist: nd, node: v });
            }
        }
    }

    dist
}

/// `dijkstra` from every node. `result[s][t]` is the cheapest `s -> t`
/// weight, `f64::INFINITY` if unreachable.
#[must_use]
pub fn all_pairs(edges: &EdgeList) -> Vec<Vec<f64>> {
    (0..edges.n as u64).map(|s| dijkstra(edges, s)).collect()
}

/// Checks that `path` is a valid witness for an `src -> dst` query result:
/// consecutive nodes are joined by an edge in `edges`, the path starts at
/// `src` and ends at `dst`, and the sum of the cheapest edge at each hop
/// equals `weight`.
///
/// `path` is a node sequence (as `Cch::query` returns), not an edge
/// list: the cheapest parallel edge is looked up per hop, matching the
/// "cheapest weight over all parallel edges" rule rather than demanding the
/// path carry a particular edge's identity.
///
/// # Panics
/// If the path is empty, does not start/end at `src`/`dst`, skips a pair
/// with no edge between them, or its weight sum does not match `weight`
/// ([`weights_agree`]).
pub fn assert_path_valid(
    edges: &EdgeList,
    src: u64,
    dst: u64,
    weight: f64,
    path: &[u64],
) {
    assert!(!path.is_empty(), "path must not be empty");
    assert_eq!(path[0], src, "path does not start at src");
    assert_eq!(*path.last().unwrap(), dst, "path does not end at dst");

    if path.len() == 1 {
        assert_eq!(src, dst, "single-node path must have src == dst");
        assert_eq!(weight, 0.0, "a trivial src == dst path must weigh 0");
        return;
    }

    let mut cheapest: HashMap<(u64, u64), f64> = HashMap::new();
    for &(u, v, w) in &edges.edges {
        if u == v {
            continue;
        }
        cheapest
            .entry((u, v))
            .and_modify(|cur| {
                if w < *cur {
                    *cur = w;
                }
            })
            .or_insert(w);
    }

    let mut sum = 0.0;
    for hop in path.windows(2) {
        let (u, v) = (hop[0], hop[1]);
        let w = *cheapest
            .get(&(u, v))
            .unwrap_or_else(|| panic!("no edge ({u}, {v}) on claimed path"));
        sum += w;
    }

    assert!(
        weights_agree(sum, weight),
        "path sums to {sum}, claimed weight is {weight}"
    );
}

/// Two path weights agree: both finite, and within `1e-6` or a relative
/// `1e-9`, whichever is looser. Summing the same edges in another order (a
/// shortcut adds its two halves first) rounds differently, and at large
/// weights the difference exceeds any fixed absolute tolerance: `1e16 + 1 + 1`
/// is `1e16` hop by hop but `10000000000000002` with the units added first.
#[must_use]
pub fn weights_agree(
    a: f64,
    b: f64,
) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-6_f64.max(1e-9 * a.abs().max(b.abs()))
}

/// Deterministic generator (SplitMix64), so a failing seed reproduces.
pub(crate) fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `[0.0, 1.0)` from the next SplitMix64 draw.
pub(crate) fn next_unit(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// `[lo, hi)` from the next SplitMix64 draw. `hi` must be `> lo`.
pub(crate) fn next_range(
    state: &mut u64,
    lo: u32,
    hi: u32,
) -> u32 {
    lo + (splitmix(state) % u64::from(hi - lo)) as u32
}

/// A node id in `[0, n)` from the next SplitMix64 draw. `n` must be `> 0`.
pub(crate) fn next_node(
    state: &mut u64,
    n: usize,
) -> u64 {
    splitmix(state) % n as u64
}

/// An integer weight drawn uniformly from `[1, wmax]`. Written as `0..wmax` plus one so that
/// `wmax == u32::MAX` does not overflow.
pub(crate) fn next_weight(
    state: &mut u64,
    wmax: u32,
) -> f64 {
    f64::from(next_range(state, 0, wmax) + 1)
}

/// An `rows x cols` grid, 4-connected, unit weights. `rows, cols >= 1`.
#[must_use]
pub fn grid(
    rows: u64,
    cols: u64,
) -> EdgeList {
    assert!(rows >= 1 && cols >= 1);
    let n = (rows * cols) as usize;
    let mut g = EdgeList::new(n);
    let id = |r: u64, c: u64| r * cols + c;
    for r in 0..rows {
        for c in 0..cols {
            let u = id(r, c);
            if c + 1 < cols {
                let v = id(r, c + 1);
                g.push(u, v, 1.0);
                g.push(v, u, 1.0);
            }
            if r + 1 < rows {
                let v = id(r + 1, c);
                g.push(u, v, 1.0);
                g.push(v, u, 1.0);
            }
        }
    }
    g
}

/// `n` nodes, `m` random directed edges, integer weights in `[1, wmax]`.
/// `u == v` is skipped when drawn (no self-loops), matching `_rand_graph` in
/// the C flow tests; deterministic in `seed`.
///
/// # Panics
/// If `m > 0` and `n < 2`: with one node every drawn pair is a self-loop, so the graph could
/// not have `m` edges.
#[must_use]
pub fn random_sparse(
    n: usize,
    m: usize,
    wmax: u32,
    seed: u64,
) -> EdgeList {
    assert!(n >= 1 && wmax >= 1);
    assert!(m == 0 || n >= 2, "{m} edges need at least 2 nodes, got {n}");
    let mut state = seed;
    let mut g = EdgeList::new(n);
    let mut remaining = m;
    // Bounded retries: a tiny `n` makes "u != v" increasingly unlikely to
    // hit on the first draw, but never impossible for `n >= 2`.
    let mut guard = m.saturating_mul(8).max(64);
    while remaining > 0 && guard > 0 {
        guard -= 1;
        let u = next_node(&mut state, n);
        let v = next_node(&mut state, n);
        if u == v {
            continue;
        }
        let w = next_weight(&mut state, wmax);
        g.push(u, v, w);
        remaining -= 1;
    }
    g
}

/// `edges` directed pairs, each duplicated `copies` times with a distinct
/// random weight in `[1, wmax]` per copy -- so a correct oracle must resolve
/// to the cheapest one, per the `CchInput` contract.
#[must_use]
pub fn with_parallel_edges(
    n: usize,
    pairs: &[(u64, u64)],
    copies: usize,
    wmax: u32,
    seed: u64,
) -> EdgeList {
    assert!(copies >= 1 && wmax >= 1);
    let mut state = seed;
    let mut g = EdgeList::new(n);
    for &(u, v) in pairs {
        for _ in 0..copies {
            let w = next_weight(&mut state, wmax);
            g.push(u, v, w);
        }
    }
    g
}

/// `base` with a self-loop of weight `w` added at every node in `at`.
#[must_use]
pub fn with_self_loops(
    mut base: EdgeList,
    at: &[u64],
    w: f64,
) -> EdgeList {
    for &node in at {
        base.push(node, node, w);
    }
    base
}

/// `count` disjoint components, each an independent `random_sparse` graph of
/// `nodes_per_component` nodes and `edges_per_component` edges, with no edge
/// crossing components.
#[must_use]
pub fn disconnected_components(
    count: usize,
    nodes_per_component: usize,
    edges_per_component: usize,
    wmax: u32,
    seed: u64,
) -> EdgeList {
    assert!(count >= 1 && nodes_per_component >= 1);
    let n = count * nodes_per_component;
    let mut g = EdgeList::new(n);
    for c in 0..count {
        let offset = (c * nodes_per_component) as u64;
        let component = random_sparse(
            nodes_per_component,
            edges_per_component,
            wmax,
            seed ^ (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        );
        for (u, v, w) in component.edges {
            g.push(u + offset, v + offset, w);
        }
    }
    g
}

/// A single node, no edges.
#[must_use]
pub fn single_node() -> EdgeList {
    EdgeList::new(1)
}

/// `n` nodes, no edges.
#[must_use]
pub fn empty(n: usize) -> EdgeList {
    EdgeList::new(n)
}

/// A star: node `0` is the hub, `leaves` outer nodes, edges both ways with
/// weight `w`.
#[must_use]
pub fn star(
    leaves: u64,
    w: f64,
) -> EdgeList {
    assert!(leaves >= 1);
    let n = (leaves + 1) as usize;
    let mut g = EdgeList::new(n);
    for leaf in 1..=leaves {
        g.push(0, leaf, w);
        g.push(leaf, 0, w);
    }
    g
}

/// A simple directed path `0 -> 1 -> ... -> n-1`, unit weights, plus the
/// reverse arcs so the underlying skeleton is connected both ways.
#[must_use]
pub fn path(n: usize) -> EdgeList {
    assert!(n >= 1);
    let mut g = EdgeList::new(n);
    for u in 0..n.saturating_sub(1) {
        let (u, v) = (u as u64, (u + 1) as u64);
        g.push(u, v, 1.0);
        g.push(v, u, 1.0);
    }
    g
}

/// `k` node-disjoint two-hop paths `0 -> mid -> dst`, each hop costing
/// `hop_w` -- every one of the `k` paths ties at `2 * hop_w`. A trap for a
/// checker that asserts path *identity* instead of weight: [`assert_path_valid`]
/// must accept whichever of the `k` middle nodes the caller passed, not a
/// single canonical one. `k >= 1`.
#[must_use]
pub fn tied_paths(
    k: u64,
    hop_w: f64,
) -> EdgeList {
    assert!(k >= 1);
    let n = (k + 2) as usize;
    let dst = k + 1;
    let mut g = EdgeList::new(n);
    for mid in 1..=k {
        g.push(0, mid, hop_w);
        g.push(mid, 0, hop_w);
        g.push(mid, dst, hop_w);
        g.push(dst, mid, hop_w);
    }
    g
}

/// A directed cycle `0 -> 1 -> ... -> len-1 -> 0`, every edge weight `0.0`.
/// Dijkstra must not loop forever relaxing a zero-weight edge (the `nd <
/// dist[v]` check must be strict, not `<=`), and every node on the cycle is
/// at distance `0.0` from every other. `len >= 2`.
#[must_use]
pub fn zero_weight_cycle(len: u64) -> EdgeList {
    assert!(len >= 2);
    let n = len as usize;
    let mut g = EdgeList::new(n);
    for u in 0..len {
        g.push(u, (u + 1) % len, 0.0);
    }
    g
}

/// `base` with one extra zero-weight directed edge `u -> v` -- a free
/// shortcut: distance to `v` can never exceed distance to `u`.
#[must_use]
pub fn with_zero_weight_edge(
    mut base: EdgeList,
    u: u64,
    v: u64,
) -> EdgeList {
    base.push(u, v, 0.0);
    base
}

/// `n` nodes, `m` random directed edges tagged with one of `rel_count`
/// relationship types (`0..rel_count`), integer weights in `[1, wmax]`.
/// Deterministic in `seed`; mirrors [`random_sparse`] but keeps edges on the
/// same `(u, v)` pair under different types distinct from each other --
/// [`TypedEdgeList::select`] is what resolves a chosen subset of types down
/// to parallel-edge semantics, not this generator.
///
/// # Panics
/// If `m > 0` and `n < 2`, as [`random_sparse`].
#[must_use]
pub fn random_multi_type(
    n: usize,
    m: usize,
    wmax: u32,
    rel_count: u16,
    seed: u64,
) -> TypedEdgeList {
    assert!(n >= 1 && wmax >= 1 && rel_count >= 1);
    assert!(m == 0 || n >= 2, "{m} edges need at least 2 nodes, got {n}");
    let mut state = seed;
    let mut g = TypedEdgeList::new(n);
    let mut remaining = m;
    let mut guard = m.saturating_mul(8).max(64);
    while remaining > 0 && guard > 0 {
        guard -= 1;
        let u = next_node(&mut state, n);
        let v = next_node(&mut state, n);
        if u == v {
            continue;
        }
        let w = next_weight(&mut state, wmax);
        let t = next_range(&mut state, 0, u32::from(rel_count)) as u16;
        g.push(u, v, w, t);
        remaining -= 1;
    }
    g
}

/// `base` with `extra` additional isolated nodes appended after its existing
/// `n`. A real `Graph -> CchInput` can have `n` larger than any node id
/// that appears in an edge -- a node with no incident edge among the indexed
/// relationship types is still a node in the graph -- and no generator above
/// produces that shape on its own, since they only draw node ids that get
/// used.
#[must_use]
pub fn with_isolated_nodes(
    mut base: EdgeList,
    extra: usize,
) -> EdgeList {
    base.n += extra;
    base
}

/// A returned path must be simple: no node appears twice. With zero-weight
/// edges a walk can revisit a node while keeping the right weight and valid
/// hops, so [`assert_path_valid`] alone does not catch it.
///
/// # Panics
/// If any node repeats.
pub fn assert_path_simple(path: &[u64]) {
    let mut seen = HashSet::new();
    for &node in path {
        assert!(seen.insert(node), "path repeats node {node}: {path:?}");
    }
}

/// Like [`random_sparse`], but about half the edges weigh exactly `0.0`
/// (drawn per edge), the rest an integer in `[1, wmax]`. Zero-weight edges
/// create zero-weight cycles and many equal-weight walks, which is where a
/// path can go wrong while its weight stays right.
#[must_use]
pub fn random_zero_heavy(
    n: usize,
    m: usize,
    wmax: u32,
    seed: u64,
) -> EdgeList {
    let mut g = random_sparse(n, m, wmax, seed);
    let mut state = seed ^ 0xA5A5_A5A5_A5A5_A5A5;
    for e in &mut g.edges {
        if splitmix(&mut state).is_multiple_of(2) {
            e.2 = 0.0;
        }
    }
    g
}

/// End-to-end differential check of an index against the oracle: for every
/// ordered pair `(s, t)`, `index_query(s, t)` must agree with [`all_pairs`]
/// on reachability and weight ([`weights_agree`]), and every returned path must
/// pass [`assert_path_valid`] (a witness check: weight only, never path
/// identity, so ties are fine) and [`assert_path_simple`] (no repeated node).
///
/// Convention for `s == t` (C's `test01_create_query_drop`): the index
/// returns weight `0` and the single-node path `[s]`; `None` there is a
/// failure.
///
/// # Panics
/// On the first disagreement, naming the pair.
pub fn check_index_against_oracle(
    index_query: impl Fn(u64, u64) -> Option<(f64, Vec<u64>)>,
    edges: &EdgeList,
) {
    let oracle = all_pairs(edges);
    for s in 0..edges.n as u64 {
        for t in 0..edges.n as u64 {
            let expected = oracle[s as usize][t as usize];
            let got = index_query(s, t);
            if s == t {
                let (w, path) =
                    got.unwrap_or_else(|| panic!("({s}, {t}): s == t must return weight 0"));
                assert_eq!(w, 0.0, "({s}, {t}): s == t must weigh 0, got {w}");
                assert_path_valid(edges, s, t, w, &path);
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
                assert_path_valid(edges, s, t, w, &path);
                assert_path_simple(&path);
            }
        }
    }
}

mod self_tests {
    use super::{
        EdgeList, TypedEdgeList, all_pairs, assert_path_valid, dijkstra, disconnected_components,
        empty, grid, next_unit, path, random_multi_type, random_sparse, single_node, splitmix,
        star, tied_paths, with_parallel_edges, with_self_loops, with_zero_weight_edge,
        zero_weight_cycle,
    };

    /// Independent second implementation used only to cross-check
    /// [`dijkstra`] (self-tests only) -- if these two agree, a shared bug is
    /// far less likely than if `dijkstra` were only checked against itself.
    fn floyd_warshall(edges: &EdgeList) -> Vec<Vec<f64>> {
        let n = edges.n;
        let mut d = vec![vec![f64::INFINITY; n]; n];
        for (i, row) in d.iter_mut().enumerate() {
            row[i] = 0.0;
        }
        for &(u, v, w) in &edges.edges {
            let (u, v) = (u as usize, v as usize);
            if w < d[u][v] {
                d[u][v] = w;
            }
        }
        for k in 0..n {
            for i in 0..n {
                if d[i][k].is_infinite() {
                    continue;
                }
                for j in 0..n {
                    let via = d[i][k] + d[k][j];
                    if via < d[i][j] {
                        d[i][j] = via;
                    }
                }
            }
        }
        d
    }

    /// `test01_create_query_drop`'s valley: a->b direct is 10, a->c->b is 2.
    #[test]
    fn valley_matches_c_test01() {
        let mut g = EdgeList::new(3);
        g.push(0, 1, 10.0);
        g.push(1, 0, 10.0);
        g.push(0, 2, 1.0);
        g.push(2, 0, 1.0);
        g.push(2, 1, 1.0);
        g.push(1, 2, 1.0);

        let dist = dijkstra(&g, 0);
        assert_eq!(dist[1], 2.0);
        assert_eq!(dist[2], 1.0);
        assert_path_valid(&g, 0, 1, 2.0, &[0, 2, 1]);
    }

    /// `test01`'s src == dst convention: weight 0, single-node path.
    #[test]
    fn src_equals_dst_is_zero_weight_single_node_path() {
        let g = EdgeList::new(3);
        assert_eq!(dijkstra(&g, 0)[0], 0.0);
        assert_path_valid(&g, 0, 0, 0.0, &[0]);
    }

    /// `test14_parallel_selfloop_unreachable`: cheapest parallel edge wins,
    /// a self-loop is a no-op, an isolated node is unreachable.
    #[test]
    fn parallel_selfloop_unreachable_matches_c_test14() {
        let mut g = EdgeList::new(4);
        g.push(0, 1, 5.0);
        g.push(0, 1, 2.0);
        g.push(1, 2, 5.0);
        g.push(1, 2, 1.0);
        g.push(0, 0, 0.5);

        let dist = dijkstra(&g, 0);
        assert_eq!(dist[1], 2.0);
        assert_eq!(dist[2], 3.0);
        assert!(dist[3].is_infinite());
        assert_path_valid(&g, 0, 2, 3.0, &[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "requires non-negative")]
    fn dijkstra_rejects_a_negative_weight() {
        let mut g = EdgeList::new(2);
        g.push(0, 1, -1.0);
        let _ = dijkstra(&g, 0);
    }

    #[test]
    #[should_panic(expected = "requires non-negative")]
    fn dijkstra_rejects_a_nan_weight() {
        let mut g = EdgeList::new(2);
        g.push(0, 1, f64::NAN);
        let _ = dijkstra(&g, 0);
    }

    #[test]
    #[should_panic(expected = "requires non-negative")]
    fn dijkstra_rejects_a_negative_infinite_weight() {
        let mut g = EdgeList::new(2);
        g.push(0, 1, f64::NEG_INFINITY);
        let _ = dijkstra(&g, 0);
    }

    #[test]
    #[should_panic(expected = "need at least 2 nodes")]
    fn random_sparse_refuses_edges_on_a_single_node() {
        let _ = random_sparse(1, 3, 5, 0);
    }

    #[test]
    fn random_sparse_allows_no_edges_on_a_single_node() {
        assert!(random_sparse(1, 0, 5, 0).edges.is_empty());
    }

    /// The weight draw must not overflow at the top of the range, and must stay in `[1, wmax]`.
    #[test]
    fn weights_stay_in_range_even_for_wmax_u32_max() {
        let g = random_sparse(4, 40, u32::MAX, 7);
        assert_eq!(g.edges.len(), 40);
        for &(_, _, w) in &g.edges {
            assert!((1.0..=f64::from(u32::MAX)).contains(&w), "weight {w}");
        }
    }

    /// Dijkstra and an independent Floyd-Warshall must agree, all-pairs, on
    /// many random graphs. A failing seed is printed so it reproduces.
    #[test]
    fn dijkstra_agrees_with_floyd_warshall() {
        for seed in 0..200u64 {
            let mut state = seed;
            let n = 2 + (splitmix(&mut state) % 12) as usize;
            let m = 1 + (splitmix(&mut state) % 40) as usize;
            let wmax = 1 + (splitmix(&mut state) % 9) as u32;
            let g = random_sparse(n, m, wmax, seed);

            let fw = floyd_warshall(&g);
            let dj = all_pairs(&g);
            for s in 0..n {
                for t in 0..n {
                    assert_eq!(
                        dj[s][t], fw[s][t],
                        "seed {seed}: dijkstra and floyd-warshall disagree on ({s}, {t})"
                    );
                }
            }
        }
    }

    /// Same cross-check over the structured generators, not just
    /// `random_sparse` -- a bug that only shows up on a grid or a star
    /// should not hide behind "only random graphs were ever checked".
    #[test]
    fn dijkstra_agrees_with_floyd_warshall_on_every_generator() {
        let graphs: Vec<EdgeList> = vec![
            grid(3, 4),
            with_parallel_edges(4, &[(0, 1), (1, 2), (2, 3)], 3, 9, 7),
            with_self_loops(grid(2, 2), &[0, 3], 0.5),
            disconnected_components(3, 4, 5, 9, 11),
            single_node(),
            empty(5),
            star(6, 2.0),
            path(5),
            tied_paths(4, 1.5),
            zero_weight_cycle(5),
            with_zero_weight_edge(grid(3, 3), 0, 8),
        ];
        for (idx, g) in graphs.iter().enumerate() {
            let fw = floyd_warshall(g);
            let dj = all_pairs(g);
            for s in 0..g.n {
                for t in 0..g.n {
                    assert_eq!(
                        dj[s][t], fw[s][t],
                        "graph #{idx}: dijkstra and floyd-warshall disagree on ({s}, {t})"
                    );
                }
            }
        }
    }

    // ---- generator sanity ------------------------------------------------

    #[test]
    fn empty_has_no_edges_and_every_node_unreachable_from_every_other() {
        let g = empty(4);
        let dist = dijkstra(&g, 0);
        assert_eq!(dist[0], 0.0);
        assert!(dist[1..].iter().all(|d| d.is_infinite()));
    }

    #[test]
    fn single_node_has_trivial_distance_to_itself() {
        let g = single_node();
        assert_eq!(dijkstra(&g, 0), vec![0.0]);
    }

    #[test]
    fn disconnected_components_cannot_reach_across() {
        let g = disconnected_components(3, 5, 6, 9, 1234);
        let dist = dijkstra(&g, 0); // component 0
        for (t, d) in dist.iter().enumerate().skip(5) {
            assert!(d.is_infinite(), "component 0 must not reach node {t}");
        }
    }

    #[test]
    fn random_sparse_is_deterministic_in_its_seed() {
        let a = random_sparse(20, 50, 9, 42);
        let b = random_sparse(20, 50, 9, 42);
        assert_eq!(a.edges, b.edges);
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

    /// `assert_path_valid` must not reject a witness just because a
    /// *cheaper* tie exists elsewhere in the graph that the witness didn't
    /// take -- it validates the path it was given, not whether a better one
    /// exists.
    #[test]
    fn assert_path_valid_checks_the_given_path_not_whether_a_cheaper_one_exists() {
        let g = tied_paths(3, 2.0);
        // every one of these is independently a valid, equally-cheap witness
        assert_path_valid(&g, 0, 4, 4.0, &[0, 1, 4]);
        assert_path_valid(&g, 0, 4, 4.0, &[0, 2, 4]);
        assert_path_valid(&g, 0, 4, 4.0, &[0, 3, 4]);
    }

    // ---- zero-weight edges and cycles --------------------------------------

    /// Every node on an all-zero-weight cycle is at distance `0.0` from
    /// every other, and `dijkstra` terminates (it must not keep relaxing a
    /// zero-weight edge forever).
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

    // ---- multi-type edges and Cypher rendering -----------------------------

    #[test]
    fn random_multi_type_is_deterministic_and_stays_in_range() {
        let a = random_multi_type(10, 30, 9, 3, 7);
        let b = random_multi_type(10, 30, 9, 3, 7);
        assert_eq!(a.edges, b.edges);
        for &(u, v, w, t) in &a.edges {
            assert!((u as usize) < a.n && (v as usize) < a.n);
            assert!((1.0..=9.0).contains(&w));
            assert!(t < 3);
        }
    }

    /// `select` keeps exactly the edges whose type is requested, parallel edges and their
    /// order included; a type never drawn gives an edgeless graph on the same `n`.
    #[test]
    fn typed_edge_list_select_keeps_only_the_given_types() {
        let mut g = TypedEdgeList::new(3);
        g.push(0, 1, 5.0, 0);
        g.push(0, 1, 2.0, 0);
        g.push(1, 2, 1.0, 1);
        assert_eq!(g.select(&[0]).edges, [(0, 1, 5.0), (0, 1, 2.0)]);
        assert_eq!(g.select(&[0, 1]).edges.len(), 3);
        let none = g.select(&[2]);
        assert!(none.edges.is_empty());
        assert_eq!(none.n, 3);
    }

    #[test]
    fn next_unit_stays_in_unit_interval() {
        let mut state = 999u64;
        for _ in 0..1000 {
            let u = next_unit(&mut state);
            assert!((0.0..1.0).contains(&u));
        }
    }
}

/// Self-tests of `check_index_against_oracle`.
mod checker_tests {
    use super::{
        EdgeList, assert_path_simple, assert_path_valid, check_index_against_oracle, grid, path,
        random_sparse, random_zero_heavy, tied_paths, weights_agree, with_isolated_nodes,
        zero_weight_cycle,
    };

    /// A correct stand-in index: Dijkstra with predecessors, written
    /// separately from the kit's `dijkstra` so the checker is not tested
    /// against its own oracle's code path.
    fn reference_query(
        g: &EdgeList,
        s: u64,
        t: u64,
    ) -> Option<(f64, Vec<u64>)> {
        let n = g.n;
        let mut dist = vec![f64::INFINITY; n];
        let mut pred = vec![u64::MAX; n];
        let mut done = vec![false; n];
        dist[s as usize] = 0.0;
        for _ in 0..n {
            let u = (0..n)
                .filter(|&i| !done[i] && dist[i].is_finite())
                .min_by(|&a, &b| dist[a].total_cmp(&dist[b]))?;
            done[u] = true;
            for &(a, b, w) in &g.edges {
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
        let mut g = EdgeList::new(4);
        g.push(3, 2, 1e16);
        g.push(2, 1, 1.0);
        g.push(1, 0, 1.0);
        let shortcut_first = 1e16 + (1.0 + 1.0);
        assert_ne!(shortcut_first, 1e16 + 1.0 + 1.0);
        assert_path_valid(&g, 3, 0, shortcut_first, &[3, 2, 1, 0]);
        assert!(weights_agree(shortcut_first, 1e16));
        assert!(!weights_agree(f64::INFINITY, f64::INFINITY));
        assert!(!weights_agree(f64::NAN, 0.0));
        assert!(!weights_agree(3.0, 4.0));
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

    #[test]
    #[should_panic(expected = "returned None")]
    fn checker_catches_a_missing_path() {
        let g = path(3);
        check_index_against_oracle(|s, t| (s == t).then(|| (0.0, vec![s])), &g);
    }

    /// Two zero-weight arcs between 0 and 1, then 1 -> 2 at cost 1.
    fn zero_loop() -> EdgeList {
        let mut g = EdgeList::new(3);
        g.push(0, 1, 0.0);
        g.push(1, 0, 0.0);
        g.push(1, 2, 1.0);
        g
    }

    /// The failure core found: right weight, every hop an edge, yet node 0
    /// and node 1 each appear twice. `assert_path_valid` accepts it.
    #[test]
    fn a_repeating_path_passes_the_witness_check_but_not_the_simplicity_check() {
        let g = zero_loop();
        assert_path_valid(&g, 0, 2, 1.0, &[0, 1, 0, 1, 2]);
        assert!(
            std::panic::catch_unwind(|| assert_path_simple(&[0, 1, 0, 1, 2])).is_err(),
            "a path repeating a node must be rejected"
        );
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
            assert!(g.edges.iter().any(|e| e.2 == 0.0), "seed {seed}");
            check_index_against_oracle(|s, t| reference_query(&g, s, t), &g);
        }
    }
}
