//! Test oracle and fixtures for CCH (#3207).
//!
//! Written from the [`CchInput`] contract and the C reference on FalkorDB's
//! `master` branch (`src/algorithms/cch.c`, `tests/flow/test_cch.py`) -- never
//! from any Rust CCH implementation.
//! This module is the oracle the CCH code is checked against, so it must
//! not share a bug with the code it checks; see the self-tests at the
//! bottom, which cross-check [`dijkstra`] against an independent
//! Floyd-Warshall and against hand-computed cases lifted from the C
//! flow tests.
//!
//! Much of this is not CCH-specific ([`EdgeList`] / [`TypedEdgeList`] as graph
//! descriptions, the generators, the Cypher scripts, [`dijkstra`], the SplitMix
//! generator). It lives here because `graph` has no shared test-support module yet, and
//! the oracle is written without reference to the code it checks; the general-purpose
//! pieces are candidates to move into such a module.

use super::{CchInput, Csr, WeightCsr};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;

/// Created before the nodes and the edge batches, so the `MATCH (a:N{v:e[0]}),(b:N{v:e[1]})`
/// that wires each edge is an index seek, not a scan of every `:N` node per endpoint (the
/// C flow tests do the same: `tests/flow/test_cch.py`, "fast grid construction").
const CREATE_NODE_INDEX: &str = "CREATE INDEX FOR (n:N) ON (n.v);\n";

/// A plain directed edge list over nodes `0..n`. May contain parallel edges
/// and self-loops (`u == v`); [`to_input`] is what resolves both into the
/// clean shape `CchInput` expects.
#[derive(Clone, Debug, Default)]
pub struct EdgeList {
    pub n: usize,
    pub edges: Vec<(u32, u32, f64)>,
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
        u: u32,
        v: u32,
        w: f64,
    ) {
        debug_assert!((u as usize) < self.n && (v as usize) < self.n);
        self.edges.push((u, v, w));
    }

    /// Renders this graph as a Cypher script that reproduces it exactly, in
    /// the idiom the C flow tests use (`tests/flow/test_cch.py`'s
    /// `_rand_graph`): an index on `:N(v)`, then nodes labeled `:N` with
    /// property `v` holding the plain `0..n` node id, then edges labeled
    /// `rel_type` with weight property `w`. One statement per step, so the
    /// script is valid even when `n == 0` (it is then empty) or there are
    /// no edges. `rel_type` is
    /// pasted verbatim into the pattern -- the caller must pass a valid
    /// Cypher relationship-type name.
    #[must_use]
    pub fn to_cypher_create(
        &self,
        rel_type: &str,
    ) -> String {
        let mut script = String::new();
        if self.n > 0 {
            script.push_str(CREATE_NODE_INDEX);
            script.push_str(&format!(
                "UNWIND range(0,{}) AS i CREATE (:N {{v:i}});\n",
                self.n - 1
            ));
        }
        if !self.edges.is_empty() {
            let items: Vec<String> = self
                .edges
                .iter()
                .map(|&(u, v, w)| format!("[{u},{v},{w}]"))
                .collect();
            script.push_str(&format!(
                "UNWIND [{}] AS e MATCH (a:N{{v:e[0]}}),(b:N{{v:e[1]}}) \
                 CREATE (a)-[:{rel_type} {{w:e[2]}}]->(b);\n",
                items.join(",")
            ));
        }
        script
    }
}

/// Like [`EdgeList`], but each edge also carries a relationship-type id
/// (`0..rel_count`) -- the C graph underneath a CCH index has many
/// relationship types and an index definition selects a subset of them
/// (`CchIndexDef::rel_types`); [`TypedEdgeList::select`]
/// performs the same selection before parallel edges are resolved to the cheapest, and
/// [`random_multi_type`] generates one directly so tests of the graph input can
/// build a matching Cypher graph with [`TypedEdgeList::to_cypher_create`].
#[derive(Clone, Debug, Default)]
pub struct TypedEdgeList {
    pub n: usize,
    pub edges: Vec<(u32, u32, f64, u16)>,
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
        u: u32,
        v: u32,
        w: f64,
        rel_type: u16,
    ) {
        debug_assert!((u as usize) < self.n && (v as usize) < self.n);
        self.edges.push((u, v, w, rel_type));
    }

    /// Projects down to a plain [`EdgeList`] holding only the edges whose
    /// `rel_type` is in `rel_types` -- the selection an index definition
    /// performs before [`to_input`]'s resolution runs over the result.
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

    /// Renders this graph as a Cypher script, one relationship type per
    /// entry in `rel_names` (`rel_names[t]` names type id `t`); otherwise
    /// the same idiom as [`EdgeList::to_cypher_create`].
    ///
    /// # Panics
    /// If an edge's `rel_type` has no entry in `rel_names`.
    #[must_use]
    pub fn to_cypher_create(
        &self,
        rel_names: &[&str],
    ) -> String {
        for &(u, v, _, t) in &self.edges {
            assert!(
                (t as usize) < rel_names.len(),
                "edge ({u}, {v}) has rel_type {t}, but only {} names were given",
                rel_names.len()
            );
        }
        let mut script = String::new();
        if self.n > 0 {
            script.push_str(CREATE_NODE_INDEX);
            script.push_str(&format!(
                "UNWIND range(0,{}) AS i CREATE (:N {{v:i}});\n",
                self.n - 1
            ));
        }
        for (t, name) in rel_names.iter().enumerate() {
            let batch: Vec<String> = self
                .edges
                .iter()
                .filter(|&&(_, _, _, et)| et as usize == t)
                .map(|&(u, v, w, _)| format!("[{u},{v},{w}]"))
                .collect();
            if batch.is_empty() {
                continue;
            }
            script.push_str(&format!(
                "UNWIND [{}] AS e MATCH (a:N{{v:e[0]}}),(b:N{{v:e[1]}}) \
                 CREATE (a)-[:{name} {{w:e[2]}}]->(b);\n",
                batch.join(",")
            ));
        }
        script
    }
}

/// [`CchInput`] as its contract defines it: a symmetric, self-loop-free skeleton for
/// Phase 1, and directed weights where each `(u, v)` is the cheapest weight
/// over every parallel edge sharing that exact direction. Self-loops carry
/// no shortest-path information (`cch.c` drops the diagonal with
/// `GrB_OFFDIAG` before `METIS_NodeND`, and `test14_parallel_selfloop_unreachable`
/// treats a self-loop as a no-op) and are dropped from both the skeleton and
/// the weights.
///
/// Weights are **not** validated here, on purpose: the tests of `Cch::build`'s
/// `InvalidWeight` path build a deliberately invalid `CchInput` through this function.
/// (The oracles, [`dijkstra`] and [`lower_rank_distances`], do reject them.)
#[must_use]
pub fn to_input(edges: &EdgeList) -> CchInput {
    let n = edges.n;

    let mut best: HashMap<(u32, u32), f64> = HashMap::new();
    for &(u, v, w) in &edges.edges {
        if u == v {
            continue;
        }
        best.entry((u, v))
            .and_modify(|cur| {
                if w < *cur {
                    *cur = w;
                }
            })
            .or_insert(w);
    }

    let weights = weight_csr_from_pairs(n, best.iter().map(|(&(u, v), &w)| (u, v, w)));

    // Symmetric union of the directed pattern and its transpose, i.e.
    // `eWiseAdd(A, A^T, ONEB_BOOL)` in cch.c -- self-loops are already
    // excluded above, so there is no diagonal to drop.
    let mut skeleton_pairs: Vec<(u32, u32)> = Vec::with_capacity(best.len() * 2);
    for &(u, v) in best.keys() {
        skeleton_pairs.push((u, v));
        skeleton_pairs.push((v, u));
    }
    let skeleton = unweighted_csr_from_pairs(n, skeleton_pairs);

    CchInput {
        n,
        skeleton,
        weights,
    }
}

fn unweighted_csr_from_pairs(
    n: usize,
    mut pairs: Vec<(u32, u32)>,
) -> Csr<u32> {
    pairs.sort_unstable();
    pairs.dedup();

    let mut p = vec![0u32; n + 1];
    for &(u, _) in &pairs {
        p[u as usize + 1] += 1;
    }
    for k in 0..n {
        p[k + 1] += p[k];
    }
    let i = pairs.into_iter().map(|(_, v)| v).collect();
    Csr { n, p, i }
}

fn weight_csr_from_pairs(
    n: usize,
    pairs: impl Iterator<Item = (u32, u32, f64)>,
) -> WeightCsr {
    let mut rows: Vec<(u32, u32, f64)> = pairs.collect();
    rows.sort_unstable_by_key(|&(u, v, _)| (u, v));

    let mut p = vec![0u32; n + 1];
    for &(u, _, _) in &rows {
        p[u as usize + 1] += 1;
    }
    for k in 0..n {
        p[k + 1] += p[k];
    }
    let i = rows.iter().map(|&(_, v, _)| v).collect();
    let w = rows.iter().map(|&(_, _, w)| w).collect();
    WeightCsr {
        csr: Csr { n, p, i },
        w,
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct HeapEntry {
    dist: f64,
    node: u32,
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
/// [`to_input`]. `f64::INFINITY` for every node `src` cannot reach.
///
/// # Panics
/// If `src` is out of range, or any edge weight is negative, infinite or NaN: the
/// oracle only promises correctness over the weight domain [`CchInput`] allows
/// ("finite and >= 0").
///
/// Not checked against FalkorDB's own Dijkstra (`dijkstra_single_path` behind
/// `algo.SPpaths`, `runtime/functions/algo_procedures.rs`), on purpose: that one is private
/// and needs a `Runtime` and a loaded `Graph`, so it is only reachable through Cypher, and
/// it silently skips edges whose weight is not finite, which this oracle must reject. It
/// is the oracle of the flow tests instead (`tests/flow/test_cch.py` compares CCH queries
/// with `algo.SPpaths`). If a pure shortest-path function over an edge list is ever added
/// to the engine, cross-check this one against it.
#[must_use]
pub fn dijkstra(
    edges: &EdgeList,
    src: u32,
) -> Vec<f64> {
    let n = edges.n;
    assert!((src as usize) < n, "src {src} out of range for {n} nodes");

    let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    for &(u, v, w) in &edges.edges {
        assert!(
            w.is_finite() && w >= 0.0,
            "dijkstra oracle requires finite, non-negative weights, got {w} on ({u}, {v})"
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
    (0..edges.n as u32).map(|s| dijkstra(edges, s)).collect()
}

/// The exact CCH customization oracle. `perm[rank] = node` is the
/// elimination order, exactly the convention a CCH's own `perm` array uses:
/// processing nodes in increasing rank order corresponds to eliminating
/// them one at a time, lowest rank first.
///
/// `result[a][b]` is the weight of the cheapest directed walk `a -> b`
/// whose *intermediate* nodes -- everything but `a` and `b` themselves --
/// all have strictly lower rank than BOTH `a` and `b`, i.e. `rank <
/// min(rank(a), rank(b))`. `a` and `b` are themselves unrestricted: either
/// may have any rank at all, including the highest in the graph.
/// `result[a][a] == 0.0` always (the trivial empty walk). `f64::INFINITY`
/// if no such walk exists. Parallel edges collapse to their minimum weight
/// per exact direction (`to_input`'s rule), and self-loops are ignored
/// entirely -- an edge `(u, u, w)` contributes nothing.
///
/// This is precisely the quantity a correctly customized CCH must hold:
/// for an arc `(a, b)` in the hierarchy with `rank(a) < rank(b)`, the
/// **up** weight of that arc must equal `result[a][b]`, and the **down**
/// weight of the reverse arc `(b, a)` must equal `result[b][a]` -- both
/// bounded by the same `rank(a)` (the lower of the two ranks), just walked
/// in opposite directions over a graph that need not be symmetric.
///
/// Computed by Floyd-Warshall, inserting nodes as allowed intermediates in
/// rank order and recording `result[a][b]`/`result[b][a]` for every pair
/// whose bound is exactly the rank about to be inserted -- the standard FW
/// invariant that the matrix before inserting the `r`-th vertex (in a given
/// order) holds shortest paths using only the first `r` as intermediates.
///
/// # Panics
/// If `perm` is not a permutation of `0..edges.n` (wrong length, a value
/// out of range, or a repeat), or if any edge weight is negative, infinite or
/// NaN (the same domain `dijkstra` requires), self-loops included.
#[must_use]
pub fn lower_rank_distances(
    edges: &EdgeList,
    perm: &[u32],
) -> Vec<Vec<f64>> {
    let n = edges.n;
    assert_eq!(perm.len(), n, "perm must have exactly n entries");
    let mut rank_of = vec![u32::MAX; n];
    for (rank, &node) in perm.iter().enumerate() {
        assert!(
            (node as usize) < n,
            "perm contains out-of-range node {node}"
        );
        assert_eq!(rank_of[node as usize], u32::MAX, "perm repeats node {node}");
        rank_of[node as usize] = rank as u32;
    }

    let mut best: HashMap<(u32, u32), f64> = HashMap::new();
    for &(u, v, w) in &edges.edges {
        assert!(
            w.is_finite() && w >= 0.0,
            "lower_rank_distances requires finite, non-negative weights, got {w} on ({u}, {v})"
        );
        if u == v {
            continue;
        }
        best.entry((u, v))
            .and_modify(|cur| {
                if w < *cur {
                    *cur = w;
                }
            })
            .or_insert(w);
    }

    let mut d = vec![vec![f64::INFINITY; n]; n];
    for (i, row) in d.iter_mut().enumerate() {
        row[i] = 0.0;
    }
    for (&(u, v), &w) in &best {
        let (u, v) = (u as usize, v as usize);
        if w < d[u][v] {
            d[u][v] = w;
        }
    }

    let mut result = vec![vec![f64::INFINITY; n]; n];
    for (r, &k) in perm.iter().enumerate() {
        let k = k as usize;
        for b in 0..n {
            if rank_of[b] as usize >= r {
                result[k][b] = d[k][b];
                result[b][k] = d[b][k];
            }
        }
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

    result
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
/// with no edge between them, or its weight sums do not match `weight`
/// within `1e-6`.
pub fn assert_path_valid(
    edges: &EdgeList,
    src: u32,
    dst: u32,
    weight: f64,
    path: &[u32],
) {
    assert!(!path.is_empty(), "path must not be empty");
    assert_eq!(path[0], src, "path does not start at src");
    assert_eq!(*path.last().unwrap(), dst, "path does not end at dst");

    if path.len() == 1 {
        assert_eq!(src, dst, "single-node path must have src == dst");
        assert_eq!(weight, 0.0, "a trivial src == dst path must weigh 0");
        return;
    }

    let mut cheapest: HashMap<(u32, u32), f64> = HashMap::new();
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
        (sum - weight).abs() < 1e-6,
        "path sums to {sum}, claimed weight is {weight}"
    );
}

/// Checks the structural invariants the contract promises about a [`CchInput`]: the
/// skeleton is symmetric and self-loop-free with sorted, duplicate-free rows;
/// the weights are directed, self-loop-free, finite and non-negative, with
/// sorted duplicate-free rows; and every weighted pair is present in the
/// skeleton (`pattern(weights)` subset of the skeleton's pattern).
///
/// Independent of how `input` was produced -- it does not assume `to_input`
/// built it, so it is also the right check to run over the real
/// `Graph -> CchInput` output, not just this kit's own.
///
/// # Panics
/// If any invariant is violated.
pub fn assert_input_invariants(input: &CchInput) {
    assert_csr_shape(&input.skeleton, input.n, "skeleton");
    assert_csr_shape(&input.weights.csr, input.n, "weights");
    assert_eq!(
        input.weights.csr.i.len(),
        input.weights.w.len(),
        "weights.csr.i and weights.w must be the same length"
    );

    let skeleton_pairs = csr_pairs(&input.skeleton);
    for &(u, v) in &skeleton_pairs {
        assert_ne!(u, v, "skeleton has a self-loop at node {u}");
    }
    let skeleton_set: HashSet<(u32, u32)> = skeleton_pairs.into_iter().collect();
    for &(u, v) in &skeleton_set {
        assert!(
            skeleton_set.contains(&(v, u)),
            "skeleton is not symmetric: ({u}, {v}) has no reverse"
        );
    }

    let weight_pairs = csr_pairs(&input.weights.csr);
    for (&(u, v), &w) in weight_pairs.iter().zip(&input.weights.w) {
        assert_ne!(u, v, "weights has a self-loop at node {u}");
        assert!(
            w.is_finite() && w >= 0.0,
            "weight on ({u}, {v}) is {w}, must be finite and >= 0"
        );
        assert!(
            skeleton_set.contains(&(u, v)),
            "weighted pair ({u}, {v}) is not present in the skeleton"
        );
    }
}

/// `csr.p` has `n + 1` non-decreasing offsets ending at `csr.i.len()`, and
/// every row of `csr.i` is sorted, duplicate-free and in range `0..n`.
fn assert_csr_shape(
    csr: &Csr<u32>,
    n: usize,
    label: &str,
) {
    assert_eq!(csr.n, n, "{label}.n must match CchInput.n");
    assert_eq!(csr.p.len(), n + 1, "{label}.p must have n+1 offsets");
    assert_eq!(*csr.p.first().unwrap_or(&0), 0, "{label}.p must start at 0");
    assert_eq!(
        *csr.p.last().unwrap_or(&0) as usize,
        csr.i.len(),
        "{label}.p must end at i.len()"
    );
    for row in 0..n {
        let (lo, hi) = (csr.p[row] as usize, csr.p[row + 1] as usize);
        assert!(lo <= hi, "{label}.p is not non-decreasing at row {row}");
        let row_cols = &csr.i[lo..hi];
        assert!(
            row_cols.windows(2).all(|pair| pair[0] < pair[1]),
            "{label} row {row} is not sorted/duplicate-free"
        );
        for &col in row_cols {
            assert!(
                (col as usize) < n,
                "{label} row {row} has out-of-range column {col}"
            );
        }
    }
}

/// `(row, col)` for every entry of `csr`, row by row.
fn csr_pairs(csr: &Csr<u32>) -> Vec<(u32, u32)> {
    let mut pairs = Vec::with_capacity(csr.i.len());
    for row in 0..csr.n {
        for &col in csr.row(row) {
            pairs.push((row as u32, col));
        }
    }
    pairs
}

/// Deterministic generator (SplitMix64), so a failing seed reproduces.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `[0.0, 1.0)` from the next SplitMix64 draw.
fn next_unit(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// `[lo, hi)` from the next SplitMix64 draw. `hi` must be `> lo`.
fn next_range(
    state: &mut u64,
    lo: u32,
    hi: u32,
) -> u32 {
    lo + (splitmix(state) % u64::from(hi - lo)) as u32
}

/// An integer weight drawn uniformly from `[1, wmax]`. Written as `0..wmax` plus one so that
/// `wmax == u32::MAX` does not overflow.
fn next_weight(
    state: &mut u64,
    wmax: u32,
) -> f64 {
    f64::from(next_range(state, 0, wmax) + 1)
}

/// An `rows x cols` grid, 4-connected, unit weights. `rows, cols >= 1`.
#[must_use]
pub fn grid(
    rows: u32,
    cols: u32,
) -> EdgeList {
    assert!(rows >= 1 && cols >= 1);
    let n = (rows * cols) as usize;
    let mut g = EdgeList::new(n);
    let id = |r: u32, c: u32| r * cols + c;
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
        let u = next_range(&mut state, 0, n as u32);
        let v = next_range(&mut state, 0, n as u32);
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
    pairs: &[(u32, u32)],
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
    at: &[u32],
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
        let offset = (c * nodes_per_component) as u32;
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
    leaves: u32,
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
        let (u, v) = (u as u32, (u + 1) as u32);
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
    k: u32,
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
pub fn zero_weight_cycle(len: u32) -> EdgeList {
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
    u: u32,
    v: u32,
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
        let u = next_range(&mut state, 0, n as u32);
        let v = next_range(&mut state, 0, n as u32);
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
pub fn assert_path_simple(path: &[u32]) {
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
/// on reachability and weight (within `1e-6`), and every returned path must
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
    index_query: impl Fn(u32, u32) -> Option<(f64, Vec<u32>)>,
    edges: &EdgeList,
) {
    let oracle = all_pairs(edges);
    for s in 0..edges.n as u32 {
        for t in 0..edges.n as u32 {
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
                    (w - expected).abs() < 1e-6,
                    "({s}, {t}): index weight {w}, oracle weight {expected}"
                );
                assert_path_valid(edges, s, t, w, &path);
                assert_path_simple(&path);
            }
        }
    }
}

/// One edit to a graph, for maintenance tests. Edits act on a directed
/// pair `(u, v)`: [`Edit::SetWeight`] and [`Edit::RemoveEdge`] touch every
/// parallel edge of that pair.
#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    /// Every `(u, v)` edge gets weight `w`. `increased` says whether the
    /// pair's cheapest weight went up (`true`) or down (`false`).
    SetWeight {
        u: u32,
        v: u32,
        w: f64,
        increased: bool,
    },
    RemoveEdge {
        u: u32,
        v: u32,
    },
    /// A new edge. `was_reachable`: `u` could already reach `v` before it.
    AddEdge {
        u: u32,
        v: u32,
        w: f64,
        was_reachable: bool,
    },
    /// One new isolated node, id = the old `n`.
    AddNode,
}

/// Applies `edit` to `edges` in place.
///
/// # Panics
/// If `SetWeight`/`RemoveEdge` names a pair with no edge.
pub fn apply_edit(
    edges: &mut EdgeList,
    edit: &Edit,
) {
    match *edit {
        Edit::SetWeight { u, v, w, .. } => {
            let mut hit = false;
            for e in edges.edges.iter_mut().filter(|e| (e.0, e.1) == (u, v)) {
                e.2 = w;
                hit = true;
            }
            assert!(hit, "SetWeight on missing edge ({u}, {v})");
        }
        Edit::RemoveEdge { u, v } => {
            let before = edges.edges.len();
            edges.edges.retain(|e| (e.0, e.1) != (u, v));
            assert!(
                edges.edges.len() < before,
                "RemoveEdge on missing edge ({u}, {v})"
            );
        }
        Edit::AddEdge { u, v, w, .. } => edges.push(u, v, w),
        Edit::AddNode => edges.n += 1,
    }
}

/// A seeded batch of 1..=4 edits drawn from: weight up, weight down, edge
/// removed, edge added between nodes already connected (`u` reaches `v`),
/// edge added between unconnected nodes, node added. Edits apply in order,
/// so later ones may use a node added earlier. Returns the graph after the
/// whole batch plus the edits; `apply_edit` over `base` reproduces it.
///
/// A drawn kind that cannot apply to the current graph (no edge to remove,
/// no unconnected pair, zero-weight edge to lower, ...) becomes `AddNode`.
/// Self-loops are never chosen as the target of an edit. Weights stay
/// finite and >= 0; "down" can reach exactly `0.0`.
#[must_use]
pub fn random_edit_batch(
    base: &EdgeList,
    seed: u64,
) -> (EdgeList, Vec<Edit>) {
    let mut state = seed;
    let mut g = base.clone();
    let mut edits = Vec::new();
    let count = 1 + (splitmix(&mut state) % 4) as usize;
    for _ in 0..count {
        let kind = splitmix(&mut state) % 6;
        let edit = draw_edit(&g, kind, &mut state).unwrap_or(Edit::AddNode);
        apply_edit(&mut g, &edit);
        edits.push(edit);
    }
    (g, edits)
}

/// Cheapest `(u, v)` weight in `g`, if any edge exists.
fn pair_min(
    g: &EdgeList,
    u: u32,
    v: u32,
) -> Option<f64> {
    g.edges
        .iter()
        .filter(|e| (e.0, e.1) == (u, v))
        .map(|e| e.2)
        .reduce(f64::min)
}

fn draw_edit(
    g: &EdgeList,
    kind: u64,
    state: &mut u64,
) -> Option<Edit> {
    let real: Vec<usize> = (0..g.edges.len())
        .filter(|&i| g.edges[i].0 != g.edges[i].1)
        .collect();
    let pick_edge = |state: &mut u64| {
        let (u, v, _) = g.edges[real[(splitmix(state) % real.len() as u64) as usize]];
        (u, v)
    };
    match kind {
        0..=2 if real.is_empty() => None,
        0 => {
            let (u, v) = pick_edge(state);
            let m = pair_min(g, u, v)?;
            let w = m + f64::from(1 + (splitmix(state) % 5) as u32);
            Some(Edit::SetWeight {
                u,
                v,
                w,
                increased: true,
            })
        }
        1 => {
            let (u, v) = pick_edge(state);
            let m = pair_min(g, u, v)?;
            if m <= 0.0 {
                return None;
            }
            let w = (m * next_unit(state)).floor();
            Some(Edit::SetWeight {
                u,
                v,
                w,
                increased: false,
            })
        }
        2 => {
            let (u, v) = pick_edge(state);
            Some(Edit::RemoveEdge { u, v })
        }
        3 | 4 => {
            if g.n < 2 {
                return None;
            }
            let want_reachable = kind == 3;
            let reach = all_pairs(g);
            let n = g.n as u32;
            let start = next_range(state, 0, n * n);
            for k in 0..n * n {
                let idx = (start + k) % (n * n);
                let (u, v) = (idx / n, idx % n);
                if u != v && reach[u as usize][v as usize].is_finite() == want_reachable {
                    let w = f64::from(next_range(state, 1, 10));
                    return Some(Edit::AddEdge {
                        u,
                        v,
                        w,
                        was_reachable: want_reachable,
                    });
                }
            }
            None
        }
        _ => Some(Edit::AddNode),
    }
}

#[cfg(test)]
mod self_tests {
    use super::{
        CchInput, Csr, EdgeList, HashMap, TypedEdgeList, WeightCsr, all_pairs,
        assert_input_invariants, assert_path_valid, dijkstra, disconnected_components, empty, grid,
        lower_rank_distances, next_unit, path, random_multi_type, random_sparse, single_node,
        splitmix, star, tied_paths, to_input, with_isolated_nodes, with_parallel_edges,
        with_self_loops, with_zero_weight_edge, zero_weight_cycle,
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

    /// Independent second derivation of [`lower_rank_distances`], used only
    /// to cross-check it (self-tests only, tiny graphs only -- it enumerates
    /// every simple path): for each `(a, b)`, DFS every simple `a -> b` walk
    /// (no repeated node, so it terminates) whose non-endpoint nodes all
    /// satisfy the rank bound, and take the cheapest.
    fn lower_rank_distances_brute_force(
        edges: &EdgeList,
        perm: &[u32],
    ) -> Vec<Vec<f64>> {
        let n = edges.n;
        let mut rank_of = vec![0u32; n];
        for (r, &node) in perm.iter().enumerate() {
            rank_of[node as usize] = r as u32;
        }

        let mut best: HashMap<(u32, u32), f64> = HashMap::new();
        for &(u, v, w) in &edges.edges {
            if u == v {
                continue;
            }
            best.entry((u, v))
                .and_modify(|cur| {
                    if w < *cur {
                        *cur = w;
                    }
                })
                .or_insert(w);
        }
        let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for (&(u, v), &w) in &best {
            adj[u as usize].push((v, w));
        }

        // Everything a single search leaves unchanged, bundled so the
        // recursive step stays under clippy's argument-count lint.
        struct Search<'a> {
            dst: u32,
            bound: u32,
            adj: &'a [Vec<(u32, f64)>],
            rank_of: &'a [u32],
        }

        fn dfs(
            s: &Search,
            cur: u32,
            acc: f64,
            visited: &mut [bool],
            best: &mut f64,
        ) {
            if cur == s.dst {
                if acc < *best {
                    *best = acc;
                }
                return;
            }
            if acc >= *best {
                return; // weights are non-negative, this branch can't improve
            }
            for &(next, w) in &s.adj[cur as usize] {
                if next != s.dst && s.rank_of[next as usize] >= s.bound {
                    continue; // intermediate violates the rank bound
                }
                if visited[next as usize] {
                    continue;
                }
                visited[next as usize] = true;
                dfs(s, next, acc + w, visited, best);
                visited[next as usize] = false;
            }
        }

        let mut result = vec![vec![f64::INFINITY; n]; n];
        for (a, row) in result.iter_mut().enumerate() {
            row[a] = 0.0;
        }
        for a in 0..n {
            for b in 0..n {
                if a == b {
                    continue;
                }
                let search = Search {
                    dst: b as u32,
                    bound: rank_of[a].min(rank_of[b]),
                    adj: &adj,
                    rank_of: &rank_of,
                };
                let mut visited = vec![false; n];
                visited[a] = true;
                let mut best_w = f64::INFINITY;
                dfs(&search, a as u32, 0.0, &mut visited, &mut best_w);
                result[a][b] = best_w;
            }
        }
        result
    }

    /// Fisher-Yates over the same seeded SplitMix64 as everything else in
    /// this kit, so a failing seed reproduces.
    fn random_perm(
        n: usize,
        state: &mut u64,
    ) -> Vec<u32> {
        let mut perm: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            let j = (splitmix(state) % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        perm
    }

    // ---- hand-computed cases, lifted from the C flow tests -----------

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

    /// Same valley graph as `valley_matches_c_test01`, but checked through
    /// `lower_rank_distances` to show the rank bound matters: when node 2
    /// (the shortcut hop) outranks both 0 and 1, the 2-hop path is not an
    /// allowed intermediate and the direct edge is all that is left -- a
    /// plain `dijkstra`/`all_pairs` would miss this entirely, since it
    /// always finds the 2-hop shortcut regardless of rank.
    #[test]
    fn lower_rank_distances_respects_the_rank_bound_on_the_valley() {
        let mut g = EdgeList::new(3);
        g.push(0, 1, 10.0);
        g.push(1, 0, 10.0);
        g.push(0, 2, 1.0);
        g.push(2, 0, 1.0);
        g.push(2, 1, 1.0);
        g.push(1, 2, 1.0);

        // rank(2) = 0 (lowest) -- the shortcut through 2 is allowed for any
        // pair, since both 0 and 1 outrank it.
        let low = lower_rank_distances(&g, &[2, 0, 1]);
        assert_eq!(low[0][1], 2.0);
        assert_eq!(low[1][0], 2.0);

        // rank(2) = 2 (highest) -- now 2 outranks both 0 and 1, so it can
        // never be an intermediate for the (0, 1) pair (bound = rank(0) =
        // rank(1) = 0 or 1, either way < rank(2) = 2). Only the direct,
        // expensive edge remains.
        let high = lower_rank_distances(&g, &[0, 1, 2]);
        assert_eq!(high[0][1], 10.0);
        assert_eq!(high[1][0], 10.0);

        // every node is its own trivial walk, regardless of rank.
        assert_eq!(high[0][0], 0.0);
        assert_eq!(high[1][1], 0.0);
        assert_eq!(high[2][2], 0.0);
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

    /// `test15_one_way_streets`: a one-way edge is never walked backwards.
    #[test]
    fn one_way_streets_respect_direction() {
        let mut g = EdgeList::new(2);
        g.push(0, 1, 1.0);
        let dist = dijkstra(&g, 1);
        assert!(dist[0].is_infinite(), "edge is one-way, 1 can't reach 0");
    }

    // ---- to_input: resolution ---------------------------------------

    #[test]
    fn to_input_takes_min_over_parallel_edges() {
        let mut g = EdgeList::new(2);
        g.push(0, 1, 5.0);
        g.push(0, 1, 2.0);
        g.push(0, 1, 9.0);
        let input = to_input(&g);
        assert_eq!(input.weights.csr.i, vec![1]);
        assert_eq!(input.weights.w, vec![2.0]);
    }

    #[test]
    fn to_input_drops_self_loops_from_weights_and_skeleton() {
        let mut g = EdgeList::new(2);
        g.push(0, 0, 1.0);
        g.push(0, 1, 3.0);
        let input = to_input(&g);
        assert_eq!(input.weights.csr.i, vec![1], "self-loop must not appear");
        // symmetrized (0,1): row 0 -> [1], row 1 -> [0]; no diagonal entry.
        assert_eq!(input.skeleton.p, vec![0, 1, 2]);
        assert_eq!(input.skeleton.i, vec![1, 0]);
    }

    #[test]
    fn to_input_skeleton_is_symmetric_even_for_a_one_way_edge() {
        let mut g = EdgeList::new(2);
        g.push(0, 1, 1.0);
        let input = to_input(&g);
        // one directed weight ...
        assert_eq!(input.weights.csr.i, vec![1]);
        assert_eq!(input.weights.csr.p, vec![0, 1, 1]);
        // ... but the skeleton carries both directions (topology only).
        assert_eq!(input.skeleton.p, vec![0, 1, 2]);
        assert_eq!(input.skeleton.i, vec![1, 0]);
    }

    #[test]
    fn to_input_skeleton_rows_are_sorted_and_duplicate_free() {
        let mut g = EdgeList::new(3);
        g.push(0, 1, 1.0);
        g.push(0, 2, 1.0);
        g.push(2, 0, 1.0); // same pair as the symmetrization of (0,2)
        let input = to_input(&g);
        assert_eq!(input.skeleton.row(0), &[1, 2]);
    }

    // ---- the oracle cross-checks itself ---------------------------------

    /// `lower_rank_distances` (the incremental-Floyd-Warshall derivation)
    /// and `lower_rank_distances_brute_force` (DFS over every simple path,
    /// an independently-written derivation of the same definition) must
    /// agree on every pair, for many random tiny graphs and random
    /// permutations. A failing seed is printed so it reproduces.
    #[test]
    fn lower_rank_distances_agrees_with_brute_force() {
        for seed in 0..300u64 {
            let mut state = seed;
            // kept tiny: the brute force enumerates every simple path.
            let n = 2 + (splitmix(&mut state) % 6) as usize;
            let m = 1 + (splitmix(&mut state) % 15) as usize;
            let wmax = 1 + (splitmix(&mut state) % 9) as u32;
            let g = random_sparse(n, m, wmax, seed);
            let perm = random_perm(n, &mut state);

            let exact = lower_rank_distances(&g, &perm);
            let brute = lower_rank_distances_brute_force(&g, &perm);
            for s in 0..n {
                for t in 0..n {
                    assert_eq!(
                        exact[s][t], brute[s][t],
                        "seed {seed}, perm {perm:?}: lower_rank_distances and the \
                         brute force disagree on ({s}, {t})"
                    );
                }
            }
        }
    }

    /// Same cross-check over the structured generators -- not just
    /// `random_sparse` -- paired with a handful of distinct permutations
    /// per graph, so a bug that only shows up on a particular rank order
    /// (e.g. the identity permutation, which happens to coincide with
    /// unrestricted shortest paths on an already-sorted structure) doesn't
    /// hide.
    #[test]
    fn lower_rank_distances_agrees_with_brute_force_on_every_generator() {
        let graphs: Vec<EdgeList> = vec![
            grid(2, 3),
            with_parallel_edges(4, &[(0, 1), (1, 2), (2, 3)], 3, 9, 7),
            with_self_loops(grid(2, 2), &[0, 3], 0.5),
            star(4, 2.0),
            path(5),
            tied_paths(3, 1.5),
            zero_weight_cycle(4),
        ];
        for (idx, g) in graphs.iter().enumerate() {
            let mut state = 0x5EED_u64.wrapping_add(idx as u64);
            for _ in 0..5 {
                let perm = random_perm(g.n, &mut state);
                let exact = lower_rank_distances(g, &perm);
                let brute = lower_rank_distances_brute_force(g, &perm);
                for s in 0..g.n {
                    for t in 0..g.n {
                        assert_eq!(
                            exact[s][t], brute[s][t],
                            "graph #{idx}, perm {perm:?}: disagree on ({s}, {t})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "finite, non-negative")]
    fn dijkstra_rejects_a_negative_weight() {
        let mut g = EdgeList::new(2);
        g.push(0, 1, -1.0);
        let _ = dijkstra(&g, 0);
    }

    #[test]
    #[should_panic(expected = "finite, non-negative")]
    fn dijkstra_rejects_a_nan_weight() {
        let mut g = EdgeList::new(2);
        g.push(0, 1, f64::NAN);
        let _ = dijkstra(&g, 0);
    }

    #[test]
    #[should_panic(expected = "finite, non-negative")]
    fn lower_rank_distances_rejects_a_nan_weight() {
        let mut g = EdgeList::new(2);
        g.push(0, 1, f64::NAN);
        let _ = lower_rank_distances(&g, &[0, 1]);
    }

    /// A self-loop is dropped from the resolved graph, but its weight is still validated.
    #[test]
    #[should_panic(expected = "finite, non-negative")]
    fn lower_rank_distances_rejects_a_bad_weight_on_a_self_loop() {
        let mut g = EdgeList::new(2);
        g.push(0, 0, -1.0);
        let _ = lower_rank_distances(&g, &[0, 1]);
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

    /// `select` must keep only the edges whose type is requested, and must
    /// resolve ties within the selection exactly like [`to_input`] does over
    /// a plain [`EdgeList`] (min over parallel edges) -- selecting a
    /// type that was never drawn gives an edgeless graph on the same `n`.
    #[test]
    fn typed_edge_list_select_filters_by_type_and_resolves_like_to_input() {
        let mut g = TypedEdgeList::new(2);
        g.push(0, 1, 5.0, 0);
        g.push(0, 1, 2.0, 0); // same type, cheaper -> should win within type 0
        g.push(0, 1, 1.0, 1); // different type -> excluded when selecting [0]

        let only_0 = g.select(&[0]);
        let resolved = to_input(&only_0);
        assert_eq!(resolved.weights.w, vec![2.0]);

        let only_2 = g.select(&[2]); // never drawn
        assert!(only_2.edges.is_empty());
        assert_eq!(only_2.n, 2);
    }

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

    #[test]
    fn edge_list_to_cypher_create_matches_the_c_flow_test_idiom() {
        let mut g = EdgeList::new(2);
        g.push(0, 1, 2.0);
        let script = g.to_cypher_create("ROAD");
        assert!(script.starts_with("CREATE INDEX FOR (n:N) ON (n.v);\n"));
        assert!(script.contains("UNWIND range(0,1) AS i CREATE (:N {v:i});"));
        assert!(script.contains("CREATE (a)-[:ROAD {w:e[2]}]->(b)"));
        assert!(script.contains("[0,1,2]"));
    }

    #[test]
    fn edge_list_to_cypher_create_handles_the_empty_graph() {
        let g = EdgeList::new(0);
        assert_eq!(g.to_cypher_create("ROAD"), "");
    }

    #[test]
    fn typed_edge_list_to_cypher_create_groups_edges_by_name() {
        let mut g = TypedEdgeList::new(3);
        g.push(0, 1, 1.0, 0);
        g.push(1, 2, 2.0, 1);
        let script = g.to_cypher_create(&["ROAD", "FERRY"]);
        assert!(script.starts_with("CREATE INDEX FOR (n:N) ON (n.v);\n"));
        assert!(script.contains(":ROAD"));
        assert!(script.contains(":FERRY"));
        assert!(script.contains("[0,1,1]"));
        assert!(script.contains("[1,2,2]"));
    }

    #[test]
    #[should_panic(expected = "rel_type 1")]
    fn typed_edge_list_to_cypher_create_panics_on_an_unnamed_type() {
        let mut g = TypedEdgeList::new(2);
        g.push(0, 1, 1.0, 1);
        let _ = g.to_cypher_create(&["ROAD"]); // only names type 0
    }

    // ---- isolated nodes beyond the max used id -----------------------------

    /// A real `Graph -> CchInput` can have `n` larger than any id that
    /// appears in an edge. `dijkstra`'s result must still be length `n`, and
    /// the extra nodes must be unreachable from (and unable to reach)
    /// anything.
    #[test]
    fn with_isolated_nodes_extends_n_past_the_max_used_id() {
        let g = with_isolated_nodes(grid(2, 2), 3); // nodes 0..3 used, 4..6 isolated
        assert_eq!(g.n, 7);

        let dist = dijkstra(&g, 0);
        assert_eq!(dist.len(), 7);
        for (t, d) in dist.iter().enumerate().skip(4) {
            assert!(d.is_infinite(), "isolated node {t} must be unreachable");
        }

        // and nothing can reach back out of isolation, either
        let dist_from_isolated = dijkstra(&g, 4);
        assert_eq!(dist_from_isolated[4], 0.0);
        for (t, d) in dist_from_isolated.iter().enumerate() {
            if t != 4 {
                assert!(d.is_infinite(), "node 4 is isolated, can't reach {t}");
            }
        }
    }

    #[test]
    fn to_input_handles_isolated_nodes_past_the_max_used_id() {
        let g = with_isolated_nodes(grid(2, 2), 3);
        let input = to_input(&g);
        assert_eq!(input.n, 7);
        assert_input_invariants(&input);
        // row for an isolated node is empty on both sides
        assert_eq!(input.skeleton.p[4], input.skeleton.p[5]);
        assert_eq!(input.weights.csr.p[4], input.weights.csr.p[5]);
    }

    // ---- assert_input_invariants --------------------------------------

    #[test]
    fn assert_input_invariants_accepts_to_input_s_own_output() {
        let graphs = [
            grid(3, 3),
            with_parallel_edges(4, &[(0, 1), (1, 2), (2, 3)], 3, 9, 7),
            disconnected_components(3, 4, 5, 9, 11),
            empty(5),
            star(6, 2.0),
            tied_paths(3, 1.5),
            zero_weight_cycle(4),
            with_isolated_nodes(grid(2, 2), 3),
        ];
        for g in &graphs {
            assert_input_invariants(&to_input(g));
        }
    }

    #[test]
    #[should_panic(expected = "not symmetric")]
    fn assert_input_invariants_catches_an_asymmetric_skeleton() {
        // (0,1) with no reverse -- to_input always symmetrizes, so this is
        // hand-built to prove the checker actually looks.
        let input = CchInput {
            n: 2,
            skeleton: Csr {
                n: 2,
                p: vec![0, 1, 1],
                i: vec![1],
            },
            weights: WeightCsr {
                csr: Csr {
                    n: 2,
                    p: vec![0, 1, 1],
                    i: vec![1],
                },
                w: vec![1.0],
            },
        };
        assert_input_invariants(&input);
    }

    #[test]
    #[should_panic(expected = "self-loop")]
    fn assert_input_invariants_catches_a_self_loop_in_weights() {
        let input = CchInput {
            n: 2,
            skeleton: Csr {
                n: 2,
                p: vec![0, 1, 2],
                i: vec![1, 0],
            },
            weights: WeightCsr {
                csr: Csr {
                    n: 2,
                    p: vec![0, 1, 1], // row 0: [0] -- a self-loop
                    i: vec![0],
                },
                w: vec![1.0],
            },
        };
        assert_input_invariants(&input);
    }

    #[test]
    #[should_panic(expected = "not present in the skeleton")]
    fn assert_input_invariants_catches_a_weight_missing_from_the_skeleton() {
        let input = CchInput {
            n: 2,
            skeleton: Csr {
                n: 2,
                p: vec![0, 0, 0], // empty skeleton
                i: vec![],
            },
            weights: WeightCsr {
                csr: Csr {
                    n: 2,
                    p: vec![0, 1, 1],
                    i: vec![1],
                },
                w: vec![1.0],
            },
        };
        assert_input_invariants(&input);
    }

    #[test]
    #[should_panic(expected = "must be finite and >= 0")]
    fn assert_input_invariants_catches_a_negative_weight() {
        let input = CchInput {
            n: 2,
            skeleton: Csr {
                n: 2,
                p: vec![0, 1, 2],
                i: vec![1, 0],
            },
            weights: WeightCsr {
                csr: Csr {
                    n: 2,
                    p: vec![0, 1, 1],
                    i: vec![1],
                },
                w: vec![-1.0],
            },
        };
        assert_input_invariants(&input);
    }

    #[test]
    #[should_panic(expected = "not sorted/duplicate-free")]
    fn assert_input_invariants_catches_an_unsorted_row() {
        let input = CchInput {
            n: 3,
            skeleton: Csr {
                n: 3,
                p: vec![0, 2, 3, 3],
                i: vec![2, 1, 0], // row 0 should be [1, 2], not [2, 1]
            },
            weights: WeightCsr {
                csr: Csr {
                    n: 3,
                    p: vec![0, 0, 0, 0],
                    i: vec![],
                },
                w: vec![],
            },
        };
        assert_input_invariants(&input);
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

/// Self-tests of `check_index_against_oracle` and `random_edit_batch`.
#[cfg(test)]
mod maintenance_tests {
    use super::{
        EdgeList, Edit, all_pairs, apply_edit, assert_path_simple, assert_path_valid,
        check_index_against_oracle, dijkstra, grid, pair_min, path, random_edit_batch,
        random_sparse, random_zero_heavy, tied_paths, with_isolated_nodes, zero_weight_cycle,
    };

    /// A correct stand-in index: Dijkstra with predecessors, written
    /// separately from the kit's `dijkstra` so the checker is not tested
    /// against its own oracle's code path.
    fn reference_query(
        g: &EdgeList,
        s: u32,
        t: u32,
    ) -> Option<(f64, Vec<u32>)> {
        let n = g.n;
        let mut dist = vec![f64::INFINITY; n];
        let mut pred = vec![u32::MAX; n];
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
    #[should_panic(expected = "no edge")]
    fn checker_catches_a_bogus_path() {
        let g = path(4);
        check_index_against_oracle(
            |s, t| reference_query(&g, s, t).map(|(w, _)| (w, vec![s, t])),
            &g,
        );
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

    #[test]
    fn edit_batches_are_deterministic_and_replayable() {
        let base = random_sparse(8, 16, 9, 3);
        for seed in 0..100u64 {
            let (after, edits) = random_edit_batch(&base, seed);
            let (again, edits_again) = random_edit_batch(&base, seed);
            assert_eq!(edits, edits_again, "seed {seed}");
            assert_eq!(after.edges, again.edges, "seed {seed}");

            let mut replay = base.clone();
            for e in &edits {
                apply_edit(&mut replay, e);
            }
            assert_eq!(replay.n, after.n, "seed {seed}");
            assert_eq!(replay.edges, after.edges, "seed {seed}");
            assert!((1..=4).contains(&edits.len()));
        }
    }

    /// Every kind must actually occur over the seeds, and each edit must do
    /// what its tag says to the graph it was drawn on.
    #[test]
    fn every_edit_kind_occurs_and_means_what_it_says() {
        let base = random_sparse(8, 16, 9, 3);
        let (mut up, mut down, mut rem, mut conn, mut unconn, mut node) = (0, 0, 0, 0, 0, 0);
        for seed in 0..400u64 {
            let (_, edits) = random_edit_batch(&base, seed);
            let mut g = base.clone();
            for e in &edits {
                match *e {
                    Edit::SetWeight { u, v, w, increased } => {
                        let old = pair_min(&g, u, v).unwrap();
                        assert!(w.is_finite() && w >= 0.0);
                        if increased {
                            assert!(w > old, "seed {seed}");
                            up += 1;
                        } else {
                            assert!(w < old, "seed {seed}");
                            down += 1;
                        }
                    }
                    Edit::RemoveEdge { u, v } => {
                        assert!(pair_min(&g, u, v).is_some());
                        rem += 1;
                    }
                    Edit::AddEdge {
                        u,
                        v,
                        was_reachable,
                        ..
                    } => {
                        assert_ne!(u, v);
                        assert_eq!(
                            dijkstra(&g, u)[v as usize].is_finite(),
                            was_reachable,
                            "seed {seed}"
                        );
                        if was_reachable {
                            conn += 1
                        } else {
                            unconn += 1
                        }
                    }
                    Edit::AddNode => node += 1,
                }
                apply_edit(&mut g, e);
            }
            for row in all_pairs(&g) {
                assert_eq!(row.len(), g.n);
            }
        }
        for (name, c) in [
            ("up", up),
            ("down", down),
            ("remove", rem),
            ("add-connected", conn),
            ("add-unconnected", unconn),
            ("add-node", node),
        ] {
            assert!(c > 0, "edit kind {name} never drawn in 400 seeds");
        }
    }

    #[test]
    fn edit_batch_on_an_empty_graph_only_adds_nodes() {
        let (after, edits) = random_edit_batch(&EdgeList::new(0), 1);
        assert!(edits.iter().all(|e| *e == Edit::AddNode));
        assert_eq!(after.n, edits.len());
    }
}

/// `Cch::build` / `Cch::customize` rejecting invalid weights. Weights must be
/// finite and `>= 0`, so a negative weight, NaN and both infinities are
/// `CchError::InvalidWeight`. Black-box tests: they use only the signatures
/// in this module's parent (`mod.rs`).
#[cfg(test)]
mod rejection_tests {
    use super::{EdgeList, grid, to_input};
    use crate::index::cch::{Cch, CchError};

    const BAD_WEIGHTS: [(&str, f64); 4] = [
        ("negative", -1.0),
        ("NaN", f64::NAN),
        ("+inf", f64::INFINITY),
        ("-inf", f64::NEG_INFINITY),
    ];

    /// `base` with its first edge's weight overwritten -- keeps the same
    /// directed-pair pattern `customize` expects (the same arcs `build`
    /// saw), changing only the one value under test.
    fn with_one_bad_weight(
        mut base: EdgeList,
        bad: f64,
    ) -> EdgeList {
        base.edges[0].2 = bad;
        base
    }

    #[test]
    fn build_rejects_every_invalid_weight() {
        for &(label, w) in &BAD_WEIGHTS {
            let input = to_input(&with_one_bad_weight(grid(2, 2), w));
            assert!(
                matches!(Cch::build(&input), Err(CchError::InvalidWeight)),
                "build should reject a {label} weight"
            );
        }
    }

    #[test]
    fn customize_rejects_every_invalid_weight() {
        let base = grid(2, 2);
        let good = to_input(&base);
        for &(label, w) in &BAD_WEIGHTS {
            let mut cch = Cch::build(&good).expect("a valid input must build");
            let bad_weights = to_input(&with_one_bad_weight(base.clone(), w)).weights;
            assert_eq!(
                cch.customize(&bad_weights),
                Err(CchError::InvalidWeight),
                "customize should reject a {label} weight"
            );
        }
    }

    /// A refused `customize` must leave the index usable: re-customizing
    /// with good weights afterward must still succeed. Whether the refused
    /// call left the *previous* metric byte-for-byte untouched is customize's
    /// own test to make (it can compare its own internals); this only
    /// checks the index isn't bricked.
    #[test]
    fn customize_refused_does_not_brick_the_index() {
        let base = grid(2, 2);
        let good = to_input(&base);
        let mut cch = Cch::build(&good).expect("a valid input must build");

        let bad_weights = to_input(&with_one_bad_weight(base.clone(), f64::NAN)).weights;
        assert_eq!(cch.customize(&bad_weights), Err(CchError::InvalidWeight));

        assert_eq!(cch.customize(&good.weights), Ok(()));
    }
}

/// The shared checker against a real `Graph` index (`Graph` -> input ->
/// Phase 1/2 -> query), on the kit's graph families.
#[cfg(test)]
mod real_index_tests {
    use std::sync::Arc;

    use roaring::RoaringTreemap;
    use rustc_hash::FxHashMap;

    use super::{
        EdgeList, assert_path_simple, check_index_against_oracle, disconnected_components, grid,
        random_zero_heavy, tied_paths, with_isolated_nodes, with_parallel_edges, with_self_loops,
        zero_weight_cycle,
    };
    use crate::graph::{graph::Graph, graphblas::test_init};
    use crate::index::cch::CchIndexDef;
    use crate::index::index_docs::IndexDocs;
    use crate::runtime::value::Value;

    /// A `Graph` holding exactly `edges` as one relationship type `R`, every
    /// relationship carrying an explicit weight attribute `w`.
    fn real_graph(
        edges: &EdgeList,
        name: &str,
    ) -> (Graph, u16) {
        test_init::ensure_init();
        let mut g = Graph::new(16, 16, 1, 0, name);
        // A fresh graph's id spaces open their first batch themselves.
        let ids: RoaringTreemap = g
            .node_id_space()
            .reserve(edges.n, &RoaringTreemap::new())
            .expect("reserved")
            .into_iter()
            .collect();
        assert_eq!(ids.len(), edges.n as u64, "a fresh graph hands out 0..n");
        g.create_nodes(&ids).expect("creatable");
        let attr = g.get_or_create_rel_attr_id(&Arc::new(String::from("w")));

        let rel_ids: Vec<u64> = g
            .relationship_id_space()
            .reserve(edges.edges.len(), &RoaringTreemap::new())
            .expect("reserved");
        let srcs: Vec<u64> = edges.edges.iter().map(|e| u64::from(e.0)).collect();
        let dsts: Vec<u64> = edges.edges.iter().map(|e| u64::from(e.1)).collect();
        g.create_relationships_bulk(&Arc::new(String::from("R")), &srcs, &dsts, &rel_ids)
            .expect("creatable");

        let mut weights: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
        for (e, &id) in edges.edges.iter().zip(&rel_ids) {
            weights.insert(id, vec![(attr, Value::Float(e.2))]);
        }
        g.import_relationship_attrs(&weights, &mut IndexDocs::default());
        (g, attr)
    }

    #[test]
    fn real_index_agrees_with_the_oracle_on_kit_graphs() {
        let graphs: Vec<(&str, EdgeList)> = vec![
            ("grid", grid(4, 4)),
            (
                "parallel",
                with_parallel_edges(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 3, 9, 7),
            ),
            ("selfloops", with_self_loops(grid(3, 3), &[0, 4, 8], 0.5)),
            ("components", disconnected_components(3, 5, 8, 9, 11)),
            ("tied", tied_paths(4, 1.5)),
            ("zero_cycle", zero_weight_cycle(5)),
            ("isolated", with_isolated_nodes(grid(3, 3), 3)),
        ];
        for (name, edges) in &graphs {
            let (mut g, attr) = real_graph(edges, &format!("kit_e2e_{name}"));
            let rel = g.get_type_id("R").expect("type R exists");
            let def = CchIndexDef::new(vec![rel], attr);
            g.create_cch_index(def.clone()).expect("create");
            let idx = g.cch_index(&def).expect("held");
            check_index_against_oracle(|s, t| idx.cch.query(s, t), edges);
        }
    }

    /// About half the edges weigh 0: zero-weight cycles and many equal-weight
    /// walks. Every returned path must be simple as well as right.
    #[test]
    fn real_index_returns_simple_correct_paths_on_zero_heavy_graphs() {
        for seed in 0..30u64 {
            let edges = random_zero_heavy(12, 40, 5, seed);
            assert!(edges.edges.iter().any(|e| e.2 == 0.0), "seed {seed}");
            let (mut g, attr) = real_graph(&edges, &format!("kit_zero_{seed}"));
            let rel = g.get_type_id("R").expect("type R exists");
            let def = CchIndexDef::new(vec![rel], attr);
            g.create_cch_index(def.clone()).expect("create");
            let idx = g.cch_index(&def).expect("held");
            check_index_against_oracle(
                |s, t| {
                    let got = idx.cch.query(s, t);
                    if let Some((_, path)) = &got {
                        assert_path_simple(path);
                    }
                    got
                },
                &edges,
            );
        }
    }
}
