//! Test fixtures for CCH (#3207), on top of the graph test kit ([`crate::testkit`], re-exported
//! here, so `cch::testkit::{grid, dijkstra, ...}` keeps working).
//!
//! Written from the [`CchInput`] contract and the C reference on FalkorDB's
//! `master` branch (`src/algorithms/cch.c`, `tests/flow/test_cch.py`) -- never
//! from any Rust CCH implementation:
//! - [`to_input`], the `CchInput` an [`EdgeList`] stands for, and [`assert_input_invariants`],
//!   the contract any `CchInput` must meet;
//! - [`lower_rank_distances`], the exact customization oracle;
//! - the Cypher renderers (`EdgeList::to_cypher_create`, `TypedEdgeList::to_cypher_create`),
//!   which build the same graph through queries;
//! - [`Edit`] and [`random_edit_batch`], the weight and structure edits the maintenance tests
//!   replay.

pub use crate::testkit::*;

use super::{CchInput, Csr, WeightCsr};
use std::collections::HashMap;
use std::collections::HashSet;

/// Created before the nodes and the edge batches, so the `MATCH (a:N{v:e[0]}),(b:N{v:e[1]})`
/// that wires each edge is an index seek, not a scan of every `:N` node per endpoint (the
/// C flow tests do the same: `tests/flow/test_cch.py`, "fast grid construction").
const CREATE_NODE_INDEX: &str = "CREATE INDEX FOR (n:N) ON (n.v);\n";

impl EdgeList {
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
                .map(|&(u, v, w)| format!("[{u},{v},{}]", cypher_number(w)))
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

impl TypedEdgeList {
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
                .map(|&(u, v, w, _)| format!("[{u},{v},{}]", cypher_number(w)))
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

/// `w` as a Cypher literal that reads back as `w`. `{w}` alone writes `1e20` as
/// `100000000000000000000`, an integer literal past `i64` that the lexer refuses, so from
/// `1e18` up the weight is written with an exponent and a decimal point (`1.0e20`), which the
/// lexer reads as a float. Below that `{w}` is exact and keeps whole weights integers.
///
/// Cypher has no infinity literal (`{w}` writes `inf`, which reads as an unbound variable), so
/// a closed edge's `+inf` is written as the float division `1.0/0.0`, which evaluates to it
/// (`-inf` likewise). A NaN is no valid weight and is left as `{w}`.
///
/// The lexer refuses a subnormal float literal too (`Float overflow`), so a subnormal weight
/// is written as the division `(w * 2^600) / 2^600` of two normal literals. Scaling by a power
/// of two is exact both ways, so the engine evaluates it back to `w` bit for bit.
fn cypher_number(w: f64) -> String {
    if w.is_infinite() {
        return String::from(if w > 0.0 { "1.0/0.0" } else { "-1.0/0.0" });
    }
    if w.is_subnormal() {
        let scale = 2f64.powi(600);
        return format!("({}/{})", float_literal(w * scale), float_literal(scale));
    }
    if w.abs() < 1e18 {
        return format!("{w}");
    }
    float_literal(w)
}

/// `w` with an exponent and a decimal point (`1.0e20`), which the lexer reads as a float.
fn float_literal(w: f64) -> String {
    let s = format!("{w:e}");
    if s.contains('.') {
        s
    } else {
        s.replacen('e', ".0e", 1)
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
/// (The oracles, [`dijkstra`] and [`lower_rank_distances`], do reject them.) So a NaN
/// among parallel edges is kept whatever their order, where `min` would drop it after a
/// finite weight; a negative one is the minimum anyway.
#[must_use]
pub fn to_input(edges: &EdgeList) -> CchInput {
    let n = edges.n;

    let mut best: HashMap<(u64, u64), f64> = HashMap::new();
    for &(u, v, w) in &edges.edges {
        if u == v {
            continue;
        }
        best.entry((u, v))
            .and_modify(|cur| {
                if w.is_nan() || w < *cur {
                    *cur = w;
                }
            })
            .or_insert(w);
    }

    let weights = weight_csr_from_pairs(n, best.iter().map(|(&(u, v), &w)| (u, v, w)));

    // Symmetric union of the directed pattern and its transpose, i.e.
    // `eWiseAdd(A, A^T, ONEB_BOOL)` in cch.c -- self-loops are already
    // excluded above, so there is no diagonal to drop.
    let mut skeleton_pairs: Vec<(u64, u64)> = Vec::with_capacity(best.len() * 2);
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
    mut pairs: Vec<(u64, u64)>,
) -> Csr {
    pairs.sort_unstable();
    pairs.dedup();

    let mut p = vec![0u64; n + 1];
    for &(u, _) in &pairs {
        p[u as usize + 1] += 1;
    }
    for k in 0..n {
        p[k + 1] += p[k];
    }
    let i = pairs.into_iter().map(|(_, v)| v).collect();
    Csr { n, p: p.into(), i }
}

fn weight_csr_from_pairs(
    n: usize,
    pairs: impl Iterator<Item = (u64, u64, f64)>,
) -> WeightCsr {
    let mut rows: Vec<(u64, u64, f64)> = pairs.collect();
    rows.sort_unstable_by_key(|&(u, v, _)| (u, v));

    let mut p = vec![0u64; n + 1];
    for &(u, _, _) in &rows {
        p[u as usize + 1] += 1;
    }
    for k in 0..n {
        p[k + 1] += p[k];
    }
    let i = rows.iter().map(|&(_, v, _)| v).collect();
    let w = rows.iter().map(|&(_, _, w)| w).collect();
    WeightCsr {
        csr: Csr { n, p: p.into(), i },
        w,
    }
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
/// out of range, or a repeat), or if any edge weight is negative or NaN (the
/// same domain `dijkstra` requires: `+inf` is a closed edge), self-loops included.
#[must_use]
pub fn lower_rank_distances(
    edges: &EdgeList,
    perm: &[u64],
) -> Vec<Vec<f64>> {
    let n = edges.n;
    assert_eq!(perm.len(), n, "perm must have exactly n entries");
    let mut rank_of = vec![u64::MAX; n];
    for (rank, &node) in perm.iter().enumerate() {
        assert!(
            (node as usize) < n,
            "perm contains out-of-range node {node}"
        );
        assert_eq!(rank_of[node as usize], u64::MAX, "perm repeats node {node}");
        rank_of[node as usize] = rank as u64;
    }

    let mut best: HashMap<(u64, u64), f64> = HashMap::new();
    for &(u, v, w) in &edges.edges {
        assert!(
            w >= 0.0,
            "lower_rank_distances requires non-negative weights, got {w} on ({u}, {v})"
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

/// Checks the structural invariants the contract promises about a [`CchInput`]: the
/// skeleton is symmetric and self-loop-free with sorted, duplicate-free rows;
/// the weights are directed, self-loop-free and non-negative (`+inf` allowed), with
/// sorted duplicate-free rows; and the skeleton is exactly the symmetric pattern of the
/// weights: every weighted pair is in it, and every pair in it is weighted one way or the
/// other.
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
    let skeleton_set: HashSet<(u64, u64)> = skeleton_pairs.into_iter().collect();
    for &(u, v) in &skeleton_set {
        assert!(
            skeleton_set.contains(&(v, u)),
            "skeleton is not symmetric: ({u}, {v}) has no reverse"
        );
    }

    let weight_pairs = csr_pairs(&input.weights.csr);
    for (&(u, v), &w) in weight_pairs.iter().zip(&input.weights.w) {
        assert_ne!(u, v, "weights has a self-loop at node {u}");
        assert!(w >= 0.0, "weight on ({u}, {v}) is {w}, must be >= 0");
        assert!(
            skeleton_set.contains(&(u, v)),
            "weighted pair ({u}, {v}) is not present in the skeleton"
        );
    }
    let weight_set: HashSet<(u64, u64)> = weight_pairs.into_iter().collect();
    for &(u, v) in &skeleton_set {
        assert!(
            weight_set.contains(&(u, v)) || weight_set.contains(&(v, u)),
            "skeleton pair ({u}, {v}) has no weight in either direction"
        );
    }
}

/// `csr.p` has `n + 1` non-decreasing offsets ending at `csr.i.len()`, and
/// every row of `csr.i` is sorted, duplicate-free and in range `0..n`.
fn assert_csr_shape(
    csr: &Csr,
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
fn csr_pairs(csr: &Csr) -> Vec<(u64, u64)> {
    let mut pairs = Vec::with_capacity(csr.i.len());
    for row in 0..csr.n {
        for &col in csr.row(row) {
            pairs.push((row as u64, col));
        }
    }
    pairs
}

/// One edit to a graph, for maintenance tests. Edits act on a directed
/// pair `(u, v)`: [`Edit::SetWeight`] and [`Edit::RemoveEdge`] touch every
/// parallel edge of that pair.
#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    /// Every `(u, v)` edge gets weight `w`. `increased` says whether the
    /// pair's cheapest weight went up (`true`) or down (`false`).
    SetWeight {
        u: u64,
        v: u64,
        w: f64,
        increased: bool,
    },
    RemoveEdge {
        u: u64,
        v: u64,
    },
    /// A new edge. `was_reachable`: `u` could already reach `v` before it.
    AddEdge {
        u: u64,
        v: u64,
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
    u: u64,
    v: u64,
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
            // From 2^53 up (and at +inf) adding a small integer can round back to `m`.
            if w <= m {
                return None;
            }
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
            // A closed edge (+inf) times a fraction is +inf or NaN, not a decrease.
            if m <= 0.0 || !m.is_finite() {
                return None;
            }
            let w = (m * next_unit(state)).floor();
            if w >= m {
                return None;
            }
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
            let n = g.n as u64;
            let start = next_node(state, g.n * g.n);
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
        CchInput, Csr, EdgeList, HashMap, TypedEdgeList, WeightCsr, assert_input_invariants,
        dijkstra, disconnected_components, empty, grid, lower_rank_distances, path, random_sparse,
        splitmix, star, tied_paths, to_input, with_isolated_nodes, with_parallel_edges,
        with_self_loops, zero_weight_cycle,
    };

    /// Independent second derivation of [`lower_rank_distances`], used only
    /// to cross-check it (self-tests only, tiny graphs only -- it enumerates
    /// every simple path): for each `(a, b)`, DFS every simple `a -> b` walk
    /// (no repeated node, so it terminates) whose non-endpoint nodes all
    /// satisfy the rank bound, and take the cheapest.
    fn lower_rank_distances_brute_force(
        edges: &EdgeList,
        perm: &[u64],
    ) -> Vec<Vec<f64>> {
        let n = edges.n;
        let mut rank_of: Vec<u64> = vec![0; n];
        for (r, &node) in perm.iter().enumerate() {
            rank_of[node as usize] = r as u64;
        }

        let mut best: HashMap<(u64, u64), f64> = HashMap::new();
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
        let mut adj: Vec<Vec<(u64, f64)>> = vec![Vec::new(); n];
        for (&(u, v), &w) in &best {
            adj[u as usize].push((v, w));
        }

        // Everything a single search leaves unchanged, bundled so the
        // recursive step stays under clippy's argument-count lint.
        struct Search<'a> {
            dst: u64,
            bound: u64,
            adj: &'a [Vec<(u64, f64)>],
            rank_of: &'a [u64],
        }

        fn dfs(
            s: &Search,
            cur: u64,
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
                    dst: b as u64,
                    bound: rank_of[a].min(rank_of[b]),
                    adj: &adj,
                    rank_of: &rank_of,
                };
                let mut visited = vec![false; n];
                visited[a] = true;
                let mut best_w = f64::INFINITY;
                dfs(&search, a as u64, 0.0, &mut visited, &mut best_w);
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
    ) -> Vec<u64> {
        let mut perm: Vec<u64> = (0..n as u64).collect();
        for i in (1..n).rev() {
            let j = (splitmix(state) % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        perm
    }

    // ---- hand-computed cases, lifted from the C flow tests -----------

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
        assert_eq!(*input.weights.csr.i, [1]);
        assert_eq!(*input.weights.w, [2.0]);
    }

    #[test]
    fn to_input_drops_self_loops_from_weights_and_skeleton() {
        let mut g = EdgeList::new(2);
        g.push(0, 0, 1.0);
        g.push(0, 1, 3.0);
        let input = to_input(&g);
        assert_eq!(*input.weights.csr.i, [1], "self-loop must not appear");
        // symmetrized (0,1): row 0 -> [1], row 1 -> [0]; no diagonal entry.
        assert_eq!(*input.skeleton.p, [0, 1, 2]);
        assert_eq!(*input.skeleton.i, [1, 0]);
    }

    #[test]
    fn to_input_skeleton_is_symmetric_even_for_a_one_way_edge() {
        let mut g = EdgeList::new(2);
        g.push(0, 1, 1.0);
        let input = to_input(&g);
        // one directed weight ...
        assert_eq!(*input.weights.csr.i, [1]);
        assert_eq!(*input.weights.csr.p, [0, 1, 1]);
        // ... but the skeleton carries both directions (topology only).
        assert_eq!(*input.skeleton.p, [0, 1, 2]);
        assert_eq!(*input.skeleton.i, [1, 0]);
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
    #[should_panic(expected = "requires non-negative")]
    fn lower_rank_distances_rejects_a_nan_weight() {
        let mut g = EdgeList::new(2);
        g.push(0, 1, f64::NAN);
        let _ = lower_rank_distances(&g, &[0, 1]);
    }

    /// A self-loop is dropped from the resolved graph, but its weight is still validated.
    #[test]
    #[should_panic(expected = "requires non-negative")]
    fn lower_rank_distances_rejects_a_bad_weight_on_a_self_loop() {
        let mut g = EdgeList::new(2);
        g.push(0, 0, -1.0);
        let _ = lower_rank_distances(&g, &[0, 1]);
    }

    /// `+inf` is a closed edge: both oracles read it as no edge, so closing one is the
    /// same as removing it, and a parallel finite edge still counts.
    #[test]
    fn a_plus_inf_edge_reads_as_no_edge_in_both_oracles() {
        let mut open = EdgeList::new(4);
        for (u, v, w) in [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 5.0), (2, 3, 1.0)] {
            open.push(u, v, w);
        }
        let mut closed = open.clone();
        closed.push(1, 3, f64::INFINITY);
        closed.push(0, 2, f64::INFINITY); // parallel to the 5.0 edge
        closed.push(3, 3, f64::INFINITY); // a closed self-loop
        let perm = [3, 1, 0, 2];
        for s in 0..4 {
            assert_eq!(dijkstra(&closed, s), dijkstra(&open, s));
        }
        assert_eq!(
            lower_rank_distances(&closed, &perm),
            lower_rank_distances(&open, &perm)
        );

        let mut cut = EdgeList::new(2);
        cut.push(0, 1, f64::INFINITY);
        assert_eq!(dijkstra(&cut, 0), vec![0.0, f64::INFINITY]);
    }

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
        assert_eq!(*resolved.weights.w, [2.0]);

        let only_2 = g.select(&[2]); // never drawn
        assert!(only_2.edges.is_empty());
        assert_eq!(only_2.n, 2);
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
                p: vec![0, 1, 1].into(),
                i: vec![1].into(),
            },
            weights: WeightCsr {
                csr: Csr {
                    n: 2,
                    p: vec![0, 1, 1].into(),
                    i: vec![1].into(),
                },
                w: vec![1.0].into(),
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
                p: vec![0, 1, 2].into(),
                i: vec![1, 0].into(),
            },
            weights: WeightCsr {
                csr: Csr {
                    n: 2,
                    p: vec![0, 1, 1].into(), // row 0: [0] -- a self-loop
                    i: vec![0].into(),
                },
                w: vec![1.0].into(),
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
                p: vec![0, 0, 0].into(), // empty skeleton
                i: vec![].into(),
            },
            weights: WeightCsr {
                csr: Csr {
                    n: 2,
                    p: vec![0, 1, 1].into(),
                    i: vec![1].into(),
                },
                w: vec![1.0].into(),
            },
        };
        assert_input_invariants(&input);
    }

    #[test]
    #[should_panic(expected = "must be >= 0")]
    fn assert_input_invariants_catches_a_negative_weight() {
        let input = CchInput {
            n: 2,
            skeleton: Csr {
                n: 2,
                p: vec![0, 1, 2].into(),
                i: vec![1, 0].into(),
            },
            weights: WeightCsr {
                csr: Csr {
                    n: 2,
                    p: vec![0, 1, 1].into(),
                    i: vec![1].into(),
                },
                w: vec![-1.0].into(),
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
                p: vec![0, 2, 3, 3].into(),
                i: vec![2, 1, 0].into(), // row 0 should be [1, 2], not [2, 1]
            },
            weights: WeightCsr {
                csr: Csr {
                    n: 3,
                    p: vec![0, 0, 0, 0].into(),
                    i: vec![].into(),
                },
                w: vec![].into(),
            },
        };
        assert_input_invariants(&input);
    }

    /// A weight from `1e18` up is written as a float literal (`{w}` would write an integer
    /// past `i64`, which the lexer refuses), and reads back as the same weight.
    #[test]
    fn to_cypher_create_writes_a_large_weight_as_a_float() {
        for w in [1e20, 2f64.powi(61), 1e300] {
            let mut g = EdgeList::new(2);
            g.push(0, 1, w);
            let mut t = TypedEdgeList::new(2);
            t.push(0, 1, w, 0);
            for script in [g.to_cypher_create("ROAD"), t.to_cypher_create(&["ROAD"])] {
                let lit = script
                    .split("[0,1,")
                    .nth(1)
                    .and_then(|r| r.split(']').next())
                    .expect("the edge item");
                assert!(lit.contains('.') && lit.contains('e'), "{lit}");
                assert_eq!(lit.parse::<f64>(), Ok(w), "{lit}");
            }
        }
    }

    /// A closed edge's `+inf` is written as `1.0/0.0`: the script parses and binds (a bare
    /// `inf` would be an unbound variable), and the engine evaluates the division to `+inf`.
    #[test]
    fn to_cypher_create_writes_infinity_as_a_division() {
        use crate::parser::cypher::Parser;
        use crate::planner::binder::Binder;
        use crate::runtime::{functions::init_functions, value::Value};

        let _ = init_functions();
        let mut g = EdgeList::new(2);
        g.push(0, 1, f64::INFINITY);
        let mut t = TypedEdgeList::new(2);
        t.push(0, 1, f64::INFINITY, 0);
        for script in [g.to_cypher_create("ROAD"), t.to_cypher_create(&["ROAD"])] {
            assert!(script.contains("[0,1,1.0/0.0]"), "{script}");
            for stmt in script.lines() {
                let ir = Parser::new(stmt)
                    .parse()
                    .unwrap_or_else(|e| panic!("{stmt}: {e}"));
                Binder::default()
                    .bind(ir)
                    .unwrap_or_else(|e| panic!("{stmt}: {e}"));
            }
        }
        assert!(
            matches!(Value::Float(1.0) / Value::Float(0.0), Ok(Value::Float(w)) if w == f64::INFINITY)
        );
    }

    /// A subnormal weight (a valid one: finite and non-negative) is written as a division of
    /// two normal literals, since the lexer refuses a subnormal literal. The script parses and
    /// binds, and the division evaluates back to the same weight bit for bit.
    #[test]
    fn to_cypher_create_writes_a_subnormal_weight_as_a_division() {
        use crate::parser::cypher::Parser;
        use crate::planner::binder::Binder;
        use crate::runtime::{functions::init_functions, value::Value};

        let _ = init_functions();
        for w in [
            1e-310,
            f64::from_bits(1),
            f64::MIN_POSITIVE - f64::from_bits(1),
        ] {
            assert!(w.is_subnormal());
            let mut g = EdgeList::new(2);
            g.push(0, 1, w);
            let mut t = TypedEdgeList::new(2);
            t.push(0, 1, w, 0);
            for script in [g.to_cypher_create("ROAD"), t.to_cypher_create(&["ROAD"])] {
                for stmt in script.lines() {
                    let ir = Parser::new(stmt)
                        .parse()
                        .unwrap_or_else(|e| panic!("{stmt}: {e}"));
                    Binder::default()
                        .bind(ir)
                        .unwrap_or_else(|e| panic!("{stmt}: {e}"));
                }
                let lit = script
                    .split("[0,1,(")
                    .nth(1)
                    .and_then(|r| r.split(")]").next())
                    .expect("the edge item");
                let (num, den) = lit.split_once('/').expect("a division");
                let (num, den) = (num.parse::<f64>().unwrap(), den.parse::<f64>().unwrap());
                assert!(num.is_normal() && den.is_normal(), "{lit}");
                assert!(
                    matches!(Value::Float(num) / Value::Float(den), Ok(Value::Float(r)) if r.to_bits() == w.to_bits()),
                    "{lit}"
                );
            }
        }
    }

    /// A NaN among parallel edges reaches the input whichever order the edges come in, so
    /// an invalid-weight fixture stays invalid.
    #[test]
    fn to_input_keeps_a_nan_among_parallel_edges() {
        for order in [[1.0, f64::NAN], [f64::NAN, 1.0]] {
            let mut g = EdgeList::new(2);
            for w in order {
                g.push(0, 1, w);
            }
            assert!(to_input(&g).weights.w[0].is_nan(), "{order:?}");
        }
    }

    #[test]
    #[should_panic(expected = "has no weight in either direction")]
    fn assert_input_invariants_catches_a_skeleton_pair_with_no_weight() {
        let input = CchInput {
            n: 2,
            skeleton: Csr {
                n: 2,
                p: vec![0, 1, 2].into(),
                i: vec![1, 0].into(),
            },
            weights: WeightCsr {
                csr: Csr {
                    n: 2,
                    p: vec![0, 0, 0].into(),
                    i: vec![].into(),
                },
                w: vec![].into(),
            },
        };
        assert_input_invariants(&input);
    }
}

/// Self-tests of `random_edit_batch`.
#[cfg(test)]
mod maintenance_tests {
    use super::{
        EdgeList, Edit, all_pairs, apply_edit, dijkstra, draw_edit, pair_min, random_edit_batch,
        random_sparse,
    };

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

    /// A weight edit is drawn only when it changes the weight the way it says: no increase
    /// on a weight `+ 1..=5` cannot move (2^60) or on a closed edge, no decrease from a
    /// closed edge. Kinds 0 and 1 then fall back to `AddNode`.
    #[test]
    fn weight_edits_that_cannot_change_the_weight_are_not_drawn() {
        for (w, kinds) in [(2f64.powi(60), &[0][..]), (f64::INFINITY, &[0, 1][..])] {
            let mut g = EdgeList::new(2);
            g.push(0, 1, w);
            for &kind in kinds {
                for seed in 0..64 {
                    let mut state = seed;
                    assert_eq!(draw_edit(&g, kind, &mut state), None, "w {w}, kind {kind}");
                }
            }
        }
    }
}

/// `Cch::build` / `Cch::customize` rejecting invalid weights. Weights must be
/// `>= 0`, so a negative weight, `-inf` and NaN are `CchError::InvalidWeight`;
/// `+inf` is legal (a closed edge). Black-box tests: they use only the signatures
/// in this module's parent (`mod.rs`).
#[cfg(test)]
mod rejection_tests {
    use super::{EdgeList, grid, to_input};
    use crate::index::falkordb::cch::{Cch, CchError};

    const BAD_WEIGHTS: [(&str, f64); 3] = [
        ("negative", -1.0),
        ("NaN", f64::NAN),
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

    /// `+inf` and `-0.0` are accepted by both, and a customize that closes an edge
    /// and then reopens it gives back the first metric.
    #[test]
    fn plus_inf_and_negative_zero_are_accepted() {
        let base = grid(2, 2);
        let good = to_input(&base);
        for w in [f64::INFINITY, -0.0] {
            let odd = to_input(&with_one_bad_weight(base.clone(), w));
            assert!(Cch::build(&odd).is_ok(), "build should accept {w}");
            let mut cch = Cch::build(&good).expect("a valid input must build");
            let before = (0..4)
                .flat_map(|s| (0..4).map(move |t| (s, t)))
                .map(|(s, t)| cch.query(s, t))
                .collect::<Vec<_>>();
            assert_eq!(cch.customize(&odd.weights), Ok(()), "customize {w}");
            assert_eq!(cch.customize(&good.weights), Ok(()));
            let after = (0..4)
                .flat_map(|s| (0..4).map(move |t| (s, t)))
                .map(|(s, t)| cch.query(s, t))
                .collect::<Vec<_>>();
            assert_eq!(before, after, "reopened after {w}");
        }
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
        random_sparse, random_zero_heavy, tied_paths, with_isolated_nodes, with_parallel_edges,
        with_self_loops, zero_weight_cycle,
    };
    use crate::graph::{graph::Graph, graphblas::test_init};
    use crate::index::falkordb::cch::CchIndexDef;
    use crate::index::index_docs::IndexDocs;
    use crate::runtime::value::Value;

    /// `edges` with every `k`-th edge closed (`+inf`): the oracle reads a closed edge
    /// as no edge, so the index must answer as if it were gone.
    fn close_every(
        mut edges: EdgeList,
        k: usize,
    ) -> EdgeList {
        for e in edges.edges.iter_mut().step_by(k) {
            e.2 = f64::INFINITY;
        }
        edges
    }

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
        g.create_nodes(&ids, &mut IndexDocs::default())
            .expect("creatable");
        let attr = g.get_or_create_rel_attr_id(&Arc::new(String::from("w")));

        let rel_ids: Vec<u64> = g
            .relationship_id_space()
            .reserve(edges.edges.len(), &RoaringTreemap::new())
            .expect("reserved");
        let srcs: Vec<u64> = edges.edges.iter().map(|e| e.0).collect();
        let dsts: Vec<u64> = edges.edges.iter().map(|e| e.1).collect();
        g.create_relationships_bulk(
            &Arc::new(String::from("R")),
            &srcs,
            &dsts,
            &rel_ids,
            &mut IndexDocs::default(),
        )
        .expect("creatable");

        let mut weights: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
        for (e, &id) in edges.edges.iter().zip(&rel_ids) {
            weights.insert(id, vec![(attr, Value::Float(e.2))]);
        }
        g.import_relationship_attrs(&weights, &mut IndexDocs::default())
            .expect("valid weights");
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
            ("closed", close_every(random_sparse(20, 60, 7, 5), 3)),
            (
                "closed_parallel",
                close_every(
                    with_parallel_edges(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 3, 9, 7),
                    2,
                ),
            ),
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
