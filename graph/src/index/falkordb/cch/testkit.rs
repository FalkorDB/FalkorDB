//! Test fixtures for CCH (#3207), on top of the graph test kit ([`crate::testkit`], re-exported
//! here, so `cch::testkit::{grid, dijkstra, ...}` keeps working).
//!
//! Written from the [`CchInput`] contract and the C reference on FalkorDB's
//! `master` branch (`src/algorithms/cch.c`, `tests/flow/test_cch.py`) -- never
//! from any Rust CCH implementation:
//! - [`to_input`], the `CchInput` a weight matrix stands for, built with `cch.c`'s GraphBLAS
//!   calls, and [`assert_input_invariants`], the contract any `CchInput` must meet;
//! - [`lower_rank_distances`], the exact customization oracle, and [`valley`], the graph of C's
//!   `test01_create_query_drop`;
//! - [`EditGraph`], [`Edit`] and [`random_edit_batch`], the weight and structure edits the
//!   maintenance tests replay, over a graph that keeps every parallel edge of a pair;
//! - [`materialize`], a real `Graph` standing for a weight matrix, built without Cypher, with
//!   [`Decoys`] that must never win, and the index definition over it; [`real_graph`], the
//!   `Graph` of an edge list, which `materialize` builds on.

pub(crate) use crate::testkit::*;

use super::{CchIndexDef, CchInput, Csr, WeightCsr};
use crate::graph::graphblas::matrix::{Descriptor, Dup, Matrix};
use crate::graph::{graph::Graph, graphblas::test_init};
use crate::index::index_docs::IndexDocs;
use crate::runtime::value::Value;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::sync::Arc;

/// [`CchInput`] as its contract defines it, for the weight matrix `g`: a symmetric,
/// self-loop-free skeleton for Phase 1, and directed weights, one per entry of `g`. Built with
/// the GraphBLAS calls `cch.c` makes, not through `CchInput::build`, which is the code under
/// test: the diagonal goes (`GrB_select` with `GrB_OFFDIAG`; a self-loop carries no
/// shortest-path information, and `test14_parallel_selfloop_unreachable` treats one as a no-op),
/// and the skeleton is the pattern of `W + Wᵀ`.
///
/// Weights are **not** validated here, on purpose: the tests of `Cch::build`'s
/// `InvalidWeight` path build a deliberately invalid `CchInput` through this function.
/// (The oracles, [`dijkstra`] and [`lower_rank_distances`], do reject them.)
#[must_use]
pub(crate) fn to_input(g: &Matrix<f64>) -> CchInput {
    let n = node_count(g);
    let mut w = g.dup();
    w.drop_diagonal();
    // A pattern: its values (1.0) are never read.
    let mut skeleton = Matrix::<f64>::new(n as u64, n as u64);
    skeleton.set_pattern(None, &w, None);
    skeleton.set_pattern(None, &w, Some(Descriptor::T0));

    let (rows, cols, w) = w.extract_tuples();
    let (skeleton_rows, skeleton_cols, _) = skeleton.extract_tuples();
    CchInput {
        n,
        skeleton: csr(n, &skeleton_rows, skeleton_cols),
        weights: WeightCsr {
            csr: csr(n, &rows, cols),
            values: w,
        },
    }
}

/// The CSR of entries extracted row by row (`rows` sorted): row `r` starts where `rows` first
/// reaches `r`.
fn csr(
    n: usize,
    rows: &[u64],
    cols: Vec<u64>,
) -> Csr {
    let row_offsets = (0..=n as u64)
        .map(|r| rows.partition_point(|&row| row < r) as u64)
        .collect();
    Csr {
        node_count: n,
        row_offsets,
        neighbors: cols,
    }
}

/// Splices one more weight entry, `row -> neighbor` of `weight`, into `input`'s weight CSR at
/// its sorted place in the row, before an equal neighbor, so it can repeat one. The skeleton
/// is left as it is: this makes weights [`CchInput::build`] never gives, a self-loop or a
/// repeated pair.
pub(super) fn insert_weight_entry(
    input: &mut CchInput,
    row: usize,
    neighbor: u64,
    weight: f64,
) {
    let csr = &mut input.weights.csr;
    let range = csr.row_range(row);
    let at = range.start + csr.neighbors[range].partition_point(|&v| v < neighbor);
    csr.neighbors.insert(at, neighbor);
    input.weights.values.insert(at, weight);
    for offset in &mut csr.row_offsets[row + 1..] {
        *offset += 1;
    }
}

/// `g`'s pattern with its `k`-th entry (row by row, as [`entries`] lists them) weighing
/// `f(k, w)`, `w` its weight in `g`.
#[must_use]
pub(crate) fn reweight(
    g: &Matrix<f64>,
    mut f: impl FnMut(usize, f64) -> f64,
) -> Matrix<f64> {
    let edges: Vec<_> = entries(g)
        .into_iter()
        .enumerate()
        .map(|(k, (u, v, w))| (u, v, f(k, w)))
        .collect();
    weights(node_count(g), &edges)
}

/// C's `test01_create_query_drop` valley, both ways: `0 - 1` direct is 10, `0 - 2 - 1` is 2.
#[must_use]
pub(super) fn valley() -> Matrix<f64> {
    weights(
        3,
        &[
            (0, 1, 10.0),
            (1, 0, 10.0),
            (0, 2, 1.0),
            (2, 0, 1.0),
            (2, 1, 1.0),
            (1, 2, 1.0),
        ],
    )
}

/// The exact CCH customization oracle. `perm[rank] = node` is the
/// elimination order, the convention of [`Topology`](super::order::Topology)'s
/// `rank_to_node` (and of C's `perm`):
/// processing nodes in increasing rank order corresponds to eliminating
/// them one at a time, lowest rank first.
///
/// `result[a][b]` is the weight of the cheapest directed walk `a -> b`
/// whose *intermediate* nodes -- everything but `a` and `b` themselves --
/// all have strictly lower rank than BOTH `a` and `b`, i.e. `rank <
/// min(rank(a), rank(b))`. `a` and `b` are themselves unrestricted: either
/// may have any rank at all, including the highest in the graph.
/// `result[a][a] == 0.0` always (the trivial empty walk). `f64::INFINITY`
/// if no such walk exists. Self-loops are ignored entirely -- a diagonal entry
/// contributes nothing.
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
/// If `perm` is not a permutation of `0..n` (wrong length, a value
/// out of range, or a repeat), or if any edge weight is negative or NaN (the
/// same domain `dijkstra` requires: `+inf` is a closed edge), self-loops included.
#[must_use]
pub(super) fn lower_rank_distances(
    g: &Matrix<f64>,
    perm: &[u64],
) -> Vec<Vec<f64>> {
    let n = node_count(g);
    assert_eq!(perm.len(), n, "perm must have exactly n entries");
    let mut rank_of = vec![usize::MAX; n];
    for (rank, &node) in perm.iter().enumerate() {
        let slot = rank_of
            .get_mut(node as usize)
            .unwrap_or_else(|| panic!("perm contains out-of-range node {node}"));
        assert_eq!(*slot, usize::MAX, "perm repeats node {node}");
        *slot = rank;
    }

    let mut d: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { 0.0 } else { f64::INFINITY })
                .collect()
        })
        .collect();
    for (u, v, w) in entries(g) {
        assert!(
            w >= 0.0,
            "lower_rank_distances requires non-negative weights, got {w} on ({u}, {v})"
        );
        if u != v {
            d[u as usize][v as usize] = w;
        }
    }

    let mut result = vec![vec![f64::INFINITY; n]; n];
    for (r, &k) in perm.iter().enumerate() {
        let k = k as usize;
        for b in (0..n).filter(|&b| rank_of[b] >= r) {
            result[k][b] = d[k][b];
            result[b][k] = d[b][k];
        }
        // Insert `k` as an intermediate. Row `k` itself cannot change (`d[k][k]` is 0), so a
        // copy of it serves every row.
        let via_k = d[k].clone();
        for row in d.iter_mut().filter(|row| row[k].is_finite()) {
            let to_k = row[k];
            for (to_j, &k_to_j) in row.iter_mut().zip(&via_k) {
                *to_j = to_j.min(to_k + k_to_j);
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
pub(super) fn assert_input_invariants(input: &CchInput) {
    assert_csr_shape(&input.skeleton, input.n, "skeleton");
    assert_csr_shape(&input.weights.csr, input.n, "weights");
    assert_eq!(
        input.weights.csr.neighbors.len(),
        input.weights.values.len(),
        "weights.csr.neighbors and weights.values must be the same length"
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
    for (&(u, v), &w) in weight_pairs.iter().zip(&input.weights.values) {
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

/// `csr.row_offsets` has `n + 1` non-decreasing offsets ending at `csr.neighbors.len()`, and
/// every row of `csr.neighbors` is sorted, duplicate-free and in range `0..n`.
fn assert_csr_shape(
    csr: &Csr,
    n: usize,
    label: &str,
) {
    assert_eq!(
        csr.node_count, n,
        "{label}.node_count must match CchInput.n"
    );
    assert_eq!(
        csr.row_offsets.len(),
        n + 1,
        "{label}.row_offsets must have n+1 offsets"
    );
    assert_eq!(
        *csr.row_offsets.first().unwrap_or(&0),
        0,
        "{label}.row_offsets must start at 0"
    );
    assert_eq!(
        *csr.row_offsets.last().unwrap_or(&0) as usize,
        csr.neighbors.len(),
        "{label}.row_offsets must end at neighbors.len()"
    );
    for row in 0..n {
        let (lo, hi) = (
            csr.row_offsets[row] as usize,
            csr.row_offsets[row + 1] as usize,
        );
        assert!(
            lo <= hi,
            "{label}.row_offsets is not non-decreasing at row {row}"
        );
        let row_cols = &csr.neighbors[lo..hi];
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
    (0..csr.node_count)
        .flat_map(|row| csr.row(row).iter().map(move |&col| (row as u64, col)))
        .collect()
}

/// A graph under edit, as the maintenance tests drive one through Cypher: every parallel edge
/// of each directed pair, by weight, on one relationship type, and the node count. Its
/// [`Self::oracle`] is the weight matrix of the cheapest edge per pair, so deleting a pair's
/// cheapest edge reveals the next one.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct EditGraph {
    pub(super) n: usize,
    /// The weights of each pair's edges, in the order they were added; never empty.
    /// Self-loops excluded.
    pub(super) pairs: BTreeMap<(u64, u64), Vec<f64>>,
}

impl EditGraph {
    /// One edge per entry of `g`, self-loops dropped.
    #[must_use]
    pub(super) fn of(g: &Matrix<f64>) -> Self {
        let pairs = entries(g)
            .into_iter()
            .filter(|e| e.0 != e.1)
            .map(|(u, v, w)| ((u, v), vec![w]))
            .collect();
        Self {
            n: node_count(g),
            pairs,
        }
    }

    /// The oracle: `n` nodes, and per pair the weight of its cheapest edge.
    ///
    /// # Panics
    /// If a weight is NaN ([`cheapest_of`]).
    #[must_use]
    pub(super) fn oracle(&self) -> Matrix<f64> {
        let cheapest: Vec<_> = self
            .pairs
            .iter()
            .map(|(&(u, v), ws)| (u, v, cheapest_of(u, v, ws)))
            .collect();
        weights(self.n, &cheapest)
    }

    /// The weight of the cheapest `(u, v)` edge, if there is one.
    ///
    /// # Panics
    /// If a weight of the pair is NaN ([`cheapest_of`]).
    #[must_use]
    pub(super) fn cheapest(
        &self,
        u: u64,
        v: u64,
    ) -> Option<f64> {
        self.pairs.get(&(u, v)).map(|ws| cheapest_of(u, v, ws))
    }
}

/// The cheapest of `ws`, the weights of the `(u, v)` edges.
///
/// # Panics
/// If one is NaN. `f64::min` drops a NaN, so it would read as no edge on a pair of its own and
/// vanish next to a finite one, and the oracles downstream, which refuse NaN, would never see
/// it. Any other weight passes as it is: a negative one is the cheapest, so it reaches them.
fn cheapest_of(
    u: u64,
    v: u64,
    ws: &[f64],
) -> f64 {
    ws.iter().fold(f64::INFINITY, |min, &w| {
        assert!(!w.is_nan(), "EditGraph: a NaN weight on ({u}, {v})");
        min.min(w)
    })
}

/// One edit to a graph, for maintenance tests. Edits act on a directed pair `(u, v)`.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Edit {
    /// Every `(u, v)` edge gets weight `w`. `increased` says whether the
    /// pair's cheapest weight went up (`true`) or down (`false`).
    SetWeight {
        u: u64,
        v: u64,
        w: f64,
        increased: bool,
    },
    /// The cheapest `(u, v)` edge goes; the pair keeps its others, if it has any.
    RemoveCheapest { u: u64, v: u64 },
    /// A new edge, parallel to the pair's others if it has any. `was_reachable`: `u` could
    /// already reach `v` before it.
    AddEdge {
        u: u64,
        v: u64,
        w: f64,
        was_reachable: bool,
    },
    /// One new isolated node, id = the old `n`.
    AddNode,
}

/// Applies `edit` to `g` in place.
///
/// # Panics
/// If `SetWeight`/`RemoveCheapest` names a pair with no edge.
pub(super) fn apply_edit(
    g: &mut EditGraph,
    edit: &Edit,
) {
    match *edit {
        Edit::SetWeight { u, v, w, .. } => {
            let ws = g
                .pairs
                .get_mut(&(u, v))
                .unwrap_or_else(|| panic!("SetWeight on missing edge ({u}, {v})"));
            ws.iter_mut().for_each(|x| *x = w);
        }
        Edit::RemoveCheapest { u, v } => {
            let ws = g
                .pairs
                .get_mut(&(u, v))
                .unwrap_or_else(|| panic!("RemoveCheapest on missing edge ({u}, {v})"));
            let at = (0..ws.len())
                .min_by(|&a, &b| ws[a].total_cmp(&ws[b]))
                .expect("a pair has an edge");
            ws.remove(at);
            if ws.is_empty() {
                g.pairs.remove(&(u, v));
            }
        }
        Edit::AddEdge { u, v, w, .. } => {
            assert_ne!(u, v, "AddEdge of a self-loop");
            g.pairs.entry((u, v)).or_default().push(w);
        }
        Edit::AddNode => g.n += 1,
    }
}

/// A seeded batch of 1..=4 edits drawn from: weight up, weight down, the cheapest edge
/// removed, edge added between nodes already connected (`u` reaches `v`; possibly parallel
/// to an edge), edge added between unconnected nodes, node added. Edits apply in order, so
/// later ones may use a node added earlier. Returns the graph after the whole batch plus the
/// edits; `apply_edit` over `base` reproduces it.
///
/// A drawn kind that cannot apply to the current graph (no edge to remove,
/// no unconnected pair, zero-weight edge to lower, ...) becomes `AddNode`.
/// Weights stay finite and >= 0; "down" can reach exactly `0.0`.
#[must_use]
pub(super) fn random_edit_batch(
    base: &EditGraph,
    seed: u64,
) -> (EditGraph, Vec<Edit>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut g = base.clone();
    let mut edits = Vec::new();
    let count = rng.random_range(1..5);
    for _ in 0..count {
        let kind = EditKind::ALL[rng.random_range(0..EditKind::ALL.len() as u64) as usize];
        let edit = draw_edit(&g, kind, &mut rng).unwrap_or(Edit::AddNode);
        apply_edit(&mut g, &edit);
        edits.push(edit);
    }
    (g, edits)
}

/// What [`random_edit_batch`] draws, before it knows whether the graph allows it.
#[derive(Clone, Copy, Debug)]
enum EditKind {
    Raise,
    Lower,
    RemoveCheapest,
    AddReachable,
    AddUnreachable,
    AddNode,
}

impl EditKind {
    const ALL: [Self; 6] = [
        Self::Raise,
        Self::Lower,
        Self::RemoveCheapest,
        Self::AddReachable,
        Self::AddUnreachable,
        Self::AddNode,
    ];
}

fn draw_edit(
    g: &EditGraph,
    kind: EditKind,
    rng: &mut SmallRng,
) -> Option<Edit> {
    let pairs: Vec<(u64, u64)> = g.pairs.keys().copied().collect();
    let pick_pair = |rng: &mut SmallRng| pairs[rng.random_range(0..pairs.len())];
    match kind {
        EditKind::Raise | EditKind::Lower | EditKind::RemoveCheapest if pairs.is_empty() => None,
        EditKind::Raise => {
            let (u, v) = pick_pair(rng);
            let m = g.cheapest(u, v)?;
            let w = m + f64::from(rng.random_range(1..6u32));
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
        EditKind::Lower => {
            let (u, v) = pick_pair(rng);
            let m = g.cheapest(u, v)?;
            // A closed edge (+inf) times a fraction is +inf or NaN, not a decrease.
            if m <= 0.0 || !m.is_finite() {
                return None;
            }
            let w = (m * rng.random::<f64>()).floor();
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
        EditKind::RemoveCheapest => {
            let (u, v) = pick_pair(rng);
            Some(Edit::RemoveCheapest { u, v })
        }
        EditKind::AddReachable | EditKind::AddUnreachable => {
            if g.n < 2 {
                return None;
            }
            let want_reachable = matches!(kind, EditKind::AddReachable);
            let reach = all_pairs(&g.oracle());
            let n = g.n as u64;
            let start = next_node(rng, g.n * g.n);
            for k in 0..n * n {
                let idx = (start + k) % (n * n);
                let (u, v) = (idx / n, idx % n);
                if u != v && reach[u as usize][v as usize].is_finite() == want_reachable {
                    let w = f64::from(rng.random_range(1..10u32));
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
        EditKind::AddNode => Some(Edit::AddNode),
    }
}

/// Weight attribute name of every [`real_graph`].
const W: &str = "w";

/// The relationship types of a [`materialize`]d graph: its index covers the two indexed ones,
/// never [`UNINDEXED`].
pub(crate) const INDEXED: [&str; 2] = ["T0", "T1"];

/// A relationship type no materialized index covers.
const UNINDEXED: &str = "U";

/// One relationship of a [`real_graph`]: `(src, dst, weight, type)`. A `None` weight writes no
/// attribute, which an index reads as 1.0.
pub(crate) type RealEdge = (u64, u64, Option<f64>, &'static str);

/// A `Graph` of `n` nodes (ids `0..n`) holding exactly `edges`, one relationship each (so
/// parallel and multi-type edges stay distinct relationships), weight attribute `w`. Every
/// type of [`INDEXED`] exists even with no edge, and the types are created in that order,
/// then any other in order of appearance; each type is one batch, in `edges`' order.
/// Returns the graph and the attribute id.
pub(crate) fn real_graph(
    n: usize,
    edges: &[RealEdge],
    name: &str,
) -> (Graph, u16) {
    test_init::ensure_init();
    let mut g = Graph::new(16, 16, 1, 0, name);
    // A fresh graph's id spaces open their first batch themselves.
    let nodes: RoaringTreemap = g
        .node_id_space()
        .reserve(n, &RoaringTreemap::new())
        .expect("reserved")
        .into_iter()
        .collect();
    assert_eq!(nodes.len(), n as u64, "a fresh graph hands out 0..n");
    g.create_nodes(&nodes).expect("creatable");
    let attr = g.get_or_create_rel_attr_id(&Arc::new(String::from(W)));

    let mut types: Vec<&str> = INDEXED.to_vec();
    for e in edges {
        if !types.contains(&e.3) {
            types.push(e.3);
        }
    }
    let mut weights: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
    for ty in types {
        let mine: Vec<usize> = (0..edges.len()).filter(|&k| edges[k].3 == ty).collect();
        // One batch per type, as one query each would be.
        g.roll_id_batches().expect("a consistent space");
        let rel_ids: Vec<u64> = g
            .relationship_id_space()
            .reserve(mine.len(), &RoaringTreemap::new())
            .expect("reserved");
        let srcs: Vec<u64> = mine.iter().map(|&k| edges[k].0).collect();
        let dsts: Vec<u64> = mine.iter().map(|&k| edges[k].1).collect();
        g.create_relationships_bulk(
            &Arc::new(String::from(ty)),
            &srcs,
            &dsts,
            &rel_ids,
            &mut IndexDocs::default(),
        )
        .expect("creatable");
        for (&k, &id) in mine.iter().zip(&rel_ids) {
            if let Some(w) = edges[k].2 {
                weights.insert(id, vec![(attr, Value::Float(w))]);
            }
        }
    }
    g.import_relationship_attrs(&weights, &mut IndexDocs::default())
        .expect("valid weights");
    (g, attr)
}

/// The index over the [`INDEXED`] types of a [`real_graph`], weighted by its attribute `attr`.
pub(crate) fn index_def(
    g: &Graph,
    attr: u16,
) -> CchIndexDef {
    let types = INDEXED
        .iter()
        .map(|t| g.get_type_id(t).expect("type exists"))
        .collect();
    CchIndexDef::new(types, attr)
}

/// What [`materialize`] adds next to each pair's own edge: edges an index must never answer
/// with. All off by default.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Decoys {
    /// Parallel edges per pair, alternating between the indexed types, each heavier than the
    /// pair's weight: the first one has no attribute (it reads as 1.0) where the weight is
    /// below 1.0. None on a closed pair.
    pub(crate) heavier: usize,
    /// A closed (`+inf`) parallel edge on every open pair, and a closed edge from each node
    /// `u` to `u + 1` where `g` has none (an arc in the hierarchy, never on a path).
    pub(crate) closed: bool,
    /// A weight-0 parallel edge of [`UNINDEXED`] on every pair, and one from each node `u` to
    /// `u + 1` where `g` has none: cheaper than anything, never usable.
    pub(crate) unindexed: bool,
    /// A weight-0 self-loop at every node.
    pub(crate) self_loops: bool,
    /// A parallel edge at exactly the pair's weight, on the other indexed type, a tie. Every
    /// other pair puts its own edge on `T1` and the tie on `T0`, so the tie wins those.
    pub(crate) ties: bool,
}

/// A real graph standing for a weight matrix.
pub(crate) struct Materialized {
    pub(crate) graph: Graph,
    /// The index over [`INDEXED`].
    pub(crate) def: CchIndexDef,
}

/// A `Graph` holding `g`: one edge per entry of weight exactly the entry's (no attribute when
/// it is 1.0, which the missing-attribute rule reads back as 1.0), plus `decoys`. Its cheapest
/// edge per pair is the matrix's weight, so `g` is the oracle of any index over it.
pub(crate) fn materialize(
    g: &Matrix<f64>,
    decoys: Decoys,
    name: &str,
) -> Materialized {
    let n = node_count(g) as u64;
    let as_attr = |w: f64| (w != 1.0).then_some(w);
    let entries = entries(g);
    let has = |u: u64, v: u64| {
        entries
            .binary_search_by(|e| (e.0, e.1).cmp(&(u, v)))
            .is_ok()
    };
    let mut edges: Vec<RealEdge> = Vec::new();
    for (k, &(u, v, w)) in entries.iter().enumerate() {
        let (own, other) = if decoys.ties && k % 2 == 1 {
            (INDEXED[1], INDEXED[0])
        } else {
            (INDEXED[0], INDEXED[1])
        };
        edges.push((u, v, as_attr(w), own));
        if u == v {
            continue;
        }
        if w.is_finite() {
            for h in 0..decoys.heavier {
                let dw = if h == 0 && w < 1.0 {
                    None
                } else {
                    Some(w + 1.0 + h as f64)
                };
                edges.push((u, v, dw, INDEXED[(k + h) % 2]));
            }
            if decoys.closed {
                edges.push((u, v, Some(f64::INFINITY), INDEXED[k % 2]));
            }
        }
        if decoys.unindexed {
            edges.push((u, v, Some(0.0), UNINDEXED));
        }
        if decoys.ties {
            edges.push((u, v, as_attr(w), other));
        }
    }
    for u in 0..n {
        let v = (u + 1) % n;
        if u != v && !has(u, v) {
            if decoys.closed {
                edges.push((u, v, Some(f64::INFINITY), INDEXED[0]));
            }
            if decoys.unindexed {
                edges.push((u, v, Some(0.0), UNINDEXED));
            }
        }
        if decoys.self_loops {
            edges.push((u, u, Some(0.0), INDEXED[0]));
        }
    }
    let (graph, attr) = real_graph(n as usize, &edges, name);
    Materialized {
        def: index_def(&graph, attr),
        graph,
    }
}

#[cfg(test)]
mod self_tests {
    use super::{
        CchInput, Csr, Matrix, WeightCsr, assert_input_invariants, csr_pairs,
        disconnected_components, empty, entries, grid, lower_rank_distances, node_count, path,
        random_sparse, star, tied_paths, to_input, valley, weights, with_isolated_nodes,
        zero_weight_cycle,
    };
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;

    /// Independent second derivation of [`lower_rank_distances`], used only
    /// to cross-check it (self-tests only, tiny graphs only -- it enumerates
    /// every simple path): for each `(a, b)`, DFS every simple `a -> b` walk
    /// (no repeated node, so it terminates) whose non-endpoint nodes all
    /// satisfy the rank bound, and take the cheapest.
    fn lower_rank_distances_brute_force(
        g: &Matrix<f64>,
        perm: &[u64],
    ) -> Vec<Vec<f64>> {
        let n = node_count(g);
        let mut rank_of: Vec<u64> = vec![0; n];
        for (r, &node) in perm.iter().enumerate() {
            rank_of[node as usize] = r as u64;
        }

        let mut adj: Vec<Vec<(u64, f64)>> = vec![Vec::new(); n];
        for (u, v, w) in entries(g) {
            if u != v {
                adj[u as usize].push((v, w));
            }
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

    /// A shuffle drawn from a seeded generator, as everything else in this kit, so a failing
    /// seed reproduces.
    fn random_perm(
        n: usize,
        rng: &mut SmallRng,
    ) -> Vec<u64> {
        let mut perm: Vec<u64> = (0..n as u64).collect();
        perm.shuffle(rng);
        perm
    }

    // ---- hand-computed cases, lifted from the C flow tests -----------

    /// The [`valley`], checked through `lower_rank_distances` to show the rank bound matters:
    /// when node 2 (the shortcut hop) outranks both 0 and 1, the 2-hop path is not an
    /// allowed intermediate and the direct edge is all that is left -- a
    /// plain `dijkstra`/`all_pairs` would miss this entirely, since it
    /// always finds the 2-hop shortcut regardless of rank.
    #[test]
    fn lower_rank_distances_respects_the_rank_bound_on_the_valley() {
        let g = valley();

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

    // ---- to_input: resolution ---------------------------------------

    #[test]
    fn to_input_drops_self_loops_from_weights_and_skeleton() {
        let g = weights(2, &[(0, 0, 1.0), (0, 1, 3.0)]);
        let input = to_input(&g);
        assert_eq!(
            *input.weights.csr.neighbors,
            [1],
            "self-loop must not appear"
        );
        // symmetrized (0,1): row 0 -> [1], row 1 -> [0]; no diagonal entry.
        assert_eq!(*input.skeleton.row_offsets, [0, 1, 2]);
        assert_eq!(*input.skeleton.neighbors, [1, 0]);
    }

    // ---- the oracle cross-checks itself ---------------------------------

    /// `lower_rank_distances` (the incremental-Floyd-Warshall derivation) and
    /// `lower_rank_distances_brute_force` (DFS over every simple path, an
    /// independently-written derivation of the same definition) must agree on every pair:
    /// over the structured generators and a few tiny random graphs, each paired with a
    /// handful of distinct permutations, so a bug that only shows up on a particular rank
    /// order (e.g. the identity permutation, which happens to coincide with unrestricted
    /// shortest paths on an already-sorted structure) doesn't hide.
    #[test]
    fn lower_rank_distances_agrees_with_brute_force_on_every_generator() {
        let mut looped = grid(2, 2);
        looped.set(0, 0, 0.5);
        looped.set(3, 3, 0.5);
        let mut graphs = vec![
            grid(2, 3),
            looped,
            star(4, 2.0),
            path(5),
            tied_paths(3, 1.5),
            zero_weight_cycle(4),
        ];
        // kept tiny: the brute force enumerates every simple path.
        graphs.extend((0..8).map(|seed| random_sparse(7, 16, 9, seed)));
        for (idx, g) in graphs.iter().enumerate() {
            let mut rng = SmallRng::seed_from_u64(0x5EED_u64.wrapping_add(idx as u64));
            for _ in 0..5 {
                let perm = random_perm(node_count(g), &mut rng);
                let exact = lower_rank_distances(g, &perm);
                let brute = lower_rank_distances_brute_force(g, &perm);
                for s in 0..node_count(g) {
                    for t in 0..node_count(g) {
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
        let _ = lower_rank_distances(&weights(2, &[(0, 1, f64::NAN)]), &[0, 1]);
    }

    /// A self-loop is dropped from the resolved graph, but its weight is still validated.
    #[test]
    #[should_panic(expected = "requires non-negative")]
    fn lower_rank_distances_rejects_a_bad_weight_on_a_self_loop() {
        let _ = lower_rank_distances(&weights(2, &[(0, 0, -1.0)]), &[0, 1]);
    }

    /// `+inf` is a closed edge: the customization oracle reads it as no edge, so closing one
    /// is the same as removing it, a closed self-loop included. (`dijkstra`'s own `+inf` cases
    /// are in the graph test kit.)
    #[test]
    fn a_plus_inf_edge_reads_as_no_edge_in_lower_rank_distances() {
        let open_edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 5.0), (2, 3, 1.0)];
        let open = weights(4, &open_edges);
        let mut closed = weights(4, &open_edges);
        closed.set(1, 3, f64::INFINITY);
        closed.set(3, 3, f64::INFINITY); // a closed self-loop
        let perm = [3, 1, 0, 2];
        assert_eq!(
            lower_rank_distances(&closed, &perm),
            lower_rank_distances(&open, &perm)
        );
    }

    // ---- assert_input_invariants --------------------------------------

    #[test]
    fn assert_input_invariants_accepts_to_input_s_own_output() {
        let graphs = [
            grid(3, 3),
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
                node_count: 2,
                row_offsets: vec![0, 1, 1],
                neighbors: vec![1],
            },
            weights: WeightCsr {
                csr: Csr {
                    node_count: 2,
                    row_offsets: vec![0, 1, 1],
                    neighbors: vec![1],
                },
                values: vec![1.0],
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
                node_count: 2,
                row_offsets: vec![0, 1, 2],
                neighbors: vec![1, 0],
            },
            weights: WeightCsr {
                csr: Csr {
                    node_count: 2,
                    row_offsets: vec![0, 1, 1], // row 0: [0] -- a self-loop
                    neighbors: vec![0],
                },
                values: vec![1.0],
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
                node_count: 2,
                row_offsets: vec![0, 0, 0], // empty skeleton
                neighbors: vec![],
            },
            weights: WeightCsr {
                csr: Csr {
                    node_count: 2,
                    row_offsets: vec![0, 1, 1],
                    neighbors: vec![1],
                },
                values: vec![1.0],
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
                node_count: 2,
                row_offsets: vec![0, 1, 2],
                neighbors: vec![1, 0],
            },
            weights: WeightCsr {
                csr: Csr {
                    node_count: 2,
                    row_offsets: vec![0, 1, 1],
                    neighbors: vec![1],
                },
                values: vec![-1.0],
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
                node_count: 3,
                row_offsets: vec![0, 2, 3, 3],
                neighbors: vec![2, 1, 0], // row 0 should be [1, 2], not [2, 1]
            },
            weights: WeightCsr {
                csr: Csr {
                    node_count: 3,
                    row_offsets: vec![0, 0, 0, 0],
                    neighbors: vec![],
                },
                values: vec![],
            },
        };
        assert_input_invariants(&input);
    }

    /// Each weight lands on its own pair: distinct weights over several rows and columns,
    /// read back out of the `WeightCsr` as triples, are `g`'s entries but the self-loops. A
    /// reordered `values` keeps every structural check green and fails this.
    #[test]
    fn to_input_puts_each_weight_on_its_own_pair() {
        let edges = [
            (0, 2, 1.5),
            (0, 3, 2.5),
            (1, 0, 3.5),
            (1, 1, 9.5), // a self-loop, dropped
            (2, 1, 4.5),
            (3, 0, 5.5),
            (3, 2, 6.5),
        ];
        let input = to_input(&weights(4, &edges));
        let got: Vec<(u64, u64, f64)> = csr_pairs(&input.weights.csr)
            .into_iter()
            .zip(input.weights.values.iter().copied())
            .map(|((u, v), w)| (u, v, w))
            .collect();
        let want: Vec<(u64, u64, f64)> = edges.into_iter().filter(|e| e.0 != e.1).collect();
        assert_eq!(got, want);
    }

    /// An invalid weight reaches the input as it is, so an invalid-weight fixture stays
    /// invalid.
    #[test]
    fn to_input_keeps_an_invalid_weight() {
        for w in [f64::NAN, -1.0, f64::NEG_INFINITY] {
            let got = to_input(&weights(2, &[(0, 1, w)])).weights.values[0];
            assert_eq!(got.to_bits(), w.to_bits(), "{w}");
        }
    }

    #[test]
    #[should_panic(expected = "has no weight in either direction")]
    fn assert_input_invariants_catches_a_skeleton_pair_with_no_weight() {
        let input = CchInput {
            n: 2,
            skeleton: Csr {
                node_count: 2,
                row_offsets: vec![0, 1, 2],
                neighbors: vec![1, 0],
            },
            weights: WeightCsr {
                csr: Csr {
                    node_count: 2,
                    row_offsets: vec![0, 0, 0],
                    neighbors: vec![],
                },
                values: vec![],
            },
        };
        assert_input_invariants(&input);
    }
}

/// Self-tests of `random_edit_batch`.
#[cfg(test)]
mod maintenance_tests {
    use super::{
        Edit, EditGraph, EditKind, all_pairs, apply_edit, dijkstra, draw_edit, entries,
        random_edit_batch, random_sparse, weights,
    };
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    /// A NaN parallel to a finite edge: `f64::min` would keep the finite one and hide it.
    #[test]
    #[should_panic(expected = "EditGraph: a NaN weight on (0, 1)")]
    fn edit_graph_oracle_refuses_a_nan_weight() {
        let g = EditGraph {
            n: 2,
            pairs: [((0, 1), vec![1.0, f64::NAN])].into(),
        };
        let _ = g.oracle();
    }

    /// A NaN alone on its pair: `f64::min` would read it as `+inf`, a closed edge.
    #[test]
    #[should_panic(expected = "EditGraph: a NaN weight on (0, 1)")]
    fn edit_graph_cheapest_refuses_a_nan_weight() {
        let g = EditGraph {
            n: 2,
            pairs: [((0, 1), vec![f64::NAN])].into(),
        };
        let _ = g.cheapest(0, 1);
    }

    #[test]
    fn edit_batches_are_deterministic_and_replayable() {
        let base = EditGraph::of(&random_sparse(8, 16, 9, 3));
        for seed in 0..100u64 {
            let (after, edits) = random_edit_batch(&base, seed);
            let (again, edits_again) = random_edit_batch(&base, seed);
            assert_eq!(edits, edits_again, "seed {seed}");
            assert_eq!(after, again, "seed {seed}");

            let mut replay = base.clone();
            for e in &edits {
                apply_edit(&mut replay, e);
            }
            assert_eq!(replay, after, "seed {seed}");
            assert!((1..=4).contains(&edits.len()));
        }
    }

    /// Every kind must actually occur over the seeds, and each edit must do
    /// what its tag says to the graph it was drawn on.
    #[test]
    fn every_edit_kind_occurs_and_means_what_it_says() {
        let base = EditGraph::of(&random_sparse(8, 16, 9, 3));
        let (mut up, mut down, mut rem, mut conn, mut unconn, mut node) = (0, 0, 0, 0, 0, 0);
        let mut parallel = 0;
        for seed in 0..400u64 {
            let (_, edits) = random_edit_batch(&base, seed);
            let mut g = base.clone();
            for e in &edits {
                match *e {
                    Edit::SetWeight { u, v, w, increased } => {
                        let old = g.cheapest(u, v).unwrap();
                        assert!(w.is_finite() && w >= 0.0);
                        if increased {
                            assert!(w > old, "seed {seed}");
                            up += 1;
                        } else {
                            assert!(w < old, "seed {seed}");
                            down += 1;
                        }
                    }
                    Edit::RemoveCheapest { u, v } => {
                        assert!(g.cheapest(u, v).is_some());
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
                            dijkstra(&g.oracle(), u)[v as usize].is_finite(),
                            was_reachable,
                            "seed {seed}"
                        );
                        parallel += usize::from(g.cheapest(u, v).is_some());
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
            for row in all_pairs(&g.oracle()) {
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
            ("add-parallel", parallel),
        ] {
            assert!(c > 0, "edit kind {name} never drawn in 400 seeds");
        }
    }

    /// The oracle is each pair's cheapest edge: removing it reveals the next one, and
    /// removing the last removes the pair. A weight edit sets every edge of the pair.
    #[test]
    fn removing_the_cheapest_edge_reveals_the_next() {
        let mut g = EditGraph::of(&weights(2, &[(0, 1, 5.0)]));
        for w in [2.0, 9.0] {
            apply_edit(
                &mut g,
                &Edit::AddEdge {
                    u: 0,
                    v: 1,
                    w,
                    was_reachable: true,
                },
            );
        }
        assert_eq!(g.cheapest(0, 1), Some(2.0));
        let edits = [2.0, 5.0, 9.0].map(|w| (w, Edit::RemoveCheapest { u: 0, v: 1 }));
        for (w, edit) in edits {
            assert_eq!(entries(&g.oracle()), [(0, 1, w)]);
            apply_edit(&mut g, &edit);
        }
        assert!(entries(&g.oracle()).is_empty());
        assert_eq!(g.oracle().nrows(), 2);

        let mut g = EditGraph::of(&weights(2, &[(0, 1, 5.0)]));
        let edits = [
            Edit::AddEdge {
                u: 0,
                v: 1,
                w: 2.0,
                was_reachable: true,
            },
            Edit::SetWeight {
                u: 0,
                v: 1,
                w: 1.0,
                increased: false,
            },
            Edit::RemoveCheapest { u: 0, v: 1 },
        ];
        for edit in &edits {
            apply_edit(&mut g, edit);
        }
        assert_eq!(g.cheapest(0, 1), Some(1.0), "the edit set both edges");
    }

    #[test]
    fn edit_batch_on_an_empty_graph_only_adds_nodes() {
        let (after, edits) = random_edit_batch(&EditGraph::default(), 0);
        assert!(edits.iter().all(|e| *e == Edit::AddNode));
        assert_eq!(after.n, edits.len());
    }

    /// A weight edit is drawn only when it changes the weight the way it says: no increase
    /// on a weight `+ 1..=5` cannot move (2^60) or on a closed edge, no decrease from a
    /// closed edge. `Raise` and `Lower` then fall back to `AddNode`.
    #[test]
    fn weight_edits_that_cannot_change_the_weight_are_not_drawn() {
        let (raise, lower) = (EditKind::Raise, EditKind::Lower);
        for (w, kinds) in [
            (2f64.powi(60), &[raise][..]),
            (f64::INFINITY, &[raise, lower][..]),
        ] {
            let g = EditGraph::of(&weights(2, &[(0, 1, w)]));
            for &kind in kinds {
                for seed in 0..64 {
                    let mut rng = SmallRng::seed_from_u64(seed);
                    assert_eq!(draw_edit(&g, kind, &mut rng), None, "w {w}, {kind:?}");
                }
            }
        }
    }
}
