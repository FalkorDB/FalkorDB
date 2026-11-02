//! Batch-mode conditional traverse operator — single-hop relationship expansion.
//!
//! For each active row in the input batch, extracts the source node and scans
//! matching relationships, producing output rows with relationship and endpoint
//! bindings.
//!
//! ```text
//!  Input batch (1 parent row with n=Node(5)):
//!  ┌──────┐
//!  │ n=5  │  ──expand_row──►  ┌─────────────────────┐
//!  └──────┘                   │ n=5, r=Rel(1,5,7)   │
//!                             │ n=5, r=Rel(2,5,9)   │
//!                             │ ...                  │
//!                             └─────────────────────┘
//! ```
//!
//! ## Execution paths
//!
//! - **Batched F·A path** (`expand_batch`): the structurally-eligible common
//!   case (anonymous edge, no sibling uniqueness, no inline attribute
//!   predicates). Builds a sparse frontier matrix and multiplies it by the
//!   relationship matrix in one `mxm`, gathering the resulting endpoints
//!   columnar. Its output batches are queued on `pending_batches`.
//! - **Per-row fallback** (`expand_row`): everything else (named edge,
//!   bidirectional, sibling-edge uniqueness, attribute filters). Each active row
//!   enumerates its `(from, to, edge)` results eagerly, and the shared
//!   [`BatchedResultEmitter`] packs those across rows into one gathered batch
//!   (replicating the parent columns once per result via `gather` instead of
//!   cloning the parent env per result).

use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::graph::graph::{LabelId, NodeId, RelationshipId};
use crate::graph::graphblas::matrix::{Dup, Matrix};
use crate::graph::graphblas::tensor::Tensor;
use crate::graph::graphblas::versioned_matrix::{Iter, VersionedMatrix};
use crate::parser::ast::{QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp, BatchRow, Column},
    row::RowView,
    runtime::Runtime,
    value::Value,
};
use itertools::Either;
use orx_tree::{Bfs, Dyn, NodeIdx, NodeRef};

use super::batched_result_emitter::{BatchedResultEmitter, EdgeEndpoints, RowIter};

/// Base matrix for the batched mxm path. Relationship matrices store inline
/// edge ids (`u64`) while the adjacency matrix is `bool`; traversal only
/// consumes the sparsity pattern (`ANY_PAIR` semiring), so both traverse
/// identically. Cloning either variant is cheap
/// (`Arc` handle clones, no data copy).
// One instance lives per traversal operator (never in collections), so the
// size gap between the variants costs nothing; boxing would add indirection
// on the hot traversal path instead.
#[allow(clippy::large_enum_variant)]
enum TraversalMatrix {
    Bool(VersionedMatrix<bool>),
    /// A single relationship `Tensor` (shallow clone, shared handles); only
    /// its forward layers are consumed.
    U64(Tensor),
    /// Several relationship types, kept apart: `F * (A1 + A2)` is computed as
    /// `F*A1 + F*A2`, the distribution C FalkorDB's algebraic-expression
    /// optimizer applies. The union `A1 + A2` used to be materialized per
    /// execution — a copy of the whole first matrix sized to node capacity —
    /// even when `F` held one row; the per-type products are only as large as
    /// the frontier. Shallow clones, like `U64`.
    Union(Vec<Tensor>),
    /// Several relationship types walked backward: the transposed pair-level
    /// adjacency of each (`dst -> src`), so the frontier follows incoming
    /// edges. Kept apart like `Union`.
    BoolUnion(Vec<VersionedMatrix<bool>>),
}

impl TraversalMatrix {
    fn ncols(&self) -> u64 {
        match self {
            Self::Bool(m) => m.ncols(),
            Self::U64(t) => t.fwd_m().ncols(),
            // Every relationship tensor is sized to the node capacity, so any
            // one of them has the shared column count.
            Self::Union(ts) => ts.first().map_or(0, |t| t.fwd_m().ncols()),
            Self::BoolUnion(vs) => vs.first().map_or(0, VersionedMatrix::ncols),
        }
    }

    /// The part of each layer set's pending additions that is new, in the order
    /// [`Self::count_lmxm_into`] consumes them (one per type). Built once per
    /// operator, since it costs a pass over the pending additions.
    fn added_layers(&self) -> Vec<Option<Matrix<bool>>> {
        match self {
            // A `bool` matrix keeps `dp` disjoint from `m`, so `dp` is the part
            // that is new.
            Self::Bool(vm) => {
                vm.dp().wait();
                vec![(vm.dp().nvals() > 0).then(|| vm.dp().clone())]
            }
            Self::U64(t) => vec![t.added_layer()],
            Self::Union(ts) => ts.iter().map(Tensor::added_layer).collect(),
            Self::BoolUnion(vs) => vs
                .iter()
                .map(|vm| {
                    vm.dp().wait();
                    (vm.dp().nvals() > 0).then(|| vm.dp().clone())
                })
                .collect(),
        }
    }

    /// `f = f * self`, counting paths (delta-aware). See
    /// [`Matrix::count_lmxm_layers`]. Each pair counts once here; the surplus of
    /// parallel edges is added by the caller from [`parallel_surplus`]. `added`
    /// comes from [`Self::added_layers`].
    fn count_lmxm_into(
        &self,
        f: &mut Matrix<u64>,
        added: &[Option<Matrix<bool>>],
        first_hop: bool,
    ) {
        match self {
            Self::Bool(vm) => f.count_lmxm_layers(vm.m(), added[0].as_ref(), vm.dm(), first_hop),
            Self::U64(t) => Self::count_tensor(f, t, added[0].as_ref(), first_hop),
            Self::BoolUnion(vs) => {
                let Some((first, rest)) = vs.split_first() else {
                    return;
                };
                let frontier = (!rest.is_empty()).then(|| f.dup());
                f.count_lmxm_layers(first.m(), added[0].as_ref(), first.dm(), first_hop);
                if let Some(frontier) = frontier {
                    for (vm, added) in rest.iter().zip(&added[1..]) {
                        let mut ft = frontier.dup();
                        ft.count_lmxm_layers(vm.m(), added.as_ref(), vm.dm(), first_hop);
                        f.add_counts(&ft);
                    }
                }
            }
            Self::Union(ts) => {
                // `f` is the frontier: keep a copy as the left operand of the
                // remaining products, then add them to the first one. A pair
                // joined by several types counts once per type.
                let Some((first, rest)) = ts.split_first() else {
                    return;
                };
                let frontier = (!rest.is_empty()).then(|| f.dup());
                Self::count_tensor(f, first, added[0].as_ref(), first_hop);
                if let Some(frontier) = frontier {
                    for (t, added) in rest.iter().zip(&added[1..]) {
                        let mut ft = frontier.dup();
                        Self::count_tensor(&mut ft, t, added.as_ref(), first_hop);
                        f.add_counts(&ft);
                    }
                }
            }
        }
    }

    /// `f = f * t`, counting paths: against the merged pattern when the tensor
    /// has one, else layer by layer.
    fn count_tensor(
        f: &mut Matrix<u64>,
        t: &Tensor,
        added: Option<&Matrix<bool>>,
        first_hop: bool,
    ) {
        match t.merged_layer() {
            Some(merged) => f.count_lmxm(&merged, first_hop),
            None => f.count_lmxm_layers(t.fwd_m(), added, t.fwd_dm(), first_hop),
        }
    }
}

/// The matrix an untyped hop multiplies by. The adjacency matrix holds a pair
/// once however many edges join it, so it is exact only while the graph has
/// fewer edges than pairs to count; otherwise the per-type tensors are used, so
/// a pair joined by several types or by parallel edges is counted per edge.
fn untyped_traversal_matrix(g: &crate::graph::graph::Graph) -> TraversalMatrix {
    let tensors = g.relationship_tensors();
    if g.relationship_count() <= g.adjacency_matrix().nvals() || tensors.is_empty() {
        TraversalMatrix::Bool(g.adjacency_matrix().clone())
    } else if tensors.len() == 1 {
        TraversalMatrix::U64(tensors[0].clone())
    } else {
        TraversalMatrix::Union(tensors.to_vec())
    }
}

/// For each type a hop matches (all of them when `types` is empty), the pairs
/// with parallel edges and the edges each holds beyond the first. Empty when no
/// pair has any. Kept per type, each a cached matrix shared by every query: a
/// sum would be rebuilt per query, a pass over every parallel pair that a
/// point query must not pay.
fn parallel_surplus(
    g: &crate::graph::graph::Graph,
    types: &[Arc<String>],
    back: bool,
) -> Vec<Matrix<u64>> {
    let tensors: Vec<&Tensor> = if types.is_empty() {
        g.relationship_tensors().iter().collect()
    } else {
        types
            .iter()
            .filter_map(|t| g.get_relationship_matrix(t))
            .collect()
    };
    tensors
        .into_iter()
        .filter_map(|t| {
            if back {
                t.parallel_edge_surplus_t_cached()
            } else {
                t.parallel_edge_surplus_cached()
            }
        })
        .collect()
}

/// Lazily resolved state — built on first `expand_row`, after any sibling
/// Commit in the subtree has had a chance to create new labels/types.
struct CtState {
    fwd_iter: std::cell::RefCell<Iter>,
    rev_iter: Option<std::cell::RefCell<Iter>>,
    /// Iterator over the TRANSPOSED pair matrix (dst-major). Built lazily on
    /// the first expansion where only the matrix-destination is bound — e.g.
    /// a planner-transposed traverse seeded from the pattern-source, or the
    /// reverse half of a bidirectional expansion. Seeking this by destination
    /// replaces what would otherwise be a full edge-matrix scan per input
    /// row (O(V·E) for the whole query).
    bwd_iter: Option<std::cell::RefCell<Iter>>,
    /// Multi-type patterns only: per-type iterators streamed as a union
    /// (`fwd_iter`/`rev_iter`/`bwd_iter` then stay empty and unused). Boxed and
    /// optional so single-type operators — the common case — keep exactly
    /// their previous state and per-row code.
    union: Option<Box<PairIters<UnionIter>>>,
    fwd_src_label_ids: Vec<LabelId>,
    fwd_dst_label_ids: Vec<LabelId>,
    rev_src_label_ids: Vec<LabelId>,
    rev_dst_label_ids: Vec<LabelId>,
    edge_type_indices: Vec<usize>,
    /// True when some requested relationship type holds a (src, dst) pair with
    /// more than one edge. Collapsing to one row per pair then changes the row
    /// count, so `emit_relationship: false` is only honoured when this is false.
    multi_edge: bool,
    /// Edges per pair, for the pairs of the requested types that hold more than
    /// one. Lets a traverse that needs no edge id emit the right number of rows
    /// without reading the tensor per pair. Built on first use, only by a
    /// traverse that needs no edge id and only when `multi_edge`; empty before.
    multi_counts: Arc<FxHashMap<(u64, u64), u32>>,
    multi_counts_built: bool,
    /// Parallel edges beyond the first per pair, for the first hop of the
    /// batched path; `None` when it has none. Built with `batched_matrix`.
    batched_surplus: Vec<Matrix<u64>>,
    /// Whether `batched_matrix` walks the backward adjacency; see
    /// `CondTraverseOp::walks_backward`.
    batched_back: bool,
    /// New pending additions of each type of the first hop, parallel to its
    /// layers; see `TraversalMatrix::added_layers`.
    batched_added: Vec<Option<Matrix<bool>>>,
    /// The same for each fused chain hop.
    chain_added: Vec<Vec<Option<Matrix<bool>>>>,
    /// The same for each fused chain hop, parallel to `chain_matrices`.
    chain_surplus: Vec<Vec<Matrix<u64>>>,
    /// Indices of the relationship types of `chain[0]`, for the self-loop
    /// correction of a two-hop chain; empty `Vec` for all types.
    chain_type_indices: Vec<usize>,
    /// True when one of the requested labels was unknown at state-build
    /// time.  We still drain the child (for side effects) but produce no
    /// output rows, since `unwrap_or_default()` would otherwise turn an
    /// unknown label into "no label restriction".
    no_match: bool,
    /// Materialized base matrix for batched mxm path. Built lazily on
    /// first `expand_batch`. Cached for op lifetime; safe because writes
    /// are serialized w.r.t. read queries.
    batched_matrix: Option<TraversalMatrix>,
    /// Per-hop matrices for fused chain (same lifetime/safety rules as
    /// `batched_matrix`). `chain_matrices[i]` corresponds to `chain[i]`.
    chain_matrices: Vec<TraversalMatrix>,
    /// Per-hop label IDs for `chain[i].to` (post-multiply dst filter).
    chain_dst_label_ids: Vec<Vec<LabelId>>,
}

pub struct CondTraverseOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    /// Holds the parent batch being expanded and the per-row edge results, and
    /// performs the shared pack-and-gather emit for the per-row fallback path.
    /// Each pushed result is a `(from, to, edge)` triple already mapped onto the
    /// pattern's endpoints; the emitter binds them (no transpose) and skips the
    /// second endpoint for self-loop patterns.
    pub(crate) emitter: BatchedResultEmitter<'a, (NodeId, NodeId, RelationshipId)>,
    /// Buffered output batches from the batched F·A path (`expand_batch`).
    pub(crate) pending_batches: VecDeque<Batch<'a>>,
    /// Whether to emit one row per edge (true) or collapse multi-edges into
    /// one row per (src, dst) pair (false). Set by the planner based on
    /// whether the edge is named or referenced in a named path.
    emit_relationship: bool,
    /// False when nothing above this operator reads the edge alias, so the
    /// per-row lookup that resolves a representative edge id is dead work and
    /// the edge column is left unbound. Distinct from `emit_relationship`,
    /// which governs row multiplicity only: an anonymous edge inside a named
    /// path collapses to one row per pair yet still has to be bound, because
    /// `PathBuilder` reads it. Lowered by the `reduce_bound_edge` pass.
    bind_relationship: bool,
    /// Alias IDs of sibling relationship variables in the same MATCH clause.
    sibling_edges: &'a [u32],
    /// When true, from/to have been swapped by the optimizer relative to the
    /// edge direction in the graph. The scan labels and node assignments are
    /// transposed accordingly.
    transposed: bool,
    /// Additional fused hops (anonymous-edge, anonymous-intermediate-node)
    /// to chain after this CT in the F·A path. Empty for single-hop. Each
    /// hop's `from` is the previous hop's `to` (anonymous & unreferenced);
    /// only the final hop's `to` alias is bound on the output env.
    chain: &'a [Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>],
    /// OPTIONAL MATCH semantics (fused from an `Optional` wrapper): input rows
    /// producing no expansion are emitted once with the edge and destination
    /// aliases bound to NULL instead of being dropped.
    optional: bool,
    /// Input-batch row indices that produced no expansion on the per-row path,
    /// buffered until the seeded batch is exhausted, then flushed as one
    /// null-padded fallback batch. Only populated when `optional` is true.
    optional_unmatched: std::cell::RefCell<Vec<usize>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// Maximum number of records this operator should produce. Once reached,
    /// subsequent `next()` calls return `None`. Set by limit propagation.
    record_cap: Option<usize>,
    /// Number of records produced so far (tracked when `record_cap` is set).
    produced: usize,
    /// Persistent forward-matrix iterator. Reused across `expand_row` calls
    /// via `seek` so we pay the GxB_Iterator allocation only once per
    /// operator (not per input row).
    ///
    /// Lazily initialized on first `expand_row`: matrices/labels referenced
    /// by this CT may be created by a sibling Commit earlier in the same
    /// query, so capturing them at construction would miss them.
    state: std::cell::RefCell<Option<CtState>>,
    /// Structural eligibility for the batched F·A path. Computed once at
    /// construction. Variants like emit_relationship, bidir, sibling-edge
    /// uniqueness, and non-empty inline attribute predicates fall back to
    /// the per-row `expand_row` path.
    batched_eligible: bool,
    /// Endpoints of the sibling edges this one must differ from, when that can
    /// be decided by comparing nodes instead of edge ids; see
    /// [`node_comparable_siblings`]. Used only while the relation has no
    /// parallel edges.
    node_unique: Option<Vec<SiblingEnds>>,
}

/// The union of several relationship types' `(src, dst)` pairs, streamed.
///
/// Each layer iterator yields its pairs in ascending `(row, col)` order (see
/// `versioned_matrix::Iter`), so a k-way merge that emits the smallest head
/// and advances every iterator sitting on it yields the sorted union with each
/// pair once — the same pairs, in the same order, as iterating the merged
/// matrix this replaces. That matrix was rebuilt per execution: a copy of the
/// first type's whole matrix plus a pattern merge per further type, ~3.6 MB
/// and ~1 ms on the benchmark graph even when the traversal seeks one row.
/// `k` is the number of types in the pattern, so the linear scan over heads is
/// cheaper than a heap.
struct UnionIter {
    iters: Vec<Iter>,
    heads: Vec<Option<(u64, u64)>>,
}

impl UnionIter {
    /// Heads start empty: every consumer seeks before reading, and priming
    /// a full-range iterator here would walk any deleted prefix of its
    /// matrix on every execution, even for a one-row lookup.
    fn new(iters: Vec<Iter>) -> Self {
        let heads = vec![None; iters.len()];
        Self { iters, heads }
    }

    fn seek(
        &mut self,
        min_row: u64,
        max_row: u64,
    ) {
        for (it, head) in self.iters.iter_mut().zip(&mut self.heads) {
            it.seek(min_row, max_row);
            *head = it.next();
        }
    }
}

impl Iterator for UnionIter {
    type Item = (u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        let min = self.heads.iter().flatten().min().copied()?;
        for (it, head) in self.iters.iter_mut().zip(&mut self.heads) {
            if *head == Some(min) {
                *head = it.next();
            }
        }
        Some(min)
    }
}

/// The per-row iterators of a multi-type pattern.
struct PairIters<I> {
    fwd: std::cell::RefCell<I>,
    rev: Option<std::cell::RefCell<I>>,
    /// Transposed (dst-major) counterpart, built lazily on the first expansion
    /// where only the matrix-destination is bound — see `CtState::bwd_iter`.
    bwd: Option<std::cell::RefCell<I>>,
}

/// The forward tensors of `types` that exist in the schema, or `None` when
/// none of them do (the caller short-circuits to an empty result). Shallow
/// clones: shared handles, no data copied.
fn relationship_tensors(
    g: &crate::graph::graph::Graph,
    types: &[Arc<String>],
) -> Option<Vec<Tensor>> {
    let tensors: Vec<Tensor> = types
        .iter()
        .filter_map(|t| g.get_relationship_matrix(t))
        .cloned()
        .collect();
    (!tensors.is_empty()).then_some(tensors)
}

/// Build an `Iter` over the pair matrix for an untyped or single-type
/// pattern; `None` when the type does not exist.
fn build_unrestricted_iter(
    g: &crate::graph::graph::Graph,
    types: &[Arc<String>],
) -> Option<Iter> {
    debug_assert!(types.len() <= 1, "several types stream through UnionIter");
    match types.first() {
        None => Some(g.adjacency_matrix().iter(0, u64::MAX)),
        Some(t) => g
            .get_relationship_matrix(t)
            .map(|t| t.structural_iter(0, u64::MAX)),
    }
}

/// Stream the union of several types' pair matrices without materializing
/// it; `None` when none of the types exist.
fn build_union_iter(
    g: &crate::graph::graph::Graph,
    types: &[Arc<String>],
) -> Option<UnionIter> {
    let iters = relationship_tensors(g, types)?
        .iter()
        .map(|t| t.structural_iter(0, u64::MAX))
        .collect();
    Some(UnionIter::new(iters))
}

fn empty_edge_iter() -> Iter {
    VersionedMatrix::<bool>::new(0, 0).iter(0, u64::MAX)
}

/// Build an `Iter` over the TRANSPOSED (dst → src) pair matrix, for
/// expansions where only the destination is bound. Tensors maintain their
/// transpose (`mt`) incrementally, so the single-type case is free; the
/// untyped case pays one O(E) GraphBLAS transpose — amortized over the whole
/// operator, unlike the per-row full scan this replaces.
fn build_transposed_iter(
    g: &crate::graph::graph::Graph,
    types: &[Arc<String>],
) -> Option<Iter> {
    debug_assert!(types.len() <= 1, "several types stream through UnionIter");
    match types.first() {
        None => Some(g.adjacency_matrix().transpose().iter(0, u64::MAX)),
        Some(t) => g
            .get_relationship_matrix(t)
            .map(|t| t.matrix_t().iter(0, u64::MAX)),
    }
}

/// The transposed counterpart of `build_union_iter`: every type's
/// incrementally maintained transpose, merged the same way.
fn build_transposed_union_iter(
    g: &crate::graph::graph::Graph,
    types: &[Arc<String>],
) -> Option<UnionIter> {
    let iters = relationship_tensors(g, types)?
        .iter()
        .map(|t| t.matrix_t().iter(0, u64::MAX))
        .collect();
    Some(UnionIter::new(iters))
}

/// Endpoint aliases of a sibling relationship, and whether its operator runs
/// against the edge direction.
struct SiblingEnds {
    from: u32,
    to: u32,
    transposed: bool,
}

/// The sibling edges of `rp`, when uniqueness among them can be checked by
/// comparing the nodes they join rather than the edge ids they bind.
///
/// That holds when the relation has no parallel edges and every edge of the
/// component (this one included) is the same single type, directed, a plain
/// unread traverse: one pair of nodes is then one edge. The decision is made
/// from the whole component, so each of its operators takes the same branch
/// and none is left comparing ids against an edge another left unbound.
fn node_comparable_siblings(
    runtime: &Runtime,
    rp: &QueryRelationship<Arc<String>, Arc<String>, Variable>,
    sibling_edges: &[u32],
) -> Option<Vec<SiblingEnds>> {
    if sibling_edges.len() < 2 || rp.types.len() != 1 {
        return None;
    }
    let mut ends = Vec::new();
    let mut found = 0;
    for idx in runtime.plan.root().indices::<Bfs>() {
        let (relationship, emit, chain, optional, transposed) = match runtime.plan.node(idx).data()
        {
            IR::CondTraverse {
                relationship,
                emit_relationship,
                bind_relationship,
                chain,
                optional,
                transposed,
                ..
            } => (
                relationship,
                *emit_relationship || *bind_relationship,
                chain,
                *optional,
                *transposed,
            ),
            IR::ExpandInto { relationship, .. }
                if sibling_edges.contains(&relationship.alias.id)
                    && relationship.alias.scope_id == rp.alias.scope_id =>
            {
                return None;
            }
            _ => continue,
        };
        if !sibling_edges.contains(&relationship.alias.id)
            || relationship.alias.scope_id != rp.alias.scope_id
        {
            continue;
        }
        if emit
            || optional
            || !chain.is_empty()
            || relationship.bidirectional
            || relationship.min_hops.is_some()
            || relationship.types != rp.types
        {
            return None;
        }
        found += 1;
        if relationship.alias.id != rp.alias.id {
            ends.push(SiblingEnds {
                from: relationship.from.alias.id,
                to: relationship.to.alias.id,
                transposed,
            });
        }
    }
    // Every sibling must be one of these operators, or it is checked by ids.
    (found == sibling_edges.len()).then_some(ends)
}

/// The node bound to `alias` on this row, if it is bound to one.
fn bound_node(
    batch: &Batch<'_>,
    alias: u32,
    row_idx: usize,
) -> Option<NodeId> {
    if !batch.is_bound_at(alias, row_idx) {
        return None;
    }
    match batch.value_at(alias, row_idx) {
        Some(Value::Node(n)) => Some(n),
        _ => None,
    }
}

/// The `(source, destination)` of every already-bound sibling edge on this row,
/// in edge direction.
fn sibling_edge_pairs<'s>(
    siblings: &'s [SiblingEnds],
    batch: &'s Batch<'_>,
    row_idx: usize,
) -> impl Iterator<Item = (NodeId, NodeId)> + 's {
    siblings.iter().filter_map(move |s| {
        let f = bound_node(batch, s.from, row_idx)?;
        let t = bound_node(batch, s.to, row_idx)?;
        Some(if s.transposed { (t, f) } else { (f, t) })
    })
}

impl<'a> CondTraverseOp<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        emit_relationship: bool,
        sibling_edges: &'a [u32],
        transposed: bool,
        chain: &'a [Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>],
        optional: bool,
        bind_relationship: bool,
        idx: NodeIdx<Dyn<IR>>,
        record_cap: Option<usize>,
    ) -> Self {
        let rp = relationship_pattern;

        // For fused chains the batched path is the ONLY correct path —
        // expand_row only handles single-hop.
        //
        // The endpoints' inline attrs no longer gate this. The planner lowers
        // both endpoints of every traverse into a Filter unconditionally, so
        // the predicate is enforced whichever path runs, and expand_row's own
        // check on them is redundant.
        //
        // An edge predicate is excluded by `!emit_relationship`: the planner
        // sets it for an edge that carries one, because expand_batch binds one
        // representative edge per (src, dst) pair and the Filter above has to
        // see every parallel edge to test them apart.
        let node_unique = (!emit_relationship
            && !bind_relationship
            && !optional
            && chain.is_empty()
            && !rp.bidirectional)
            .then(|| node_comparable_siblings(runtime, rp, sibling_edges))
            .flatten();
        let batched_eligible = !emit_relationship
            && !rp.bidirectional
            && (sibling_edges.is_empty()
                || node_unique.is_some()
                || (chain.len() == 1
                    && sibling_edges
                        .iter()
                        .all(|&id| id == rp.alias.id || id == chain[0].alias.id)))
            && chain.iter().all(|hop| !hop.bidirectional);

        // Self-loop patterns like `MATCH (n)-[r:T]->(n)` share one alias on both
        // endpoints; bind it once (via `from`) and skip the `to` column so the
        // second insert can't overwrite the first. The per-row path resolves
        // `from`/`to` to the pattern endpoints itself, so the emitter binds them
        // untransposed.
        let to = (relationship_pattern.to.alias != relationship_pattern.from.alias)
            .then_some(relationship_pattern.to.alias.id);
        let emitter = BatchedResultEmitter::with_binding(
            EdgeEndpoints {
                from: relationship_pattern.from.alias.id,
                to,
                edge: relationship_pattern.alias.id,
                transposed: false,
            },
            record_cap,
        );

        Self {
            runtime,
            child,
            relationship_pattern,
            emitter,
            pending_batches: VecDeque::new(),
            emit_relationship,
            bind_relationship,
            sibling_edges,
            transposed,
            chain,
            optional,
            optional_unmatched: std::cell::RefCell::new(Vec::new()),
            idx,
            record_cap,
            produced: 0,
            state: std::cell::RefCell::new(None),
            batched_eligible,
            node_unique,
        }
    }

    /// Build the lazy state from current graph contents.  Called on the
    /// first `expand_row`/`expand_batch` invocation, after any sibling Commit in
    /// the subtree has executed.
    fn build_state(
        runtime: &Runtime,
        rp: &QueryRelationship<Arc<String>, Arc<String>, Variable>,
        transposed: bool,
    ) -> CtState {
        let g = runtime.graph();

        let (fwd_src_labels, fwd_dst_labels) = if transposed {
            (&rp.to.labels, &rp.from.labels)
        } else {
            (&rp.from.labels, &rp.to.labels)
        };

        let fwd_src_label_ids = g.resolve_label_ids(fwd_src_labels);
        let fwd_dst_label_ids = g.resolve_label_ids(fwd_dst_labels);
        let (rev_src_label_ids, rev_dst_label_ids) = if rp.bidirectional {
            let (rev_src_labels, rev_dst_labels) = if transposed {
                (&rp.from.labels, &rp.to.labels)
            } else {
                (&rp.to.labels, &rp.from.labels)
            };
            (
                g.resolve_label_ids(rev_src_labels),
                g.resolve_label_ids(rev_dst_labels),
            )
        } else {
            (Some(Vec::new()), Some(Vec::new()))
        };

        // Several types stream their union through `union`; the plain
        // iterators are then left empty. A missing type makes the builder
        // return `None`, and so `no_match` below.
        let multi_type = rp.types.len() > 1;
        let (fwd_iter_opt, rev_iter_opt, union) = if multi_type {
            let fwd = build_union_iter(&g, &rp.types);
            let rev = rp.bidirectional.then(|| build_union_iter(&g, &rp.types));
            let fwd_ok = fwd.is_some().then(empty_edge_iter);
            let rev_ok = rev
                .as_ref()
                .and_then(|r| r.as_ref().map(|_| empty_edge_iter()));
            let empty = || UnionIter::new(Vec::new());
            let union = PairIters {
                fwd: std::cell::RefCell::new(fwd.unwrap_or_else(empty)),
                rev: rev.map(|r| std::cell::RefCell::new(r.unwrap_or_else(empty))),
                bwd: None,
            };
            (fwd_ok, rev_ok, Some(Box::new(union)))
        } else {
            let fwd = build_unrestricted_iter(&g, &rp.types);
            let rev = if rp.bidirectional {
                build_unrestricted_iter(&g, &rp.types)
            } else {
                None
            };
            (fwd, rev, None)
        };

        let no_match = fwd_src_label_ids.is_none()
            || fwd_dst_label_ids.is_none()
            || rev_src_label_ids.is_none()
            || rev_dst_label_ids.is_none()
            || fwd_iter_opt.is_none()
            || (rp.bidirectional && rev_iter_opt.is_none());

        let fwd_src_label_ids = fwd_src_label_ids.unwrap_or_default();
        let fwd_dst_label_ids = fwd_dst_label_ids.unwrap_or_default();
        let rev_src_label_ids = rev_src_label_ids.unwrap_or_default();
        let rev_dst_label_ids = rev_dst_label_ids.unwrap_or_default();

        let fwd_iter = fwd_iter_opt.unwrap_or_else(empty_edge_iter);
        // Multi-type: the union carries the reverse half, so `rev_iter` stays
        // `None` and the single-type reverse block never runs.
        let rev_iter = if rp.bidirectional && !multi_type {
            Some(rev_iter_opt.unwrap_or_else(empty_edge_iter))
        } else {
            None
        };

        let edge_type_indices: Vec<usize> = if rp.types.is_empty() {
            (0..g.relationship_tensors().len()).collect()
        } else {
            rp.types
                .iter()
                .filter_map(|t| g.get_type_id(t).map(|tid| tid.0))
                .collect()
        };

        let multi_edge = edge_type_indices
            .iter()
            .any(|&tidx| g.relationship_tensors()[tidx].has_multi_edge());

        CtState {
            fwd_iter: std::cell::RefCell::new(fwd_iter),
            rev_iter: rev_iter.map(std::cell::RefCell::new),
            bwd_iter: None,
            union,
            fwd_src_label_ids,
            fwd_dst_label_ids,
            rev_src_label_ids,
            rev_dst_label_ids,
            edge_type_indices,
            multi_edge,
            multi_counts: Arc::new(FxHashMap::default()),
            multi_counts_built: false,
            batched_surplus: Vec::new(),
            batched_back: false,
            batched_added: Vec::new(),
            chain_added: Vec::new(),
            chain_surplus: Vec::new(),
            chain_type_indices: Vec::new(),
            no_match,
            batched_matrix: None,
            chain_matrices: Vec::new(),
            chain_dst_label_ids: Vec::new(),
        }
    }

    /// How many self-loops at `v` both hops of a two-hop chain can bind: edges
    /// at `(v, v)` of a type matched by the entry hop and by the chain hop. The
    /// path `v -> v -> v` over such an edge uses it twice, which relationship
    /// uniqueness forbids and the path count includes. Any other repeat of an
    /// edge needs a chain of three hops or more, or a hop that is not directed.
    fn shared_self_loops(
        g: &crate::graph::graph::Graph,
        state: &CtState,
        v: NodeId,
    ) -> u64 {
        let v = u64::from(v);
        let tensors = g.relationship_tensors();
        // An edge has exactly one type, so the edges both hops can bind are the
        // self-loops of the types they share.
        state
            .edge_type_indices
            .iter()
            .filter(|t| state.chain_type_indices.contains(t))
            .map(|&t| tensors[t].get(v, v).count() as u64)
            .sum()
    }

    /// Whether the batched path should walk the backward adjacency for this
    /// input: a traverse the planner transposed (`rel.from` is the node it
    /// starts from, the edge is stored `to -> from`) whose start is bound and
    /// whose end is not. The frontier then follows incoming edges, which is
    /// `F * A^T` over the maintained transpose. Anything else keeps the
    /// forward product (and, for a transposed traverse, the per-row path when
    /// its start is not the pattern's `to`).
    fn walks_backward(
        &self,
        batch: &Batch<'a>,
        active_subset: &[usize],
    ) -> bool {
        let rp = self.relationship_pattern;
        let Some(&row) = active_subset.first() else {
            return false;
        };
        self.transposed
            && self.chain.is_empty()
            && !self.optional
            && rp.from.alias != rp.to.alias
            && batch.is_bound_at(rp.from.alias.id, row)
            && !batch.is_bound_at(rp.to.alias.id, row)
    }

    /// Batched F·A traversal — mirrors C FalkorDB's `_traverse` in
    /// `op_conditional_traverse.c`. For an input slice of envs:
    /// 1. build sparse F[i, src_id] = true (one bulk FFI call),
    /// 2. compute M = F * A in one mxm,
    /// 3. extract tuples (row_i, dest_id) and emit one row per pair.
    ///
    /// Eligibility was checked structurally at op construction
    /// (`self.batched_eligible`); callers must not invoke this for ineligible
    /// ops. State-level eligibility (matrix existence) is checked here and
    /// returns `false` to signal "fall back to slow path for this batch".
    fn expand_batch(
        &self,
        batch: &Batch<'a>,
        active_subset: &[usize],
        out_pending: &mut VecDeque<Batch<'a>>,
    ) -> bool {
        // Under a small record cap (`LIMIT 10`) the product over a whole input
        // batch is mostly wasted: run it over a few rows at a time and stop once
        // enough rows are out. Only when every row has its source bound to a
        // node, so no slice can fall back to the per-row path after another
        // has emitted.
        const SLICE: usize = 64;
        let rp = self.relationship_pattern;
        let seed_alias = if self.walks_backward(batch, active_subset) {
            &rp.from.alias
        } else if self.transposed {
            &rp.to.alias
        } else {
            &rp.from.alias
        };
        let cap_left = self.record_cap.map(|c| c.saturating_sub(self.produced));
        if let Some(cap) = cap_left
            && cap <= 256
            && !self.optional
            && active_subset.len() > SLICE
            && active_subset
                .iter()
                .all(|&r| bound_node(batch, seed_alias.id, r).is_some())
        {
            let mut emitted = 0usize;
            for rows in active_subset.chunks(SLICE) {
                let before = out_pending.len();
                if !self.expand_batch_rows(batch, rows, out_pending) {
                    return false;
                }
                emitted += out_pending
                    .iter()
                    .skip(before)
                    .map(Batch::active_len)
                    .sum::<usize>();
                if emitted >= cap {
                    break;
                }
            }
            return true;
        }
        self.expand_batch_rows(batch, active_subset, out_pending)
    }

    /// [`Self::expand_batch`] over `active_subset` as one product.
    fn expand_batch_rows(
        &self,
        batch: &Batch<'a>,
        active_subset: &[usize],
        out_pending: &mut VecDeque<Batch<'a>>,
    ) -> bool {
        let runtime = self.runtime;
        let rp = self.relationship_pattern;

        let mut state_ref = self.state.borrow_mut();
        if state_ref.is_none() {
            drop(state_ref);
            let new_state =
                Self::build_state(self.runtime, self.relationship_pattern, self.transposed);
            *self.state.borrow_mut() = Some(new_state);
            state_ref = self.state.borrow_mut();
        }
        let state = state_ref.as_mut().unwrap();
        if state.no_match {
            if self.optional {
                out_pending.push_back(self.null_pad(batch, active_subset));
            }
            return true;
        }

        let node_mode = self.node_unique.as_deref();
        let back = self.walks_backward(batch, active_subset);
        if state.batched_matrix.is_some() && state.batched_back != back {
            // The matrix was built for the other direction.
            return false;
        }

        let g = runtime.graph();

        if state.batched_matrix.is_none() {
            state.batched_back = back;
            let m = if back {
                let tensors: Vec<&Tensor> = if rp.types.is_empty() {
                    g.relationship_tensors().iter().collect()
                } else {
                    let found: Vec<&Tensor> = rp
                        .types
                        .iter()
                        .filter_map(|t| g.get_relationship_matrix(t))
                        .collect();
                    found
                };
                // Unknown types match nothing, as in the forward union; the
                // traverse matches nothing only when every type is unknown.
                if tensors.is_empty() {
                    state.no_match = true;
                    return true;
                }
                TraversalMatrix::BoolUnion(tensors.iter().map(|t| t.matrix_t().clone()).collect())
            } else if rp.types.is_empty() {
                untyped_traversal_matrix(&g)
            } else if rp.types.len() == 1 {
                if let Some(t) = g.get_relationship_matrix(&rp.types[0]) {
                    TraversalMatrix::U64(t.clone())
                } else {
                    state.no_match = true;
                    if self.optional {
                        out_pending.push_back(self.null_pad(batch, active_subset));
                    }
                    return true;
                }
            } else {
                if let Some(ts) = relationship_tensors(&g, &rp.types) {
                    TraversalMatrix::Union(ts)
                } else {
                    state.no_match = true;
                    if self.optional {
                        out_pending.push_back(self.null_pad(batch, active_subset));
                    }
                    return true;
                }
            };
            state.batched_added = m.added_layers();
            state.batched_matrix = Some(m);
            state.batched_surplus = parallel_surplus(&g, &rp.types, back);

            // Build per-hop matrices and label-id vectors for the fused chain.
            // If any hop's relationship type is unknown, mark no_match — that
            // hop can produce no rows, so the overall fused traversal can't
            // either.
            for hop in self.chain {
                let hm = if hop.types.is_empty() {
                    untyped_traversal_matrix(&g)
                } else if hop.types.len() == 1 {
                    if let Some(t) = g.get_relationship_matrix(&hop.types[0]) {
                        TraversalMatrix::U64(t.clone())
                    } else {
                        state.no_match = true;
                        if self.optional {
                            out_pending.push_back(self.null_pad(batch, active_subset));
                        }
                        return true;
                    }
                } else {
                    if let Some(ts) = relationship_tensors(&g, &hop.types) {
                        TraversalMatrix::Union(ts)
                    } else {
                        state.no_match = true;
                        if self.optional {
                            out_pending.push_back(self.null_pad(batch, active_subset));
                        }
                        return true;
                    }
                };
                state.chain_added.push(hm.added_layers());
                state.chain_matrices.push(hm);
                state
                    .chain_surplus
                    .push(parallel_surplus(&g, &hop.types, false));
                let Some(dst_labels) = g.resolve_label_ids(&hop.to.labels) else {
                    state.no_match = true;
                    if self.optional {
                        out_pending.push_back(self.null_pad(batch, active_subset));
                    }
                    return true;
                };
                state.chain_dst_label_ids.push(dst_labels);
            }
        }
        if state.chain_type_indices.is_empty()
            && let Some(hop) = self.chain.first()
        {
            state.chain_type_indices = if hop.types.is_empty() {
                (0..g.relationship_tensors().len()).collect()
            } else {
                hop.types
                    .iter()
                    .filter_map(|t| g.get_type_id(t).map(|tid| tid.0))
                    .collect()
            };
        }
        let m_merged = state.batched_matrix.as_ref().unwrap();
        let ncols = m_merged.ncols();

        let transposed = self.transposed;
        let nrows = active_subset.len() as u64;
        // The matrix-source and matrix-destination ends of the pattern. A
        // backward walk starts at `rel.from` (the matrix destination) and ends
        // at `rel.to`; every other batched traversal starts at the matrix
        // source.
        let (seed_labels, end_labels) = if back {
            (&state.fwd_dst_label_ids, &state.fwd_src_label_ids)
        } else {
            (&state.fwd_src_label_ids, &state.fwd_dst_label_ids)
        };

        // Collect (row_i, src_id) for rows where the from-alias is bound to
        // a Node and (post-label-filter) the src has all required labels.
        // Rows where from is bound to a non-Node are dropped (mirror line
        // 311 early return). Rows where from is unbound trigger fall-back.
        let mut row_idx_buf: Vec<u64> = Vec::with_capacity(active_subset.len());
        let mut col_idx_buf: Vec<u64> = Vec::with_capacity(active_subset.len());
        let from_alias = if transposed && !back {
            &rp.to.alias
        } else {
            &rp.from.alias
        };
        for (i, &row_idx) in active_subset.iter().enumerate() {
            // Bail to slow path if the matrix-src side isn't explicitly
            // bound on this row. `env.get` alone would silently return a
            // colliding outer-scope slot value; `is_bound` is authoritative.
            if !batch.is_bound_at(from_alias.id, row_idx) {
                drop(g);
                return false;
            }
            let Some(Value::Node(src_id)) = batch.value_at(from_alias.id, row_idx) else {
                // Optional traverses null-pad rows whose source is bound to a
                // non-Node (e.g. NULL from a preceding OPTIONAL MATCH) instead
                // of bailing to the per-row path.
                if self.optional {
                    continue;
                }
                drop(g);
                return false;
            };
            // Pre-filter src by label (= L_src * F in C's algebra).
            if !seed_labels
                .iter()
                .all(|&lid| g.node_has_label_id(src_id, lid))
            {
                continue;
            }
            row_idx_buf.push(i as u64);
            col_idx_buf.push(u64::from(src_id));
        }

        if row_idx_buf.is_empty() {
            drop(g);
            if self.optional {
                // No source expanded, so every active row is unmatched.
                out_pending.push_back(self.null_pad(batch, active_subset));
            }
            return true;
        }

        // Path counts, not reachability: `f[i, v]` is the number of paths from
        // row `i`'s source to `v`. A pair stores its edge once, so the product
        // is short by the surplus of any pair with parallel edges, which a
        // second product over `PLUS_TIMES` adds back.
        let ones = vec![1u64; row_idx_buf.len()];
        let mut f = Matrix::<u64>::new(nrows, ncols);
        f.build(&row_idx_buf, &col_idx_buf, &ones);
        let first = (m_merged, &state.batched_surplus, &state.batched_added);
        let rest = state
            .chain_matrices
            .iter()
            .zip(&state.chain_surplus)
            .zip(&state.chain_added)
            .map(|((m, s), a)| (m, s, a));
        for (hop, (hop_m, surplus, added)) in std::iter::once(first).chain(rest).enumerate() {
            let prev = (!surplus.is_empty()).then(|| f.dup());
            hop_m.count_lmxm_into(&mut f, added, hop == 0);
            if let Some(prev) = prev {
                for surplus in surplus {
                    f.add_counted_product(&prev, surplus);
                }
            }
        }

        // Flush pending mxm work before attaching the row iterator.
        f.wait();
        // For fused chains all hops are storage-direction (the fusion pass
        // refuses to fuse when transposed differs), so `transposed` applies
        // only to the first hop and the final destination alias comes from
        // the last hop in `self.chain`.
        let from_alias = if transposed && !back {
            &rp.to.alias
        } else {
            &rp.from.alias
        };
        let (to_alias, dst_label_ids) = if let Some(last_hop) = self.chain.last() {
            (
                &last_hop.to.alias,
                state.chain_dst_label_ids.last().unwrap().as_slice(),
            )
        } else if back {
            (&rp.to.alias, end_labels.as_slice())
        } else if transposed {
            (&rp.from.alias, end_labels.as_slice())
        } else {
            (&rp.to.alias, end_labels.as_slice())
        };
        let chain_is_empty = self.chain.is_empty();
        // A chain-less traverse binds the edge column, and getting a value for
        // it costs a tensor lookup per surviving row. Skip both when nothing
        // above reads the alias.
        let bind_edge = chain_is_empty && self.bind_relationship;
        let mut out_indices = Vec::new();
        let mut out_dest_ids = Vec::new();
        let mut out_edge_ids = Vec::new();
        // Per-active-row match flags (indexed by active_subset position),
        // used to build the null-padded fallback batch for optional traverses.
        let mut matched = if self.optional {
            vec![false; active_subset.len()]
        } else {
            Vec::new()
        };

        // A two-hop chain over overlapping types is fused only when a hop pair
        // can reuse an edge solely at a self-loop; see `shared_self_loops`.
        let chain_overlap = self.chain.len() == 1
            && (rp.types.is_empty()
                || self.chain[0].types.is_empty()
                || rp.types.iter().any(|t| self.chain[0].types.contains(t)));
        let mut sibling_row = u64::MAX;
        let mut sibling_seed: Option<NodeId> = None;
        let mut sibling_pairs: Vec<(NodeId, NodeId)> = Vec::new();
        // Rows the record cap still allows, so an entry that fans out into a
        // huge multiplicity is not materialized past it.
        let cap_left = self.record_cap.map(|c| c.saturating_sub(self.produced));
        let mut emitted = 0usize;
        for (row_i, dest_raw, count) in f.iter(0, u64::MAX) {
            if cap_left.is_some_and(|left| emitted + out_indices.len() >= left) {
                break;
            }
            let dest_id = NodeId::from(dest_raw);
            // Post-filter final-hop dst label (= F * A * R_dst in C's algebra).
            if !dst_label_ids
                .iter()
                .all(|&lid| g.node_has_label_id(dest_id, lid))
            {
                continue;
            }
            let row_idx = active_subset[row_i as usize];
            // If the to-alias is already bound on the input batch row, the
            // planner should have inserted ExpandInto, not CondTraverse —
            // but be defensive and skip mismatches.
            if let Some(Value::Node(bound)) = batch.value_at(to_alias.id, row_idx)
                && bound != dest_id
            {
                continue;
            }

            // Edges joining the same pair of nodes: a sibling bound to one of
            // them leaves one fewer for this edge, and a pair of single edges
            // is none. The siblings' nodes are read once per input row.
            let mut shared = 0u64;
            if let Some(siblings) = node_mode {
                if sibling_row != row_i {
                    sibling_row = row_i;
                    sibling_seed = bound_node(batch, from_alias.id, row_idx);
                    sibling_pairs.clear();
                    sibling_pairs.extend(sibling_edge_pairs(siblings, batch, row_idx));
                }
                if let Some(seed) = sibling_seed {
                    shared = sibling_pairs
                        .iter()
                        .filter(|&&pair| {
                            pair == if back {
                                (dest_id, seed)
                            } else {
                                (seed, dest_id)
                            }
                        })
                        .count() as u64;
                }
            }

            // `count` is the number of paths from this row's source to `dest_id`;
            // each is one output row. A chain-less hop knows which edge each
            // row is; a fused chain binds none.
            let mut count = count.saturating_sub(shared);
            if chain_overlap
                && let Some(Value::Node(src_id)) = batch.value_at(from_alias.id, row_idx)
                && src_id == dest_id
            {
                // Both hops can use the same self-loop at this node; one edge
                // may be matched only once per pattern.
                count -= Self::shared_self_loops(&g, state, src_id);
            }
            if count == 0 {
                continue;
            }

            if bind_edge {
                // Needed whenever the alias is read downstream — including by
                // PathBuilder, which reads it even when emit_relationship is
                // false, so `emit_relationship` alone cannot decide this.
                // Storage matrix orientation: src=F's seed (matrix-src),
                // dst=F*A result (matrix-dst), regardless of self.transposed
                // (which only affects alias→storage mapping, not the
                // underlying matrix orientation: every `TraversalMatrix`
                // variant multiplies by forward layers).
                let Some(Value::Node(src_id)) = batch.value_at(from_alias.id, row_idx) else {
                    continue;
                };
                let (mat_src, mat_dst) = if back {
                    (u64::from(dest_id), u64::from(src_id))
                } else {
                    (u64::from(src_id), u64::from(dest_id))
                };
                let before = out_indices.len();
                for &tidx in &state.edge_type_indices {
                    for raw_id in g.relationship_tensors()[tidx].get(mat_src, mat_dst) {
                        // The cap may fall inside one pair's parallel edges.
                        if cap_left.is_some_and(|left| emitted + out_indices.len() >= left) {
                            break;
                        }
                        out_indices.push(row_idx);
                        out_dest_ids.push(dest_id);
                        out_edge_ids.push(RelationshipId::from(raw_id));
                        // One path, one edge: stop at the representative.
                        if count == 1 {
                            break;
                        }
                    }
                    if count == 1 && out_indices.len() > before {
                        break;
                    }
                }
                if self.optional && out_indices.len() > before {
                    matched[row_i as usize] = true;
                }
            } else {
                // No edge column to fill: either a fused chain, whose per-hop
                // edges are anonymous & unreferenced by construction (the
                // fusion pass enforces it) and whose intermediate node aliases
                // are not exposed, or a single hop whose alias nothing reads.
                // `f` is built from these same tensors, so every pair it
                // yields has at least one edge in one of them.
                let mut left = count as usize;
                if let Some(cap) = cap_left {
                    left = left.min(cap.saturating_sub(emitted + out_indices.len()));
                }
                while left > 0 {
                    // A batch holds at most `BATCH_SIZE` rows: its selection
                    // vector indexes them with a `u16`.
                    let n = left.min(BATCH_SIZE - out_indices.len());
                    out_indices.extend(std::iter::repeat_n(row_idx, n));
                    out_dest_ids.extend(std::iter::repeat_n(dest_id, n));
                    left -= n;
                    if out_indices.len() >= BATCH_SIZE {
                        let out_batch = Self::take_batch(
                            batch,
                            &mut out_indices,
                            &mut out_dest_ids,
                            None,
                            to_alias.id,
                        );
                        emitted += out_batch.active_len();
                        out_pending.push_back(out_batch);
                    }
                }
                if self.optional {
                    matched[row_i as usize] = true;
                }
            }

            while out_indices.len() >= BATCH_SIZE {
                let out_batch = Self::take_batch(
                    batch,
                    &mut out_indices,
                    &mut out_dest_ids,
                    bind_edge.then_some((rp.alias.id, &mut out_edge_ids)),
                    to_alias.id,
                );
                emitted += out_batch.active_len();
                out_pending.push_back(out_batch);
            }
        }

        if !out_indices.is_empty() {
            let mut out_batch = batch.gather(&out_indices);
            if bind_edge {
                out_batch.set_column(rp.alias.id, Column::RelIds(out_edge_ids));
            }
            out_batch.set_column(to_alias.id, Column::NodeIds(out_dest_ids));
            out_pending.push_back(out_batch);
        }

        if self.optional {
            let unmatched: Vec<usize> = matched
                .iter()
                .enumerate()
                .filter(|&(_, &m)| !m)
                .map(|(i, _)| active_subset[i])
                .collect();
            if !unmatched.is_empty() {
                out_pending.push_back(self.null_pad(batch, &unmatched));
            }
        }

        drop(g);
        true
    }

    /// Moves the first `BATCH_SIZE` rows buffered by `expand_batch` out as a
    /// batch: the input rows gathered at `indices`, with the destination (and
    /// the edge, when bound) columns.
    fn take_batch(
        batch: &Batch<'a>,
        indices: &mut Vec<usize>,
        dest_ids: &mut Vec<NodeId>,
        edge_ids: Option<(u32, &mut Vec<RelationshipId>)>,
        to_alias: u32,
    ) -> Batch<'a> {
        let n = indices.len().min(BATCH_SIZE);
        let mut out = batch.gather(&indices[..n]);
        indices.drain(..n);
        if let Some((alias, ids)) = edge_ids {
            out.set_column(alias, Column::RelIds(ids.drain(..n).collect()));
        }
        out.set_column(to_alias, Column::NodeIds(dest_ids.drain(..n).collect()));
        out
    }

    /// Enumerate the single-hop results for `batch[row_idx]`, pushing each as a
    /// `(from, to, edge)` triple (already mapped onto the pattern's endpoints)
    /// into `out`. Callers pass the op's fields explicitly so this can run inside
    /// the emitter closure without borrowing the emitter through `&self`.
    #[allow(clippy::too_many_arguments)]
    fn expand_row(
        runtime: &Runtime,
        rp: &QueryRelationship<Arc<String>, Arc<String>, Variable>,
        emit_relationship: bool,
        sibling_edges: &[u32],
        need_edge_id: bool,
        node_unique: Option<&[SiblingEnds]>,
        transposed: bool,
        state_cell: &std::cell::RefCell<Option<CtState>>,
        batch: &Batch<'a>,
        row_idx: usize,
        out: &mut Vec<(NodeId, NodeId, RelationshipId)>,
    ) -> Result<(), String> {
        let env = BatchRow::new(batch, row_idx);

        let from_id = env.value_at(rp.from.alias.id).and_then(|v| match v {
            Value::Node(id) => Some(id),
            _ => None,
        });
        if from_id.is_none() && batch.is_bound_at(rp.from.alias.id, row_idx) {
            return Ok(());
        }
        let to_id = env.value_at(rp.to.alias.id).and_then(|v| match v {
            Value::Node(id) => Some(id),
            _ => None,
        });
        if to_id.is_none() && batch.is_bound_at(rp.to.alias.id, row_idx) {
            return Ok(());
        }

        let g = runtime.graph();

        // Lazily initialize state from current graph contents on first use,
        // ensuring sibling Commits in the subtree have already had a chance
        // to add new labels/relationship types.
        let mut state_ref = state_cell.borrow_mut();
        if state_ref.is_none() {
            drop(state_ref);
            drop(g);
            let new_state = Self::build_state(runtime, rp, transposed);
            *state_cell.borrow_mut() = Some(new_state);
            state_ref = state_cell.borrow_mut();
            // Re-borrow after building state.
            // (g re-borrowed below)
        }
        let state = state_ref.as_mut().unwrap();
        if state.no_match {
            return Ok(());
        }
        let g = runtime.graph();
        // Collapsing parallel edges to one row per pair is only exact when the
        // relation has none.
        // A pair joined by edges of different types holds several edges too,
        // although no single tensor stores a parallel one.
        let emit_relationship =
            emit_relationship || state.multi_edge || state.edge_type_indices.len() > 1;
        if !need_edge_id && state.multi_edge && !state.multi_counts_built {
            let mut counts: FxHashMap<(u64, u64), u32> = FxHashMap::default();
            for &tidx in &state.edge_type_indices {
                for (src, dst, _) in g.relationship_tensors()[tidx].iter_multi_edges() {
                    *counts.entry((src, dst)).or_default() += 1;
                }
            }
            state.multi_counts = Arc::new(counts);
            state.multi_counts_built = true;
        }
        let id_free = (!need_edge_id).then(|| Arc::clone(&state.multi_counts));
        // Siblings compared by node (the caller filters the results) need no id
        // check here.
        let sibling_edges = if node_unique.is_some() {
            &[]
        } else {
            sibling_edges
        };

        // Map from_id/to_id to the matrix's src/dst dimensions.
        let (fwd_src_id, fwd_dst_id) = if transposed {
            (to_id, from_id)
        } else {
            (from_id, to_id)
        };
        let (rev_src_id, rev_dst_id) = if transposed {
            (from_id, to_id)
        } else {
            (to_id, from_id)
        };
        // When only the matrix-destination is bound, enumerating that node's
        // incoming edges via the transposed matrix is O(in-degree); the
        // forward iterator would have to scan EVERY edge and filter on dest
        // (quadratic across input rows). Build the shared transposed
        // iterator once, lazily.
        let need_bwd_fwd = fwd_src_id.is_none() && fwd_dst_id.is_some();
        if state.union.is_some() {
            Self::expand_union_row(
                state,
                &g,
                rp,
                transposed,
                (from_id, to_id),
                (fwd_src_id, fwd_dst_id),
                (rev_src_id, rev_dst_id),
                need_bwd_fwd,
                batch,
                row_idx,
                out,
                emit_relationship,
                sibling_edges,
                id_free.as_deref(),
            );
        } else {
            let need_bwd_rev =
                state.rev_iter.is_some() && rev_src_id.is_none() && rev_dst_id.is_some();
            if (need_bwd_fwd || need_bwd_rev) && state.bwd_iter.is_none() {
                state.bwd_iter = Some(std::cell::RefCell::new(
                    build_transposed_iter(&g, &rp.types).unwrap_or_else(empty_edge_iter),
                ));
            }

            // Reuse the persistent forward iterator: seek the row range
            // instead of allocating a fresh GxB_Iterator per input row.
            {
                let mut fwd_borrow;
                let mut bwd_borrow;
                let pairs = if need_bwd_fwd {
                    let d = u64::from(fwd_dst_id.expect("need_bwd_fwd implies dst bound"));
                    bwd_borrow = state.bwd_iter.as_ref().expect("built above").borrow_mut();
                    bwd_borrow.seek(d, d);
                    Either::Left(
                        (&mut *bwd_borrow)
                            .map(|(dest, src)| (NodeId::from(src), NodeId::from(dest))),
                    )
                } else {
                    fwd_borrow = state.fwd_iter.borrow_mut();
                    let (min_row, max_row) =
                        fwd_src_id.map_or((0, u64::MAX), |id| (u64::from(id), u64::from(id)));
                    fwd_borrow.seek(min_row, max_row);
                    let fwd_dst_filter = fwd_dst_id.map(u64::from);
                    Either::Right(
                        (&mut *fwd_borrow)
                            .filter(move |(_, dest)| fwd_dst_filter.is_none_or(|d| d == *dest))
                            .map(|(src, dest)| (NodeId::from(src), NodeId::from(dest))),
                    )
                };
                Self::process_pairs(
                    pairs,
                    transposed,
                    from_id,
                    to_id,
                    &g,
                    rp,
                    batch,
                    row_idx,
                    out,
                    emit_relationship,
                    sibling_edges,
                    id_free.as_deref(),
                    &state.edge_type_indices,
                    &state.fwd_src_label_ids,
                    &state.fwd_dst_label_ids,
                );
            }

            // Process reverse relationships for bidirectional patterns.
            if let Some(ref rev_iter_cell) = state.rev_iter {
                let mut rev_borrow;
                let mut bwd_borrow;
                let pairs = if need_bwd_rev {
                    let d = u64::from(rev_dst_id.expect("need_bwd_rev implies dst bound"));
                    bwd_borrow = state.bwd_iter.as_ref().expect("built above").borrow_mut();
                    bwd_borrow.seek(d, d);
                    Either::Left(
                        (&mut *bwd_borrow)
                            .map(|(dest, src)| (NodeId::from(src), NodeId::from(dest)))
                            .filter(|(s, d)| s != d),
                    )
                } else {
                    rev_borrow = rev_iter_cell.borrow_mut();
                    let (min_row, max_row) =
                        rev_src_id.map_or((0, u64::MAX), |id| (u64::from(id), u64::from(id)));
                    rev_borrow.seek(min_row, max_row);
                    let rev_dst_filter = rev_dst_id.map(u64::from);
                    Either::Right(
                        (&mut *rev_borrow)
                            .filter(move |(_, dest)| rev_dst_filter.is_none_or(|d| d == *dest))
                            .map(|(src, dest)| (NodeId::from(src), NodeId::from(dest)))
                            .filter(|(s, d)| s != d),
                    )
                };
                Self::process_pairs(
                    pairs,
                    !transposed,
                    from_id,
                    to_id,
                    &g,
                    rp,
                    batch,
                    row_idx,
                    out,
                    emit_relationship,
                    sibling_edges,
                    id_free.as_deref(),
                    &state.edge_type_indices,
                    &state.rev_src_label_ids,
                    &state.rev_dst_label_ids,
                );
            }
        }

        drop(g);

        Ok(())
    }

    /// `expand_row`'s expansion for a multi-type pattern: the single-type
    /// forward/reverse code run over the union iterators. Out of line on
    /// purpose — single-type traversals never take it, and a second inlined
    /// copy in `expand_row` measured +2.5–12% instructions on them.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn expand_union_row(
        state: &mut CtState,
        g: &crate::graph::graph::Graph,
        rp: &QueryRelationship<Arc<String>, Arc<String>, Variable>,
        transposed: bool,
        (from_id, to_id): (Option<NodeId>, Option<NodeId>),
        (fwd_src_id, fwd_dst_id): (Option<NodeId>, Option<NodeId>),
        (rev_src_id, rev_dst_id): (Option<NodeId>, Option<NodeId>),
        need_bwd_fwd: bool,
        batch: &Batch<'a>,
        row_idx: usize,
        out: &mut Vec<(NodeId, NodeId, RelationshipId)>,
        emit_relationship: bool,
        sibling_edges: &[u32],
        id_free: Option<&FxHashMap<(u64, u64), u32>>,
    ) {
        let iters = state.union.as_deref_mut().expect("multi-type state");
        let need_bwd_rev = iters.rev.is_some() && rev_src_id.is_none() && rev_dst_id.is_some();
        if (need_bwd_fwd || need_bwd_rev) && iters.bwd.is_none() {
            iters.bwd = Some(std::cell::RefCell::new(
                build_transposed_union_iter(g, &rp.types)
                    .unwrap_or_else(|| UnionIter::new(Vec::new())),
            ));
        }
        let state = &*state;
        let iters = state.union.as_deref().expect("multi-type state");
        // Reuse the persistent forward iterator: seek the row range
        // instead of allocating a fresh GxB_Iterator per input row.
        {
            let mut fwd_borrow;
            let mut bwd_borrow;
            let pairs = if need_bwd_fwd {
                let d = u64::from(fwd_dst_id.expect("need_bwd_fwd implies dst bound"));
                bwd_borrow = iters.bwd.as_ref().expect("built above").borrow_mut();
                bwd_borrow.seek(d, d);
                Either::Left(
                    (&mut *bwd_borrow).map(|(dest, src)| (NodeId::from(src), NodeId::from(dest))),
                )
            } else {
                fwd_borrow = iters.fwd.borrow_mut();
                let (min_row, max_row) =
                    fwd_src_id.map_or((0, u64::MAX), |id| (u64::from(id), u64::from(id)));
                fwd_borrow.seek(min_row, max_row);
                let fwd_dst_filter = fwd_dst_id.map(u64::from);
                Either::Right(
                    (&mut *fwd_borrow)
                        .filter(move |(_, dest)| fwd_dst_filter.is_none_or(|d| d == *dest))
                        .map(|(src, dest)| (NodeId::from(src), NodeId::from(dest))),
                )
            };
            Self::process_pairs(
                pairs,
                transposed,
                from_id,
                to_id,
                g,
                rp,
                batch,
                row_idx,
                out,
                emit_relationship,
                sibling_edges,
                id_free,
                &state.edge_type_indices,
                &state.fwd_src_label_ids,
                &state.fwd_dst_label_ids,
            );
        }

        // Process reverse relationships for bidirectional patterns.
        if let Some(ref rev_iter_cell) = iters.rev {
            let mut rev_borrow;
            let mut bwd_borrow;
            let pairs = if need_bwd_rev {
                let d = u64::from(rev_dst_id.expect("need_bwd_rev implies dst bound"));
                bwd_borrow = iters.bwd.as_ref().expect("built above").borrow_mut();
                bwd_borrow.seek(d, d);
                Either::Left(
                    (&mut *bwd_borrow)
                        .map(|(dest, src)| (NodeId::from(src), NodeId::from(dest)))
                        .filter(|(s, d)| s != d),
                )
            } else {
                rev_borrow = rev_iter_cell.borrow_mut();
                let (min_row, max_row) =
                    rev_src_id.map_or((0, u64::MAX), |id| (u64::from(id), u64::from(id)));
                rev_borrow.seek(min_row, max_row);
                let rev_dst_filter = rev_dst_id.map(u64::from);
                Either::Right(
                    (&mut *rev_borrow)
                        .filter(move |(_, dest)| rev_dst_filter.is_none_or(|d| d == *dest))
                        .map(|(src, dest)| (NodeId::from(src), NodeId::from(dest)))
                        .filter(|(s, d)| s != d),
                )
            };
            Self::process_pairs(
                pairs,
                !transposed,
                from_id,
                to_id,
                g,
                rp,
                batch,
                row_idx,
                out,
                emit_relationship,
                sibling_edges,
                id_free,
                &state.edge_type_indices,
                &state.rev_src_label_ids,
                &state.rev_dst_label_ids,
            );
        }
    }

    /// Processes relationship pairs from an iterator without materializing them.
    #[allow(clippy::too_many_arguments)]
    fn process_pairs(
        pairs: impl Iterator<Item = (crate::graph::graph::NodeId, crate::graph::graph::NodeId)>,
        is_reverse: bool,
        from_id: Option<crate::graph::graph::NodeId>,
        to_id: Option<crate::graph::graph::NodeId>,
        g: &crate::graph::graph::Graph,
        rp: &QueryRelationship<Arc<String>, Arc<String>, Variable>,
        batch: &Batch<'a>,
        row_idx: usize,
        out: &mut Vec<(NodeId, NodeId, RelationshipId)>,
        emit_relationship: bool,
        sibling_edges: &[u32],
        id_free: Option<&FxHashMap<(u64, u64), u32>>,
        edge_type_indices: &[usize],
        src_label_ids: &[LabelId],
        dst_label_ids: &[LabelId],
    ) {
        for (src, dst) in pairs {
            // Per-pair label validation replaces the per-query rmxm/lmxm
            // restriction on the relationship matrix. `(src, dst)` here are
            // the raw matrix coordinates (before `is_reverse` swap).
            if !src_label_ids
                .iter()
                .all(|&lid| g.node_has_label_id(src, lid))
            {
                continue;
            }
            if !dst_label_ids
                .iter()
                .all(|&lid| g.node_has_label_id(dst, lid))
            {
                continue;
            }
            let (from_node, to_node) = if is_reverse { (dst, src) } else { (src, dst) };
            if from_id.is_some() && from_id.unwrap() != from_node {
                continue;
            }
            if to_id.is_some() && to_id.unwrap() != to_node {
                continue;
            }
            // When emit_relationship is false — an anonymous edge, outside any
            // named path, that no predicate reads (the planner sets it for an
            // edge with one) — skip per-edge iteration and emit one row per
            // (src, dst) pair. The outer `get_relationships` iterator already
            // returns unique matrix-level pairs, so one representative edge per
            // pair is sufficient.
            let mat_src = u64::from(src);
            let mat_dst = u64::from(dst);
            if !emit_relationship {
                if id_free.is_some() {
                    // Nothing reads the edge and nothing compares ids: the pair
                    // is the match, so skip the tensor lookup for a
                    // representative.
                    out.push((from_node, to_node, RelationshipId::from(0u64)));
                    continue;
                }
                let mut found_id: Option<RelationshipId> = None;
                let env = BatchRow::new(batch, row_idx);
                'outer: for &tidx in edge_type_indices {
                    for raw_id in g.relationship_tensors()[tidx].get(mat_src, mat_dst) {
                        let id = RelationshipId::from(raw_id);
                        if !super::edge_already_used(&env, id, rp.alias.id, sibling_edges) {
                            found_id = Some(id);
                            break 'outer;
                        }
                    }
                }
                if let Some(id) = found_id {
                    // The parent columns are replicated by the emitter's gather;
                    // here we only record the produced endpoints and edge.
                    out.push((from_node, to_node, id));
                }
                continue;
            }

            // Scan edges
            if let Some(counts) = id_free {
                // No id is read or compared: a pair holds one edge unless it is
                // in `counts`, and each edge is one row.
                let edges = counts.get(&(mat_src, mat_dst)).copied().unwrap_or(1);
                for _ in 0..edges {
                    out.push((from_node, to_node, RelationshipId::from(0u64)));
                }
                continue;
            }
            let env = BatchRow::new(batch, row_idx);
            for &tidx in edge_type_indices {
                for raw_id in g.relationship_tensors()[tidx].get(mat_src, mat_dst) {
                    let id = RelationshipId::from(raw_id);
                    // Relationship uniqueness: skip edges already bound to other
                    // relationship variables in this MATCH clause.
                    if super::edge_already_used(&env, id, rp.alias.id, sibling_edges) {
                        continue;
                    }
                    out.push((from_node, to_node, id));
                }
            }
        }
    }

    /// Trim a produced batch to the remaining `record_cap` budget (when set),
    /// advancing `produced`. Takes the fields explicitly so it can be called
    /// while disjoint fields of `self` are borrowed by the emitter closure.
    fn trim_to_cap(
        record_cap: Option<usize>,
        produced: &mut usize,
        batch: Batch<'a>,
    ) -> Batch<'a> {
        let Some(cap) = record_cap else {
            return batch;
        };
        let remaining = cap.saturating_sub(*produced);
        let active_len = batch.active_len();
        if active_len > remaining {
            let active: Vec<usize> = batch.active_indices().take(remaining).collect();
            let trimmed = batch.gather(&active);
            *produced += trimmed.active_len();
            trimmed
        } else {
            *produced += active_len;
            batch
        }
    }

    /// Alias id of the endpoint this traverse introduces (respecting
    /// transposition and any fused chain).
    fn out_alias_id(&self) -> u32 {
        if let Some(last_hop) = self.chain.last() {
            last_hop.to.alias.id
        } else if self.transposed {
            self.relationship_pattern.from.alias.id
        } else {
            self.relationship_pattern.to.alias.id
        }
    }

    /// OPTIONAL MATCH fallback: gather the unexpanded input rows and bind the
    /// edge and destination aliases to NULL (mirrors `OptionalOp`'s fallback).
    fn null_pad(
        &self,
        batch: &Batch<'a>,
        rows: &[usize],
    ) -> Batch<'a> {
        let mut fb = batch.gather(rows);
        fb.set_column(
            self.relationship_pattern.alias.id,
            Column::Values(vec![Value::Null; rows.len()]),
        );
        fb.set_column(
            self.out_alias_id(),
            Column::Values(vec![Value::Null; rows.len()]),
        );
        fb
    }
}

impl<'a> Iterator for CondTraverseOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if record_cap already reached.
        if let Some(cap) = self.record_cap
            && self.produced >= cap
        {
            return None;
        }

        // NOTE: we no longer short-circuit when requested labels/types are
        // unknown.  A sibling Commit in the subtree may add them while this
        // operator is running, and skipping the child here would skip its
        // side effects.  Empty matrices/iters in the lazy state naturally
        // produce zero rows, which is the desired behavior.

        // Pre-bind disjoint field borrows so the emitter closure doesn't borrow
        // `self` (which also owns the emitter). The per-row expansion runs
        // through these instead of `&self`.
        let runtime = self.runtime;
        let rp = self.relationship_pattern;
        let emit_relationship = self.emit_relationship;
        let sibling_edges = self.sibling_edges;
        let node_unique = self.node_unique.as_deref();
        let need_edge_id = node_unique.is_none();
        let transposed = self.transposed;
        let state_cell = &self.state;
        let record_cap = self.record_cap;
        let optional = self.optional;
        let unmatched_cell = &self.optional_unmatched;

        loop {
            // 1. Emit any batches produced by the batched F·A path first.
            if let Some(out) = self.pending_batches.pop_front() {
                return Some(Ok(Self::trim_to_cap(record_cap, &mut self.produced, out)));
            }

            // 2. Drive the per-row fallback emitter over the seeded batch. Each
            //    active row's single-hop results are enumerated eagerly into a
            //    `Vec` (the edge iterators borrow the graph, so they can't stream
            //    lazily), which the emitter packs across rows into one gathered
            //    batch — replacing the per-result env clone + row-builder
            //    transpose with a single columnar `gather`.
            match self.emitter.emit_lazy(|batch, row_idx| {
                let mut expanded = Vec::new();
                Self::expand_row(
                    runtime,
                    rp,
                    emit_relationship,
                    sibling_edges,
                    need_edge_id,
                    node_unique,
                    transposed,
                    state_cell,
                    batch,
                    row_idx,
                    &mut expanded,
                )?;
                if let Some(siblings) = node_unique {
                    // As in the batched path: each sibling bound to a pair of
                    // nodes takes one of the edges that join them.
                    let pairs: Vec<(NodeId, NodeId)> =
                        sibling_edge_pairs(siblings, batch, row_idx).collect();
                    if !pairs.is_empty() {
                        let mut taken: Vec<((NodeId, NodeId), usize)> = Vec::new();
                        expanded.retain(|&(from, to, _)| {
                            let own = if transposed { (to, from) } else { (from, to) };
                            let shared = pairs.iter().filter(|&&p| p == own).count();
                            if shared == 0 {
                                return true;
                            }
                            let n = match taken.iter_mut().find(|(p, _)| *p == own) {
                                Some((_, n)) => n,
                                None => {
                                    taken.push((own, 0));
                                    &mut taken.last_mut().unwrap().1
                                }
                            };
                            *n += 1;
                            *n > shared
                        });
                    }
                }
                if expanded.is_empty() {
                    // Optional traverse: remember the unexpanded row so it can
                    // be null-padded once the seeded batch is exhausted.
                    if optional {
                        unmatched_cell.borrow_mut().push(row_idx);
                    }
                    Ok(None)
                } else {
                    Ok(Some(RowIter::many(Box::new(expanded.into_iter()))))
                }
            }) {
                Ok(Some(out)) => {
                    return Some(Ok(Self::trim_to_cap(record_cap, &mut self.produced, out)));
                }
                Ok(None) => {
                    // Seeded batch exhausted: flush any null-padded fallback
                    // rows accumulated by the optional per-row path first.
                    if optional {
                        let unmatched = std::mem::take(&mut *unmatched_cell.borrow_mut());
                        if !unmatched.is_empty() {
                            let src = self
                                .emitter
                                .batch()
                                .expect("unmatched rows imply a seeded batch");
                            let fb = self.null_pad(src, &unmatched);
                            return Some(Ok(Self::trim_to_cap(record_cap, &mut self.produced, fb)));
                        }
                    }
                    // Seeded batch exhausted (or none installed). Pull the next
                    // child batch and decide fast vs. per-row path for it.
                    match self.child.next() {
                        Some(Ok(b)) => {
                            // Try the batched mxm path on the whole batch. When it
                            // fully handles the batch (`true`) its output is
                            // queued on `pending_batches` and the emitter is left
                            // idle; on fallback (`false`) seed the emitter to
                            // expand the batch row-by-row.
                            let mut handled = false;
                            if self.batched_eligible {
                                let active: Vec<usize> = b.active_indices().collect();
                                let mut pending = std::mem::take(&mut self.pending_batches);
                                handled = self.expand_batch(&b, &active, &mut pending);
                                self.pending_batches = pending;
                            }
                            if !handled {
                                self.emitter.seed(b);
                            }
                        }
                        Some(Err(e)) => return Some(Err(e)),
                        None => return None,
                    }
                }
                Err(e) => return Some(Err(e)),
            }
        }
    }
}
