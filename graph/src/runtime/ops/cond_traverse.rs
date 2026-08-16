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
use std::sync::Arc;

use crate::graph::graph::{LabelId, NodeId, RelationshipId};
use crate::graph::graphblas::matrix::Matrix;
use crate::graph::graphblas::tensor::Tensor;
use crate::graph::graphblas::versioned_matrix::{Iter, VersionedMatrix};
use crate::parser::ast::{QueryExpr, QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp, BatchRow, Column},
    row::RowView,
    runtime::Runtime,
    value::Value,
};
use itertools::Either;
use orx_tree::NodeRef as _;
use orx_tree::{Dyn, NodeIdx};

use super::batched_result_emitter::{BatchedResultEmitter, EdgeEndpoints, RowIter};

/// Base matrix for the batched mxm path. Relationship matrices store inline
/// edge ids (`u64`) while the adjacency matrix and merged multi-type matrices
/// are `bool`; traversal only consumes the sparsity pattern (`ANY_PAIR`
/// semiring), so both traverse identically. Cloning either variant is cheap
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
}

impl TraversalMatrix {
    fn ncols(&self) -> u64 {
        match self {
            Self::Bool(m) => m.ncols(),
            Self::U64(t) => t.fwd_m().ncols(),
        }
    }

    /// `f = f * self` (delta-aware, structural). See [`Matrix::delta_lmxm`].
    fn delta_lmxm_into(
        &self,
        f: &mut Matrix<bool>,
    ) {
        match self {
            Self::Bool(vm) => f.delta_lmxm(vm.m(), vm.dp(), vm.dm()),
            Self::U64(t) => f.delta_lmxm(t.fwd_m(), t.fwd_dp(), t.fwd_dm()),
        }
    }
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
    fwd_src_label_ids: Vec<LabelId>,
    fwd_dst_label_ids: Vec<LabelId>,
    rev_src_label_ids: Vec<LabelId>,
    rev_dst_label_ids: Vec<LabelId>,
    edge_type_indices: Vec<usize>,
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
    /// Predicate every candidate edge must satisfy, applied during iteration
    /// so a rejected edge never becomes an output row. Its presence also means
    /// parallel edges are individually distinguishable, so the collapse to one
    /// representative per (src, dst) pair is unsound.
    edge_filter: Option<QueryExpr<Variable>>,
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
    /// For bidirectional anonymous-edge CTs, tracks (source, dest) pairs
    /// already emitted to deduplicate rows that reach the same pair via
    /// different intermediate nodes — matching C FalkorDB's matrix-multiply
    /// semantics.
    bidir_dedup: Option<std::cell::RefCell<std::collections::HashSet<(u64, u64)>>>,
    /// When the child is also an anonymous bidir CT, stores the child's
    /// from-alias so the dedup key uses the original scan source (not the
    /// intermediate node).  When None, dedup uses this CT's own from-alias.
    dedup_source_alias: Option<Variable>,
    /// Structural eligibility for the batched F·A path. Computed once at
    /// construction. Variants like emit_relationship, bidir, sibling-edge
    /// uniqueness, and non-empty inline attribute predicates fall back to
    /// the per-row `expand_row` path.
    batched_eligible: bool,
}

/// Build an `EdgeIter` over the union of relationship matrices for `types`,
/// avoiding the dup+merge cost of materializing a fresh `Matrix`.
fn build_unrestricted_iter(
    g: &crate::graph::graph::Graph,
    types: &[Arc<String>],
) -> Option<Iter> {
    if types.is_empty() {
        return Some(g.adjacency_matrix().iter(0, u64::MAX));
    }
    if types.len() == 1 {
        return g
            .get_relationship_matrix(&types[0])
            .map(|t| t.structural_iter(0, u64::MAX));
    }
    let merged = g.build_relationship_matrix_unrestricted(types)?;
    Some(VersionedMatrix::from_matrix(merged).iter(0, u64::MAX))
}

fn empty_edge_iter() -> Iter {
    VersionedMatrix::<bool>::new(0, 0).iter(0, u64::MAX)
}

/// Build an `EdgeIter` over the TRANSPOSED (dst → src) pair matrix for
/// `types`, for expansions where only the destination is bound. Tensors
/// maintain their transpose (`mt`) incrementally, so the single-type case is
/// free; the untyped/multi-type cases pay one O(E) GraphBLAS transpose —
/// amortized over the whole operator, unlike the per-row full scan this
/// replaces.
fn build_transposed_iter(
    g: &crate::graph::graph::Graph,
    types: &[Arc<String>],
) -> Option<Iter> {
    if types.is_empty() {
        return Some(g.adjacency_matrix().transpose().iter(0, u64::MAX));
    }
    if types.len() == 1 {
        return g
            .get_relationship_matrix(&types[0])
            .map(|t| t.matrix_t().iter(0, u64::MAX));
    }
    let merged = g.build_relationship_matrix_unrestricted(types)?;
    Some(VersionedMatrix::from_matrix(merged.transpose()).iter(0, u64::MAX))
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
        edge_filter: Option<QueryExpr<Variable>>,
        idx: NodeIdx<Dyn<IR>>,
        record_cap: Option<usize>,
    ) -> Self {
        let rp = relationship_pattern;

        // When this CT and its child CT are both anonymous bidirectional
        // edges, enable cross-row (from, to) deduplication to replicate
        // C FalkorDB's matrix-multiply semantics.  Use the child's
        // from-alias as the dedup source so we deduplicate by
        // (original_scan_source, final_destination).
        //
        // Only enable when the intermediate node (this CT's from-alias,
        // which is the child CT's to-alias) is anonymous.  If the
        // intermediate is a user-named variable it may be referenced by
        // downstream operators (e.g. ExpandInto), so collapsing rows
        // with different intermediate values would lose valid results.
        let intermediate_is_anon = rp
            .from
            .alias
            .name
            .as_ref()
            .is_some_and(|n| n.starts_with("_anon"));
        let (bidir_dedup, dedup_source_alias) =
            if !emit_relationship && rp.bidirectional && intermediate_is_anon {
                if let BatchOp::CondTraverse(ref child_ct) = *child {
                    if !child_ct.emit_relationship && child_ct.relationship_pattern.bidirectional {
                        (
                            Some(std::cell::RefCell::new(std::collections::HashSet::<(
                                u64,
                                u64,
                            )>::new(
                            ))),
                            Some(child_ct.relationship_pattern.from.alias.clone()),
                        )
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

        // For fused chains the batched path is the ONLY correct path —
        // expand_row only handles single-hop.
        //
        // The endpoints' inline attrs no longer gate this. The planner lowers
        // both endpoints of every traverse into a Filter unconditionally, so
        // the predicate is enforced whichever path runs, and expand_row's own
        // check on them is redundant.
        //
        // An edge predicate still gates it: with `emit_relationship` false,
        // expand_batch binds one representative edge per (src, dst) pair, so a
        // predicate that tells parallel edges apart has to be applied during
        // the per-row scan, which iterates all of them.
        let chain_is_empty = chain.is_empty();
        let batched_eligible = !emit_relationship
            && !rp.bidirectional
            && bidir_dedup.is_none()
            && sibling_edges.is_empty()
            && (!chain_is_empty || edge_filter.is_none())
            && chain.iter().all(|hop| !hop.bidirectional);

        // Self-loop patterns like `MATCH (n)-[r:T]->(n)` share one alias on both
        // endpoints; bind it once (via `from`) and skip the `to` column so the
        // second insert can't overwrite the first. The per-row path resolves
        // `from`/`to` to the pattern endpoints itself, so the emitter binds them
        // untransposed.
        let to = (relationship_pattern.to.alias != relationship_pattern.from.alias)
            .then_some(relationship_pattern.to.alias.id);
        let mut emitter = BatchedResultEmitter::with_binding(EdgeEndpoints {
            from: relationship_pattern.from.alias.id,
            to,
            edge: relationship_pattern.alias.id,
            transposed: false,
        });
        // A downstream Skip/Limit lowers how many rows are needed; shrink the
        // pack ceiling so the first emit returns a small batch instead of a full
        // BATCH_SIZE worth of work.
        emitter.apply_record_cap(record_cap);

        Self {
            runtime,
            child,
            relationship_pattern,
            emitter,
            pending_batches: VecDeque::new(),
            emit_relationship,
            bind_relationship,
            edge_filter,
            sibling_edges,
            transposed,
            chain,
            optional,
            optional_unmatched: std::cell::RefCell::new(Vec::new()),
            idx,
            record_cap,
            produced: 0,
            state: std::cell::RefCell::new(None),
            bidir_dedup,
            dedup_source_alias,
            batched_eligible,
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
        let g = runtime.g.borrow();

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

        let fwd_iter_opt = build_unrestricted_iter(&g, &rp.types);
        let rev_iter_opt = if rp.bidirectional {
            build_unrestricted_iter(&g, &rp.types)
        } else {
            None
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
        let rev_iter = if rp.bidirectional {
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

        CtState {
            fwd_iter: std::cell::RefCell::new(fwd_iter),
            rev_iter: rev_iter.map(std::cell::RefCell::new),
            bwd_iter: None,
            fwd_src_label_ids,
            fwd_dst_label_ids,
            rev_src_label_ids,
            rev_dst_label_ids,
            edge_type_indices,
            no_match,
            batched_matrix: None,
            chain_matrices: Vec::new(),
            chain_dst_label_ids: Vec::new(),
        }
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

        let g = runtime.g.borrow();

        if state.batched_matrix.is_none() {
            let m = if rp.types.is_empty() {
                TraversalMatrix::Bool(g.adjacency_matrix().clone())
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
                if let Some(m) = g.build_relationship_matrix_unrestricted(&rp.types) {
                    TraversalMatrix::Bool(VersionedMatrix::from_matrix(m))
                } else {
                    state.no_match = true;
                    if self.optional {
                        out_pending.push_back(self.null_pad(batch, active_subset));
                    }
                    return true;
                }
            };
            state.batched_matrix = Some(m);

            // Build per-hop matrices and label-id vectors for the fused chain.
            // If any hop's relationship type is unknown, mark no_match — that
            // hop can produce no rows, so the overall fused traversal can't
            // either.
            for hop in self.chain {
                let hm = if hop.types.is_empty() {
                    TraversalMatrix::Bool(g.adjacency_matrix().clone())
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
                    if let Some(m) = g.build_relationship_matrix_unrestricted(&hop.types) {
                        TraversalMatrix::Bool(VersionedMatrix::from_matrix(m))
                    } else {
                        state.no_match = true;
                        if self.optional {
                            out_pending.push_back(self.null_pad(batch, active_subset));
                        }
                        return true;
                    }
                };
                state.chain_matrices.push(hm);
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
        let m_merged = state.batched_matrix.as_ref().unwrap();
        let ncols = m_merged.ncols();

        let transposed = self.transposed;
        let nrows = active_subset.len() as u64;

        // Collect (row_i, src_id) for rows where the from-alias is bound to
        // a Node and (post-label-filter) the src has all required labels.
        // Rows where from is bound to a non-Node are dropped (mirror line
        // 311 early return). Rows where from is unbound trigger fall-back.
        let mut row_idx_buf: Vec<u64> = Vec::with_capacity(active_subset.len());
        let mut col_idx_buf: Vec<u64> = Vec::with_capacity(active_subset.len());
        let from_alias = if transposed {
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
            if !state
                .fwd_src_label_ids
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

        let mut f = Matrix::<bool>::new(nrows, ncols);
        f.build(&row_idx_buf, &col_idx_buf);
        m_merged.delta_lmxm_into(&mut f);
        for hop_m in &state.chain_matrices {
            hop_m.delta_lmxm_into(&mut f);
        }

        // Flush pending mxm work before attaching the row iterator.
        f.wait();
        // For fused chains all hops are storage-direction (the fusion pass
        // refuses to fuse when transposed differs), so `transposed` applies
        // only to the first hop and the final destination alias comes from
        // the last hop in `self.chain`.
        let from_alias = if transposed {
            &rp.to.alias
        } else {
            &rp.from.alias
        };
        let (to_alias, dst_label_ids) = if let Some(last_hop) = self.chain.last() {
            (
                &last_hop.to.alias,
                state.chain_dst_label_ids.last().unwrap().as_slice(),
            )
        } else if transposed {
            (&rp.from.alias, state.fwd_dst_label_ids.as_slice())
        } else {
            (&rp.to.alias, state.fwd_dst_label_ids.as_slice())
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

        for (row_i, dest_raw) in f.iter(0, u64::MAX) {
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

            if bind_edge {
                // Look up one representative edge id (mirrors expand_row's
                // anonymous-edge fast path). Needed whenever the alias is read
                // downstream — including by PathBuilder, which reads it even
                // when emit_relationship is false, so `emit_relationship`
                // alone cannot decide this. Storage matrix orientation:
                // src=F's seed (matrix-src), dst=F*A result (matrix-dst),
                // regardless of self.transposed (which only affects
                // alias→storage mapping, not the underlying matrix orientation
                // since build_relationship_matrix_unrestricted is
                // non-transposed).
                let Some(Value::Node(src_id)) = batch.value_at(from_alias.id, row_idx) else {
                    continue;
                };
                let mat_src = u64::from(src_id);
                let mat_dst = u64::from(dest_id);
                let mut found_id: Option<RelationshipId> = None;
                for &tidx in &state.edge_type_indices {
                    if let Some(raw_id) =
                        g.relationship_tensors()[tidx].get(mat_src, mat_dst).next()
                    {
                        found_id = Some(RelationshipId::from(raw_id));
                        break;
                    }
                }
                if let Some(edge_id) = found_id {
                    out_indices.push(row_idx);
                    out_dest_ids.push(dest_id);
                    out_edge_ids.push(edge_id);
                    if self.optional {
                        matched[row_i as usize] = true;
                    }
                }
            } else {
                // No edge column to fill: either a fused chain, whose per-hop
                // edges are anonymous & unreferenced by construction (the
                // fusion pass enforces it) and whose intermediate node aliases
                // are not exposed, or a single hop whose alias nothing reads.
                // Dropping the lookup also drops its `found_id` guard, which
                // never fired: `f` is built from these same tensors, so every
                // pair it yields has at least one edge in one of them.
                out_indices.push(row_idx);
                out_dest_ids.push(dest_id);
                if self.optional {
                    matched[row_i as usize] = true;
                }
            }

            if out_indices.len() >= BATCH_SIZE {
                let mut out_batch = batch.gather(&out_indices);
                if bind_edge {
                    out_batch.set_column(
                        rp.alias.id,
                        Column::RelIds(std::mem::take(&mut out_edge_ids)),
                    );
                }
                out_batch.set_column(
                    to_alias.id,
                    Column::NodeIds(std::mem::take(&mut out_dest_ids)),
                );
                out_pending.push_back(out_batch);
                out_indices.clear();
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

    /// Enumerate the single-hop results for `batch[row_idx]`, pushing each as a
    /// `(from, to, edge)` triple (already mapped onto the pattern's endpoints)
    /// into `out`. Callers pass the op's fields explicitly so this can run inside
    /// the emitter closure without borrowing the emitter through `&self`.
    #[allow(clippy::too_many_arguments)]
    fn expand_row(
        runtime: &Runtime,
        rp: &QueryRelationship<Arc<String>, Arc<String>, Variable>,
        emit_relationship: bool,
        edge_filter: Option<&QueryExpr<Variable>>,
        sibling_edges: &[u32],
        transposed: bool,
        state_cell: &std::cell::RefCell<Option<CtState>>,
        bidir_dedup: Option<&std::cell::RefCell<std::collections::HashSet<(u64, u64)>>>,
        dedup_source_alias: Option<&Variable>,
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

        let g = runtime.g.borrow();

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
        let g = runtime.g.borrow();

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
        let need_bwd_rev = state.rev_iter.is_some() && rev_src_id.is_none() && rev_dst_id.is_some();
        if (need_bwd_fwd || need_bwd_rev) && state.bwd_iter.is_none() {
            state.bwd_iter = Some(std::cell::RefCell::new(
                build_transposed_iter(&g, &rp.types).unwrap_or_else(empty_edge_iter),
            ));
        }

        // Reuse the persistent forward iterator: seek the row range
        // instead of allocating a fresh GxB_Iterator per input row.
        let start = out.len();
        {
            let mut fwd_borrow;
            let mut bwd_borrow;
            let pairs = if need_bwd_fwd {
                let d = u64::from(fwd_dst_id.expect("need_bwd_fwd implies dst bound"));
                bwd_borrow = state.bwd_iter.as_ref().expect("built above").borrow_mut();
                bwd_borrow.seek(d, d);
                Either::Left(
                    (&mut *bwd_borrow).map(|(dest, src)| (NodeId::from(src), NodeId::from(dest))),
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
                edge_filter.map(|f| (f, env.to_owned_row())),
                runtime,
                &g,
                rp,
                batch,
                row_idx,
                out,
                emit_relationship,
                sibling_edges,
                &state.edge_type_indices,
                &state.fwd_src_label_ids,
                &state.fwd_dst_label_ids,
            )?;
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
                edge_filter.map(|f| (f, env.to_owned_row())),
                runtime,
                &g,
                rp,
                batch,
                row_idx,
                out,
                emit_relationship,
                sibling_edges,
                &state.edge_type_indices,
                &state.rev_src_label_ids,
                &state.rev_dst_label_ids,
            )?;
        }

        // When both this CT and its child are anonymous bidirectional,
        // deduplicate output rows by (scan_source, final_dest) across
        // expand_row calls — matching C FalkorDB's matrix-multiply semantics.
        if let Some(dedup) = bidir_dedup {
            let source_alias = dedup_source_alias.unwrap();
            // The scan source is a parent-carried column, constant for every
            // result of this row, so read it once.
            let src_key = match batch.value_at(source_alias.id, row_idx) {
                Some(Value::Node(id)) => Some(u64::from(id)),
                _ => None,
            };
            let mut seen = dedup.borrow_mut();
            let mut i = start;
            while i < out.len() {
                // `out[i].1` is the `to` endpoint bound on the produced row.
                let key = src_key.map(|s| (s, u64::from(out[i].1)));
                if let Some(k) = key
                    && !seen.insert(k)
                {
                    out.swap_remove(i);
                    continue;
                }
                i += 1;
            }
        }

        drop(g);

        Ok(())
    }

    /// Processes relationship pairs from an iterator without materializing them.
    #[allow(clippy::too_many_arguments)]
    fn process_pairs(
        pairs: impl Iterator<Item = (crate::graph::graph::NodeId, crate::graph::graph::NodeId)>,
        is_reverse: bool,
        from_id: Option<crate::graph::graph::NodeId>,
        to_id: Option<crate::graph::graph::NodeId>,
        mut edge_filter: Option<(&QueryExpr<Variable>, crate::runtime::row::Row)>,
        runtime: &Runtime,
        g: &crate::graph::graph::Graph,
        rp: &QueryRelationship<Arc<String>, Arc<String>, Variable>,
        batch: &Batch<'a>,
        row_idx: usize,
        out: &mut Vec<(NodeId, NodeId, RelationshipId)>,
        emit_relationship: bool,
        sibling_edges: &[u32],
        edge_type_indices: &[usize],
        src_label_ids: &[LabelId],
        dst_label_ids: &[LabelId],
    ) -> Result<(), String> {
        // Hoisted: constructing the evaluator per candidate edge shows up
        // directly in the instruction count.
        let evaluator = ExprEval::from_runtime(runtime);
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
            // When emit_relationship is false (anonymous edge not in a named
            // path) and there are no edge attribute filters, skip per-edge
            // iteration and emit one row per (src, dst) pair.  The outer
            // `get_relationships` iterator already returns unique matrix-level
            // pairs, so one representative edge per pair is sufficient.
            let has_edge_filter = edge_filter.is_some();
            let mat_src = u64::from(src);
            let mat_dst = u64::from(dst);
            if !emit_relationship && !has_edge_filter {
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
            let env = BatchRow::new(batch, row_idx);
            for &tidx in edge_type_indices {
                for raw_id in g.relationship_tensors()[tidx].get(mat_src, mat_dst) {
                    let id = RelationshipId::from(raw_id);
                    // Relationship uniqueness: skip edges already bound to other
                    // relationship variables in this MATCH clause.
                    if super::edge_already_used(&env, id, rp.alias.id, sibling_edges) {
                        continue;
                    }
                    // Reject here rather than downstream: a row built for an
                    // edge that fails is pure waste, and the more selective the
                    // predicate the more of it there would be.
                    if let Some((filter_expr, filter_env)) = &mut edge_filter {
                        filter_env.insert(&rp.alias, Value::Relationship(id));
                        let ok = evaluator.eval(
                            filter_expr,
                            filter_expr.root().idx(),
                            Some(&*filter_env),
                            None,
                        );
                        match ok {
                            Ok(Value::Bool(true)) => {}
                            Ok(_) => continue,
                            Err(e) => return Err(e),
                        }
                    }
                    out.push((from_node, to_node, id));
                }
            }
        }
        Ok(())
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
        let edge_filter = self.edge_filter.as_ref();
        let sibling_edges = self.sibling_edges;
        let transposed = self.transposed;
        let state_cell = &self.state;
        let bidir_dedup = self.bidir_dedup.as_ref();
        let dedup_source_alias = self.dedup_source_alias.as_ref();
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
                    edge_filter,
                    sibling_edges,
                    transposed,
                    state_cell,
                    bidir_dedup,
                    dedup_source_alias,
                    batch,
                    row_idx,
                    &mut expanded,
                )?;
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
                            // Reset bidirectional dedup for each new input batch
                            // so Apply/Optional scopes see fresh state.
                            if let Some(ref dedup) = self.bidir_dedup {
                                dedup.borrow_mut().clear();
                            }
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
