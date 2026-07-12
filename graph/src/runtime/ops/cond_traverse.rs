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
use crate::graph::graphblas::matrix::{Matrix, New, Size, Transpose};
use crate::graph::graphblas::tensor::compound_key;
use crate::graph::graphblas::versioned_matrix::{Iter as EdgeIter, VersionedMatrix};
use crate::parser::ast::{ExprIR, QueryExpr, QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp, BatchRow, Column},
    row::RowView,
    runtime::Runtime,
    value::Value,
};
use itertools::Either;
use orx_tree::{Dyn, NodeIdx, NodeRef};

use super::batched_result_emitter::{BatchedResultEmitter, EdgeEndpoints, RowIter};

/// Lazily resolved state — built on first `expand_row`, after any sibling
/// Commit in the subtree has had a chance to create new labels/types.
struct CtState {
    fwd_iter: std::cell::RefCell<EdgeIter>,
    rev_iter: Option<std::cell::RefCell<EdgeIter>>,
    /// Iterator over the TRANSPOSED pair matrix (dst-major). Built lazily on
    /// the first expansion where only the matrix-destination is bound — e.g.
    /// a planner-transposed traverse seeded from the pattern-source, or the
    /// reverse half of a bidirectional expansion. Seeking this by destination
    /// replaces what would otherwise be a full edge-matrix scan per input
    /// row (O(V·E) for the whole query).
    bwd_iter: Option<std::cell::RefCell<EdgeIter>>,
    fwd_src_label_ids: Vec<LabelId>,
    fwd_dst_label_ids: Vec<LabelId>,
    rev_src_label_ids: Vec<LabelId>,
    rev_dst_label_ids: Vec<LabelId>,
    edge_iters: Vec<std::cell::RefCell<EdgeIter>>,
    /// True when one of the requested labels was unknown at state-build
    /// time.  We still drain the child (for side effects) but produce no
    /// output rows, since `unwrap_or_default()` would otherwise turn an
    /// unknown label into "no label restriction".
    no_match: bool,
    /// Materialized base matrix for batched mxm path. Built lazily on
    /// first `expand_batch`. Cached for op lifetime; safe because writes
    /// are serialized w.r.t. read queries.
    batched_matrix: Option<VersionedMatrix>,
    /// Per-hop matrices for fused chain (same lifetime/safety rules as
    /// `batched_matrix`). `chain_matrices[i]` corresponds to `chain[i]`.
    chain_matrices: Vec<VersionedMatrix>,
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
) -> Option<EdgeIter> {
    if types.is_empty() {
        return Some(g.adjacency_matrix().iter(0, u64::MAX));
    }
    if types.len() == 1 {
        return g
            .get_relationship_matrix(&types[0])
            .map(|t| t.matrix().iter(0, u64::MAX));
    }
    let merged = g.build_relationship_matrix_unrestricted(types)?;
    Some(VersionedMatrix::from_matrix(merged).iter(0, u64::MAX))
}

fn empty_edge_iter() -> EdgeIter {
    use crate::graph::graphblas::matrix::New;
    VersionedMatrix::new(0, 0).iter(0, u64::MAX)
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
) -> Option<EdgeIter> {
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

/// Returns true when an inline-attributes tree is structurally an empty
/// `Map` literal (`{}`). Such expressions never reference outer variables,
/// so the F·A batched path can skip evaluating them per row.
fn attrs_is_static_empty(attrs: &QueryExpr<Variable>) -> bool {
    let root = attrs.root();
    matches!(root.data(), ExprIR::Map) && root.children().next().is_none()
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
        // expand_row only handles single-hop. The planner inserts a Filter
        // for any non-empty inline attrs, so attribute predicates are
        // enforced by surrounding Filter nodes regardless of which path runs.
        // Single-hop ops keep the existing strict check (expand_row applies
        // attrs redundantly, but the Filter is still the source of truth).
        let chain_is_empty = chain.is_empty();
        let batched_eligible = !emit_relationship
            && !rp.bidirectional
            && bidir_dedup.is_none()
            && sibling_edges.is_empty()
            && (!chain_is_empty
                || (attrs_is_static_empty(&rp.attrs)
                    && attrs_is_static_empty(&rp.from.attrs)
                    && attrs_is_static_empty(&rp.to.attrs)))
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
        if let Some(cap) = record_cap
            && cap < BATCH_SIZE
        {
            emitter.set_pack_ceiling(cap.max(1));
        }

        Self {
            runtime,
            child,
            relationship_pattern,
            emitter,
            pending_batches: VecDeque::new(),
            emit_relationship,
            sibling_edges,
            transposed,
            chain,
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

        let edge_iters: Vec<_> = if rp.types.is_empty() {
            g.relationship_matrices_iter()
                .map(|tensor| std::cell::RefCell::new(tensor.edge_iter(0, u64::MAX)))
                .collect()
        } else {
            rp.types
                .iter()
                .filter_map(|t| g.get_relationship_matrix(t))
                .map(|tensor| std::cell::RefCell::new(tensor.edge_iter(0, u64::MAX)))
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
            edge_iters,
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
    ) -> Result<bool, String> {
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
            return Ok(true);
        }

        let g = runtime.g.borrow();

        if state.batched_matrix.is_none() {
            let m = if rp.types.is_empty() {
                g.adjacency_matrix().clone()
            } else if rp.types.len() == 1 {
                if let Some(t) = g.get_relationship_matrix(&rp.types[0]) {
                    t.matrix().clone()
                } else {
                    state.no_match = true;
                    return Ok(true);
                }
            } else {
                if let Some(m) = g.build_relationship_matrix_unrestricted(&rp.types) {
                    VersionedMatrix::from_matrix(m)
                } else {
                    state.no_match = true;
                    return Ok(true);
                }
            };
            state.batched_matrix = Some(m);

            // Build per-hop matrices and label-id vectors for the fused chain.
            // If any hop's relationship type is unknown, mark no_match — that
            // hop can produce no rows, so the overall fused traversal can't
            // either.
            for hop in self.chain {
                let hm = if hop.types.is_empty() {
                    g.adjacency_matrix().clone()
                } else if hop.types.len() == 1 {
                    if let Some(t) = g.get_relationship_matrix(&hop.types[0]) {
                        t.matrix().clone()
                    } else {
                        state.no_match = true;
                        return Ok(true);
                    }
                } else {
                    if let Some(m) = g.build_relationship_matrix_unrestricted(&hop.types) {
                        VersionedMatrix::from_matrix(m)
                    } else {
                        state.no_match = true;
                        return Ok(true);
                    }
                };
                state.chain_matrices.push(hm);
                let Some(dst_labels) = g.resolve_label_ids(&hop.to.labels) else {
                    state.no_match = true;
                    return Ok(true);
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
                return Ok(false);
            }
            let src_id = if let Some(Value::Node(id)) = batch.value_at(from_alias.id, row_idx) {
                id
            } else {
                drop(g);
                return Ok(false);
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
            return Ok(true);
        }

        let mut f = Matrix::new(nrows, ncols);
        f.build_bool(&row_idx_buf, &col_idx_buf);
        f.delta_lmxm(m_merged);
        for hop_m in &state.chain_matrices {
            f.delta_lmxm(hop_m);
        }

        let (row_is, col_is) = f.extract_tuples_bool();
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
        let mut out_indices = Vec::new();
        let mut out_dest_ids = Vec::new();
        let mut out_edge_ids = Vec::new();
        let mut out_src_ids = Vec::new();

        for (row_i, dest_raw) in row_is.into_iter().zip(col_is) {
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

            if chain_is_empty {
                // Look up one representative edge id (mirrors expand_row's
                // anonymous-edge fast path). Required because downstream
                // PathBuilder reads the edge alias even when emit_relationship
                // is false. Storage matrix orientation: src=F's seed
                // (matrix-src), dst=F*A result (matrix-dst), regardless of
                // self.transposed (which only affects alias→storage mapping,
                // not the underlying matrix orientation since
                // build_relationship_matrix_unrestricted is non-transposed).
                let src_id = match batch.value_at(from_alias.id, row_idx) {
                    Some(Value::Node(id)) => id,
                    _ => continue,
                };
                let mat_src = u64::from(src_id);
                let mat_dst = u64::from(dest_id);
                let key = compound_key(mat_src, mat_dst);
                let mut found_id: Option<RelationshipId> = None;
                for cell in &state.edge_iters {
                    let mut it = cell.borrow_mut();
                    it.seek(key, key);
                    if let Some((_, raw_id)) = it.next() {
                        found_id = Some(RelationshipId::from(raw_id));
                        break;
                    }
                }
                if let Some(edge_id) = found_id {
                    out_indices.push(row_idx);
                    out_dest_ids.push(dest_id);
                    out_edge_ids.push(edge_id);
                    out_src_ids.push(src_id);
                }
            } else {
                // Fused chain: edges across hops are anonymous & unreferenced
                // (enforced by the fusion pass), so no edge id is bound and
                // no intermediate node alias is exposed.
                out_indices.push(row_idx);
                out_dest_ids.push(dest_id);
            }

            if out_indices.len() >= BATCH_SIZE {
                let mut out_batch = batch.gather(&out_indices);
                if chain_is_empty {
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
                out_src_ids.clear();
            }
        }

        if !out_indices.is_empty() {
            let mut out_batch = batch.gather(&out_indices);
            if chain_is_empty {
                out_batch.set_column(rp.alias.id, Column::RelIds(out_edge_ids));
            }
            out_batch.set_column(to_alias.id, Column::NodeIds(out_dest_ids));
            out_pending.push_back(out_batch);
        }

        drop(g);
        Ok(true)
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
        transposed: bool,
        state_cell: &std::cell::RefCell<Option<CtState>>,
        bidir_dedup: &Option<std::cell::RefCell<std::collections::HashSet<(u64, u64)>>>,
        dedup_source_alias: &Option<Variable>,
        batch: &Batch<'a>,
        row_idx: usize,
        out: &mut Vec<(NodeId, NodeId, RelationshipId)>,
    ) -> Result<(), String> {
        let env = BatchRow::new(batch, row_idx);

        let filter_attrs = ExprEval::from_runtime(runtime).eval(
            &rp.attrs,
            rp.attrs.root().idx(),
            Some(&env),
            None,
        )?;
        let from_node_attrs = ExprEval::from_runtime(runtime).eval(
            &rp.from.attrs,
            rp.from.attrs.root().idx(),
            Some(&env),
            None,
        )?;
        let to_node_attrs = ExprEval::from_runtime(runtime).eval(
            &rp.to.attrs,
            rp.to.attrs.root().idx(),
            Some(&env),
            None,
        )?;

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
                &from_node_attrs,
                &to_node_attrs,
                &filter_attrs,
                &g,
                rp,
                batch,
                row_idx,
                out,
                emit_relationship,
                sibling_edges,
                &state.edge_iters,
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
                &from_node_attrs,
                &to_node_attrs,
                &filter_attrs,
                &g,
                rp,
                batch,
                row_idx,
                out,
                emit_relationship,
                sibling_edges,
                &state.edge_iters,
                &state.rev_src_label_ids,
                &state.rev_dst_label_ids,
            );
        }

        // When both this CT and its child are anonymous bidirectional,
        // deduplicate output rows by (scan_source, final_dest) across
        // expand_row calls — matching C FalkorDB's matrix-multiply semantics.
        if let Some(dedup) = bidir_dedup {
            let source_alias = dedup_source_alias.as_ref().unwrap();
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
        from_node_attrs: &Value,
        to_node_attrs: &Value,
        filter_attrs: &Value,
        g: &crate::graph::graph::Graph,
        rp: &QueryRelationship<Arc<String>, Arc<String>, Variable>,
        batch: &Batch<'a>,
        row_idx: usize,
        out: &mut Vec<(NodeId, NodeId, RelationshipId)>,
        emit_relationship: bool,
        sibling_edges: &[u32],
        edge_iters: &[std::cell::RefCell<EdgeIter>],
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
            // Check from node attrs
            if let Value::Map(attrs) = from_node_attrs
                && !attrs.is_empty()
            {
                let mut skip = false;
                for (attr, avalue) in attrs.iter() {
                    match g.get_node_attribute(from_node, attr) {
                        Some(pvalue) if pvalue == *avalue => {}
                        _ => {
                            skip = true;
                            break;
                        }
                    }
                }
                if skip {
                    continue;
                }
            }
            // Check to node attrs
            if let Value::Map(attrs) = to_node_attrs
                && !attrs.is_empty()
            {
                let mut skip = false;
                for (attr, avalue) in attrs.iter() {
                    match g.get_node_attribute(to_node, attr) {
                        Some(pvalue) if pvalue == *avalue => {}
                        _ => {
                            skip = true;
                            break;
                        }
                    }
                }
                if skip {
                    continue;
                }
            }
            // When emit_relationship is false (anonymous edge not in a named
            // path) and there are no edge attribute filters, skip per-edge
            // iteration and emit one row per (src, dst) pair.  The outer
            // `get_relationships` iterator already returns unique matrix-level
            // pairs, so one representative edge per pair is sufficient.
            let has_edge_filter = matches!(filter_attrs, Value::Map(m) if !m.is_empty());
            let key = compound_key(u64::from(src), u64::from(dst));
            if !emit_relationship && !has_edge_filter {
                let mut found_id: Option<RelationshipId> = None;
                let env = BatchRow::new(batch, row_idx);
                'outer: for cell in edge_iters {
                    let mut it = cell.borrow_mut();
                    it.seek(key, key);
                    for (_, raw_id) in &mut *it {
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
            for cell in edge_iters {
                let mut it = cell.borrow_mut();
                it.seek(key, key);
                for (_, raw_id) in &mut *it {
                    let id = RelationshipId::from(raw_id);
                    // Relationship uniqueness: skip edges already bound to other
                    // relationship variables in this MATCH clause.
                    if super::edge_already_used(&env, id, rp.alias.id, sibling_edges) {
                        continue;
                    }
                    if let Value::Map(filter_map) = filter_attrs
                        && !filter_map.is_empty()
                    {
                        let mut matches = true;
                        for (attr, avalue) in filter_map.iter() {
                            if let Some(pvalue) = g.get_relationship_attribute(id, attr) {
                                if *avalue == pvalue {
                                    continue;
                                }
                                matches = false;
                                break;
                            }
                            matches = false;
                            break;
                        }
                        if !matches {
                            continue;
                        }
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
        let transposed = self.transposed;
        let state_cell = &self.state;
        let bidir_dedup = &self.bidir_dedup;
        let dedup_source_alias = &self.dedup_source_alias;
        let record_cap = self.record_cap;

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
                    transposed,
                    state_cell,
                    bidir_dedup,
                    dedup_source_alias,
                    batch,
                    row_idx,
                    &mut expanded,
                )?;
                if expanded.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(RowIter::many(Box::new(expanded.into_iter()))))
                }
            }) {
                Ok(Some(out)) => {
                    return Some(Ok(Self::trim_to_cap(record_cap, &mut self.produced, out)));
                }
                Ok(None) => {
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
                            // fully handles the batch (`Ok(true)`) its output is
                            // queued on `pending_batches` and the emitter is left
                            // idle; on fallback (`Ok(false)`) seed the emitter to
                            // expand the batch row-by-row.
                            let mut handled = false;
                            if self.batched_eligible {
                                let active: Vec<usize> = b.active_indices().collect();
                                let mut pending = std::mem::take(&mut self.pending_batches);
                                match self.expand_batch(&b, &active, &mut pending) {
                                    Ok(true) => {
                                        self.pending_batches = pending;
                                        handled = true;
                                    }
                                    Ok(false) => {
                                        self.pending_batches = pending;
                                    }
                                    Err(e) => {
                                        self.pending_batches = pending;
                                        return Some(Err(e));
                                    }
                                }
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
