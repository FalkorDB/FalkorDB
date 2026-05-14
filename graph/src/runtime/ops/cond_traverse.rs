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
//!                             (buffered in `pending`, drained to output batches)
//! ```
//!
//! ## State Machine
//!
//! The operator maintains three pieces of state across `next()` calls:
//! - `current_batch` / `current_pos`: position within the current input batch
//! - `pending`: VecDeque of already-expanded output rows awaiting emission
//!
//! Each `next()` drains pending rows, then continues expanding input rows
//! until `BATCH_SIZE` output rows are collected or input is exhausted.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::graph::graph::{LabelId, NodeId, RelationshipId};
use crate::graph::graphblas::matrix::{Matrix, New, Size};
use crate::graph::graphblas::tensor::compound_key;
use crate::graph::graphblas::versioned_matrix::{Iter as EdgeIter, VersionedMatrix};
use crate::parser::ast::{ExprIR, QueryExpr, QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

/// Lazily resolved state — built on first `expand_row`, after any sibling
/// Commit in the subtree has had a chance to create new labels/types.
struct CtState {
    fwd_iter: std::cell::RefCell<EdgeIter>,
    rev_iter: Option<std::cell::RefCell<EdgeIter>>,
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
    /// Buffered output rows from a partial expansion.
    pending: VecDeque<Env<'a>>,
    /// Current input batch being expanded (may span multiple output batches).
    current_batch: Option<Batch<'a>>,
    /// Index into active rows of the current batch.
    current_pos: usize,
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

        Self {
            runtime,
            child,
            relationship_pattern,
            pending: VecDeque::new(),
            current_batch: None,
            current_pos: 0,
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
    /// first `expand_row` invocation, after any sibling Commit in the
    /// subtree has executed.
    fn build_state(&self) -> CtState {
        let rp = self.relationship_pattern;
        let g = self.runtime.g.borrow();

        let (fwd_src_labels, fwd_dst_labels) = if self.transposed {
            (&rp.to.labels, &rp.from.labels)
        } else {
            (&rp.from.labels, &rp.to.labels)
        };

        let fwd_src_label_ids = g.resolve_label_ids(fwd_src_labels);
        let fwd_dst_label_ids = g.resolve_label_ids(fwd_dst_labels);
        let (rev_src_label_ids, rev_dst_label_ids) = if rp.bidirectional {
            let (rev_src_labels, rev_dst_labels) = if self.transposed {
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
        rows: &[&Env<'a>],
        out: &mut Vec<Env<'a>>,
    ) -> Result<bool, String> {
        let runtime = self.runtime;
        let rp = self.relationship_pattern;

        let mut state_ref = self.state.borrow_mut();
        if state_ref.is_none() {
            drop(state_ref);
            let new_state = self.build_state();
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
                match g.get_relationship_matrix(&rp.types[0]) {
                    Some(t) => t.matrix().clone(),
                    None => {
                        state.no_match = true;
                        return Ok(true);
                    }
                }
            } else {
                match g.build_relationship_matrix_unrestricted(&rp.types) {
                    Some(m) => VersionedMatrix::from_matrix(m),
                    None => {
                        state.no_match = true;
                        return Ok(true);
                    }
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
                    match g.get_relationship_matrix(&hop.types[0]) {
                        Some(t) => t.matrix().clone(),
                        None => {
                            state.no_match = true;
                            return Ok(true);
                        }
                    }
                } else {
                    match g.build_relationship_matrix_unrestricted(&hop.types) {
                        Some(m) => VersionedMatrix::from_matrix(m),
                        None => {
                            state.no_match = true;
                            return Ok(true);
                        }
                    }
                };
                state.chain_matrices.push(hm);
                let dst_labels = match g.resolve_label_ids(&hop.to.labels) {
                    Some(ids) => ids,
                    None => {
                        state.no_match = true;
                        return Ok(true);
                    }
                };
                state.chain_dst_label_ids.push(dst_labels);
            }
        }
        let m_merged = state.batched_matrix.as_ref().unwrap();
        let ncols = m_merged.ncols();

        let transposed = self.transposed;
        let nrows = rows.len() as u64;

        // Collect (row_i, src_id) for rows where the from-alias is bound to
        // a Node and (post-label-filter) the src has all required labels.
        // Rows where from is bound to a non-Node are dropped (mirror line
        // 311 early return). Rows where from is unbound trigger fall-back.
        let mut row_idx_buf: Vec<u64> = Vec::with_capacity(rows.len());
        let mut col_idx_buf: Vec<u64> = Vec::with_capacity(rows.len());
        for (i, env) in rows.iter().enumerate() {
            let from_alias = if transposed {
                &rp.to.alias
            } else {
                &rp.from.alias
            };
            // Bail to slow path if the matrix-src side isn't explicitly
            // bound on this row. `env.get` alone would silently return a
            // colliding outer-scope slot value; `is_bound` is authoritative.
            if !env.is_bound(from_alias) {
                drop(g);
                return Ok(false);
            }
            let src_id = match env.get(from_alias) {
                Some(Value::Node(id)) => *id,
                _ => {
                    drop(g);
                    return Ok(false);
                }
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
        for (row_i, dest_raw) in row_is.into_iter().zip(col_is) {
            let dest_id = NodeId::from(dest_raw);
            // Post-filter final-hop dst label (= F * A * R_dst in C's algebra).
            if !dst_label_ids
                .iter()
                .all(|&lid| g.node_has_label_id(dest_id, lid))
            {
                continue;
            }
            let env = rows[row_i as usize];
            // If the to-alias is already bound on the input env, the
            // planner should have inserted ExpandInto, not CondTraverse —
            // but be defensive and skip mismatches.
            if let Some(Value::Node(bound)) = env.get(to_alias)
                && *bound != dest_id
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
                let src_id = match env.get(from_alias) {
                    Some(Value::Node(id)) => *id,
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
                let Some(edge_id) = found_id else { continue };
                let mut row = env.clone_pooled(runtime.env_pool);
                row.insert(to_alias, Value::Node(dest_id));
                row.insert(
                    &rp.alias,
                    Value::Relationship(Box::new((
                        edge_id,
                        NodeId::from(mat_src),
                        NodeId::from(mat_dst),
                    ))),
                );
                out.push(row);
            } else {
                // Fused chain: edges across hops are anonymous & unreferenced
                // (enforced by the fusion pass), so no edge id is bound and
                // no intermediate node alias is exposed.
                let mut row = env.clone_pooled(runtime.env_pool);
                row.insert(to_alias, Value::Node(dest_id));
                out.push(row);
            }
        }

        drop(g);
        Ok(true)
    }

    fn expand_row(
        &self,
        env: &Env<'a>,
        out: &mut Vec<Env<'a>>,
    ) -> Result<(), String> {
        let runtime = self.runtime;
        let rp = self.relationship_pattern;

        let filter_attrs = ExprEval::from_runtime(runtime).eval(
            &rp.attrs,
            rp.attrs.root().idx(),
            Some(env),
            None,
        )?;
        let from_node_attrs = ExprEval::from_runtime(runtime).eval(
            &rp.from.attrs,
            rp.from.attrs.root().idx(),
            Some(env),
            None,
        )?;
        let to_node_attrs = ExprEval::from_runtime(runtime).eval(
            &rp.to.attrs,
            rp.to.attrs.root().idx(),
            Some(env),
            None,
        )?;

        let from_id = env.get(&rp.from.alias).and_then(|v| match v {
            Value::Node(id) => Some(*id),
            _ => None,
        });
        if from_id.is_none() && env.is_bound(&rp.from.alias) {
            return Ok(());
        }
        let to_id = env.get(&rp.to.alias).and_then(|v| match v {
            Value::Node(id) => Some(*id),
            _ => None,
        });
        if to_id.is_none() && env.is_bound(&rp.to.alias) {
            return Ok(());
        }

        let g = runtime.g.borrow();

        // Lazily initialize state from current graph contents on first use,
        // ensuring sibling Commits in the subtree have already had a chance
        // to add new labels/relationship types.
        let mut state_ref = self.state.borrow_mut();
        if state_ref.is_none() {
            drop(state_ref);
            drop(g);
            let new_state = self.build_state();
            *self.state.borrow_mut() = Some(new_state);
            state_ref = self.state.borrow_mut();
            // Re-borrow after building state.
            // (g re-borrowed below)
        }
        let state = state_ref.as_mut().unwrap();
        if state.no_match {
            return Ok(());
        }
        let g = runtime.g.borrow();

        let transposed = self.transposed;

        // Map from_id/to_id to the matrix's src/dst dimensions.
        let (fwd_src_id, fwd_dst_id) = if transposed {
            (to_id, from_id)
        } else {
            (from_id, to_id)
        };
        // Reuse the persistent forward iterator: seek the row range
        // instead of allocating a fresh GxB_Iterator per input row.
        let start = out.len();
        {
            let mut iter = state.fwd_iter.borrow_mut();
            let (min_row, max_row) =
                fwd_src_id.map_or((0, u64::MAX), |id| (u64::from(id), u64::from(id)));
            iter.seek(min_row, max_row);
            let fwd_dst_filter = fwd_dst_id.map(u64::from);
            let pairs = (&mut *iter)
                .filter(|(_, dest)| fwd_dst_filter.is_none_or(|d| d == *dest))
                .map(|(src, dest)| (NodeId::from(src), NodeId::from(dest)));
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
                env,
                runtime,
                out,
                self.emit_relationship,
                self.sibling_edges,
                &state.edge_iters,
                &state.fwd_src_label_ids,
                &state.fwd_dst_label_ids,
            );
        }

        // Process reverse relationships for bidirectional patterns.
        if let Some(ref rev_iter_cell) = state.rev_iter {
            let (rev_src_id, rev_dst_id) = if transposed {
                (from_id, to_id)
            } else {
                (to_id, from_id)
            };
            let mut iter = rev_iter_cell.borrow_mut();
            let (min_row, max_row) =
                rev_src_id.map_or((0, u64::MAX), |id| (u64::from(id), u64::from(id)));
            iter.seek(min_row, max_row);
            let rev_dst_filter = rev_dst_id.map(u64::from);
            let pairs = (&mut *iter)
                .filter(move |(_, dest)| rev_dst_filter.is_none_or(|d| d == *dest))
                .map(|(src, dest)| (NodeId::from(src), NodeId::from(dest)))
                .filter(|(s, d)| s != d);
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
                env,
                runtime,
                out,
                self.emit_relationship,
                self.sibling_edges,
                &state.edge_iters,
                &state.rev_src_label_ids,
                &state.rev_dst_label_ids,
            );
        }

        // When both this CT and its child are anonymous bidirectional,
        // deduplicate output rows by (scan_source, final_dest) across
        // expand_row calls — matching C FalkorDB's matrix-multiply semantics.
        if let Some(ref dedup) = self.bidir_dedup {
            let source_alias = self.dedup_source_alias.as_ref().unwrap();
            let mut seen = dedup.borrow_mut();
            let mut i = start;
            while i < out.len() {
                let key = {
                    let f = out[i].get(source_alias);
                    let t = out[i].get(&rp.to.alias);
                    match (f, t) {
                        (Some(Value::Node(fid)), Some(Value::Node(tid))) => {
                            Some((u64::from(*fid), u64::from(*tid)))
                        }
                        _ => None,
                    }
                };
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
        env: &Env<'a>,
        runtime: &'a Runtime<'a>,
        out: &mut Vec<Env<'a>>,
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
                'outer: for cell in edge_iters {
                    let mut it = cell.borrow_mut();
                    it.seek(key, key);
                    for (_, raw_id) in &mut *it {
                        let id = RelationshipId::from(raw_id);
                        if !super::edge_already_used(env, id, rp.alias.id, sibling_edges) {
                            found_id = Some(id);
                            break 'outer;
                        }
                    }
                }
                if let Some(id) = found_id {
                    let mut row = env.clone_pooled(runtime.env_pool);
                    row.insert(&rp.alias, Value::Relationship(Box::new((id, src, dst))));
                    row.insert(&rp.from.alias, Value::Node(from_node));
                    row.insert(&rp.to.alias, Value::Node(to_node));
                    out.push(row);
                }
                continue;
            }

            // Scan edges
            for cell in edge_iters {
                let mut it = cell.borrow_mut();
                it.seek(key, key);
                for (_, raw_id) in &mut *it {
                    let id = RelationshipId::from(raw_id);
                    // Relationship uniqueness: skip edges already bound to other
                    // relationship variables in this MATCH clause.
                    if super::edge_already_used(env, id, rp.alias.id, sibling_edges) {
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
                    let mut row = env.clone_pooled(runtime.env_pool);
                    row.insert(&rp.alias, Value::Relationship(Box::new((id, src, dst))));
                    row.insert(&rp.from.alias, Value::Node(from_node));
                    row.insert(&rp.to.alias, Value::Node(to_node));
                    out.push(row);
                }
            }
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

        let mut envs = Vec::with_capacity(BATCH_SIZE);

        // Drain leftover rows from previous call.
        super::drain_pending(&mut self.pending, &mut envs);

        loop {
            if envs.len() >= BATCH_SIZE {
                break;
            }

            // Get or fetch current batch.
            if self.current_batch.is_none() {
                match self.child.next() {
                    Some(Ok(b)) => {
                        self.current_batch = Some(b);
                        self.current_pos = 0;
                        // Reset bidirectional dedup for each new input batch so
                        // that Apply/Optional scopes see fresh state.
                        if let Some(ref dedup) = self.bidir_dedup {
                            dedup.borrow_mut().clear();
                        }
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => break,
                }
            }

            {
                let batch = self.current_batch.as_ref().unwrap();
                let active: Vec<usize> = batch.active_indices().collect();

                let mut used_batched = false;
                if self.batched_eligible && self.current_pos < active.len() {
                    let envs_slice: Vec<&Env<'a>> = active[self.current_pos..]
                        .iter()
                        .map(|&i| batch.env_ref(i))
                        .collect();
                    let mut expanded = Vec::new();
                    match self.expand_batch(&envs_slice, &mut expanded) {
                        Ok(true) => {
                            self.current_pos = active.len();
                            self.pending.extend(expanded);
                            used_batched = true;
                        }
                        Ok(false) => {}
                        Err(e) => return Some(Err(e)),
                    }
                }

                if !used_batched {
                    while self.current_pos < active.len() {
                        let row_idx = active[self.current_pos];
                        self.current_pos += 1;
                        let env = batch.env_ref(row_idx);
                        let mut expanded = Vec::new();
                        if let Err(e) = self.expand_row(env, &mut expanded) {
                            return Some(Err(e));
                        }
                        self.pending.extend(expanded);

                        if self.pending.len() >= BATCH_SIZE {
                            break;
                        }
                    }
                }
            }

            super::drain_pending(&mut self.pending, &mut envs);

            // Check if batch is exhausted.
            if let Some(ref batch) = self.current_batch
                && self.current_pos >= batch.active_len()
            {
                self.current_batch = None;
            }
        }

        if envs.is_empty() {
            None
        } else {
            // Trim to record_cap if set.
            if let Some(cap) = self.record_cap {
                let remaining = cap - self.produced;
                if envs.len() > remaining {
                    envs.truncate(remaining);
                }
            }
            self.produced += envs.len();
            Some(Ok(Batch::from_envs(envs)))
        }
    }
}
