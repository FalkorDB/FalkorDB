//! Batch-mode expand-into operator — checks for relationships between
//! two already-bound nodes.
//!
//! Unlike `CondTraverse` (which scans all relationships for a label pair),
//! `ExpandInto` only checks edges between two specific already-bound endpoints.
//! This is used when both sides of a relationship pattern have been resolved
//! by prior operators.
//!
//! ```text
//!  Input: row where from=Node(5), to=Node(7)
//!  ──refill──►  checks edges between 5→7 (and 7→5 if bidirectional)
//!              queues one matched-edge iterator per matching input row
//! ```
//!
//! Emission reuses the shared [`BatchedResultEmitter`]: each refill processes
//! exactly one child batch — queueing a per-input-row iterator over that row's
//! matched edge ids — and `emit` packs the queued rows into `≤ BATCH_SIZE`
//! gathered batches. The operator therefore never materialises every matching
//! edge of an entire input stream up front; it pulls one parent batch at a time
//! and yields as soon as a batch is ready.

use std::sync::Arc;

use crate::graph::graph::{Graph, RelationshipId};
use crate::parser::ast::{QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    pending::Pending,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx};

use super::batched_result_emitter::{BatchedResultEmitter, RowIter};

pub struct ExpandIntoOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    /// Shared incremental emitter: holds the current input batch plus a queue
    /// of per-input-row matched-edge-id iterators, and packs them into gathered
    /// output batches. For the synthetic label-check it binds no relationship
    /// column.
    emitter: BatchedResultEmitter<'a, RelationshipId>,
    /// Synthetic multi-label self-loop (`MATCH (a:A:B:C)` lowered to
    /// `LabelScan(:A) + ExpandInto` checking `:B:C`): no edge is bound, the op
    /// only verifies the remaining labels on the single endpoint.
    synthetic_label: bool,
    /// Whether to emit one row per edge (true) or collapse multi-edges into
    /// one row per (src, dst) pair (false). Set by the planner.
    emit_relationship: bool,
    /// True when a predicate on the edge sits above this operator, so the
    /// collapse to one representative edge per (src, dst) pair would test an
    /// arbitrary member of the group. See `IR::CondTraverse::edge_predicate`.
    edge_predicate: bool,
    /// Alias IDs of sibling relationship variables in the same MATCH clause.
    sibling_edges: &'a [u32],
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// Maximum number of records this operator should produce. Once reached,
    /// subsequent `next()` calls return `None`. Set by limit propagation.
    record_cap: Option<usize>,
    /// Number of records produced so far (tracked when `record_cap` is set).
    produced: usize,
    /// Persistent per-relationship-type edge-id iterators. Reused via `seek`
    /// to fetch edge IDs for a specific (src, dst) pair without allocating
    /// fresh GxB_Iterators per pair.
    ///
    /// Lazily initialized on first use rather than at construction: the
    /// relationship type may be created by a sibling Commit earlier in the
    /// same query, so capturing matrices in `new()` would miss them.
    edge_type_indices: std::cell::RefCell<Option<Vec<usize>>>,
}

impl<'a> ExpandIntoOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        emit_relationship: bool,
        edge_predicate: bool,
        sibling_edges: &'a [u32],
        idx: NodeIdx<Dyn<IR>>,
        record_cap: Option<usize>,
    ) -> Self {
        // Synthetic multi-label self-loop (MATCH (a:A:B:C) lowered to
        // LabelScan(:A) + ExpandInto checking :B:C): no edge is bound, the op
        // only verifies the remaining labels on the single endpoint, so the
        // emitter binds no relationship column.
        let synthetic_label = relationship_pattern.from.alias.id
            == relationship_pattern.to.alias.id
            && relationship_pattern.from.labels.is_empty()
            && !relationship_pattern.to.labels.is_empty();
        let emitter: BatchedResultEmitter<'a, RelationshipId> = if synthetic_label {
            BatchedResultEmitter::new_without_alias()
        } else {
            BatchedResultEmitter::new(relationship_pattern.alias.id)
        };
        Self {
            runtime,
            child,
            relationship_pattern,
            emitter,
            synthetic_label,
            emit_relationship,
            edge_predicate,
            sibling_edges,
            idx,
            record_cap,
            produced: 0,
            edge_type_indices: std::cell::RefCell::new(None),
        }
    }

    /// Probe one input row's already-bound `from`/`to` endpoints for matching
    /// relationships and return them as a [`RowIter`] (`None` skips the row).
    /// The reverse direction is probed too when the pattern is bidirectional and
    /// not a self-loop. The matched edge ids are collected eagerly while the
    /// caller holds the graph borrow; the returned `RowIter` owns its ids, so
    /// nothing queued in the emitter borrows the graph.
    ///
    /// The synthetic label-check returns a single placeholder id — discarded by
    /// the no-alias filter emitter, which only gathers the row forward — when
    /// `from` carries all the required labels.
    #[allow(clippy::too_many_arguments)]
    fn expand_row(
        _runtime: &'a Runtime<'a>,
        rp: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        synthetic_label: bool,
        emit_relationship: bool,
        edge_predicate: bool,
        sibling_edges: &'a [u32],
        g: &Graph,
        pending: &Pending,
        iters_ref: &mut Option<Vec<usize>>,
        batch: &Batch<'a>,
        row_idx: usize,
    ) -> Result<Option<RowIter<'a, RelationshipId>>, String> {
        let src = match batch.value_at(rp.from.alias.id, row_idx) {
            Some(Value::Node(id)) => id,
            Some(Value::Null) | None => return Ok(None),
            _ => {
                return Err(String::from(
                    "Invalid node id for 'from' in relationship pattern",
                ));
            }
        };
        let dst = match batch.value_at(rp.to.alias.id, row_idx) {
            Some(Value::Node(id)) => id,
            Some(Value::Null) | None => return Ok(None),
            _ => {
                return Err(String::from(
                    "Invalid node id for 'to' in relationship pattern",
                ));
            }
        };

        if synthetic_label {
            let has_all_labels = rp
                .to
                .labels
                .iter()
                .all(|label| g.get_node_labels(src).any(|nl| nl == *label));
            return if has_all_labels {
                Ok(Some(RowIter::one(RelationshipId::from(0u64))))
            } else {
                Ok(None)
            };
        }

        let env = BatchRow::new(batch, row_idx);
        // The predicate itself is the Filter the planner put above this
        // operator; the flag only says parallel edges are distinguishable, so
        // this path must yield every candidate rather than one per pair.
        let has_edge_filter = edge_predicate;

        // Edge directions to probe: forward, plus reverse when the pattern is
        // bidirectional and not a self-loop. NodeId is Copy, so this fixed array
        // is stack-only.
        let pairs = [(src, dst), (dst, src)];
        let npairs = if rp.bidirectional && src != dst { 2 } else { 1 };

        let edge_type_indices = iters_ref.get_or_insert_with(|| {
            if rp.types.is_empty() {
                (0..g.relationship_tensors().len()).collect()
            } else {
                rp.types
                    .iter()
                    .filter_map(|t| g.get_type_id(t).map(|tid| tid.0))
                    .collect()
            }
        });

        // Matched edge ids for this input row, drained eagerly while the caller
        // holds the graph borrow (the tensor edge lookups are consumed here, never
        // parked in the emitter).
        let mut row_edges: Vec<RelationshipId> = Vec::new();
        for &(edge_src, edge_dst) in &pairs[..npairs] {
            let mat_src = u64::from(edge_src);
            let mat_dst = u64::from(edge_dst);
            if !emit_relationship && !has_edge_filter {
                // One representative edge per (src, dst) pair.
                let mut found_id: Option<RelationshipId> = None;
                'outer: for &tidx in edge_type_indices.iter() {
                    for raw_id in g.relationship_tensors()[tidx].get(mat_src, mat_dst) {
                        let id = RelationshipId::from(raw_id);
                        if !pending.is_relationship_deleted(id)
                            && !super::edge_already_used(&env, id, rp.alias.id, sibling_edges)
                        {
                            found_id = Some(id);
                            break 'outer;
                        }
                    }
                }
                if let Some(id) = found_id {
                    row_edges.push(id);
                }
                continue;
            }
            // One row per matching edge.
            for &tidx in edge_type_indices.iter() {
                for raw_id in g.relationship_tensors()[tidx].get(mat_src, mat_dst) {
                    let id = RelationshipId::from(raw_id);
                    if pending.is_relationship_deleted(id) {
                        continue;
                    }
                    if super::edge_already_used(&env, id, rp.alias.id, sibling_edges) {
                        continue;
                    }
                    row_edges.push(id);
                }
            }
        }
        // A single matched edge (the collapsed/anonymous common case) is queued
        // inline without a boxed-iterator allocation; several matched edges box an
        // iterator.
        match row_edges.len() {
            0 => Ok(None),
            1 => Ok(Some(RowIter::one(row_edges[0]))),
            _ => Ok(Some(RowIter::many(Box::new(row_edges.into_iter())))),
        }
    }
}

impl<'a> Iterator for ExpandIntoOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Stop once the record cap is reached.
        if let Some(cap) = self.record_cap
            && self.produced >= cap
        {
            return None;
        }

        let runtime = self.runtime;
        let rp = self.relationship_pattern;
        let synthetic_label = self.synthetic_label;
        let emit_relationship = self.emit_relationship;
        let edge_predicate = self.edge_predicate;
        let sibling_edges = self.sibling_edges;

        loop {
            // Pack one output batch, probing each active input row's endpoints on
            // demand. The graph/pending read borrows and the cached edge iterators
            // are held only for this pack and captured by the closure as locals, so
            // `self.emitter` stays exclusively borrowed by `emit_lazy` while nothing
            // queued in the emitter borrows the graph. The borrows drop at the end
            // of the block, before the refill arm pulls the next child batch.
            let result = {
                let g = runtime.g.borrow();
                let pending = runtime.pending.borrow();
                let mut iters_ref = self.edge_type_indices.borrow_mut();
                self.emitter.emit_lazy(|batch, row_idx| {
                    Self::expand_row(
                        runtime,
                        rp,
                        synthetic_label,
                        emit_relationship,
                        edge_predicate,
                        sibling_edges,
                        &g,
                        &pending,
                        &mut iters_ref,
                        batch,
                        row_idx,
                    )
                })
            };

            match result {
                Ok(Some(mut out)) => {
                    // Trim the final batch to the record cap if set.
                    if let Some(cap) = self.record_cap {
                        let remaining = cap - self.produced;
                        if out.active_len() > remaining {
                            out.set_selection((0..remaining as u16).collect());
                        }
                    }
                    self.produced += out.active_len();
                    return Some(Ok(out));
                }
                Ok(None) => match self.child.next() {
                    Some(Ok(batch)) => self.emitter.seed(batch),
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None,
                },
                Err(e) => return Some(Err(e)),
            }
        }
    }
}
