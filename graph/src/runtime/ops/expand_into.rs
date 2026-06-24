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

use crate::graph::graph::RelationshipId;
use crate::graph::graphblas::tensor::compound_key;
use crate::graph::graphblas::versioned_matrix::Iter as EdgeIter;
use crate::parser::ast::{QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

use super::batched_result_emitter::BatchedResultEmitter;

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
    edge_iters: std::cell::RefCell<Option<Vec<std::cell::RefCell<EdgeIter>>>>,
}

impl<'a> ExpandIntoOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        emit_relationship: bool,
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
            sibling_edges,
            idx,
            record_cap,
            produced: 0,
            edge_iters: std::cell::RefCell::new(None),
        }
    }

    /// Pull the edge matches for one input `batch` into the emitter. For each
    /// active input row, probes the edges between the already-bound `from`/`to`
    /// endpoints (reverse too when bidirectional) using a borrowed [`BatchRow`]
    /// view (zero owned-`Row` allocation), collects the matched edge ids — bounded
    /// by the parallel edges between two specific endpoints, which is small — and
    /// queues a single per-row iterator over them. Only matching rows are queued,
    /// so the dominant non-matching probes on a cycle closure add nothing.
    ///
    /// The synthetic label-check queues one placeholder per row whose `from`
    /// carries all the required labels; the emitter (a filter emitter) gathers
    /// that input row forward without binding any relationship column.
    fn refill(
        &mut self,
        batch: &Batch<'a>,
    ) -> Result<(), String> {
        let runtime = self.runtime;
        let rp = self.relationship_pattern;
        let synthetic_label = self.synthetic_label;
        let emit_relationship = self.emit_relationship;
        let sibling_edges = self.sibling_edges;

        let g = runtime.g.borrow();
        let pending = runtime.pending.borrow();
        let mut iters_ref = self.edge_iters.borrow_mut();

        for row_idx in batch.active_indices() {
            let src = match batch.value_at(rp.from.alias.id, row_idx) {
                Some(Value::Node(id)) => id,
                Some(Value::Null) | None => continue,
                _ => {
                    return Err(String::from(
                        "Invalid node id for 'from' in relationship pattern",
                    ));
                }
            };
            let dst = match batch.value_at(rp.to.alias.id, row_idx) {
                Some(Value::Node(id)) => id,
                Some(Value::Null) | None => continue,
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
                if has_all_labels {
                    // The filter emitter binds no relationship; this placeholder
                    // id is discarded and only makes `emit` gather exactly one
                    // row forward for this matching input row.
                    self.emitter.push_one(row_idx, RelationshipId::from(0u64));
                }
                continue;
            }

            let env = BatchRow::new(batch, row_idx);
            let filter_attrs = ExprEval::from_runtime(runtime).eval(
                &rp.attrs,
                rp.attrs.root().idx(),
                Some(&env),
                None,
            )?;
            let has_edge_filter = matches!(filter_attrs, Value::Map(ref m) if !m.is_empty());

            // Edge directions to probe: forward, plus reverse when the pattern
            // is bidirectional and not a self-loop. NodeId is Copy, so this
            // fixed array is stack-only.
            let pairs = [(src, dst), (dst, src)];
            let npairs = if rp.bidirectional && src != dst { 2 } else { 1 };

            let edge_iters = iters_ref.get_or_insert_with(|| {
                if rp.types.is_empty() {
                    g.relationship_matrices_iter()
                        .map(|tensor| std::cell::RefCell::new(tensor.edge_iter(0, u64::MAX)))
                        .collect()
                } else {
                    rp.types
                        .iter()
                        .filter_map(|t| g.get_relationship_matrix(t))
                        .map(|tensor| std::cell::RefCell::new(tensor.edge_iter(0, u64::MAX)))
                        .collect()
                }
            });

            // Matched edge ids for this input row, collected eagerly within the
            // graph borrow scope (the lazy GraphBLAS iterators can't outlive it).
            let mut row_edges: Vec<RelationshipId> = Vec::new();
            for &(edge_src, edge_dst) in &pairs[..npairs] {
                let key = compound_key(u64::from(edge_src), u64::from(edge_dst));
                if !emit_relationship && !has_edge_filter {
                    // One representative edge per (src, dst) pair.
                    let mut found_id: Option<RelationshipId> = None;
                    'outer: for cell in edge_iters.iter() {
                        let mut it = cell.borrow_mut();
                        it.seek(key, key);
                        for (_, raw_id) in &mut *it {
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
                for cell in edge_iters.iter() {
                    let mut it = cell.borrow_mut();
                    it.seek(key, key);
                    for (_, raw_id) in &mut *it {
                        let id = RelationshipId::from(raw_id);
                        if pending.is_relationship_deleted(id) {
                            continue;
                        }
                        if super::edge_already_used(&env, id, rp.alias.id, sibling_edges) {
                            continue;
                        }
                        if let Value::Map(ref filter_map) = filter_attrs
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
                        row_edges.push(id);
                    }
                }
            }
            // One queued row per matching input row: a single matched edge
            // (the collapsed/anonymous common case) is stored inline without a
            // boxed-iterator allocation; several matched edges box an iterator.
            match row_edges.len() {
                0 => {}
                1 => self.emitter.push_one(row_idx, row_edges[0]),
                _ => self.emitter.push(row_idx, Box::new(row_edges.into_iter())),
            }
        }
        drop(g);
        drop(pending);
        drop(iters_ref);
        Ok(())
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

        loop {
            // Refill from one child batch at a time — never buffer the whole
            // input stream up front.
            if self.emitter.needs_refill() {
                match self.child.next() {
                    Some(Ok(batch)) => {
                        if let Err(e) = self.refill(&batch) {
                            return Some(Err(e));
                        }
                        self.emitter.set_batch(batch);
                        continue;
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None,
                }
            }

            let Some(mut out) = self.emitter.emit() else {
                continue;
            };

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
    }
}
