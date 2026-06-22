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
//!  ──expand_batch──►  checks edges between 5→7 (and 7→5 if bidirectional)
//!                     emits one output row per matching edge
//! ```
//!
//! Uses the same `pending_batches` / `current_batch` / `current_pos` state
//! machine as [`CondTraverseOp`] for buffered batch emission.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::graph::graph::RelationshipId;
use crate::graph::graphblas::tensor::compound_key;
use crate::graph::graphblas::versioned_matrix::Iter as EdgeIter;
use crate::parser::ast::{QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::batch::Column;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct ExpandIntoOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    pending_batches: VecDeque<Batch<'a>>,
    current_batch: Option<Batch<'a>>,
    current_pos: usize,
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
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        emit_relationship: bool,
        sibling_edges: &'a [u32],
        idx: NodeIdx<Dyn<IR>>,
        record_cap: Option<usize>,
    ) -> Self {
        Self {
            runtime,
            child,
            relationship_pattern,
            pending_batches: VecDeque::new(),
            current_batch: None,
            current_pos: 0,
            emit_relationship,
            sibling_edges,
            idx,
            record_cap,
            produced: 0,
            edge_iters: std::cell::RefCell::new(None),
        }
    }

    /// Columnar expansion: checks edges between the already-bound `from`/`to`
    /// endpoints of every row in `active_subset` using a borrowed [`BatchRow`]
    /// view (zero owned-`Row` allocation), accumulates the matched input row
    /// indices plus their edge ids, and emits gathered [`Batch`]es into
    /// `out_pending`. The input columns are carried forward verbatim via
    /// [`Batch::gather`]; the relationship variable is attached as a single
    /// [`Column::RelIds`]. A row that produces N matching edges pushes its
    /// index N times so `gather` replicates the carried-forward bindings.
    fn expand_batch(
        &self,
        batch: &Batch<'a>,
        active_subset: &[usize],
        out_pending: &mut VecDeque<Batch<'a>>,
    ) -> Result<(), String> {
        let runtime = self.runtime;
        let rp = self.relationship_pattern;

        // Synthetic multi-label self-loop (MATCH (a:A:B:C) lowered to
        // LabelScan(:A) + ExpandInto checking :B:C): no edge is bound, the op
        // only verifies the remaining labels on the single endpoint.
        let synthetic_label = rp.from.alias.id == rp.to.alias.id
            && rp.from.labels.is_empty()
            && !rp.to.labels.is_empty();

        let mut out_indices: Vec<usize> = Vec::new();
        let mut out_edge_ids: Vec<RelationshipId> = Vec::new();

        let g = runtime.g.borrow();
        let pending = runtime.pending.borrow();
        let mut iters_ref = self.edge_iters.borrow_mut();

        // Flush the accumulated rows into a gathered batch. For non-synthetic
        // ops the edge column is attached; the synthetic label-check op emits
        // no relationship binding.
        macro_rules! flush {
            () => {
                if !out_indices.is_empty() {
                    let mut b = batch.gather(&out_indices);
                    if !synthetic_label {
                        b.set_column(
                            rp.alias.id,
                            Column::RelIds(std::mem::take(&mut out_edge_ids)),
                        );
                    }
                    out_pending.push_back(b);
                    out_indices.clear();
                }
            };
        }

        for &row_idx in active_subset {
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
                    out_indices.push(row_idx);
                    if out_indices.len() >= BATCH_SIZE {
                        flush!();
                    }
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

            for &(edge_src, edge_dst) in &pairs[..npairs] {
                let key = compound_key(u64::from(edge_src), u64::from(edge_dst));
                if !self.emit_relationship && !has_edge_filter {
                    // One representative edge per (src, dst) pair.
                    let mut found_id: Option<RelationshipId> = None;
                    'outer: for cell in edge_iters.iter() {
                        let mut it = cell.borrow_mut();
                        it.seek(key, key);
                        for (_, raw_id) in &mut *it {
                            let id = RelationshipId::from(raw_id);
                            if !pending.is_relationship_deleted(id)
                                && !super::edge_already_used(
                                    &env,
                                    id,
                                    rp.alias.id,
                                    self.sibling_edges,
                                )
                            {
                                found_id = Some(id);
                                break 'outer;
                            }
                        }
                    }
                    if let Some(id) = found_id {
                        out_indices.push(row_idx);
                        out_edge_ids.push(id);
                        if out_indices.len() >= BATCH_SIZE {
                            flush!();
                        }
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
                        if super::edge_already_used(&env, id, rp.alias.id, self.sibling_edges) {
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
                        out_indices.push(row_idx);
                        out_edge_ids.push(id);
                        if out_indices.len() >= BATCH_SIZE {
                            flush!();
                        }
                    }
                }
            }
        }

        flush!();
        drop(g);
        drop(pending);
        Ok(())
    }
}

impl<'a> Iterator for ExpandIntoOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if record_cap already reached.
        if let Some(cap) = self.record_cap
            && self.produced >= cap
        {
            return None;
        }

        let mut builder = BatchBuilder::new();

        // Drain leftover batches from a previous call.
        super::drain_pending_batches(&mut self.pending_batches, &mut builder);

        loop {
            if builder.len() >= BATCH_SIZE {
                break;
            }

            if self.current_batch.is_none() {
                match self.child.next() {
                    Some(Ok(b)) => {
                        self.current_batch = Some(b);
                        self.current_pos = 0;
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => break,
                }
            }

            {
                let batch = self.current_batch.as_ref().unwrap();
                let active: Vec<usize> = batch.active_indices().collect();

                if self.current_pos < active.len() {
                    let active_subset = &active[self.current_pos..];
                    let mut pending = std::mem::take(&mut self.pending_batches);
                    let result = self.expand_batch(batch, active_subset, &mut pending);
                    self.pending_batches = pending;
                    if let Err(e) = result {
                        return Some(Err(e));
                    }
                    self.current_pos = active.len();
                }
            }

            super::drain_pending_batches(&mut self.pending_batches, &mut builder);

            // Check if batch is exhausted.
            if let Some(ref batch) = self.current_batch
                && self.current_pos >= batch.active_len()
            {
                self.current_batch = None;
            }
        }

        if builder.is_empty() {
            None
        } else {
            // Trim to record_cap if set.
            if let Some(cap) = self.record_cap {
                let remaining = cap - self.produced;
                if builder.len() > remaining {
                    builder.truncate(remaining);
                }
            }
            self.produced += builder.len();
            Some(Ok(builder.finish()))
        }
    }
}
