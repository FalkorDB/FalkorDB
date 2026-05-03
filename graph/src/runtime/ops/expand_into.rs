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
//!  ──expand_row──►  checks edges between 5→7 (and 7→5 if bidirectional)
//!                   emits one output row per matching edge
//! ```
//!
//! Uses the same `pending` / `current_batch` / `current_pos` state machine
//! as [`CondTraverseOp`] for buffered batch emission.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::graph::graph::RelationshipId;
use crate::graph::graphblas::tensor::compound_key;
use crate::graph::graphblas::versioned_matrix::Iter as EdgeIter;
use crate::parser::ast::{QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct ExpandIntoOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    pending: VecDeque<Env<'a>>,
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
    pub fn new(
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
            pending: VecDeque::new(),
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

    fn expand_row(
        &self,
        env: &Env<'a>,
        out: &mut Vec<Env<'a>>,
    ) -> Result<(), String> {
        let runtime = self.runtime;
        let rp = self.relationship_pattern;

        let src = match env.get(&rp.from.alias) {
            Some(Value::Node(id)) => *id,
            Some(Value::Null) | None => return Ok(()),
            _ => {
                return Err(String::from(
                    "Invalid node id for 'from' in relationship pattern",
                ));
            }
        };
        let dst = match env.get(&rp.to.alias) {
            Some(Value::Node(id)) => *id,
            Some(Value::Null) | None => return Ok(()),
            _ => {
                return Err(String::from(
                    "Invalid node id for 'to' in relationship pattern",
                ));
            }
        };

        let filter_attrs = ExprEval::from_runtime(runtime).eval(
            &rp.attrs,
            rp.attrs.root().idx(),
            Some(env),
            None,
        )?;

        // Synthetic multi-label check: the planner splits MATCH (a:A:B:C) into
        // LabelScan(:A) + ExpandInto(self-loop) to verify remaining labels.
        // In that synthetic case from.labels is empty and to.labels holds the
        // remaining labels.  We must NOT take this shortcut for real self-loop
        // relationship patterns like MATCH (a:A)-[r]->(a) where from.labels
        // is non-empty, because those need actual edge matching.
        if rp.from.alias.id == rp.to.alias.id
            && rp.from.labels.is_empty()
            && !rp.to.labels.is_empty()
        {
            let g = runtime.g.borrow();
            let has_all_labels = rp
                .to
                .labels
                .iter()
                .all(|label| g.get_node_labels(src).any(|nl| nl == *label));
            if has_all_labels {
                let mut row = env.clone_pooled(runtime.env_pool);
                row.insert(&rp.from.alias, Value::Node(src));
                out.push(row);
            }
            return Ok(());
        }

        let mut edge_pairs = vec![(src, dst)];
        if rp.bidirectional && src != dst {
            edge_pairs.push((dst, src));
        }

        // When emit_relationship is false (anonymous edge not in a named
        // path) and there are no type or attribute filters, emit one row per
        // (src, dst) pair with the first matching edge.
        let has_edge_filter = matches!(filter_attrs, Value::Map(ref m) if !m.is_empty());

        let g = runtime.g.borrow();
        let pending = runtime.pending.borrow();
        let mut iters_ref = self.edge_iters.borrow_mut();
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
        for (edge_src, edge_dst) in &edge_pairs {
            let key = compound_key(u64::from(*edge_src), u64::from(*edge_dst));
            if !self.emit_relationship && !has_edge_filter {
                let mut found_id: Option<RelationshipId> = None;
                'outer: for cell in edge_iters.iter() {
                    let mut it = cell.borrow_mut();
                    it.seek(key, key);
                    for (_, raw_id) in &mut *it {
                        let id = RelationshipId::from(raw_id);
                        if !pending.is_relationship_deleted(id, *edge_src, *edge_dst)
                            && !super::edge_already_used(env, id, rp.alias.id, self.sibling_edges)
                        {
                            found_id = Some(id);
                            break 'outer;
                        }
                    }
                }
                if let Some(id) = found_id {
                    let mut row = env.clone_pooled(runtime.env_pool);
                    row.insert(
                        &rp.alias,
                        Value::Relationship(Box::new((id, *edge_src, *edge_dst))),
                    );
                    row.insert(&rp.from.alias, Value::Node(src));
                    row.insert(&rp.to.alias, Value::Node(dst));
                    out.push(row);
                }
                continue;
            }
            for cell in edge_iters.iter() {
                let mut it = cell.borrow_mut();
                it.seek(key, key);
                for (_, raw_id) in &mut *it {
                    let id = RelationshipId::from(raw_id);
                    if pending.is_relationship_deleted(id, *edge_src, *edge_dst) {
                        continue;
                    }
                    // Relationship uniqueness: skip edges already bound to other
                    // relationship variables in this MATCH clause.
                    if super::edge_already_used(env, id, rp.alias.id, self.sibling_edges) {
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
                    let mut row = env.clone_pooled(runtime.env_pool);
                    row.insert(
                        &rp.alias,
                        Value::Relationship(Box::new((id, *edge_src, *edge_dst))),
                    );
                    row.insert(&rp.from.alias, Value::Node(src));
                    row.insert(&rp.to.alias, Value::Node(dst));
                    out.push(row);
                }
            }
        }
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

        let mut envs = Vec::with_capacity(BATCH_SIZE);

        // Drain leftover rows from previous call.
        super::drain_pending(&mut self.pending, &mut envs);

        loop {
            if envs.len() >= BATCH_SIZE {
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
