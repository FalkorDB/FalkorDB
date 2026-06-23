//! Batch-mode label-and-ID scan operator — retrieves nodes by label filtered by ID range.
//!
//! Combines a label scan with an ID range constraint from the optimizer.
//! For each active row in each input batch, evaluates the ID filter to
//! determine the candidate range, scans nodes with the given label starting
//! from the minimum candidate ID, and yields those whose ID falls within
//! the evaluated filter range.

use std::collections::VecDeque;
use std::sync::Arc;

use roaring::RoaringTreemap;

use crate::graph::graph::NodeId;
use crate::parser::ast::{ExprIR, QueryExpr, QueryNode, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    row::{Row, RowView},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx};

pub struct NodeByLabelAndIdScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    pending: VecDeque<(Row, Box<dyn Iterator<Item = NodeId>>, RoaringTreemap)>,
    node_pattern: &'a QueryNode<Arc<String>, Variable>,
    filter: &'a Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> NodeByLabelAndIdScanOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        node_pattern: &'a QueryNode<Arc<String>, Variable>,
        filter: &'a Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            pending: VecDeque::new(),
            node_pattern,
            filter,
            idx,
        }
    }

    /// Drains rows from `self.pending` into `envs` until `BATCH_SIZE` is reached
    /// or all pending scans are exhausted.
    fn drain_pending(
        &mut self,
        builder: &mut BatchBuilder,
    ) {
        while builder.len() < BATCH_SIZE {
            let Some((env, iter, range)) = self.pending.front_mut() else {
                break;
            };
            let Some(max) = range.max() else {
                self.pending.pop_front();
                continue;
            };
            let mut found = false;
            for nid in iter.by_ref() {
                let id = u64::from(nid);
                if id > max {
                    break;
                }
                if range.contains(id) {
                    let mut row = env.clone();
                    row.insert(&self.node_pattern.alias, Value::Node(nid));
                    builder.push_row(&row);
                    found = true;
                    if builder.len() >= BATCH_SIZE {
                        break;
                    }
                }
            }
            if !found {
                self.pending.pop_front();
            }
        }
    }

    /// Drains label-and-range-matching node IDs from leading no-binding pending
    /// scans into a flat `Vec<NodeId>`, mirroring `NodeByLabelScan`'s
    /// `from_node_ids` path. Valid only when the parent row carries no bindings,
    /// so every output row is just `Value::Node(id)`. Applies the same
    /// `range.contains` / `range.max()` filtering as the row path. Stops at
    /// `BATCH_SIZE`, a binding-carrying env, or exhaustion.
    fn drain_pending_columnar(
        &mut self,
        ids: &mut Vec<NodeId>,
    ) {
        while ids.len() < BATCH_SIZE {
            let Some((env, iter, range)) = self.pending.front_mut() else {
                break;
            };
            if env.has_bindings() || env.origin_row != 0 {
                break;
            }
            let Some(max) = range.max() else {
                self.pending.pop_front();
                continue;
            };
            let mut found = false;
            for nid in iter.by_ref() {
                let id = u64::from(nid);
                if id > max {
                    break;
                }
                if range.contains(id) {
                    ids.push(nid);
                    found = true;
                    if ids.len() >= BATCH_SIZE {
                        break;
                    }
                }
            }
            if !found {
                self.pending.pop_front();
            }
        }
    }
}

impl<'a> Iterator for NodeByLabelAndIdScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let alias_id = self.node_pattern.alias.id;

        loop {
            // Refill pending scans from the child when we've run dry.
            if self.pending.is_empty() {
                match self.child.next() {
                    Some(Ok(batch)) => {
                        for row in batch.active_indices() {
                            let view = BatchRow::new(&batch, row);
                            match self.runtime.evaluate_id_filter(self.filter, &view) {
                                Ok(Some(range)) => {
                                    if range.min().is_some() {
                                        let iter = self.runtime.g.borrow().get_nodes(
                                            &self.node_pattern.labels,
                                            range.min().unwrap(),
                                        );
                                        self.pending.push_back((view.to_owned_row(), iter, range));
                                    }
                                }
                                Ok(None) => {}
                                Err(e) => return Some(Err(e)),
                            }
                        }
                        continue;
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None,
                }
            }

            // Dispatch on whether the front parent row qualifies for the
            // columnar fast path. It requires no bindings AND a default
            // correlation origin: `Batch::from_node_ids` emits `origin_rows:
            // None`, so a parent carrying a non-zero `origin_row` must take the
            // row path to preserve its correlation lineage.
            let (env, ..) = self.pending.front().expect("pending is non-empty");
            if !env.has_bindings() && env.origin_row == 0 {
                // Columnar fast path: emit a `Column::NodeIds` batch directly,
                // skipping per-row `Row` construction and the `BatchBuilder`
                // transpose.
                let mut ids: Vec<NodeId> = Vec::with_capacity(BATCH_SIZE);
                self.drain_pending_columnar(&mut ids);
                if !ids.is_empty() {
                    return Some(Ok(Batch::from_node_ids(alias_id, ids)));
                }
            } else {
                // Row path: parent bindings or origin present, so build one env
                // per node.
                let mut builder = BatchBuilder::new();
                self.drain_pending(&mut builder);
                if !builder.is_empty() {
                    return Some(Ok(builder.finish()));
                }
            }
        }
    }
}
