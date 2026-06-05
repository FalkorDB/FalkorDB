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
}

impl<'a> Iterator for NodeByLabelAndIdScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut builder = BatchBuilder::new();

        // Drain leftover scans from previous call.
        self.drain_pending(&mut builder);

        while builder.len() < BATCH_SIZE {
            let batch = match self.child.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            };

            for row in batch.active_indices() {
                let view = BatchRow::new(&batch, row);
                match self.runtime.evaluate_id_filter(self.filter, &view) {
                    Ok(Some(range)) => {
                        if range.min().is_some() {
                            let iter = self
                                .runtime
                                .g
                                .borrow()
                                .get_nodes(&self.node_pattern.labels, range.min().unwrap());

                            self.pending.push_back((
                                BatchRow::new(&batch, row).to_owned_row(),
                                iter,
                                range,
                            ));
                        }
                    }
                    Ok(None) => {}
                    Err(e) => return Some(Err(e)),
                }
            }

            self.drain_pending(&mut builder);
        }

        if builder.is_empty() {
            None
        } else {
            Some(Ok(builder.finish()))
        }
    }
}
