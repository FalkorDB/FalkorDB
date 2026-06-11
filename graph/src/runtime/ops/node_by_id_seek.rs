//! Batch-mode node-by-ID-seek operator — retrieves nodes by internal ID.
//!
//! Implements optimizer-generated plans for `WHERE id(n) = expr` or
//! `WHERE id(n) IN [...]`. For each active row in the input batch, evaluates
//! the ID filter expression to produce a candidate `RoaringTreemap` of node IDs,
//! removes deleted nodes, and yields non-deleted nodes matching the range.

use std::collections::VecDeque;
use std::sync::Arc;

use roaring::treemap::IntoIter as RoaringIntoIter;

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

pub struct NodeByIdSeekOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    node_pattern: &'a QueryNode<Arc<String>, Variable>,
    filter: &'a Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
    pending: VecDeque<(Row, RoaringIntoIter)>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> NodeByIdSeekOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        node_pattern: &'a QueryNode<Arc<String>, Variable>,
        filter: &'a Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            node_pattern,
            filter,
            pending: VecDeque::new(),
            idx,
        }
    }

    /// Drains rows from `self.pending` into `envs` until `BATCH_SIZE` is reached
    /// or all pending ranges are exhausted.
    fn drain_pending(
        &mut self,
        builder: &mut BatchBuilder,
    ) {
        while builder.len() < BATCH_SIZE {
            let Some((env, iter)) = self.pending.front_mut() else {
                break;
            };
            if let Some(nid) = iter.next() {
                let mut row = env.clone();
                row.insert(&self.node_pattern.alias, Value::Node(NodeId::from(nid)));
                builder.push_row(&row);
            } else {
                self.pending.pop_front();
            }
        }
    }
}

impl<'a> Iterator for NodeByIdSeekOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut builder = BatchBuilder::new();

        // Drain leftover ranges from previous call.
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
                    Ok(Some(mut range)) => {
                        // Remove all deleted nodes at once.
                        range -= self.runtime.g.borrow().deleted_nodes();
                        if !range.is_empty() {
                            self.pending.push_back((
                                BatchRow::new(&batch, row).to_owned_row(),
                                range.into_iter(),
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
