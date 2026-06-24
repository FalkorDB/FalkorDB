//! Batch-mode node-by-ID-seek operator — retrieves nodes by internal ID.
//!
//! Implements optimizer-generated plans for `WHERE id(n) = expr` or
//! `WHERE id(n) IN [...]`. For each active row in the input batch, evaluates
//! the ID filter expression to produce a candidate `RoaringTreemap` of node IDs,
//! removes deleted nodes, and yields non-deleted nodes matching the range.

use std::sync::Arc;

use crate::graph::graph::NodeId;
use crate::parser::ast::{ExprIR, QueryExpr, QueryNode, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    runtime::Runtime,
};
use orx_tree::{Dyn, NodeIdx};

use super::batched_result_emitter::BatchedResultEmitter;

pub struct NodeByIdSeekOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    filter: &'a Vec<(QueryExpr<Variable>, ExprIR<Variable>)>,
    /// Holds the parent batch being expanded and the per-row id iterators, and
    /// performs the shared pack-and-gather emit.
    emitter: BatchedResultEmitter<'a, NodeId>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> NodeByIdSeekOp<'a> {
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
            filter,
            emitter: BatchedResultEmitter::new(node_pattern.alias.id),
            idx,
        }
    }
}

impl<'a> Iterator for NodeByIdSeekOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Refill the pending ranges from the child when we've run dry. For
            // each active parent row, evaluate the id filter into a candidate
            // range, drop deleted nodes, and queue the range as this row's
            // node-id iterator.
            if self.emitter.needs_refill() {
                match self.child.next() {
                    Some(Ok(batch)) => {
                        for row in batch.active_indices() {
                            let view = BatchRow::new(&batch, row);
                            match self.runtime.evaluate_id_filter(self.filter, &view) {
                                Ok(Some(mut range)) => {
                                    // Remove all deleted nodes at once.
                                    range -= self.runtime.g.borrow().deleted_nodes();
                                    if !range.is_empty() {
                                        self.emitter.push(
                                            row,
                                            Box::new(range.into_iter().map(NodeId::from)),
                                        );
                                    }
                                }
                                Ok(None) => {}
                                Err(e) => return Some(Err(e)),
                            }
                        }
                        self.emitter.set_batch(batch);
                        continue;
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None,
                }
            }

            if let Some(out) = self.emitter.emit() {
                return Some(Ok(out));
            }
        }
    }
}
