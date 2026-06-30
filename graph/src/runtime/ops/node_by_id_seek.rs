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

use super::batched_result_emitter::{BatchedResultEmitter, RowResult};

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
            // For each active parent row, evaluate the id filter into a candidate
            // range, drop deleted nodes, and queue the range as this row's
            // node-id iterator — built lazily, one row at a time. When the batch
            // is exhausted (`Ok(None)`), pull and seed the next child batch.
            match self.emitter.emit_lazy(|b, row| {
                let view = BatchRow::new(b, row);
                let Some(mut range) = self.runtime.evaluate_id_filter(self.filter, &view)? else {
                    return Ok(None);
                };
                // Remove all deleted nodes at once.
                range -= self.runtime.g.borrow().deleted_nodes();
                if range.is_empty() {
                    return Ok(None);
                }
                Ok(Some(RowResult::many(Box::new(
                    range.into_iter().map(NodeId::from),
                ))))
            }) {
                Ok(Some(out)) => return Some(Ok(out)),
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
