//! Batch-mode label-and-ID scan operator — retrieves nodes by label filtered by ID range.
//!
//! Combines a label scan with an ID range constraint from the optimizer.
//! For each active row in each input batch, evaluates the ID filter to
//! determine the candidate range, scans nodes with the given label starting
//! from the minimum candidate ID, and yields those whose ID falls within
//! the evaluated filter range.

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

pub struct NodeByLabelAndIdScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    /// Holds the parent batch being expanded and the per-row id iterators, and
    /// performs the shared pack-and-gather emit.
    pub(crate) emitter: BatchedResultEmitter<'a, NodeId>,
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
            emitter: BatchedResultEmitter::new(node_pattern.alias.id),
            node_pattern,
            filter,
            idx,
        }
    }
}

impl<'a> Iterator for NodeByLabelAndIdScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Refill the pending scans from the child when we've run dry. For
            // each active parent row, evaluate the id filter to a candidate
            // range and queue a label scan from the range minimum, folding the
            // `id <= max` cutoff and `range.contains` membership into the
            // iterator so the shared emit stays generic.
            if self.emitter.needs_refill() {
                match self.child.next() {
                    Some(Ok(batch)) => {
                        for row in batch.active_indices() {
                            let view = BatchRow::new(&batch, row);
                            match self.runtime.evaluate_id_filter(self.filter, &view) {
                                Ok(Some(range)) => {
                                    if let Some(min) = range.min() {
                                        let max =
                                            range.max().expect("range has a min, so it has a max");
                                        let iter = self
                                            .runtime
                                            .g
                                            .borrow()
                                            .get_nodes(&self.node_pattern.labels, min)
                                            .take_while(move |nid| u64::from(*nid) <= max)
                                            .filter(move |nid| range.contains(u64::from(*nid)));
                                        self.emitter.push(row, Box::new(iter));
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
