//! Batch-mode fulltext scan operator — retrieves nodes via a fulltext index query.
//!
//! Implements `CALL db.idx.fulltext.queryNodes(label, query)`. For each active
//! parent row, evaluates the label and query expressions, delegates to the
//! graph's fulltext index, and queues the matching `(node, score)` iterator on
//! the shared [`BatchedResultEmitter`], which packs up to
//! [`BATCH_SIZE`](super::super::batch::BATCH_SIZE) results into one columnar
//! batch. The matched node binds to the node variable and, when a score yield
//! variable is present, the relevance score binds to it as a float column.

use crate::graph::graph::NodeId;
use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

use super::batched_result_emitter::{BatchedResultEmitter, ScoredColumn};

pub struct NodeByFulltextScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    /// Holds the parent batch being expanded and the per-row `(node, score)`
    /// iterators, and performs the shared pack-and-gather emit.
    pub(crate) emitter: BatchedResultEmitter<'a, (NodeId, f64)>,
    label: &'a QueryExpr<Variable>,
    query: &'a QueryExpr<Variable>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> NodeByFulltextScanOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        node: &'a Variable,
        label: &'a QueryExpr<Variable>,
        query: &'a QueryExpr<Variable>,
        score: &'a Option<Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            emitter: BatchedResultEmitter::with_binding(ScoredColumn {
                id: node.id,
                score: score.as_ref().map(|v| v.id),
            }),
            label,
            query,
            idx,
        }
    }
}

impl<'a> Iterator for NodeByFulltextScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Refill the per-row scans from the child when we've run dry: queue
            // one fulltext-query iterator per active parent row.
            if self.emitter.needs_refill() {
                match self.child.next() {
                    Some(Ok(batch)) => {
                        for row in batch.active_indices() {
                            let view = BatchRow::new(&batch, row);
                            let label_str = match ExprEval::from_runtime(self.runtime).eval(
                                self.label,
                                self.label.root().idx(),
                                Some(&view),
                                None,
                            ) {
                                Ok(Value::String(s)) => s,
                                Ok(_) => {
                                    return Some(Err(
                                        "fulltext query expects a string label".into()
                                    ));
                                }
                                Err(e) => return Some(Err(e)),
                            };
                            let query_str = match ExprEval::from_runtime(self.runtime).eval(
                                self.query,
                                self.query.root().idx(),
                                Some(&view),
                                None,
                            ) {
                                Ok(Value::String(s)) => s,
                                Ok(_) => {
                                    return Some(Err(
                                        "fulltext query expects a string query".into()
                                    ));
                                }
                                Err(e) => return Some(Err(e)),
                            };
                            let g = self.runtime.g.borrow();
                            let iter = match g.fulltext_query_nodes(&label_str, &query_str) {
                                Ok(iter) => {
                                    Box::new(iter) as Box<dyn Iterator<Item = (NodeId, f64)>>
                                }
                                Err(e) => return Some(Err(e)),
                            };
                            drop(g);
                            self.emitter.push(row, iter);
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
