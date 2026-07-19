//! Batch-mode fulltext scan operator — retrieves nodes via a fulltext index query.
//!
//! Implements `CALL db.idx.fulltext.queryNodes(label, query)`. For each active
//! parent row, evaluates the label and query expressions, delegates to the
//! graph's fulltext index, and queues the matching `(node, score)` iterator on
//! the shared [`BatchedResultEmitter`], which packs up to
//! [`BATCH_SIZE`](super::super::batch::BATCH_SIZE) results into one columnar
//! batch. The matched node binds to the node variable and, when a score yield
//! variable is present, the relevance score binds to it as a float column.
//!
//! A downstream `Skip`/`Limit` lowers the emitter's pack ceiling (via
//! `record_cap`), so `CALL db.idx.fulltext.queryNodes(..) ... LIMIT k` drains
//! only about `k` results from the RediSearch iterator instead of eagerly
//! packing a whole `BATCH_SIZE` worth of matches per call.

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

use super::batched_result_emitter::{BatchedResultEmitter, RowIter, ScoredColumn};

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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        node: &'a Variable,
        label: &'a QueryExpr<Variable>,
        query: &'a QueryExpr<Variable>,
        score: &'a Option<Variable>,
        record_cap: Option<usize>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let mut emitter = BatchedResultEmitter::with_binding(ScoredColumn {
            id: node.id,
            score: score.as_ref().map(|v| v.id),
        });
        // A downstream Skip/Limit lowers how many rows are needed; shrink the
        // pack ceiling so the first emit drains just enough index results
        // instead of a full BATCH_SIZE worth of work.
        emitter.apply_record_cap(record_cap);
        Self {
            runtime,
            child,
            emitter,
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
            // One fulltext-query iterator per active parent row, built lazily one
            // row at a time. When the batch is exhausted (`Ok(None)`), pull and
            // seed the next child batch.
            match self.emitter.emit_lazy(|b, row| {
                let view = BatchRow::new(b, row);
                let Value::String(label_str) = ExprEval::from_runtime(self.runtime).eval(
                    self.label,
                    self.label.root().idx(),
                    Some(&view),
                    None,
                )?
                else {
                    return Err("fulltext query expects a string label".into());
                };
                let Value::String(query_str) = ExprEval::from_runtime(self.runtime).eval(
                    self.query,
                    self.query.root().idx(),
                    Some(&view),
                    None,
                )?
                else {
                    return Err("fulltext query expects a string query".into());
                };
                let g = self.runtime.g.borrow();
                let iter = Box::new(g.fulltext_query_nodes(&label_str, &query_str)?)
                    as Box<dyn Iterator<Item = (NodeId, f64)>>;
                Ok(Some(RowIter::many(iter)))
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
