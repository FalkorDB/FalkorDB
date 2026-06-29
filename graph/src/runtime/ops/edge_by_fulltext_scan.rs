//! Batch-mode edge fulltext scan operator — retrieves edges via a fulltext index query.
//!
//! Implements `CALL db.idx.fulltext.queryRelationships(type, query)`. For each
//! active parent row, evaluates the relationship-type and query expressions,
//! delegates to the graph's edge fulltext index, and queues the matching
//! `(edge, score)` iterator on the shared [`BatchedResultEmitter`], which packs
//! up to [`BATCH_SIZE`](super::super::batch::BATCH_SIZE) results into one
//! columnar batch.
//!
//! The matched edge binds to the relationship variable as a
//! `Value::Relationship`, carrying enough information for downstream property
//! lookups (e.g. `RETURN r.name`) without binding endpoint variables — the
//! procedure only yields `relationship` / `score`. When a score yield variable
//! is present, the relevance score binds to it as a float column.

use crate::graph::graph::RelationshipId;
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

pub struct EdgeByFulltextScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    /// Holds the parent batch being expanded and the per-row `(edge, score)`
    /// iterators, and performs the shared pack-and-gather emit.
    pub(crate) emitter: BatchedResultEmitter<'a, (RelationshipId, f64)>,
    label: &'a QueryExpr<Variable>,
    query: &'a QueryExpr<Variable>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> EdgeByFulltextScanOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        edge: &'a Variable,
        label: &'a QueryExpr<Variable>,
        query: &'a QueryExpr<Variable>,
        score: &'a Option<Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            emitter: BatchedResultEmitter::with_binding(ScoredColumn {
                id: edge.id,
                score: score.as_ref().map(|v| v.id),
            }),
            label,
            query,
            idx,
        }
    }
}

impl<'a> Iterator for EdgeByFulltextScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Refill the per-row scans from the child when we've run dry: queue
            // one fulltext-query iterator per active parent row.
            if self.emitter.needs_refill() {
                match self.child.next() {
                    Some(Ok(batch)) => {
                        if let Err(e) = self.emitter.seed(batch, |b, row| {
                            let view = BatchRow::new(b, row);
                            let label_str = match ExprEval::from_runtime(self.runtime).eval(
                                self.label,
                                self.label.root().idx(),
                                Some(&view),
                                None,
                            )? {
                                Value::String(s) => s,
                                _ => {
                                    return Err(
                                        "fulltext query expects a string relationship type".into(),
                                    );
                                }
                            };
                            let query_str = match ExprEval::from_runtime(self.runtime).eval(
                                self.query,
                                self.query.root().idx(),
                                Some(&view),
                                None,
                            )? {
                                Value::String(s) => s,
                                _ => return Err("fulltext query expects a string query".into()),
                            };
                            let g = self.runtime.g.borrow();
                            let iter = Box::new(
                                g.fulltext_query_edges(&label_str, &query_str)?
                                    .map(|(_src, _dst, edge_id, score)| (edge_id, score)),
                            )
                                as Box<dyn Iterator<Item = (RelationshipId, f64)>>;
                            Ok(Some(iter))
                        }) {
                            return Some(Err(e));
                        }
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
