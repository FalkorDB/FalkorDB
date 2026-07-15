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
//!
//! A downstream `Skip`/`Limit` lowers the emitter's pack ceiling (via
//! `record_cap`), so `CALL db.idx.fulltext.queryRelationships(..) ... LIMIT k`
//! drains only about `k` results from the RediSearch iterator instead of
//! eagerly packing a whole `BATCH_SIZE` worth of matches per call.

use crate::graph::graph::RelationshipId;
use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp, BatchRow},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

use super::batched_result_emitter::{BatchedResultEmitter, RowIter, ScoredColumn};

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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        edge: &'a Variable,
        label: &'a QueryExpr<Variable>,
        query: &'a QueryExpr<Variable>,
        score: &'a Option<Variable>,
        record_cap: Option<usize>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let mut emitter = BatchedResultEmitter::with_binding(ScoredColumn {
            id: edge.id,
            score: score.as_ref().map(|v| v.id),
        });
        // A downstream Skip/Limit lowers how many rows are needed; shrink the
        // pack ceiling so the first emit drains just enough index results
        // instead of a full BATCH_SIZE worth of work.
        if let Some(cap) = record_cap
            && cap < BATCH_SIZE
        {
            emitter.set_pack_ceiling(cap.max(1));
        }
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

impl<'a> Iterator for EdgeByFulltextScanOp<'a> {
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
                    return Err("fulltext query expects a string relationship type".into());
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
                let iter = Box::new(
                    g.fulltext_query_edges(&label_str, &query_str)?
                        .map(|(_src, _dst, edge_id, score)| (edge_id, score)),
                ) as Box<dyn Iterator<Item = (RelationshipId, f64)>>;
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
