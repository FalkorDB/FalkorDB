//! Batch-mode KNN vector scan operator — retrieves edges via a vector index.
//!
//! Implements `CALL db.idx.vector.queryRelationships(type, attr, k, vector)`.
//! Mirrors [`NodeByVectorScanOp`](super::node_by_vector_scan) but binds the
//! matched edge as a `Value::Relationship` so downstream property lookups (e.g.
//! `RETURN r.name`) work without binding endpoint variables — the procedure only
//! yields `relationship` and `score`. The underlying index iterator yields the
//! endpoints too, which are discarded here. When a score yield variable is
//! present, the distance binds to it as a float column.

use crate::graph::graph::RelationshipId;
use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::ops::node_by_vector_scan::eval_vector_args;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    runtime::Runtime,
};
use orx_tree::{Dyn, NodeIdx};

use super::batched_result_emitter::{BatchedResultEmitter, ScoredColumn};

pub struct EdgeByVectorScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    /// Holds the parent batch being expanded and the per-row `(edge, distance)`
    /// iterators, and performs the shared pack-and-gather emit. See the
    /// corresponding field on `NodeByVectorScanOp` for why this must be reset on
    /// `set_argument_batch`.
    pub(crate) emitter: BatchedResultEmitter<'a, (RelationshipId, f64)>,
    label: &'a QueryExpr<Variable>,
    attr: &'a QueryExpr<Variable>,
    k: &'a QueryExpr<Variable>,
    vector: &'a QueryExpr<Variable>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> EdgeByVectorScanOp<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        edge: &'a Variable,
        label: &'a QueryExpr<Variable>,
        attr: &'a QueryExpr<Variable>,
        k: &'a QueryExpr<Variable>,
        vector: &'a QueryExpr<Variable>,
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
            attr,
            k,
            vector,
            idx,
        }
    }
}

impl<'a> Iterator for EdgeByVectorScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Refill the per-row scans from the child when we've run dry: queue
            // one KNN iterator per active parent row, discarding the endpoints
            // the index yields and keeping only `(edge, distance)`.
            if self.emitter.needs_refill() {
                match self.child.next() {
                    Some(Ok(batch)) => {
                        if let Err(e) = self.emitter.seed(batch, |b, row| {
                            let view = BatchRow::new(b, row);
                            let (label_str, attr_str, k_val, vec_arc) = eval_vector_args(
                                self.runtime,
                                self.label,
                                self.attr,
                                self.k,
                                self.vector,
                                &view,
                                "db.idx.vector.queryRelationships",
                            )?;
                            let g = self.runtime.g.borrow();
                            let iter = Box::new(
                                g.vector_query_edges(&label_str, &attr_str, vec_arc, k_val)?
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
