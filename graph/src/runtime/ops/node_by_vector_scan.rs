//! Batch-mode KNN vector scan operator — retrieves nodes via a vector index.
//!
//! Implements `CALL db.idx.vector.queryNodes(label, attr, k, vector)`. For each
//! active parent row, evaluates the four input expressions (label / attribute
//! name / k / query vector), delegates to the graph's vector index, and queues
//! the matching `(node, distance)` iterator on the shared
//! [`BatchedResultEmitter`], which packs up to
//! [`BATCH_SIZE`](super::super::batch::BATCH_SIZE) results into one columnar
//! batch. The matched node binds to the node variable and, when a score yield
//! variable is present, the distance binds to it as a float column. Results are
//! ordered by ascending distance — RediSearch's `RediSearch_CreateVecSimNode`
//! defaults to `BY_SCORE` ordering, which the graph index layer preserves.

use std::sync::Arc;

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
use thin_vec::ThinVec;

use super::batched_result_emitter::{BatchedResultEmitter, ScoredColumn};

pub struct NodeByVectorScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    /// Holds the parent batch being expanded and the per-row `(node, distance)`
    /// iterators, and performs the shared pack-and-gather emit. Must be reset on
    /// `set_argument_batch` so a correlated plan (Apply) doesn't replay stale
    /// rows from a previous outer iteration when the inner side stops early.
    pub(crate) emitter: BatchedResultEmitter<'a, (NodeId, f64)>,
    label: &'a QueryExpr<Variable>,
    attr: &'a QueryExpr<Variable>,
    k: &'a QueryExpr<Variable>,
    vector: &'a QueryExpr<Variable>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> NodeByVectorScanOp<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        node: &'a Variable,
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
                id: node.id,
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

impl<'a> Iterator for NodeByVectorScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Refill the per-row scans from the child when we've run dry: queue
            // one KNN iterator per active parent row.
            if self.emitter.needs_refill() {
                match self.child.next() {
                    Some(Ok(batch)) => {
                        for row in batch.active_indices() {
                            let view = BatchRow::new(&batch, row);
                            let (label_str, attr_str, k_val, vec_arc) = match eval_vector_args(
                                self.runtime,
                                self.label,
                                self.attr,
                                self.k,
                                self.vector,
                                &view,
                                "db.idx.vector.queryNodes",
                            ) {
                                Ok(t) => t,
                                Err(e) => return Some(Err(e)),
                            };

                            let g = self.runtime.g.borrow();
                            let iter =
                                match g.vector_query_nodes(&label_str, &attr_str, vec_arc, k_val) {
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

/// Evaluate the four runtime expressions feeding a vector-index scan
/// (label, attribute, k, vector) and validate them against the
/// procedure's argument contract. The error string is pinned by the
/// `test_vecsim::test06_validate_arguments` negative cases.
pub(crate) fn eval_vector_args<'a, R: crate::runtime::row::RowView + ?Sized>(
    runtime: &'a Runtime<'a>,
    label: &QueryExpr<Variable>,
    attr: &QueryExpr<Variable>,
    k: &QueryExpr<Variable>,
    vector: &QueryExpr<Variable>,
    vars: &R,
    procedure: &str,
) -> Result<(Arc<String>, Arc<String>, usize, Arc<ThinVec<f32>>), String> {
    let invalid = || format!("Invalid arguments for procedure '{procedure}'");

    let label_str =
        match ExprEval::from_runtime(runtime).eval(label, label.root().idx(), Some(vars), None) {
            Ok(Value::String(s)) => s,
            Ok(_) => return Err(invalid()),
            Err(e) => return Err(e),
        };
    let attr_str =
        match ExprEval::from_runtime(runtime).eval(attr, attr.root().idx(), Some(vars), None) {
            Ok(Value::String(s)) => s,
            Ok(_) => return Err(invalid()),
            Err(e) => return Err(e),
        };
    let k_val = match ExprEval::from_runtime(runtime).eval(k, k.root().idx(), Some(vars), None) {
        Ok(Value::Int(n)) if n > 0 => n as usize,
        Ok(_) => return Err(invalid()),
        Err(e) => return Err(e),
    };
    let vec_arc =
        match ExprEval::from_runtime(runtime).eval(vector, vector.root().idx(), Some(vars), None) {
            Ok(Value::VecF32(v)) => v,
            Ok(_) => return Err(invalid()),
            Err(e) => return Err(e),
        };
    Ok((label_str, attr_str, k_val, vec_arc))
}
