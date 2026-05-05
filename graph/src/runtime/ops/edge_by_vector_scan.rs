//! Batch-mode KNN vector scan operator — retrieves edges via a vector index.
//!
//! Implements
//! `CALL db.idx.vector.queryRelationships(type, attr, k, vector)`.
//! Mirrors [`NodeByVectorScanOp`](super::node_by_vector_scan) but binds
//! the matched edge as `Value::Relationship((edge_id, src, dst))` so
//! downstream property lookups (e.g. `RETURN r.name`) work without
//! binding endpoint variables — the procedure only yields `relationship`
//! and `score`.

use std::collections::VecDeque;

use crate::graph::graph::{NodeId, RelationshipId};
use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::ops::node_by_vector_scan::eval_vector_args;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx};

pub struct EdgeByVectorScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    /// Buffered KNN results pending emission — see the corresponding
    /// field on `NodeByVectorScanOp` for why this must be cleared on
    /// `set_argument_batch`.
    pub(crate) pending: VecDeque<(
        Env<'a>,
        Box<dyn Iterator<Item = (NodeId, NodeId, RelationshipId, f64)>>,
    )>,
    edge: &'a Variable,
    label: &'a QueryExpr<Variable>,
    attr: &'a QueryExpr<Variable>,
    k: &'a QueryExpr<Variable>,
    vector: &'a QueryExpr<Variable>,
    score: &'a Option<Variable>,
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
            pending: VecDeque::new(),
            edge,
            label,
            attr,
            k,
            vector,
            score,
            idx,
        }
    }

    fn drain_pending(
        &mut self,
        envs: &mut Vec<Env<'a>>,
    ) {
        while envs.len() < BATCH_SIZE {
            let Some((env, iter)) = self.pending.front_mut() else {
                break;
            };
            if let Some((src, dst, edge_id, s)) = iter.next() {
                let mut row = env.clone_pooled(self.runtime.env_pool);
                row.insert(
                    self.edge,
                    Value::Relationship(Box::new((edge_id, src, dst))),
                );
                if let Some(score) = self.score {
                    row.insert(score, Value::Float(s));
                }
                envs.push(row);
            } else {
                self.pending.pop_front();
            }
        }
    }
}

impl<'a> Iterator for EdgeByVectorScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut envs = Vec::with_capacity(BATCH_SIZE);

        self.drain_pending(&mut envs);

        while envs.len() < BATCH_SIZE {
            let batch = match self.child.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            };

            for vars in batch.active_env_iter() {
                let (label_str, attr_str, k_val, vec_arc) = match eval_vector_args(
                    self.runtime,
                    self.label,
                    self.attr,
                    self.k,
                    self.vector,
                    vars,
                    "db.idx.vector.queryRelationships",
                ) {
                    Ok(t) => t,
                    Err(e) => return Some(Err(e)),
                };

                let g = self.runtime.g.borrow();
                let iter = match g.vector_query_edges(&label_str, &attr_str, vec_arc, k_val) {
                    Ok(iter) => Box::new(iter)
                        as Box<dyn Iterator<Item = (NodeId, NodeId, RelationshipId, f64)>>,
                    Err(e) => return Some(Err(e)),
                };
                drop(g);

                self.pending
                    .push_back((vars.clone_pooled(self.runtime.env_pool), iter));
            }

            self.drain_pending(&mut envs);
        }

        if envs.is_empty() {
            None
        } else {
            Some(Ok(Batch::from_envs(envs)))
        }
    }
}
