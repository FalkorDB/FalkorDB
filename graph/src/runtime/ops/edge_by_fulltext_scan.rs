//! Batch-mode edge fulltext scan operator — retrieves edges via a fulltext index query.
//!
//! Implements `CALL db.idx.fulltext.queryRelationships(type, query)`.
//! For each active row in each input batch, evaluates the relationship-type
//! and query expressions, delegates to the graph's edge fulltext index,
//! and expands matching edges into output rows accumulated into batches
//! of up to `BATCH_SIZE`.
//!
//! Each result row includes the matched edge as
//! `Value::Relationship((edge_id, src, dst))` and optionally a
//! relevance score (float) when a score yield variable is specified.
//! The `Value::Relationship` carries enough information for downstream
//! property lookups (e.g. `RETURN r.name`) without binding endpoint
//! variables — the procedure only yields `relationship` / `score`.

use std::collections::VecDeque;

use crate::graph::graph::{NodeId, RelationshipId};
use crate::parser::ast::{QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct EdgeByFulltextScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    pending: VecDeque<(
        Env<'a>,
        Box<dyn Iterator<Item = (NodeId, NodeId, RelationshipId, f64)>>,
    )>,
    edge: &'a Variable,
    label: &'a QueryExpr<Variable>,
    query: &'a QueryExpr<Variable>,
    score: &'a Option<Variable>,
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
            pending: VecDeque::new(),
            edge,
            label,
            query,
            score,
            idx,
        }
    }

    /// Drains rows from `self.pending` into `envs` until `BATCH_SIZE` is reached
    /// or all pending scans are exhausted.
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

impl<'a> Iterator for EdgeByFulltextScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut envs = Vec::with_capacity(BATCH_SIZE);

        // Drain leftover scans from previous call.
        self.drain_pending(&mut envs);

        while envs.len() < BATCH_SIZE {
            let batch = match self.child.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            };

            for vars in batch.active_env_iter() {
                let label_str = match ExprEval::from_runtime(self.runtime).eval(
                    self.label,
                    self.label.root().idx(),
                    Some(vars),
                    None,
                ) {
                    Ok(Value::String(s)) => s,
                    Ok(_) => {
                        return Some(Err(
                            "fulltext query expects a string relationship type".into()
                        ));
                    }
                    Err(e) => return Some(Err(e)),
                };
                let query_str = match ExprEval::from_runtime(self.runtime).eval(
                    self.query,
                    self.query.root().idx(),
                    Some(vars),
                    None,
                ) {
                    Ok(Value::String(s)) => s,
                    Ok(_) => {
                        return Some(Err("fulltext query expects a string query".into()));
                    }
                    Err(e) => return Some(Err(e)),
                };
                let g = self.runtime.g.borrow();
                let iter = match g.fulltext_query_edges(&label_str, &query_str) {
                    Ok(iter) => Box::new(iter),
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
