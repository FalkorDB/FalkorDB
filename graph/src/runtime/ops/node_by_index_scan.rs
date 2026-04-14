//! Batch-mode index scan operator — retrieves nodes using a secondary index.
//!
//! For each active row in the input batch, evaluates the index query
//! parameters and collects matching nodes. Supports equality, range,
//! point-radius, and conjunctive (AND) index queries.
//!
//! ```text
//!  Input row ──► evaluate index query params
//!                      │
//!             ┌────────▼────────┐
//!             │ IndexQuery::    │
//!             │  Equal          │  key = value
//!             │  Range          │  min <= key <= max
//!             │  Point          │  within radius of point
//!             │  And([...])     │  conjunction of the above
//!             └────────┬────────┘
//!                      │
//!            graph.get_indexed_nodes()
//!                      │
//!             output rows with node IDs
//! ```

use std::collections::VecDeque;
use std::sync::Arc;

use crate::graph::graph::NodeId;
use crate::index::indexer::IndexQuery;
use crate::parser::ast::{QueryExpr, QueryNode, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct NodeByIndexScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    node_pattern: &'a QueryNode<Arc<String>, Variable>,
    index: &'a Arc<String>,
    query: &'a IndexQuery<QueryExpr<Variable>>,
    pending: VecDeque<(Env<'a>, Box<dyn Iterator<Item = NodeId>>)>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> NodeByIndexScanOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        node_pattern: &'a QueryNode<Arc<String>, Variable>,
        index: &'a Arc<String>,
        query: &'a IndexQuery<QueryExpr<Variable>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            node_pattern,
            index,
            query,
            pending: VecDeque::new(),
            idx,
        }
    }

    fn evaluate_index_query(
        runtime: &Runtime,
        query: &IndexQuery<QueryExpr<Variable>>,
        vars: &Env<'_>,
    ) -> Result<IndexQuery<Value>, String> {
        match query {
            IndexQuery::Equal { key, value } => {
                let value = {
                    ExprEval::from_runtime(runtime).eval(
                        value,
                        value.root().idx(),
                        Some(vars),
                        None,
                    )
                }?;
                Ok(IndexQuery::Equal {
                    key: key.clone(),
                    value,
                })
            }
            IndexQuery::Range {
                key,
                min,
                max,
                include_min,
                include_max,
            } => {
                let (min, max) = match (min, max) {
                    (Some(min), Some(max)) => {
                        let min = ExprEval::from_runtime(runtime).eval(
                            min,
                            min.root().idx(),
                            Some(vars),
                            None,
                        )?;
                        let max = ExprEval::from_runtime(runtime).eval(
                            max,
                            max.root().idx(),
                            Some(vars),
                            None,
                        )?;
                        (Some(min), Some(max))
                    }
                    (Some(min), None) => {
                        let min = ExprEval::from_runtime(runtime).eval(
                            min,
                            min.root().idx(),
                            Some(vars),
                            None,
                        )?;
                        (Some(min), None)
                    }
                    (None, Some(max)) => {
                        let max = ExprEval::from_runtime(runtime).eval(
                            max,
                            max.root().idx(),
                            Some(vars),
                            None,
                        )?;
                        (None, Some(max))
                    }
                    (None, None) => (None, None),
                };
                Ok(IndexQuery::Range {
                    key: key.clone(),
                    min,
                    max,
                    include_min: *include_min,
                    include_max: *include_max,
                })
            }
            IndexQuery::Point { key, point, radius } => {
                let point = ExprEval::from_runtime(runtime).eval(
                    point,
                    point.root().idx(),
                    Some(vars),
                    None,
                )?;
                let radius = ExprEval::from_runtime(runtime).eval(
                    radius,
                    radius.root().idx(),
                    Some(vars),
                    None,
                )?;
                Ok(IndexQuery::Point {
                    key: key.clone(),
                    point,
                    radius,
                })
            }
            IndexQuery::And(queries) => {
                let mut evaluated = Vec::with_capacity(queries.len());
                for q in queries {
                    evaluated.push(Self::evaluate_index_query(runtime, q, vars)?);
                }
                Ok(IndexQuery::And(evaluated))
            }
            IndexQuery::Or(queries) => {
                let mut evaluated = Vec::with_capacity(queries.len());
                for q in queries {
                    evaluated.push(Self::evaluate_index_query(runtime, q, vars)?);
                }
                Ok(IndexQuery::Or(evaluated))
            }
            IndexQuery::ArrayContains { key, value } => {
                let value = {
                    ExprEval::from_runtime(runtime).eval(
                        value,
                        value.root().idx(),
                        Some(vars),
                        None,
                    )
                }?;
                Ok(IndexQuery::ArrayContains {
                    key: key.clone(),
                    value,
                })
            }
            IndexQuery::InList { key, list } => {
                let list_val = {
                    ExprEval::from_runtime(runtime).eval(list, list.root().idx(), Some(vars), None)
                }?;
                match list_val {
                    Value::List(items) => {
                        let equals = items
                            .iter()
                            .filter(|v| {
                                matches!(
                                    v,
                                    Value::Int(_)
                                        | Value::Float(_)
                                        | Value::String(_)
                                        | Value::Bool(_)
                                )
                            })
                            .map(|v| IndexQuery::Equal {
                                key: key.clone(),
                                value: v.clone(),
                            })
                            .collect::<Vec<_>>();
                        Ok(IndexQuery::Or(equals))
                    }
                    _ => Err("IN operator requires a list".into()),
                }
            }
        }
    }

    /// Check if an evaluated index query can be satisfied by the index.
    /// Returns false when runtime values are types the index can't handle
    /// (e.g. List, Map, large Int64), in which case the caller should fall
    /// back to a label scan.
    fn can_utilize_index(q: &IndexQuery<Value>) -> bool {
        use crate::index::Index;

        const fn is_indexable(v: &Value) -> bool {
            match v {
                Value::Int(i) => !Index::int_loses_f64_precision(*i),
                Value::Float(_)
                | Value::String(_)
                | Value::Bool(_)
                | Value::Point(_)
                | Value::Null => true,
                _ => false, // List, Map, Node, Relationship, Path, etc.
            }
        }
        match q {
            IndexQuery::Equal { value, .. } => is_indexable(value),
            IndexQuery::Range { min, max, .. } => {
                min.as_ref().is_none_or(is_indexable) && max.as_ref().map_or(true, is_indexable)
            }
            IndexQuery::And(children) | IndexQuery::Or(children) => {
                children.iter().all(Self::can_utilize_index)
            }
            IndexQuery::ArrayContains { value, .. } => {
                matches!(
                    value,
                    Value::Int(_) | Value::Float(_) | Value::String(_) | Value::Bool(_)
                )
            }
            _ => true,
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
            if let Some(nid) = iter.next() {
                let mut row = env.clone_pooled(self.runtime.env_pool);
                row.insert(&self.node_pattern.alias, Value::Node(nid));
                envs.push(row);
            } else {
                self.pending.pop_front();
            }
        }
    }
}

impl<'a> Iterator for NodeByIndexScanOp<'a> {
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
                let q = match Self::evaluate_index_query(self.runtime, self.query, vars) {
                    Ok(q) => q,
                    Err(e) => return Some(Err(e)),
                };

                // Check if the index can satisfy this query. If not (e.g.
                // non-indexable value types), fall back to a label scan.
                let iter: Box<dyn Iterator<Item = NodeId>> = if Self::can_utilize_index(&q) {
                    Box::new(self.runtime.g.borrow().get_indexed_nodes(self.index, q))
                } else {
                    Box::new(
                        self.runtime
                            .g
                            .borrow()
                            .get_nodes(&self.node_pattern.labels, 0),
                    )
                };
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
