//! Batch-mode edge index scan operator — retrieves edges using a secondary index.
//!
//! For each active row in the input batch, evaluates the index query
//! parameters and collects matching edges. Supports equality, range,
//! and conjunctive (AND/OR) index queries.
//!
//! This operator replaces `CondTraverse` when the optimizer detects that
//! a filter on an edge property can be pushed into an edge index.
//!
//! ```text
//!  Input row ──► evaluate index query params
//!                      │
//!             ┌────────▼────────┐
//!             │ IndexQuery::    │
//!             │  Equal          │  key = value
//!             │  Range          │  min <= key <= max
//!             │  And([...])     │  conjunction of the above
//!             └────────┬────────┘
//!                      │
//!            graph.get_indexed_edges()
//!                      │
//!             output rows with edge + endpoint IDs
//! ```

use std::collections::VecDeque;
use std::sync::Arc;

use crate::graph::graph::{NodeId, RelationshipId};
use crate::index::indexer::IndexQuery;
use crate::parser::ast::{QueryExpr, QueryRelationship, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct EdgeByIndexScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    query: &'a IndexQuery<QueryExpr<Variable>>,
    transposed: bool,
    pending: VecDeque<(
        Env<'a>,
        std::vec::IntoIter<(NodeId, NodeId, RelationshipId)>,
    )>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> EdgeByIndexScanOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
        query: &'a IndexQuery<QueryExpr<Variable>>,
        transposed: bool,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            relationship_pattern,
            query,
            transposed,
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
                _ => false,
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

    /// Drains rows from `self.pending` into `envs` until `BATCH_SIZE` is reached.
    fn drain_pending(
        &mut self,
        envs: &mut Vec<Env<'a>>,
    ) {
        let rp = self.relationship_pattern;
        while envs.len() < BATCH_SIZE {
            let Some((env, iter)) = self.pending.front_mut() else {
                break;
            };
            if let Some((src, dst, edge_id)) = iter.next() {
                let mut row = env.clone_pooled(self.runtime.env_pool);
                // Bind from/to nodes according to transposed flag
                let (from_node, to_node) = if self.transposed {
                    (dst, src)
                } else {
                    (src, dst)
                };
                row.insert(&rp.from.alias, Value::Node(from_node));
                row.insert(&rp.to.alias, Value::Node(to_node));
                // Relationship value always stores (edge_id, src, dst) in graph order
                row.insert(
                    &rp.alias,
                    Value::Relationship(Box::new((edge_id, src, dst))),
                );
                envs.push(row);
            } else {
                self.pending.pop_front();
            }
        }
    }
}

impl<'a> Iterator for EdgeByIndexScanOp<'a> {
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

            let label = &self.relationship_pattern.types[0];
            let rp = self.relationship_pattern;

            for vars in batch.active_env_iter() {
                let q = match Self::evaluate_index_query(self.runtime, self.query, vars) {
                    Ok(q) => q,
                    Err(e) => return Some(Err(e)),
                };

                let mut edges: Vec<(NodeId, NodeId, RelationshipId)> =
                    if Self::can_utilize_index(&q) {
                        self.runtime.g.borrow().get_indexed_edges(label, q)
                    } else {
                        // Fall back to scanning all edges of this type
                        self.runtime.g.borrow().get_all_edges(label)
                    };

                // If the child already bound the from-node (e.g. from
                // NodeByLabelScan or AllNodeScan), filter edges to only
                // those originating from that node.
                if let Some(Value::Node(bound_from)) = vars.get(&rp.from.alias) {
                    let bound_id = *bound_from;
                    if self.transposed {
                        edges.retain(|(_, dst, _)| *dst == bound_id);
                    } else {
                        edges.retain(|(src, _, _)| *src == bound_id);
                    }
                }

                self.pending
                    .push_back((vars.clone_pooled(self.runtime.env_pool), edges.into_iter()));
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
