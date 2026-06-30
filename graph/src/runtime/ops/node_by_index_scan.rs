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

use std::sync::Arc;

use crate::graph::graph::NodeId;
use crate::index::indexer::IndexQuery;
use crate::parser::ast::{QueryExpr, QueryNode, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

use super::batched_result_emitter::{BatchedResultEmitter, RowResult};

pub struct NodeByIndexScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    node_pattern: &'a QueryNode<Arc<String>, Variable>,
    index: &'a Arc<String>,
    query: &'a IndexQuery<QueryExpr<Variable>>,
    /// Additional labels (beyond the index's primary label) that must be
    /// verified post-hoc, since the index only filters on the primary. Computed
    /// once; folded into each row's iterator as a `filter`.
    extra_labels: Option<Vec<Arc<String>>>,
    /// Holds the parent batch being expanded and the per-row node iterators, and
    /// performs the shared pack-and-gather emit.
    emitter: BatchedResultEmitter<'a, NodeId>,
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
        let extra_labels = if node_pattern.labels.len() > 1 {
            Some(node_pattern.labels.iter().skip(1).cloned().collect())
        } else {
            None
        };
        Self {
            runtime,
            child,
            node_pattern,
            index,
            query,
            extra_labels,
            emitter: BatchedResultEmitter::new(node_pattern.alias.id),
            idx,
        }
    }

    fn evaluate_index_query<R: crate::runtime::row::RowView + ?Sized>(
        runtime: &Runtime,
        query: &IndexQuery<QueryExpr<Variable>>,
        vars: &R,
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
                min.as_ref().is_none_or(is_indexable) && max.as_ref().is_none_or(is_indexable)
            }
            IndexQuery::And(children) | IndexQuery::Or(children) => {
                // Empty `Or([])` / `And([])` is not a valid index
                // query: the RediSearch backend treats it as
                // match-all, which is unsafe when the optimizer has
                // already pushed an IN-list whose elements were all
                // filtered out at runtime (e.g. `IN [NULL]`). Force
                // the fallback path so the retained post-filter
                // re-establishes correctness.
                !children.is_empty() && children.iter().all(Self::can_utilize_index)
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
}

impl<'a> Iterator for NodeByIndexScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // For each active parent row, evaluate the index query and queue the
            // matching node ids (falling back to a label scan when the index
            // can't satisfy the query), folding any extra-label verification
            // into the iterator so the shared emit stays generic. Iterators are
            // built lazily, one row at a time. When the batch is exhausted
            // (`Ok(None)`), pull and seed the next child batch.
            match self.emitter.emit_lazy(|b, row| {
                let view = BatchRow::new(b, row);
                let q = Self::evaluate_index_query(self.runtime, self.query, &view)?;

                // Check if the index can satisfy this query. If not
                // (e.g. non-indexable value types), fall back to a
                // label scan.
                let base: Box<dyn Iterator<Item = NodeId>> = if Self::can_utilize_index(&q) {
                    Box::new(self.runtime.g.borrow().get_indexed_nodes(self.index, q))
                } else {
                    Box::new(
                        self.runtime
                            .g
                            .borrow()
                            .get_nodes(&self.node_pattern.labels, 0),
                    )
                };
                let iter: Box<dyn Iterator<Item = NodeId> + 'a> = match &self.extra_labels {
                    Some(extra) => {
                        let extra = extra.clone();
                        let runtime = self.runtime;
                        Box::new(base.filter(move |nid| {
                            let node_labels = runtime.get_node_labels(*nid);
                            extra.iter().all(|l| node_labels.iter().any(|nl| nl == l))
                        }))
                    }
                    None => base,
                };
                Ok(Some(RowResult::many(iter)))
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
