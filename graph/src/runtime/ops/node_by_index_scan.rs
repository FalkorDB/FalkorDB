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
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    row::{Row, RowView},
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
    pending: VecDeque<(Row, Box<dyn Iterator<Item = NodeId>>)>,
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

    /// Drains rows from `self.pending` into `envs` until `BATCH_SIZE` is reached
    /// or all pending scans are exhausted.
    fn drain_pending(
        &mut self,
        builder: &mut BatchBuilder,
    ) {
        // Additional labels (beyond the index's primary label) must be
        // verified post-hoc since the index only filters on the primary.
        let extra_labels = if self.node_pattern.labels.len() > 1 {
            Some(self.node_pattern.labels.iter().skip(1).collect::<Vec<_>>())
        } else {
            None
        };
        while builder.len() < BATCH_SIZE {
            let Some((env, iter)) = self.pending.front_mut() else {
                break;
            };
            if let Some(nid) = iter.next() {
                if let Some(extra_labels) = &extra_labels {
                    let node_labels = self.runtime.get_node_labels(nid);
                    if !extra_labels
                        .iter()
                        .all(|l| node_labels.iter().any(|nl| nl == *l))
                    {
                        continue;
                    }
                }
                let mut row = env.clone();
                row.insert(&self.node_pattern.alias, Value::Node(nid));
                builder.push_row(&row);
            } else {
                self.pending.pop_front();
            }
        }
    }

    /// Columnar fast path: drains node IDs from leading no-binding pending
    /// scans into a flat `Vec<NodeId>`, mirroring `NodeByLabelScan`'s
    /// `from_node_ids` path. Only valid when the pattern has a single label
    /// (no post-hoc label verification) and the parent row carries no
    /// bindings, so every output row is just `Value::Node(id)`. Stops at
    /// `BATCH_SIZE`, a binding-carrying env, or pending exhaustion.
    fn drain_pending_columnar(
        &mut self,
        ids: &mut Vec<NodeId>,
    ) {
        while ids.len() < BATCH_SIZE {
            let Some((env, iter)) = self.pending.front_mut() else {
                break;
            };
            if env.has_bindings() {
                break;
            }
            if let Some(nid) = iter.next() {
                ids.push(nid);
            } else {
                self.pending.pop_front();
            }
        }
    }
}

impl<'a> Iterator for NodeByIndexScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let single_label = self.node_pattern.labels.len() <= 1;
        let alias_id = self.node_pattern.alias.id;

        loop {
            // Refill pending scans from the child when we've run dry.
            if self.pending.is_empty() {
                match self.child.next() {
                    Some(Ok(batch)) => {
                        for row in batch.active_indices() {
                            let view = BatchRow::new(&batch, row);
                            let q =
                                match Self::evaluate_index_query(self.runtime, self.query, &view) {
                                    Ok(q) => q,
                                    Err(e) => return Some(Err(e)),
                                };

                            // Check if the index can satisfy this query. If not
                            // (e.g. non-indexable value types), fall back to a
                            // label scan.
                            let iter: Box<dyn Iterator<Item = NodeId>> =
                                if Self::can_utilize_index(&q) {
                                    Box::new(
                                        self.runtime.g.borrow().get_indexed_nodes(self.index, q),
                                    )
                                } else {
                                    Box::new(
                                        self.runtime
                                            .g
                                            .borrow()
                                            .get_nodes(&self.node_pattern.labels, 0),
                                    )
                                };
                            self.pending.push_back((view.to_owned_row(), iter));
                        }
                        continue;
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None,
                }
            }

            // Columnar fast path: single label and the front parent row carries
            // no bindings → emit a `Column::NodeIds` batch directly, skipping
            // per-row `Row` construction and the `BatchBuilder` transpose.
            if single_label && !self.pending.front().unwrap().0.has_bindings() {
                let mut ids: Vec<NodeId> = Vec::with_capacity(BATCH_SIZE);
                self.drain_pending_columnar(&mut ids);
                if !ids.is_empty() {
                    return Some(Ok(Batch::from_node_ids(alias_id, ids)));
                }
                continue;
            }

            // Row path: parent bindings present (or multi-label verification),
            // so build one env per matching node.
            let mut builder = BatchBuilder::new();
            self.drain_pending(&mut builder);
            if !builder.is_empty() {
                return Some(Ok(builder.finish()));
            }
        }
    }
}
