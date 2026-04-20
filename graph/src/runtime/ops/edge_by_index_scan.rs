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
// `NodeRef` supplies the `.root()` / `.idx()` methods used below via
// trait import — it appears unused in signatures but removing it
// breaks `value.root().idx()` etc.
use orx_tree::{Dyn, NodeIdx, NodeRef};

/// Invariant: `relationship.from` is always the bound endpoint of the
/// edge scan; `transposed` flips which graph-side endpoint the edge
/// tensor keys on (src vs dst) but the bound endpoint is always read
/// via `rp.from.alias`. The planner's `select_scan_node` pass enforces
/// this by swapping the relationship and setting `transposed`
/// accordingly before any rewrite to `EdgeByIndexScan`.
pub struct EdgeByIndexScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    relationship_pattern: &'a QueryRelationship<Arc<String>, Arc<String>, Variable>,
    query: &'a IndexQuery<QueryExpr<Variable>>,
    transposed: bool,
    pending: VecDeque<(
        Env<'a>,
        Box<dyn Iterator<Item = (NodeId, NodeId, RelationshipId)>>,
    )>,
    /// Lazily-populated cache of all edges of `relationship_pattern.types[0]`
    /// for the non-indexable-runtime-value fallback. Materialized once
    /// on the first fallback and shared across subsequent input rows
    /// so we don't pay O(|E_type|) per row when the index can't serve
    /// the query (e.g. value is a list or date). The cache is
    /// dropped when the op is dropped.
    all_edges_cache: std::cell::RefCell<Option<Arc<Vec<(NodeId, NodeId, RelationshipId)>>>>,
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
            all_edges_cache: std::cell::RefCell::new(None),
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
            IndexQuery::ArrayContains { value, .. } => match value {
                // Match `Equal` / `Range`: reject int64s that lose
                // f64 precision — the index can't represent them
                // exactly so the query must fall back to post-filter.
                Value::Int(i) => !Index::int_loses_f64_precision(*i),
                Value::Float(_) | Value::String(_) | Value::Bool(_) => true,
                _ => false,
            },
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
                // Self-loop patterns share the same alias on both
                // endpoints; the second `insert` would overwrite the
                // first with the (equal, per the filter above)
                // destination value. Skip when aliases are the same
                // to avoid redundant work and to preserve the
                // semantics that the single variable is bound once.
                if rp.to.alias != rp.from.alias {
                    row.insert(&rp.to.alias, Value::Node(to_node));
                }
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

                // Stream results instead of collecting: the child
                // batch may have many rows and each row would
                // otherwise materialize the full edge set before any
                // emission. Matches `NodeByIndexScanOp`'s pattern.
                //
                // On the fallback path (non-indexable runtime value),
                // `get_all_edges` is materialized once into the op's
                // cache and shared by all subsequent rows via `Arc`,
                // so we don't rebuild the full-type Vec per row.
                let base: Box<dyn Iterator<Item = (NodeId, NodeId, RelationshipId)>> =
                    if Self::can_utilize_index(&q) {
                        Box::new(self.runtime.g.borrow().get_indexed_edges(label, q))
                    } else {
                        let cached = {
                            let mut cache = self.all_edges_cache.borrow_mut();
                            if cache.is_none() {
                                *cache =
                                    Some(Arc::new(self.runtime.g.borrow().get_all_edges(label)));
                            }
                            Arc::clone(cache.as_ref().unwrap())
                        };
                        let mut idx = 0usize;
                        Box::new(std::iter::from_fn(move || {
                            if idx < cached.len() {
                                let v = cached[idx];
                                idx += 1;
                                Some(v)
                            } else {
                                None
                            }
                        }))
                    };

                // Filter edges by *both* endpoints when the child has
                // already bound them. `transposed` flips which
                // graph-side (src/dst) role the pattern's from/to
                // endpoints play in the tensor, so the binding check
                // swaps correspondingly.
                //
                // Self-loop patterns like `MATCH (n)-[r:T]->(n)` have
                // `rp.from.alias == rp.to.alias`. Without filtering
                // for `from_id == to_id`, non-loop edges leak through
                // and `drain_pending`'s second `row.insert(to.alias)`
                // overwrites the `from.alias` binding.
                let bound_from = match vars.get(&rp.from.alias) {
                    Some(Value::Node(id)) => Some(*id),
                    _ => None,
                };
                let bound_to = match vars.get(&rp.to.alias) {
                    Some(Value::Node(id)) => Some(*id),
                    _ => None,
                };
                let transposed = self.transposed;
                let same_endpoint_alias = rp.from.alias == rp.to.alias;
                let edges: Box<dyn Iterator<Item = (NodeId, NodeId, RelationshipId)>> =
                    if bound_from.is_some() || bound_to.is_some() || same_endpoint_alias {
                        Box::new(base.filter(move |(src, dst, _)| {
                            let (from_id, to_id) = if transposed {
                                (*dst, *src)
                            } else {
                                (*src, *dst)
                            };
                            bound_from.is_none_or(|id| id == from_id)
                                && bound_to.is_none_or(|id| id == to_id)
                                && (!same_endpoint_alias || from_id == to_id)
                        }))
                    } else {
                        base
                    };

                self.pending
                    .push_back((vars.clone_pooled(self.runtime.env_pool), edges));
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
