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

use super::batched_result_emitter::{BatchedResultEmitter, RowIter};

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
        // Kept in the signature, unused for now: see the emitter below.
        _record_cap: Option<usize>,
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
            // `None`, and not for want of a win: capping this measured
            // `LIMIT 10` at 240,138 instructions against 1,079,312, a 4.5x
            // saving on a full batch packed and discarded.
            //
            // It is off because it hangs `tests/flow/test_constraint.py`.
            // Bisected to this line and to the same line in
            // `edge_by_index_scan`, each of which hangs the file on its own,
            // while the caps on `cond_var_len_traverse` and
            // `node_by_label_and_id_scan` leave all 23 tests passing. Constraint
            // validation drives index scans and its tests poll for a constraint
            // to become operational, so a scan that does not finish hangs
            // rather than fails. The mechanism is not yet understood — the
            // `next()` loop here looks correct for a partial batch — so this
            // stays off until it is, rather than shipping a 4.5x win that
            // deadlocks constraint creation. Tracked in #2790.
            emitter: BatchedResultEmitter::new(node_pattern.alias.id, None),
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
                    Value::List(items) => Ok(IndexQuery::in_list(key, &items)),
                    _ => Err("IN operator requires a list".into()),
                }
            }
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
                let base: Box<dyn Iterator<Item = NodeId>> = if q.can_be_served() {
                    let g = self.runtime.graph();
                    // Index: a numeric Equal/Range on a column we own is served by the
                    // CoW B-tree; everything else (string/geo on the same Range index, composite, or a
                    // missing column) falls through to RediSearch during the dark-launch.
                    #[cfg(feature = "index-falkordb")]
                    let it: Box<dyn Iterator<Item = NodeId>> =
                        if let Some(hit) = g.query_index_numeric_nodes(self.index, &q) {
                            Box::new(hit)
                        } else {
                            Box::new(g.get_indexed_nodes(self.index, q))
                        };
                    #[cfg(not(feature = "index-falkordb"))]
                    let it: Box<dyn Iterator<Item = NodeId>> =
                        Box::new(g.get_indexed_nodes(self.index, q));
                    it
                } else {
                    Box::new(self.runtime.graph().get_nodes(&self.node_pattern.labels, 0))
                };
                let iter: Box<dyn Iterator<Item = NodeId> + 'a> = match &self.extra_labels {
                    Some(extra) => {
                        let extra = extra.clone();
                        let runtime = self.runtime;
                        Box::new(base.filter(move |nid| {
                            extra.iter().all(|l| runtime.node_has_label(*nid, l))
                        }))
                    }
                    None => base,
                };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> Arc<String> {
        Arc::new("v".to_string())
    }

    fn in_list(values: &[Value]) -> IndexQuery<Value> {
        IndexQuery::InList {
            key: key(),
            list: Value::List(Arc::new(values.iter().cloned().collect())),
        }
    }

    /// The same members as an `Or` of `Equal`s — the shape an `IN` list used to reach the index in.
    fn or_of_equals(values: &[Value]) -> IndexQuery<Value> {
        IndexQuery::Or(
            values
                .iter()
                .map(|v| IndexQuery::Equal {
                    key: key(),
                    value: v.clone(),
                })
                .collect(),
        )
    }

    /// An `IN` list reaches the index only when it is non-empty and every member is one the index
    /// holds exactly; otherwise the op scans. That is the rule its members got as an `Or` of
    /// `Equal`s, so which queries fall back is unchanged by handing the list over whole.
    #[test]
    fn in_list_reaches_the_index_on_the_same_terms_as_an_or_of_its_members() {
        let lossy = Value::Int(9_007_199_254_740_993);
        // A member the index cannot hold (here a list; a `date()` alike) is kept by
        // `IndexQuery::in_list` so that the whole query falls back (#2958).
        let unservable = Value::List(Arc::new(vec![Value::Int(1)].into_iter().collect()));
        let cases: [(&[Value], bool); 6] = [
            (&[Value::Int(1), Value::Float(2.5), Value::Bool(true)], true),
            (
                &[Value::Int(1), Value::String(Arc::new("a".to_string()))],
                true,
            ),
            (&[], false),
            (&[Value::Int(1), lossy.clone()], false),
            (&[lossy], false),
            (&[Value::Int(1), unservable], false),
        ];
        for (values, expected) in cases {
            assert_eq!(in_list(values).can_be_served(), expected, "IN {values:?}");
            assert_eq!(
                in_list(values).can_be_served(),
                or_of_equals(values).can_be_served(),
                "IN {values:?} must decide as its members did as an Or"
            );
        }
        assert!(
            !IndexQuery::InList {
                key: key(),
                list: Value::Int(1),
            }
            .can_be_served(),
            "an InList that does not hold a list never reaches the index"
        );
    }
}
