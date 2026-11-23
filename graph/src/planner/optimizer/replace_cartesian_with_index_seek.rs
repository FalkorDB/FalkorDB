//! Index-seek replacement for a Cartesian Product joined on an indexed lookup.
//!
//! A later `MATCH (p:L {k: e.x})` whose inline property reads a variable bound
//! by an earlier clause shares no variable with it, so the planner makes it a
//! `CartesianProduct` branch of the earlier clause, with the predicate in a
//! `Filter` above (see `Planner::plan_query`). Left to
//! `replace_cartesian_with_hash_join`, that becomes a `ValueHashJoin` whose
//! build side is a scan of the whole label: its cost grows with the label,
//! however few rows the earlier clause produces.
//!
//! When the label has an index on the joined property, the other branches can
//! instead feed the scan, like any later clause stitched below an earlier one:
//! each of their rows seeks the index for its own key.
//!
//! ```text
//! Before:                          After:
//!
//! Filter(p.k = e.x)                Filter(p.k = e.x)       (rewritten to an index scan
//!   |                                |                      by `utilize_index`)
//!   v                                v
//! CartesianProduct                 NodeByLabelScan(p:L)
//!   |           |                    |
//!   v           v                    v
//! ChildA      NodeByLabelScan(p)   ChildA
//! ```
//!
//! Without such an index the hash join stays: it reads the label once instead
//! of once per row. It also stays when the other branches are estimated to
//! produce so many rows that one index lookup each costs more than reading the
//! label once (a bulk `UNWIND` of a large batch): see [`SEEK_COST_IN_SCANNED_NODES`].
//!
//! A later `MATCH (p) WHERE id(p) = e.x` is the same join on the node's id. Its
//! scan (an `AllNodeScan`, or a `NodeByLabelScan` when `p` has a label) is fed
//! the same way, and `utilize_node_by_id` turns the moved `id()` comparisons
//! into a `NodeByIdSeek` (or a `NodeByLabelAndIdScan`): a direct lookup that
//! needs no index.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use orx_tree::{Bfs, Dfs, DynNode, DynTree, NodeRef, PostOrder};

use crate::{
    graph::graph::Graph,
    parser::ast::{ExprIR, QueryExpr, QueryNode, Variable},
    runtime::{functions::FnType, value::Value},
    tree,
};

use super::super::IR;
use super::eliminate_true_filters::substitute_params;
use super::utilize_index::{can_seek_index, seek_index_label};
use super::utilize_node_by_id::get_id_filter;
use super::{collect_expr_variables, collect_subtree_variables, conjuncts, min_label_count};

/// What one index lookup costs, in nodes of a label scan. Measured on a 1M-node
/// label: reading it once (the hash join's build side) takes about 52 ms, a
/// lookup about 17 us, a ratio of 330. Rounded down: a lookup also keeps the
/// result clustered per outer row.
const SEEK_COST_IN_SCANNED_NODES: f64 = 256.0;

/// What one id lookup costs, in nodes of a scan. Measured against the hash
/// join it replaces, on graphs of 1k to 200k nodes: the hash join is faster
/// once a batch has more rows than about half the nodes it would scan (no
/// label) or about a quarter of them (one label). Where 4 picks the hash join
/// over a faster seek, it was at most about 5% slower.
const ID_SEEK_COST_IN_SCANNED_NODES: f64 = 4.0;

/// Rewrites `Filter -> CartesianProduct(.., [Filter ->] NodeByLabelScan)` so
/// the scan reads the other branches' rows and the filter can become an index
/// seek, wherever the label has an index for a predicate that reads them, or
/// an id seek, wherever the predicate is `id(p) = ..` (on any node scan).
pub(super) fn replace_cartesian_with_index_seek(
    optimized_plan: &mut DynTree<IR>,
    graph: &Graph,
    params: &HashMap<String, Value>,
) {
    // In post-order: a later clause's product holds the earlier one, and the
    // rows it is estimated to feed depend on whether the earlier clause's
    // scan already became a lookup. A rewrite moves nodes, so the walk starts
    // over after each one; a product left as it was is left again.
    loop {
        let rewrite = optimized_plan
            .root()
            .indices::<PostOrder>()
            .find_map(|idx| {
                try_rewrite_at(&optimized_plan.node(idx), graph, params).map(|tree| (idx, tree))
            });
        let Some((idx, tree)) = rewrite else {
            break;
        };
        // Dropping the walk over the replaced nodes removes them.
        let _ = optimized_plan.node_mut(idx).replace_with::<Dfs, _>(tree);
    }
}

/// The rewrite of `node`, when it is a `Filter` over a `CartesianProduct`
/// that reads no `Argument` rows.
fn try_rewrite_at(
    node: &DynNode<IR>,
    graph: &Graph,
    params: &HashMap<String, Value>,
) -> Option<DynTree<IR>> {
    let IR::Filter(filter) = node.data() else {
        return None;
    };
    let child = node.get_child(0)?;
    if !matches!(child.data(), IR::CartesianProduct)
        || child
            .walk::<Bfs>()
            .any(|data| matches!(data, IR::Argument(_)))
    {
        return None;
    }
    try_seek_rewrite(filter, &child, graph, params)
}

/// The `Filter*` chain above a childless `NodeByLabelScan` or `AllNodeScan`
/// that makes up `branch`, as the filters (top first), the scan and the
/// scanned node.
fn scan_chain(
    branch: &DynNode<IR>
) -> Option<(
    Vec<QueryExpr<Variable>>,
    IR,
    Arc<QueryNode<Arc<String>, Variable>>,
)> {
    let mut filters = Vec::new();
    let mut cur: DynTree<IR> = branch.clone_as_tree();
    loop {
        let root = cur.root();
        match root.data() {
            IR::Filter(f) if root.num_children() == 1 => {
                filters.push(f.clone());
                let next = root.child(0).clone_as_tree();
                cur = next;
            }
            IR::NodeByLabelScan { node } | IR::AllNodeScan(node) if root.num_children() == 0 => {
                return Some((filters, root.data().clone(), node.clone()));
            }
            _ => return None,
        }
    }
}

fn try_seek_rewrite(
    filter: &QueryExpr<Variable>,
    cp_node: &DynNode<IR>,
    graph: &Graph,
    params: &HashMap<String, Value>,
) -> Option<DynTree<IR>> {
    let conjuncts: Vec<DynTree<ExprIR<Variable>>> = conjuncts(filter)
        .iter()
        .map(NodeRef::clone_as_tree)
        .collect();
    let children: Vec<DynNode<IR>> = cp_node.children().collect();
    if children.len() < 2 {
        return None;
    }
    let child_vars: Vec<HashSet<u32>> = children.iter().map(collect_subtree_variables).collect();

    for (bi, branch) in children.iter().enumerate() {
        let Some((chain_filters, scan, scan_node)) = scan_chain(branch) else {
            continue;
        };
        let p = scan_node.alias.id;
        let others: HashSet<u32> = child_vars
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != bi)
            .flat_map(|(_, v)| v.iter().copied())
            .collect();
        // Conjuncts that read the scanned node and a variable of the other
        // branches, and nothing else: they can be checked once the scan has
        // the other branches' rows as its input.
        let (moved, kept): (Vec<_>, Vec<_>) = conjuncts.iter().cloned().partition(|c| {
            let vars = collect_expr_variables(c);
            vars.contains(&p)
                && vars.iter().any(|v| others.contains(v))
                && vars.iter().all(|v| *v == p || others.contains(v))
        });
        if moved.is_empty() {
            continue;
        }
        // A branch filter that is not deterministic would run once per input
        // row instead of once per scanned node.
        if chain_filters.iter().any(|f| has_non_deterministic(f)) {
            continue;
        }
        // What the seek evaluates for a conjunct, before the scan: a key that
        // is not deterministic would be evaluated by the scan and again by
        // its retained filter. That runs before the branch's own filters, so
        // a key that can fail stays behind them, and an `OR` is evaluated
        // whole by the index query.
        let safe_key = |c: &DynTree<ExprIR<Variable>>| {
            !has_non_deterministic(c)
                && !((!chain_filters.is_empty() || matches!(c.root().data(), ExprIR::Or))
                    && may_fail(c))
        };
        // The key of `id(p) <op> key`, when the seek can evaluate it.
        let safe_id_key = |c: &DynTree<ExprIR<Variable>>| {
            get_id_filter(&c.root(), &scan_node.alias).filter(|(key, _)| safe_key(key))
        };
        // `id(p) = key` finds at most one node per input row, with no index:
        // the `id()` comparisons are moved and `utilize_node_by_id` turns them
        // into one lookup. Not on more than one label: that lookup intersects
        // the label matrices for every row (`Graph::get_nodes`), which costs
        // more than the hash join's one scan. Otherwise only the conjuncts one
        // index serves are moved onto the scan; all must use the same label's
        // index, as `utilize_index` merges them into one lookup, and a key
        // that reads the scanned node itself cannot be evaluated before the
        // scan. The other cross-branch conjuncts stay above the branch's own
        // filters, where the hash-join plan evaluated them.
        let by_id = scan_node.labels.len() <= 1
            && moved
                .iter()
                .any(|c| matches!(safe_id_key(c), Some((_, ExprIR::Eq))));
        let (moved, rest): (Vec<_>, Vec<_>) = if by_id {
            moved.into_iter().partition(|c| safe_id_key(c).is_some())
        } else {
            let mut seek_label: Option<Arc<String>> = None;
            moved.into_iter().partition(|c| {
                if !safe_key(c) || reads_node_bare(c, p) {
                    return false;
                }
                let Some(label) = seek_index_label(&scan_node, c, graph) else {
                    return false;
                };
                match &seek_label {
                    None => {
                        seek_label = Some(label);
                        true
                    }
                    Some(first) => *first == label,
                }
            })
        };
        if moved.is_empty() {
            continue;
        }
        let kept: Vec<_> = kept.into_iter().chain(rest).collect();
        let moved_filter = if moved.len() == 1 {
            moved[0].clone()
        } else {
            ExprIR::and_of(moved)
        };
        if !by_id && !can_seek_index(&scan_node, &moved_filter, graph) {
            continue;
        }
        let seek_cost = if by_id {
            ID_SEEK_COST_IN_SCANNED_NODES
        } else {
            SEEK_COST_IN_SCANNED_NODES
        };
        let feeder_rows = children
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != bi)
            .map(|(_, c)| estimate_rows(c, graph, params))
            .try_fold(1.0_f64, |acc, rows| rows.map(|r| acc * r));
        // An estimate we cannot make is taken as a handful of rows, the
        // shape of an OLTP lookup: the plan before consecutive MATCH clauses
        // were kept apart always seeked.
        let scanned = min_label_count(&scan_node, graph).unwrap_or_else(|| graph.node_count());
        #[allow(clippy::cast_precision_loss, reason = "a row-count estimate")]
        if feeder_rows.is_some_and(|rows| rows * seek_cost > scanned as f64) {
            continue;
        }

        // Feed the scan with the other branches.
        let mut feeders: Vec<DynTree<IR>> = children
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != bi)
            .map(|(_, c)| c.clone_as_tree())
            .collect();
        let feed = if feeders.len() == 1 {
            feeders.remove(0)
        } else {
            tree!(IR::CartesianProduct; feeders)
        };
        let mut seek = DynTree::new(scan);
        seek.root_mut().push_child_tree(feed);
        // The moved predicate sits directly on the scan, where
        // `utilize_index` and `utilize_node_by_id` look for it, below the
        // branch's own filters.
        let mut chain = tree!(IR::Filter(Arc::new(moved_filter)), seek);
        for f in chain_filters.into_iter().rev() {
            chain = tree!(IR::Filter(f), chain);
        }
        if !kept.is_empty() {
            let kept_filter = Arc::new(ExprIR::and_of(kept));
            chain = tree!(IR::Filter(kept_filter), chain);
        }
        return Some(chain);
    }
    None
}

/// Number of rows the plan below `node` is expected to produce, or `None`
/// when it cannot be told from the plan, the graph and the parameters.
///
/// Only the operators that make a branch large are modelled: a list that is
/// unwound, a scan of a label, and the product of branches. An index or id
/// lookup counts as one row per input row and every other operator passes its input
/// through, so the estimate is an upper bound for a selective clause and
/// `None` for the rest (a traversal, a call).
fn estimate_rows(
    node: &DynNode<IR>,
    graph: &Graph,
    params: &HashMap<String, Value>,
) -> Option<f64> {
    let inputs = || -> Option<f64> {
        node.children()
            .map(|c| estimate_rows(&c, graph, params))
            .try_fold(1.0_f64, |acc, rows| rows.map(|r| acc * r))
    };
    #[allow(clippy::cast_precision_loss, reason = "a row-count estimate")]
    match node.data() {
        IR::Unwind { expr, .. } => {
            let expr = substitute_params(expr, params);
            let len = unwound_len(&expr)?;
            Some(inputs()? * len as f64)
        }
        IR::NodeByLabelScan { node: n } | IR::AllNodeScan(n) => {
            // `utilize_index` and `utilize_node_by_id` run after this pass: a
            // scan under a filter they will serve is a lookup, not a scan.
            if let Some(parent) = node.parent()
                && let IR::Filter(f) = parent.data()
                && is_point_lookup(n, f, graph)
            {
                return inputs();
            }
            let nodes = min_label_count(n, graph).unwrap_or_else(|| graph.node_count());
            Some(inputs()? * nodes as f64)
        }
        IR::NodeByIndexScan { .. }
        | IR::Filter(_)
        | IR::Project { .. }
        | IR::Aggregate { .. }
        | IR::CartesianProduct => inputs(),
        _ => None,
    }
}

/// Length of the list an `UNWIND` expression (parameters already substituted)
/// produces, when it is a list value, a list literal or `range` over integer
/// constants.
fn unwound_len(expr: &DynTree<ExprIR<Variable>>) -> Option<usize> {
    let root = expr.root();
    match root.data() {
        ExprIR::Constant(Value::List(l)) => Some(l.len()),
        ExprIR::List => Some(root.num_children()),
        ExprIR::FuncInvocation(f) if f.name.as_str() == "range" => {
            let int = |i: usize| match root.get_child(i)?.data() {
                ExprIR::Constant(Value::Int(v)) => Some(i128::from(*v)),
                _ => None,
            };
            let (from, to) = (int(0)?, int(1)?);
            let step = if root.num_children() > 2 { int(2)? } else { 1 };
            if step == 0 {
                return None;
            }
            let n = (to - from) / step + 1;
            usize::try_from(n.max(0)).ok()
        }
        _ => None,
    }
}

/// Whether the scan of `node` under `filter` becomes a lookup that the row
/// estimate counts as one row per input row: one of the conjuncts is
/// `id(node) = key`, or an index serves `filter` and one of its conjuncts is
/// an equality the index serves. A range lookup can return the whole label,
/// so it does not count.
fn is_point_lookup(
    node: &Arc<QueryNode<Arc<String>, Variable>>,
    filter: &QueryExpr<Variable>,
    graph: &Graph,
) -> bool {
    let conjuncts = conjuncts(filter);
    conjuncts
        .iter()
        .any(|c| matches!(get_id_filter(c, &node.alias), Some((_, ExprIR::Eq))))
        || (can_seek_index(node, filter, graph)
            && conjuncts.iter().any(|c| {
                matches!(c.data(), ExprIR::Eq) && can_seek_index(node, &c.clone_as_tree(), graph)
            }))
}

/// Whether evaluating `expr` can raise an error (arithmetic on a value of the
/// wrong type or out of range, a call, an element access), which a guard on
/// the scanned branch may be there to avoid.
fn may_fail(expr: &DynTree<ExprIR<Variable>>) -> bool {
    expr.root().walk::<Bfs>().any(|data| {
        matches!(
            data,
            ExprIR::Add
                | ExprIR::Sub
                | ExprIR::Mul
                | ExprIR::Negate
                | ExprIR::Div
                | ExprIR::Modulo
                | ExprIR::Pow
                | ExprIR::GetElement
                | ExprIR::GetElements
                | ExprIR::FuncInvocation(_)
        )
    })
}

/// Whether `expr` calls a function whose result can differ between calls.
fn has_non_deterministic(expr: &DynTree<ExprIR<Variable>>) -> bool {
    expr.root()
        .walk::<Bfs>()
        .any(|data| matches!(data, ExprIR::FuncInvocation(f) if f.non_deterministic || matches!(f.fn_type, FnType::Udf)))
}

/// Whether `expr` reads the node `p` other than through a property access
/// (`id(p)`, `p` itself): a seek key must not depend on the scanned node.
fn reads_node_bare(
    expr: &DynTree<ExprIR<Variable>>,
    p: u32,
) -> bool {
    expr.root().indices::<Bfs>().any(|idx| {
        let node = expr.node(idx);
        matches!(node.data(), ExprIR::Variable(v) if v.id == p)
            && !node
                .parent()
                .is_some_and(|parent| matches!(parent.data(), ExprIR::Property(_)))
    })
}
