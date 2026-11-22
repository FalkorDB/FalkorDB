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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use orx_tree::{DynNode, DynTree, NodeRef};

use crate::{
    graph::graph::Graph,
    parser::ast::{ExprIR, QueryExpr, QueryNode, Variable},
    runtime::{functions::FnType, value::Value},
    tree,
};

use super::super::IR;
use super::eliminate_true_filters::substitute_params;
use super::utilize_index::{can_seek_index, seek_index_label};
use super::{collect_expr_variables, collect_subtree_variables};

/// What one index lookup costs, in nodes of a label scan. Measured on a 1M-node
/// label: reading it once (the hash join's build side) takes about 52 ms, a
/// lookup about 17 us, a ratio of 330. Rounded down: a lookup also keeps the
/// result clustered per outer row.
const SEEK_COST_IN_SCANNED_NODES: f64 = 256.0;

/// Rewrites `Filter -> CartesianProduct(.., [Filter ->] NodeByLabelScan)` so
/// the scan reads the other branches' rows and the filter can become an index
/// seek, wherever the label has an index for a predicate that reads them.
pub(super) fn replace_cartesian_with_index_seek(
    optimized_plan: &mut DynTree<IR>,
    graph: &Graph,
    params: &HashMap<String, Value>,
) {
    let new_plan = rebuild(&optimized_plan.root(), graph, params);
    *optimized_plan = new_plan;
}

fn rebuild(
    node: &DynNode<IR>,
    graph: &Graph,
    params: &HashMap<String, Value>,
) -> DynTree<IR> {
    if let IR::Filter(filter) = node.data()
        && let Some(child) = node.get_child(0)
        && matches!(child.data(), IR::CartesianProduct)
        && !child
            .walk::<orx_tree::Bfs>()
            .any(|data| matches!(data, IR::Argument(_)))
        && let Some(result) = try_seek_rewrite(filter, &child, graph, params)
    {
        return rebuild(&result.root(), graph, params);
    }
    let mut new_tree = DynTree::new(node.data().clone());
    for child in node.children() {
        new_tree
            .root_mut()
            .push_child_tree(rebuild(&child, graph, params));
    }
    new_tree
}

/// The `Filter*` chain above a childless `NodeByLabelScan` that makes up
/// `branch`, as the filters (top first) and the scanned node.
fn scan_chain(
    branch: &DynNode<IR>
) -> Option<(
    Vec<QueryExpr<Variable>>,
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
            IR::NodeByLabelScan { node } if root.num_children() == 0 => {
                return Some((filters, node.clone()));
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
    let conjuncts: Vec<DynTree<ExprIR<Variable>>> = if matches!(filter.root().data(), ExprIR::And) {
        filter
            .root()
            .children()
            .map(|c| c.clone_as_tree())
            .collect()
    } else {
        vec![(**filter).clone()]
    };
    let children: Vec<DynNode<IR>> = cp_node.children().collect();
    if children.len() < 2 {
        return None;
    }
    let child_vars: Vec<HashSet<u32>> = children.iter().map(collect_subtree_variables).collect();

    for (bi, branch) in children.iter().enumerate() {
        let Some((chain_filters, scan_node)) = scan_chain(branch) else {
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
        // Only the conjuncts one index serves are moved onto the scan; all
        // must use the same label's index, as `utilize_index` merges them
        // into one lookup. A key that is not deterministic would be evaluated
        // by the scan and again by its retained filter, and a key that reads
        // the scanned node itself cannot be evaluated before the scan. The
        // other cross-branch conjuncts stay above the branch's own filters,
        // where the hash-join plan evaluated them.
        let mut seek_label: Option<Arc<String>> = None;
        let (moved, rest): (Vec<_>, Vec<_>) = moved.into_iter().partition(|c| {
            if has_non_deterministic(c)
                || reads_node_bare(c, p)
                || (!chain_filters.is_empty() && may_fail(c))
            {
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
        });
        if moved.is_empty() {
            continue;
        }
        let kept: Vec<_> = kept.into_iter().chain(rest).collect();
        let moved_filter = if moved.len() == 1 {
            moved[0].clone()
        } else {
            ExprIR::and_of(moved)
        };
        if !can_seek_index(&scan_node, &moved_filter, graph) {
            continue;
        }
        let label_nodes = scan_node
            .labels
            .iter()
            .map(|l| graph.label_node_count(l))
            .min()
            .unwrap_or(0);
        let feeder_rows = children
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != bi)
            .map(|(_, c)| estimate_rows(c, graph, params))
            .try_fold(1.0_f64, |acc, rows| rows.map(|r| acc * r));
        // An estimate we cannot make is taken as a handful of rows, the
        // shape of an OLTP lookup: the plan before consecutive MATCH clauses
        // were kept apart always seeked.
        #[allow(clippy::cast_precision_loss, reason = "a row-count estimate")]
        if feeder_rows.is_some_and(|rows| rows * SEEK_COST_IN_SCANNED_NODES > label_nodes as f64) {
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
        let mut seek = DynTree::new(IR::NodeByLabelScan { node: scan_node });
        seek.root_mut().push_child_tree(feed);
        // The moved predicate sits directly on the scan, where
        // `utilize_index` looks for it, below the branch's own filters.
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
/// unwound, a scan of a label, and the product of branches. An index lookup
/// counts as one row per input row and every other operator passes its input
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
        IR::NodeByLabelScan { node: n } => {
            // `utilize_index` runs after this pass: a scan under a filter it
            // will serve from an index is a lookup, not a scan.
            // A range lookup can return the whole label, so only an
            // equality counts as one row per input row.
            if let Some(parent) = node.parent()
                && let IR::Filter(f) = parent.data()
                && can_seek_index(n, f, graph)
                && has_seekable_equality(n, f, graph)
            {
                return inputs();
            }
            let nodes = n
                .labels
                .iter()
                .map(|l| graph.label_node_count(l))
                .min()
                .unwrap_or(0);
            Some(inputs()? * nodes as f64)
        }
        IR::AllNodeScan(_) => Some(inputs()? * graph.node_count() as f64),
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

/// Whether an equality that an index serves is among the conjuncts of `filter`.
fn has_seekable_equality(
    node: &Arc<QueryNode<Arc<String>, Variable>>,
    filter: &QueryExpr<Variable>,
    graph: &Graph,
) -> bool {
    let root = filter.root();
    match root.data() {
        ExprIR::Eq => true,
        ExprIR::And => root.children().any(|c| {
            matches!(c.data(), ExprIR::Eq) && can_seek_index(node, &c.clone_as_tree(), graph)
        }),
        _ => false,
    }
}

/// Whether evaluating `expr` can raise an error (a division, a call, an
/// element access), which a guard on the scanned branch may be there to avoid.
fn may_fail(expr: &DynTree<ExprIR<Variable>>) -> bool {
    expr.root().walk::<orx_tree::Bfs>().any(|data| {
        matches!(
            data,
            ExprIR::Div
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
        .walk::<orx_tree::Bfs>()
        .any(|data| matches!(data, ExprIR::FuncInvocation(f) if f.non_deterministic || matches!(f.fn_type, FnType::Udf)))
}

/// Whether `expr` reads the node `p` other than through a property access
/// (`id(p)`, `p` itself): a seek key must not depend on the scanned node.
fn reads_node_bare(
    expr: &DynTree<ExprIR<Variable>>,
    p: u32,
) -> bool {
    expr.root().indices::<orx_tree::Bfs>().any(|idx| {
        let node = expr.node(idx);
        matches!(node.data(), ExprIR::Variable(v) if v.id == p)
            && !node
                .parent()
                .is_some_and(|parent| matches!(parent.data(), ExprIR::Property(_)))
    })
}
