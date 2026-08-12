//! Query plan optimization passes.
//!
//! The optimizer transforms the logical execution plan produced by the planner
//! to improve performance. It applies a fixed sequence of rewrite passes, each
//! making local transformations to the IR tree.
//!
//! ## Pass Ordering
//!
//! Passes run in the following order:
//!
//! ```text
//! Input plan (from Planner)
//!       |
//!       v
//! 1. eliminate_true_filters   -- Remove trivial Filter(true) nodes
//!       |
//!       v
//! 2. select_scan_node         -- Pick the best starting node for traversal
//!       |                        chains, possibly reversing chain direction
//!       v
//! 3. push_filters_down        -- Move Filter conjuncts closer to the
//!       |                        operators that produce their variables
//!       v
//! 4. replace_cartesian_       -- Convert CartesianProduct + equality
//!    with_hash_join               Filter into ValueHashJoin
//!       |
//!       v
//! 5. absorb_edge_filters_     -- Fold edge-only filters into
//!    into_vlt                     CondVarLenTraverse's per-hop filter
//!       |
//!       v
//! 6. utilize_index            -- Replace NodeByLabelScan + Filter with
//!       |                        NodeByIndexScan when an index exists
//!       v
//! 7. utilize_node_by_id       -- Replace label scan + id() filter with
//!       |                        NodeByLabelAndIdScan or NodeByIdSeek
//!       v
//! Optimized plan
//! ```
//!
//! ## Implementation Pattern
//!
//! Each pass uses a collect-then-iterate loop: collect candidate node indices
//! via a BFS traversal, attempt one transformation, then restart the traversal
//! if the tree structure changed. This avoids issues with invalidated indices
//! after in-place tree mutations.

mod eliminate_true_filters;
mod fuse_anonymous_traverse;
mod fuse_optional_traverse;
mod push_filters_down;
mod reduce_bound_edge;
mod reduce_count;
mod reduce_expand_into;
mod reduce_var_len_path;
mod reorder_labels;
mod replace_cartesian_with_hash_join;
mod select_scan_node;
mod utilize_index;
mod utilize_node_by_id;

use std::collections::{HashMap, HashSet};

use orx_tree::{Bfs, DynTree, NodeRef};

use crate::{
    graph::graph::Graph,
    parser::ast::{ExprIR, Variable},
    runtime::value::Value,
};

use super::IR;

use eliminate_true_filters::eliminate_true_filters;
use fuse_anonymous_traverse::fuse_anonymous_traverse;
use fuse_optional_traverse::fuse_optional_traverse;
use push_filters_down::push_filters_down;
use reduce_bound_edge::reduce_bound_edge;
use reduce_count::reduce_count;
use reduce_expand_into::reduce_expand_into;
use reduce_var_len_path::reduce_var_len_path;
use reorder_labels::reorder_labels;
use replace_cartesian_with_hash_join::replace_cartesian_with_hash_join;
use select_scan_node::select_scan_node;
use utilize_index::utilize_index;
use utilize_node_by_id::utilize_node_by_id;

/// Collects all variable IDs referenced in an expression tree.
pub(crate) fn collect_expr_variables(expr: &DynTree<ExprIR<Variable>>) -> HashSet<u32> {
    let mut vars = HashSet::new();
    for idx in expr.root().indices::<Bfs>() {
        if let ExprIR::Variable(var) = expr.node(idx).data() {
            vars.insert(var.id);
        }
    }
    vars
}

/// Collects all variable IDs provided by a plan subtree.
pub(crate) fn collect_subtree_variables(node: &orx_tree::DynNode<IR>) -> HashSet<u32> {
    use crate::runtime::runtime::GetVariables;
    let mut vars = HashSet::new();
    for var in node.get_variables() {
        vars.insert(var.id);
    }
    vars
}

/// Optimizes a query execution plan.
///
/// Applies all optimization passes to the plan and returns the optimized version.
/// The original plan is not modified.
///
/// # Arguments
/// * `plan` - The unoptimized execution plan
/// * `graph` - The graph (needed to check for index availability)
///
/// # Returns
/// An optimized copy of the plan
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn optimize(
    plan: &DynTree<IR>,
    graph: &Graph,
    params: &HashMap<String, Value>,
) -> DynTree<IR> {
    let mut optimized_plan = plan.clone();

    reduce_count(&mut optimized_plan, graph);
    reduce_expand_into(&mut optimized_plan);
    reduce_var_len_path(&mut optimized_plan);
    eliminate_true_filters(&mut optimized_plan, params);
    select_scan_node(&mut optimized_plan, graph);
    push_filters_down(&mut optimized_plan);
    fuse_anonymous_traverse(&mut optimized_plan);
    replace_cartesian_with_hash_join(&mut optimized_plan);
    // Re-run path reduction: folding an edge-only filter into a
    // CondVarLenTraverse can remove the last ancestor that consumed the path
    // alias, so a path kept by the first pass may now be skippable. Safe here
    // because `ir_references_variable` inspects `ValueHashJoin` keys, the only
    // path consumer `replace_cartesian_with_hash_join` adds in between.
    reduce_var_len_path(&mut optimized_plan);
    utilize_index(&mut optimized_plan, graph);
    utilize_node_by_id(&mut optimized_plan);

    reorder_labels(&mut optimized_plan, graph);

    // Runs last so no earlier pass has to reason about optional traverses.
    fuse_optional_traverse(&mut optimized_plan);

    // After every pass that rebuilds a CondTraverse or moves its ancestors, so
    // the "does anything read this edge" answer is the final plan's.
    reduce_bound_edge(&mut optimized_plan);

    debug_assert_no_pattern_attrs(&optimized_plan);

    optimized_plan
}

/// Asserts that no match-side operator still carries inline attributes on its
/// pattern.
///
/// A MATCH pattern's `{k: v}` is a predicate, and the planner lowers it into
/// exactly one place: an `IR::Filter` for nodes and for the fixed-length
/// traverses, or `edge_filter` for the walks, which cannot express it as a
/// Filter. Anything left on the pattern is a second source of truth. That is
/// not hypothetical — the two used to disagree, `push_filters_down` merged the
/// duplicates into `And(p, p)`, and no index could serve the result.
///
/// `IR::Create` and `IR::Merge` are exempt and must stay so: there the attrs
/// are the property template for the entity being built, read by
/// `runtime/ops/create.rs` and `runtime/ops/merge.rs`.
#[cfg(debug_assertions)]
fn debug_assert_no_pattern_attrs(plan: &DynTree<IR>) {
    use crate::parser::ast::{QueryNode, QueryRelationship};
    use std::sync::Arc;

    fn node_is_clean(node: &QueryNode<Arc<String>, Variable>) -> bool {
        node.attrs.root().num_children() == 0
    }
    fn rel_is_clean(rel: &QueryRelationship<Arc<String>, Arc<String>, Variable>) -> bool {
        rel.attrs.root().num_children() == 0 && node_is_clean(&rel.from) && node_is_clean(&rel.to)
    }

    for idx in plan.root().indices::<Bfs>() {
        let (what, clean) = match plan.node(idx).data() {
            IR::AllNodeScan(node)
            | IR::NodeByLabelScan { node }
            | IR::IncludePending { node }
            | IR::NodeByIndexScan { node, .. }
            | IR::NodeByLabelAndIdScan { node, .. }
            | IR::NodeByIdSeek { node, .. } => ("node scan", node_is_clean(node)),
            IR::EdgeByIndexScan { relationship, .. }
            | IR::CondVarLenTraverse { relationship, .. }
            | IR::ExpandInto { relationship, .. } => ("traverse", rel_is_clean(relationship)),
            IR::AllShortestPaths(relationship) => ("traverse", rel_is_clean(relationship)),
            IR::CondTraverse {
                relationship,
                chain,
                ..
            } => (
                "traverse",
                rel_is_clean(relationship) && chain.iter().all(|hop| rel_is_clean(hop)),
            ),
            _ => continue,
        };
        debug_assert!(
            clean,
            "{what} still carries inline pattern attrs after planning; they must be \
             lowered exactly once (Filter, or edge_filter for the walks) and stripped: \
             {}",
            plan.node(idx).data()
        );
    }
}

#[cfg(not(debug_assertions))]
const fn debug_assert_no_pattern_attrs(_plan: &DynTree<IR>) {}
