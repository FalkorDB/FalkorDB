//! Hash join replacement optimizer pass.
//!
//! Replaces a `CartesianProduct` paired with an equality `Filter` by a
//! `ValueHashJoin`, turning an O(N*M) nested-loop cross product into an
//! O(N+M) hash-based equi-join.
//!
//! ## Transformation
//!
//! The pass looks for this pattern:
//!
//! ```text
//! Before:                               After:
//!
//! Filter(a.name = b.name)               ValueHashJoin(a.name, b.name)
//!   |                                     |           |
//!   v                                     v           v
//! CartesianProduct                      ChildA      ChildB
//!   |           |
//!   v           v
//! ChildA      ChildB
//! ```
//!
//! The equality conjunct's left-hand side must reference variables produced
//! exclusively by one CartesianProduct child, and the right-hand side must
//! reference variables produced exclusively by a different child.
//!
//! **AND filter with extra conjuncts:**
//!
//! When the filter is an AND, exactly one equality conjunct is consumed for
//! the hash join; the remaining conjuncts stay as a reduced Filter above the
//! new ValueHashJoin.
//!
//! **Multi-child CartesianProduct:**
//!
//! If the CartesianProduct has more than two children, the two children
//! involved in the join are pulled out into a ValueHashJoin and the rest
//! remain as CartesianProduct children:
//!
//! ```text
//! Before:                     After:
//!
//! Filter(a.x = b.x)          CartesianProduct
//!   |                           |           |
//!   v                           v           v
//! CartesianProduct            ChildC      ValueHashJoin(a.x, b.x)
//!   |     |     |                           |           |
//!   v     v     v                           v           v
//! ChildA ChildB ChildC                    ChildA      ChildB
//! ```
//!
//! The pass is applied recursively so nested join opportunities are found.

use std::collections::HashSet;
use std::sync::Arc;

use orx_tree::{DynNode, DynTree, NodeRef};

use crate::{
    parser::ast::{ExprIR, QueryExpr, Variable},
    tree,
};

use super::super::IR;
use super::{collect_expr_variables, collect_subtree_variables};

/// Replaces `Filter(a = b) -> CartesianProduct(ChildA, ChildB)` with
/// `ValueHashJoin(ChildA, ChildB)` when the equality predicate's left side
/// references only variables from one child and the right side references
/// only variables from the other.
///
/// Uses a recursive tree-rebuild approach to cleanly restructure multi-child
/// CartesianProduct nodes.
pub(super) fn replace_cartesian_with_hash_join(optimized_plan: &mut DynTree<IR>) {
    let new_plan = rebuild_with_hash_joins(&optimized_plan.root());
    *optimized_plan = new_plan;
}

/// Recursively rebuilds the plan tree, converting Filter -> CartesianProduct
/// patterns into ValueHashJoin where applicable.
fn rebuild_with_hash_joins(node: &DynNode<IR>) -> DynTree<IR> {
    if let IR::Filter(filter) = node.data()
        && let Some(child) = node.get_child(0)
        && matches!(child.data(), IR::CartesianProduct)
        && let Some(result) = try_hash_join_rewrite(filter, &child)
    {
        // Recursively apply to the result (may find more join opportunities).
        return rebuild_with_hash_joins(&result.root());
    }

    // Default: clone this node's data and recursively rebuild children.
    let mut new_tree = DynTree::new(node.data().clone());
    for child in node.children() {
        let child_tree = rebuild_with_hash_joins(&child);
        new_tree.root_mut().push_child_tree(child_tree);
    }
    new_tree
}

/// Attempts to find one equality conjunct in the filter that can be converted
/// to a ValueHashJoin. Returns the rebuilt subtree if successful.
fn try_hash_join_rewrite(
    filter: &QueryExpr<Variable>,
    cp_node: &DynNode<IR>,
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

    let cp_children_vars: Vec<HashSet<u32>> = cp_node
        .children()
        .map(|c| collect_subtree_variables(&c))
        .collect();

    if cp_children_vars.len() < 2 {
        return None;
    }

    let mut found = None;
    for (ci, conjunct) in conjuncts.iter().enumerate() {
        if !matches!(conjunct.root().data(), ExprIR::Eq) || conjunct.root().num_children() < 2 {
            continue;
        }
        let lhs_tree = conjunct.root().child(0).clone_as_tree();
        let rhs_tree = conjunct.root().child(1).clone_as_tree();
        let lhs_vars = collect_expr_variables(&lhs_tree);
        let rhs_vars = collect_expr_variables(&rhs_tree);
        if lhs_vars.is_empty() || rhs_vars.is_empty() {
            continue;
        }

        'outer: for (li, l_vars) in cp_children_vars.iter().enumerate() {
            for (ri, r_vars) in cp_children_vars.iter().enumerate() {
                if li == ri {
                    continue;
                }
                if lhs_vars.iter().all(|v| l_vars.contains(v))
                    && rhs_vars.iter().all(|v| r_vars.contains(v))
                {
                    found = Some((
                        ci,
                        li,
                        ri,
                        Arc::new(lhs_tree.clone()),
                        Arc::new(rhs_tree.clone()),
                    ));
                    break 'outer;
                }
                if rhs_vars.iter().all(|v| l_vars.contains(v))
                    && lhs_vars.iter().all(|v| r_vars.contains(v))
                {
                    found = Some((
                        ci,
                        li,
                        ri,
                        Arc::new(rhs_tree.clone()),
                        Arc::new(lhs_tree.clone()),
                    ));
                    break 'outer;
                }
            }
        }
        if found.is_some() {
            break;
        }
    }

    let (conj_idx, left_pos, right_pos, lhs_exp, rhs_exp) = found?;

    let cp_child_trees: Vec<DynTree<IR>> = cp_node.children().map(|c| c.clone_as_tree()).collect();

    let vhj = tree!(
        IR::ValueHashJoin { lhs_exp, rhs_exp },
        cp_child_trees[left_pos].clone(),
        cp_child_trees[right_pos].clone()
    );

    let other_children: Vec<DynTree<IR>> = cp_child_trees
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i != left_pos && *i != right_pos)
        .map(|(_, t)| t)
        .collect();

    let join_subtree = if other_children.is_empty() {
        vhj
    } else {
        let mut children = other_children;
        children.push(vhj);
        tree!(IR::CartesianProduct; children)
    };

    let remaining: Vec<DynTree<ExprIR<Variable>>> = conjuncts
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i != conj_idx)
        .map(|(_, c)| c)
        .collect();

    if remaining.is_empty() {
        Some(join_subtree)
    } else {
        let remaining_filter = if remaining.len() == 1 {
            Arc::new(remaining.into_iter().next().unwrap())
        } else {
            Arc::new(tree!(ExprIR::And; remaining))
        };
        let mut result = DynTree::new(IR::Filter(remaining_filter));
        result.root_mut().push_child_tree(join_subtree);
        Some(result)
    }
}
