//! Filter push-down optimizer pass.
//!
//! Moves filter conjuncts as close as possible to the operators that produce
//! their referenced variables. This reduces the number of intermediate rows
//! flowing through the plan by filtering early.
//!
//! ## Main Transformation
//!
//! Given a Filter with an AND predicate sitting above a multi-child operator
//! (e.g. CartesianProduct), each conjunct is routed to the deepest child
//! whose output variables fully cover the conjunct's referenced variables.
//!
//! ```text
//! Before:                              After:
//!
//! Filter(AND(cond_a, cond_b))          CartesianProduct
//!   |                                    |           |
//!   v                                    v           v
//! CartesianProduct                     Filter(a)   Filter(b)
//!   |           |                        |           |
//!   v           v                        v           v
//! ChildA      ChildB                   ChildA      ChildB
//! ```
//!
//! Conjuncts that reference variables from multiple children (cross-product
//! predicates) remain at the original Filter level.
//!
//! ## Additional behaviors
//!
//! - **Filter merging**: Two stacked Filter nodes are merged into a single
//!   AND filter before push-down is attempted.
//! - **Apply awareness**: When a Filter is inside an Apply's right branch,
//!   variables from the left branch (propagated via Argument) are included
//!   in the available variable set so that filters can be pushed down into
//!   sub-plans that receive those variables through Argument leaves.

use std::collections::HashSet;
use std::sync::Arc;

use orx_tree::{Bfs, DynTree, NodeRef};

use crate::{
    parser::ast::{ExprIR, Variable},
    tree,
};

use super::super::{IR, subtree_contains};
use super::{collect_expr_variables, collect_subtree_variables};

/// Pushes filter conjuncts down through nodes.
///
/// Transforms:
/// ```text
/// Filter(AND(cond_a, cond_b))
///   └─ CartesianProduct
///        ├─ ChildA
///        └─ ChildB
/// ```
/// Into:
/// ```text
/// CartesianProduct
///   ├─ Filter(cond_a)
///   │    └─ ChildA
///   └─ Filter(cond_b)
///        └─ ChildB
/// ```
///
/// Each conjunct is routed to the child whose variables fully cover the
/// conjunct's referenced variables. Conjuncts that span multiple children
/// remain at the current level.
pub(super) fn push_filters_down(optimized_plan: &mut DynTree<IR>) {
    loop {
        let mut changed = false;
        let indices = optimized_plan.root().indices::<Bfs>().collect::<Vec<_>>();
        for idx in indices {
            let IR::Filter(filter) = optimized_plan.node(idx).data() else {
                continue;
            };

            // Merge stacked filters: if this filter's child is also a filter,
            // combine their conjuncts into a single AND filter.
            if let Some(child) = optimized_plan.node(idx).get_child(0)
                && let IR::Filter(child_filter) = child.data()
            {
                let child_filter = child_filter.clone();
                let filter = filter.clone();
                let child_idx = child.idx();

                // Flatten conjuncts from both filters
                let mut conjuncts: Vec<DynTree<ExprIR<Variable>>> = vec![];
                for f in [&filter, &child_filter] {
                    if matches!(f.root().data(), ExprIR::And) {
                        conjuncts.extend(f.root().children().map(|c| c.clone_as_tree()));
                    } else {
                        conjuncts.push((**f).clone());
                    }
                }

                let merged = if conjuncts.len() == 1 {
                    Arc::new(conjuncts.into_iter().next().unwrap())
                } else {
                    Arc::new(tree!(ExprIR::And; conjuncts))
                };
                *optimized_plan.node_mut(idx).data_mut() = IR::Filter(merged);
                optimized_plan.node_mut(child_idx).take_out();

                changed = true;
                break;
            }

            if !optimized_plan
                .node(idx)
                .children()
                .any(|c| c.num_children() > 0)
            {
                continue; // Skip if filter already is downstream
            }
            let filter = filter.clone();

            // Split filter into individual conjuncts
            let conjuncts: Vec<DynTree<ExprIR<Variable>>> =
                if matches!(filter.root().data(), ExprIR::And) {
                    filter
                        .root()
                        .children()
                        .map(|c| c.clone_as_tree())
                        .collect()
                } else {
                    vec![(*filter).clone()]
                };

            // Collect children and the variables they provide
            let mut children = Vec::new();
            for child in optimized_plan.node(idx).children().filter(|c| {
                c.num_children() > 0
                    && !matches!(
                        c.data(),
                        IR::Project { .. }
                            | IR::Aggregate { .. }
                            | IR::Merge { .. }
                            | IR::Argument(_)
                            | IR::IncludePending { .. }
                            | IR::SemiApply
                            | IR::AntiSemiApply
                            | IR::OrApplyMultiplexer(_)
                            | IR::Optional(_)
                    )
            }) {
                for grandchild in child.children() {
                    children.push((grandchild.idx(), collect_subtree_variables(&grandchild)));
                }
            }

            // Compute inherited variables from Apply context.
            // When Apply propagates bound variables via Argument leaves,
            // the right branch effectively has access to the left branch's
            // variables. We augment variable sets accordingly so filters
            // referencing bound variables can be pushed down.
            let mut inherited = HashSet::new();

            // Case 1: Filter's child is Apply — left branch vars are
            // available in the right branch via Argument.
            if let Some(child) = optimized_plan.node(idx).get_child(0)
                && matches!(child.data(), IR::Apply)
                && let Some((_, left_vars)) = children.first()
            {
                inherited.extend(left_vars.iter());
            }

            // Case 2: Filter is inside an Apply's right branch.
            // The left branch's variables are available in the right branch
            // via Argument propagation. Only applies when the filter is
            // actually in the right subtree (not the left subtree itself).
            // Stop the ancestor walk at Merge nodes — Merge's internal
            // match sub-plan has its own Argument leaves that receive
            // variables from Merge's input, not from an enclosing Apply.
            {
                let mut ancestor = idx;
                while let Some(parent) = optimized_plan.node(ancestor).parent() {
                    if matches!(parent.data(), IR::Apply) {
                        // Only inherit left-branch variables if the filter
                        // is NOT inside the left branch (child 0) itself.
                        let left_child_idx = parent.child(0).idx();
                        let filter_in_left = {
                            let mut cur = idx;
                            loop {
                                if cur == left_child_idx {
                                    break true;
                                }
                                if let Some(p) = optimized_plan.node(cur).parent() {
                                    if p.idx() == parent.idx() {
                                        break false;
                                    }
                                    cur = p.idx();
                                } else {
                                    break false;
                                }
                            }
                        };
                        if !filter_in_left {
                            let left_vars = collect_subtree_variables(&parent.child(0));
                            inherited.extend(left_vars);
                        }
                        break;
                    }
                    if matches!(parent.data(), IR::Merge { .. }) {
                        break;
                    }
                    ancestor = parent.idx();
                }
            }

            // Augment variable sets for subtrees containing Argument leaves.
            if !inherited.is_empty() {
                for (child_idx, vars) in &mut children {
                    if subtree_contains(optimized_plan, *child_idx, |ir| {
                        matches!(ir, IR::Argument(_))
                    }) {
                        vars.extend(&inherited);
                    }
                }
            }

            // Route each conjunct to the child that provides all its variables
            let mut child_conjuncts: Vec<Vec<DynTree<ExprIR<Variable>>>> =
                vec![vec![]; children.len()];
            let mut remaining: Vec<DynTree<ExprIR<Variable>>> = vec![];

            for conjunct in conjuncts {
                let conj_vars = collect_expr_variables(&conjunct);
                let mut matched_any = false;
                for (i, (_, child_vars)) in children.iter().enumerate() {
                    if conj_vars.iter().all(|v| child_vars.contains(v)) {
                        child_conjuncts[i].push(conjunct.clone());
                        matched_any = true;
                    }
                }
                if !matched_any {
                    remaining.push(conjunct);
                }
            }

            // Try CP splitting for remaining cross-branch conjuncts.
            // When a conjunct references variables from a proper subset of CP
            // branches (>=2 but < all), extract those branches into an inner
            // CP wrapped with that conjunct as a filter.
            if !remaining.is_empty()
                && child_conjuncts.iter().all(Vec::is_empty)
                && let Some(filter_child) = optimized_plan.node(idx).get_child(0)
                && matches!(filter_child.data(), IR::CartesianProduct)
                && filter_child.num_children() > 2
            {
                let total_children = children.len();
                let mut split_idx = None;

                for (ci, conjunct) in remaining.iter().enumerate() {
                    let conj_vars = collect_expr_variables(conjunct);
                    // Find solving branches: children that contribute >=1
                    // variable referenced by this conjunct.
                    let solving: Vec<usize> = children
                        .iter()
                        .enumerate()
                        .filter(|(_, (_, child_vars))| {
                            conj_vars.iter().any(|v| child_vars.contains(v))
                        })
                        .map(|(i, _)| i)
                        .collect();

                    if solving.len() >= 2 && solving.len() < total_children {
                        split_idx = Some((ci, solving));
                        break;
                    }
                }

                if let Some((ci, solving)) = split_idx {
                    // Rebuild the entire plan using a recursive approach
                    // to avoid in-place tree mutation issues with prune.
                    let split_filter_idx = idx;
                    let conjunct_to_split = remaining[ci].clone();
                    let solving_set: HashSet<usize> = solving.iter().copied().collect();
                    let mut remaining_conjuncts = remaining.clone();
                    remaining_conjuncts.remove(ci);

                    *optimized_plan = rebuild_with_cp_split(
                        optimized_plan,
                        split_filter_idx,
                        &conjunct_to_split,
                        &solving_set,
                        &remaining_conjuncts,
                    );

                    changed = true;
                    break;
                }
            }

            // Skip if nothing can be pushed down
            if child_conjuncts.iter().all(Vec::is_empty) {
                continue;
            }

            // For each child with matching conjuncts: add a Filter-wrapped
            // clone as a sibling, then prune the original.
            for (i, conjuncts) in child_conjuncts.into_iter().enumerate() {
                if conjuncts.is_empty() {
                    continue;
                }

                let child_idx = children[i].0;

                // Build the filter expression for this child
                let filter_expr = if conjuncts.len() == 1 {
                    Arc::new(conjuncts.into_iter().next().unwrap())
                } else {
                    Arc::new(tree!(ExprIR::And; conjuncts))
                };

                // Insert the new filter node above the child
                optimized_plan
                    .node_mut(child_idx)
                    .push_parent(IR::Filter(filter_expr));
            }

            // Update or remove the original Filter
            if remaining.is_empty() {
                optimized_plan.node_mut(idx).take_out();
            } else if remaining.len() == 1 {
                *optimized_plan.node_mut(idx).data_mut() =
                    IR::Filter(Arc::new(remaining.into_iter().next().unwrap()));
            } else {
                *optimized_plan.node_mut(idx).data_mut() =
                    IR::Filter(Arc::new(tree!(ExprIR::And; remaining)));
            }

            changed = true;
            break; // Restart traversal after structural modification
        }

        if !changed {
            break;
        }
    }
}

/// Rebuilds the plan tree, splitting one CP node at the given filter index.
///
/// Instead of in-place tree mutation (which can trigger crashes in
/// orx-tree's unsafe prune code), this rebuilds the entire tree,
/// substituting the target filter+CP node with the split version.
fn rebuild_with_cp_split(
    plan: &DynTree<IR>,
    target_filter_idx: orx_tree::NodeIdx<orx_tree::Dyn<IR>>,
    conjunct: &DynTree<ExprIR<Variable>>,
    solving_set: &HashSet<usize>,
    remaining_conjuncts: &[DynTree<ExprIR<Variable>>],
) -> DynTree<IR> {
    fn rebuild_node(
        node: &orx_tree::DynNode<IR>,
        target_filter_idx: orx_tree::NodeIdx<orx_tree::Dyn<IR>>,
        conjunct: &DynTree<ExprIR<Variable>>,
        solving_set: &HashSet<usize>,
        remaining_conjuncts: &[DynTree<ExprIR<Variable>>],
    ) -> DynTree<IR> {
        use orx_tree::DynTree;

        if node.idx() == target_filter_idx {
            // This is the filter node we want to transform.
            // Its child(0) is the CartesianProduct.
            let cp = node.child(0);
            let cp_children: Vec<DynTree<IR>> = cp.children().map(|c| c.clone_as_tree()).collect();

            let extracted: Vec<DynTree<IR>> = cp_children
                .iter()
                .enumerate()
                .filter(|(i, _)| solving_set.contains(i))
                .map(|(_, t)| t.clone())
                .collect();
            let others: Vec<DynTree<IR>> = cp_children
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !solving_set.contains(i))
                .map(|(_, t)| t)
                .collect();

            let inner_cp = tree!(IR::CartesianProduct; extracted);
            let filter_expr = Arc::new(conjunct.clone());
            let inner_filtered = tree!(IR::Filter(filter_expr); [inner_cp]);

            let mut new_children = others;
            new_children.push(inner_filtered);

            if remaining_conjuncts.is_empty() {
                tree!(IR::CartesianProduct; new_children)
            } else {
                let remaining_filter = if remaining_conjuncts.len() == 1 {
                    Arc::new(remaining_conjuncts[0].clone())
                } else {
                    #[allow(clippy::unnecessary_to_owned)]
                    Arc::new(tree!(ExprIR::And; remaining_conjuncts.to_vec()))
                };
                let cp = tree!(IR::CartesianProduct; new_children);
                tree!(IR::Filter(remaining_filter), cp)
            }
        } else {
            // Default: clone this node and recursively rebuild children.
            let mut new_tree = DynTree::new(node.data().clone());
            for child in node.children() {
                let child_tree = rebuild_node(
                    &child,
                    target_filter_idx,
                    conjunct,
                    solving_set,
                    remaining_conjuncts,
                );
                new_tree.root_mut().push_child_tree(child_tree);
            }
            new_tree
        }
    }

    rebuild_node(
        &plan.root(),
        target_filter_idx,
        conjunct,
        solving_set,
        remaining_conjuncts,
    )
}
