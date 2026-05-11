//! Count reduction optimization pass.
//!
//! Replaces scan + aggregate patterns that simply count nodes or relationships
//! with a single `Project` node that reads the count directly from graph
//! metadata. This avoids scanning and aggregating all entities.
//!
//! ## Transformations
//!
//! **Node count (unlabeled):**
//!
//! ```text
//! Before:                     After:
//!
//! Aggregate(COUNT(n))         Project(node_count)
//!   AllNodeScan(n)
//! ```
//!
//! **Node count (labeled):**
//!
//! ```text
//! Before:                        After:
//!
//! Aggregate(COUNT(n))            Project(label_node_count)
//!   NodeByLabelScan(n:Label)
//! ```
//!
//! **Edge count:**
//!
//! ```text
//! Before:                                After:
//!
//! Aggregate(COUNT(r))                    Project(edge_count)
//!   CondTraverse(()-[r]->())
//! ```

use std::sync::Arc;

use orx_tree::{DynTree, NodeRef};

use crate::{
    graph::graph::Graph,
    parser::ast::{ExprIR, Variable},
    tree,
};

use super::super::IR;

/// Attempts to reduce a count aggregation to a direct graph metadata lookup.
///
/// Detects patterns where the entire query is a simple `MATCH ... RETURN COUNT(x)`
/// with no filters, and replaces the Aggregate + scan subtree with a Project
/// that emits the count as a constant integer.
pub(super) fn reduce_count(
    optimized_plan: &mut DynTree<IR>,
    graph: &Graph,
) {
    // Walk the plan looking for Aggregate nodes.
    let indices = optimized_plan
        .root()
        .indices::<orx_tree::Bfs>()
        .collect::<Vec<_>>();

    for idx in indices {
        let IR::Aggregate {
            keys,
            aggregations,
            projections,
            ..
        } = optimized_plan.node(idx).data()
        else {
            continue;
        };

        // Must have zero group-by keys, exactly one aggregation, zero projections.
        if !keys.is_empty() || aggregations.len() != 1 || !projections.is_empty() {
            continue;
        }

        let (agg_var, agg_expr) = &aggregations[0];

        // The aggregation expression must be a single count() function call
        // on a variable.
        let Some(count_var_id) = extract_count_variable(agg_expr) else {
            continue;
        };

        // Examine the child of the Aggregate node.
        let agg_node = optimized_plan.node(idx);
        if agg_node.num_children() != 1 {
            continue;
        }
        let child = agg_node.child(0);

        let count = match child.data() {
            // MATCH (n) RETURN COUNT(n)
            IR::AllNodeScan(node) if node.alias.id == count_var_id && child.num_children() == 0 => {
                Some(graph.node_count() as i64)
            }
            // MATCH (n:Label) RETURN COUNT(n)
            IR::NodeByLabelScan { node, .. }
                if node.alias.id == count_var_id
                    && child.num_children() == 0
                    && node.labels.len() == 1 =>
            {
                // Get count for the single label (the scan label).
                let label = node.labels.iter().next().unwrap();
                Some(graph.label_node_count(label.as_str()) as i64)
            }
            // MATCH ()-[r]->() RETURN COUNT(r) or MATCH ()-[r:Type]->() RETURN COUNT(r)
            // Plan shape: CondTraverse { rel } (leaf, before select_scan_node adds the scan child)
            // Only safe when both endpoints are unlabeled and direction is not reversed.
            IR::CondTraverse {
                relationship,
                transposed,
                ..
            } if relationship.alias.id == count_var_id && !transposed => {
                // The CondTraverse must be a leaf (no children) — this means
                // the pattern is a simple scan, not part of a longer chain.
                if child.num_children() != 0 {
                    continue;
                }
                // Skip when endpoints have labels — the graph-level count
                // doesn't account for endpoint label filtering.
                if !relationship.from.labels.is_empty() || !relationship.to.labels.is_empty() {
                    continue;
                }
                // Skip bidirectional patterns — the graph-level count is directional.
                if relationship.bidirectional {
                    continue;
                }
                // Compute the edge count based on relationship types.
                if relationship.types.is_empty() {
                    // Untyped: total relationship count.
                    Some(graph.relationship_count() as i64)
                } else {
                    // Typed: sum counts per type.
                    let mut total: i64 = 0;
                    for type_name in &relationship.types {
                        if let Some(type_id) = graph.get_type_id(type_name.as_str()) {
                            total += graph.type_edge_count(usize::from(type_id)) as i64;
                        }
                        // If the type doesn't exist, its count is 0.
                    }
                    Some(total)
                }
            }
            _ => None,
        };

        let Some(count_value) = count else {
            continue;
        };

        // Replace the Aggregate subtree with a Project that emits the count
        // as a constant integer.
        let agg_var = agg_var.clone();
        let count_expr: Arc<DynTree<ExprIR<Variable>>> =
            Arc::new(tree!(ExprIR::Integer(count_value)));

        // First prune all children.
        while optimized_plan.node(idx).num_children() > 0 {
            let child_idx = optimized_plan.node(idx).child(0).idx();
            optimized_plan.node_mut(child_idx).prune();
        }

        // Then replace data.
        *optimized_plan.node_mut(idx).data_mut() = IR::Project {
            exprs: vec![(agg_var, count_expr)],
            copies: vec![],
        };
        // Only handle one reduction per plan (the plan shouldn't have multiple
        // such patterns, but breaking is safer after tree mutation).
        break;
    }
}

/// If the expression is `count(var)`, returns `Some(var.id)`.
/// Returns `None` for any other expression shape.
fn extract_count_variable(expr: &DynTree<ExprIR<Variable>>) -> Option<u32> {
    let ExprIR::FuncInvocation(func) = expr.root().data() else {
        return None;
    };
    if func.name != "count" {
        return None;
    }
    // count(var) has the counted variable as the first child.
    // There may be an additional __agg_order_by_placeholder__ child.
    let root = expr.root();
    if root.num_children() == 0 {
        return None;
    }
    let child = root.child(0);
    if let ExprIR::Variable(var) = child.data() {
        Some(var.id)
    } else {
        None
    }
}
