//! Reduces ExpandInto/CondTraverse `emit_relationship` to false when the
//! edge variable is not consumed by any ancestor operator.
//!
//! When `emit_relationship` is true, these operators produce one output row
//! per matching edge. When false, they collapse multi-edges into one row per
//! (src, dst) pair. The planner conservatively sets it to true for all
//! non-anonymous edges, but many queries name an edge variable without
//! actually consuming it (e.g. `MATCH (a)-[e]->(b) RETURN a, b`).
//!
//! This pass walks ancestors of each ExpandInto/CondTraverse to check
//! whether any expression references the edge alias. If the edge is never
//! consumed, `emit_relationship` is set to false.

use orx_tree::{Bfs, DynTree, NodeRef};

use crate::parser::ast::SetItem;

use super::super::IR;

/// Check if any expression in an IR node references a variable with the
/// given (id, scope_id) pair.
fn ir_references_variable(
    ir: &IR,
    var_id: u32,
    scope_id: u32,
) -> bool {
    match ir {
        IR::Project { exprs, copies } => {
            exprs
                .iter()
                .any(|(_, expr)| expr_references_variable(expr, var_id, scope_id))
                || copies
                    .iter()
                    .any(|(v, _)| v.id == var_id && v.scope_id == scope_id)
        }
        IR::Filter(expr) => expr_references_variable(expr, var_id, scope_id),
        IR::Sort(exprs) => exprs
            .iter()
            .any(|(expr, _)| expr_references_variable(expr, var_id, scope_id)),
        IR::Aggregate {
            keys, aggregations, ..
        } => {
            keys.iter()
                .any(|(_, expr)| expr_references_variable(expr, var_id, scope_id))
                || aggregations
                    .iter()
                    .any(|(_, expr)| expr_references_variable(expr, var_id, scope_id))
        }
        IR::PathBuilder(paths) => paths.iter().any(|p| {
            p.vars
                .iter()
                .any(|v| v.id == var_id && v.scope_id == scope_id)
        }),
        IR::Unwind { var, .. } | IR::ForEach { var, .. } => {
            var.id == var_id && var.scope_id == scope_id
        }
        IR::Delete { exprs, .. } | IR::Remove(exprs) => exprs
            .iter()
            .any(|expr| expr_references_variable(expr, var_id, scope_id)),
        IR::Set(items) => set_items_reference_variable(items, var_id, scope_id),
        IR::Merge {
            on_create,
            on_match,
            ..
        } => {
            set_items_reference_variable(on_create, var_id, scope_id)
                || set_items_reference_variable(on_match, var_id, scope_id)
        }
        _ => false,
    }
}

fn set_items_reference_variable(
    items: &[SetItem<std::sync::Arc<String>, crate::parser::ast::Variable>],
    var_id: u32,
    scope_id: u32,
) -> bool {
    items.iter().any(|item| match item {
        SetItem::Attribute { target, value, .. } => {
            expr_references_variable(target, var_id, scope_id)
                || expr_references_variable(value, var_id, scope_id)
        }
        SetItem::Label { var, .. } => var.id == var_id && var.scope_id == scope_id,
    })
}

fn expr_references_variable(
    expr: &orx_tree::DynTree<crate::parser::ast::ExprIR<crate::parser::ast::Variable>>,
    var_id: u32,
    scope_id: u32,
) -> bool {
    for idx in expr.root().indices::<Bfs>() {
        if let crate::parser::ast::ExprIR::Variable(v) = expr.node(idx).data()
            && v.id == var_id
            && v.scope_id == scope_id
        {
            return true;
        }
    }
    false
}

pub(super) fn reduce_expand_into(plan: &mut DynTree<IR>) {
    let indices: Vec<_> = plan.root().indices::<Bfs>().collect();
    for idx in indices {
        let (edge_id, edge_scope_id) = match plan.node(idx).data() {
            IR::ExpandInto {
                emit_relationship: true,
                relationship,
                ..
            }
            | IR::CondTraverse {
                emit_relationship: true,
                relationship,
                ..
            } => (relationship.alias.id, relationship.alias.scope_id),
            _ => continue,
        };

        // Walk ancestors to check if the edge variable is referenced.
        let mut referenced = false;
        let mut cur = idx;
        while let Some(parent) = plan.node(cur).parent() {
            if ir_references_variable(parent.data(), edge_id, edge_scope_id) {
                referenced = true;
                break;
            }
            cur = parent.idx();
        }

        if !referenced {
            match plan.node_mut(idx).data_mut() {
                IR::ExpandInto {
                    emit_relationship, ..
                }
                | IR::CondTraverse {
                    emit_relationship, ..
                } => {
                    *emit_relationship = false;
                }
                _ => {}
            }
        }
    }
}
