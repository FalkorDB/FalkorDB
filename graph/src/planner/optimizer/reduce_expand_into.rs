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

use std::sync::Arc;

use crate::index::indexer::IndexQuery;
use crate::parser::ast::{QueryExpr, QueryGraph, SetItem, Variable};

use super::super::IR;

/// Check if any expression in an IR node references a variable with the
/// given (id, scope_id) pair.
///
/// The match is exhaustive on purpose — no `_` arm. Several passes decide
/// whether an optimization is safe by asking this, and every one of them acts
/// on a `false`: `reduce_expand_into` collapses multi-edges, `reduce_bound_edge`
/// stops binding the edge, `reduce_var_len_path` drops the path, and
/// `fuse_anonymous_traverse` erases the intermediate. So an unlisted variant
/// silently reads as "references nothing" and licenses all four — the
/// dangerous direction. A wildcard would let the next variant join that arm
/// without anyone deciding it should; without one, adding a variant stops this
/// function compiling until someone classifies it.
///
/// This matters more than it looks, because `IR::Filter` is not the only place
/// a predicate lives: `utilize_index` moves conjuncts into an index scan's
/// `query` and `utilize_node_by_id` into `NodeByIdSeek`'s `filter`. Those are
/// the same predicate, relocated — they reference variables exactly as they did
/// while they were Filters. Edge predicates stay in `IR::Filter`; the traverses
/// only absorb them when the runtime builds its operators, which is after every
/// caller of this function has run.
pub(super) fn ir_references_variable(
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
        IR::ValueHashJoin { lhs_exp, rhs_exp } => {
            expr_references_variable(lhs_exp, var_id, scope_id)
                || expr_references_variable(rhs_exp, var_id, scope_id)
        }
        IR::ProcedureCall { args, .. } => args
            .iter()
            .any(|expr| expr_references_variable(expr, var_id, scope_id)),
        // Predicates relocated out of a Filter by an optimizer pass.
        IR::NodeByIndexScan { query, .. } | IR::EdgeByIndexScan { query, .. } => {
            index_query_references_variable(query, var_id, scope_id)
        }
        IR::NodeByLabelAndIdScan { filter, .. } | IR::NodeByIdSeek { filter, .. } => filter
            .iter()
            .any(|(expr, _)| expr_references_variable(expr, var_id, scope_id)),
        IR::CondVarLenTraverse { path_var, .. } => path_var
            .as_ref()
            .is_some_and(|v| v.id == var_id && v.scope_id == scope_id),
        IR::NodeByFulltextScan { label, query, .. }
        | IR::EdgeByFulltextScan { label, query, .. } => {
            expr_references_variable(label, var_id, scope_id)
                || expr_references_variable(query, var_id, scope_id)
        }
        IR::NodeByVectorScan {
            label,
            attr,
            k,
            vector,
            ..
        }
        | IR::EdgeByVectorScan {
            label,
            attr,
            k,
            vector,
            ..
        } => {
            expr_references_variable(label, var_id, scope_id)
                || expr_references_variable(attr, var_id, scope_id)
                || expr_references_variable(k, var_id, scope_id)
                || expr_references_variable(vector, var_id, scope_id)
        }
        IR::LoadCsv {
            file_path,
            delimiter,
            ..
        } => {
            expr_references_variable(file_path, var_id, scope_id)
                || expr_references_variable(delimiter, var_id, scope_id)
        }
        IR::Skip(expr) | IR::Limit(expr) => expr_references_variable(expr, var_id, scope_id),
        // Constructor patterns: `CREATE (n {p: x})` reads `x`.
        IR::Create(pattern) => query_graph_references_variable(pattern, var_id, scope_id),
        // Structural or variable-free: they bind and route rows, and hold no
        // expression that could name this variable.
        IR::Argument(_)
        | IR::Optional(_)
        | IR::AllNodeScan(_)
        | IR::NodeByLabelScan { .. }
        | IR::IncludePending { .. }
        | IR::ExpandInto { .. }
        | IR::CondTraverse { .. }
        | IR::AllShortestPaths(_)
        | IR::CartesianProduct
        | IR::Apply
        | IR::SemiApply
        | IR::AntiSemiApply
        | IR::OrApplyMultiplexer(_)
        | IR::Distinct
        | IR::Union
        | IR::Commit
        | IR::CreateIndex { .. }
        | IR::DropIndex { .. } => false,
    }
}

/// Whether an index query's operands reference the variable. The operands are
/// the conjuncts `utilize_index` lifted out of a Filter, so they can name
/// runtime-bound values — `WHERE d.v > x` becomes a `Range` whose `min` reads
/// `x`.
fn index_query_references_variable(
    query: &IndexQuery<QueryExpr<Variable>>,
    var_id: u32,
    scope_id: u32,
) -> bool {
    let refs = |e: &QueryExpr<Variable>| expr_references_variable(e, var_id, scope_id);
    match query {
        IndexQuery::Equal { value, .. } | IndexQuery::ArrayContains { value, .. } => refs(value),
        IndexQuery::InList { list, .. } => refs(list),
        IndexQuery::Range { min, max, .. } => {
            min.as_ref().is_some_and(&refs) || max.as_ref().is_some_and(&refs)
        }
        IndexQuery::Point { point, radius, .. } => refs(point) || refs(radius),
        IndexQuery::And(qs) | IndexQuery::Or(qs) => qs
            .iter()
            .any(|q| index_query_references_variable(q, var_id, scope_id)),
    }
}

/// Whether a CREATE/MERGE pattern's inline attributes reference the variable.
fn query_graph_references_variable(
    pattern: &QueryGraph<Arc<String>, Arc<String>, Variable>,
    var_id: u32,
    scope_id: u32,
) -> bool {
    pattern
        .nodes()
        .iter()
        .any(|n| expr_references_variable(&n.attrs, var_id, scope_id))
        || pattern
            .relationships()
            .iter()
            .any(|r| expr_references_variable(&r.attrs, var_id, scope_id))
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
