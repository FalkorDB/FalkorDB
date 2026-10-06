//! Does an IR node, or an expression, read a given variable?
//!
//! One answer for every optimizer pass. Several passes decide whether an
//! optimization is safe by asking it — and they used to ask four separate,
//! slightly different copies, two of which ignored the variable's scope.
//!
//! A variable is its `(id, scope_id)` pair, never its id alone: the binder
//! allocates ids per scope (`Variable::id` is an index into its own scope's
//! env), so the same number names different variables in different scopes.
//! Note that `Variable`'s `PartialEq` compares ids only, so `==` on two
//! variables is the wrong test here.

use std::sync::Arc;

use orx_tree::{Bfs, DynNode, DynTree, NodeRef};

use crate::index::indexer::IndexQuery;
use crate::parser::ast::{ExprIR, QueryExpr, QueryGraph, SetItem, Variable};

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
/// while they were Filters. A fixed-length traverse's edge predicate stays in
/// `IR::Filter`, the only place it is evaluated.
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
            keys,
            aggregations,
            projections,
            ..
        } => {
            keys.iter()
                .any(|(_, expr)| expr_references_variable(expr, var_id, scope_id))
                || aggregations
                    .iter()
                    .any(|(_, expr)| expr_references_variable(expr, var_id, scope_id))
                // Variables carried through from the input. Both sides are
                // counted: over-reporting a reader only makes a caller keep
                // something it could have dropped.
                || projections.iter().any(|(a, b)| {
                    (a.id == var_id && a.scope_id == scope_id)
                        || (b.id == var_id && b.scope_id == scope_id)
                })
        }
        IR::PathBuilder(paths) => paths.iter().any(|p| {
            p.vars
                .iter()
                .any(|v| v.id == var_id && v.scope_id == scope_id)
        }),
        // The list they iterate is read; `UNWIND [e] AS x` reads `e`.
        IR::Unwind { expr: list, var } | IR::ForEach { list, var } => {
            expr_references_variable(list, var_id, scope_id)
                || (var.id == var_id && var.scope_id == scope_id)
        }
        IR::Delete { exprs, .. } | IR::Remove(exprs) => exprs
            .iter()
            .any(|expr| expr_references_variable(expr, var_id, scope_id)),
        IR::Set(items) => set_items_reference_variable(items, var_id, scope_id),
        // The pattern's property expressions are read too:
        // `MERGE (x {w: e.w})` reads `e`, as `CREATE` does below.
        IR::Merge {
            pattern,
            on_create,
            on_match,
        } => {
            query_graph_references_variable(pattern, var_id, scope_id)
                || set_items_reference_variable(on_create, var_id, scope_id)
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
        // A walk prunes per edge on its pattern's own attrs (`-[:R* {w: p.v}]->`
        // reads `p`) and, for CVLT, on a WHERE predicate absorbed into
        // `edge_filter`.
        IR::CondVarLenTraverse {
            relationship,
            edge_filter,
            path_var,
            ..
        } => {
            expr_references_variable(&relationship.attrs, var_id, scope_id)
                || edge_filter
                    .as_ref()
                    .is_some_and(|f| expr_references_variable(f, var_id, scope_id))
                || path_var
                    .as_ref()
                    .is_some_and(|v| v.id == var_id && v.scope_id == scope_id)
        }
        IR::AllShortestPaths(relationship) => {
            expr_references_variable(&relationship.attrs, var_id, scope_id)
        }
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
        // Holds no expression: its children are the query's plan and the nested
        // plans, each visited as its own node. What a nested plan reads is not
        // lost either — the `ExprIR::NestedPlan` that calls it carries those
        // variables as children, so the expression holding the call reports
        // them. (`optimize` also strips this root before any pass runs.)
        IR::NestedPlans => false,
    }
}

/// Whether an index query's operands reference the variable. The operands are
/// the conjuncts `utilize_index` lifted out of a Filter, so they can name
/// runtime-bound values — `WHERE d.v > x` becomes a `Range` whose `min` reads
/// `x`.
pub(super) fn index_query_references_variable(
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
    items: &[SetItem<Arc<String>, Variable>],
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

/// Whether an expression names the variable anywhere in it.
pub(super) fn expr_references_variable(
    expr: &DynTree<ExprIR<Variable>>,
    var_id: u32,
    scope_id: u32,
) -> bool {
    subtree_references_variable(&expr.root(), var_id, scope_id)
}

/// The same question for one subtree of an expression — `utilize_node_by_id`
/// asks it of a single operand of a comparison.
pub(super) fn subtree_references_variable(
    node: &DynNode<ExprIR<Variable>>,
    var_id: u32,
    scope_id: u32,
) -> bool {
    node.walk::<Bfs>()
        .any(|e| matches!(e, ExprIR::Variable(v) if v.id == var_id && v.scope_id == scope_id))
}
