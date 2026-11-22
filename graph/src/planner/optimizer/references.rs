//! Does an IR node, or an expression, read a given variable?
//!
//! One answer for every optimizer pass. Several passes decide whether an
//! optimization is safe by asking it — and they used to ask four separate,
//! slightly different copies, two of which ignored the variable's scope.
//!
//! A variable is identified by its `VarKey` (`id` and `scope_id`), never by its
//! id alone: the binder allocates ids per scope (`Variable::id` is an index into
//! its own scope's env), so the same number names different variables in
//! different scopes.
//! Note that `Variable`'s `PartialEq` compares ids only, so `==` on two
//! variables is the wrong test here.

use std::sync::Arc;

use orx_tree::{Bfs, Dyn, DynNode, DynTree, NodeIdx, NodeRef};

use crate::index::indexer::IndexQuery;
use crate::parser::ast::{ExprIR, QueryExpr, QueryGraph, SetItem, VarKey, Variable};

use super::super::IR;

/// Check if any expression in an IR node references the variable `key`.
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
pub(in crate::planner) fn ir_references_variable(
    ir: &IR,
    key: VarKey,
) -> bool {
    match ir {
        IR::Project { exprs, copies } => {
            exprs
                .iter()
                .any(|(_, expr)| expr_references_variable(expr, key))
                || copies.iter().any(|(v, _)| v.key() == key)
        }
        IR::Filter(expr) | IR::Skip(expr) | IR::Limit(expr) => expr_references_variable(expr, key),
        IR::Sort(exprs) => exprs
            .iter()
            .any(|(expr, _)| expr_references_variable(expr, key)),
        IR::Aggregate {
            keys,
            aggregations,
            projections,
            ..
        } => {
            keys.iter()
                .any(|(_, expr)| expr_references_variable(expr, key))
                || aggregations
                    .iter()
                    .any(|(_, expr)| expr_references_variable(expr, key))
                // Variables carried through from the input. Both sides are
                // counted: over-reporting a reader only makes a caller keep
                // something it could have dropped.
                || projections.iter().any(|(a, b)| {
                    a.key() == key
                        || b.key() == key
                })
        }
        IR::PathBuilder(paths) => paths.iter().any(|p| p.vars.iter().any(|v| v.key() == key)),
        // The list they iterate is read; `UNWIND [e] AS x` reads `e`.
        IR::Unwind { expr: list, var } => expr_references_variable(list, key) || var.key() == key,
        // The loop variable is not a read: it is bound only into the body's
        // own environment (the operator outputs its input rows), and `FOREACH
        // (r IN [1] | ...)` may reuse an outer edge's name, so a body that
        // never reads it must not keep that edge per-edge. A body read of it
        // shares the outer edge's `VarKey` and is conservatively
        // treated as an edge read.
        IR::ForEach { list, .. } => expr_references_variable(list, key),
        IR::Delete { exprs, .. } | IR::Remove(exprs) => {
            exprs.iter().any(|expr| expr_references_variable(expr, key))
        }
        IR::Set(items) => set_items_reference_variable(items, key),
        // The pattern's property expressions are read too:
        // `MERGE (x {w: e.w})` reads `e`, as `CREATE` does below.
        IR::Merge {
            pattern,
            on_create,
            on_match,
        } => {
            query_graph_references_variable(pattern, key)
                || set_items_reference_variable(on_create, key)
                || set_items_reference_variable(on_match, key)
        }
        IR::ValueHashJoin { lhs_exp, rhs_exp } => {
            expr_references_variable(lhs_exp, key) || expr_references_variable(rhs_exp, key)
        }
        IR::ProcedureCall { args, .. } => {
            args.iter().any(|expr| expr_references_variable(expr, key))
        }
        // Predicates relocated out of a Filter by an optimizer pass.
        IR::NodeByIndexScan { query, .. } | IR::EdgeByIndexScan { query, .. } => {
            index_query_references_variable(query, key)
        }
        IR::NodeByLabelAndIdScan { filter, .. } | IR::NodeByIdSeek { filter, .. } => filter
            .iter()
            .any(|(expr, _)| expr_references_variable(expr, key)),
        // A walk prunes per edge on its pattern's own attrs (`-[:R* {w: p.v}]->`
        // reads `p`) and, for CVLT, on a WHERE predicate absorbed into
        // `edge_filter`.
        IR::CondVarLenTraverse {
            relationship,
            edge_filter,
            path_var,
            ..
        } => {
            expr_references_variable(&relationship.attrs, key)
                || edge_filter
                    .as_ref()
                    .is_some_and(|f| expr_references_variable(f, key))
                || path_var.as_ref().is_some_and(|v| v.key() == key)
        }
        IR::AllShortestPaths(relationship) => expr_references_variable(&relationship.attrs, key),
        IR::NodeByFulltextScan { label, query, .. }
        | IR::EdgeByFulltextScan { label, query, .. } => {
            expr_references_variable(label, key) || expr_references_variable(query, key)
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
            expr_references_variable(label, key)
                || expr_references_variable(attr, key)
                || expr_references_variable(k, key)
                || expr_references_variable(vector, key)
        }
        IR::LoadCsv {
            file_path,
            delimiter,
            ..
        } => expr_references_variable(file_path, key) || expr_references_variable(delimiter, key),
        // Constructor patterns: `CREATE (n {p: x})` reads `x`.
        IR::Create(pattern) => query_graph_references_variable(pattern, key),
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
        | IR::DropIndex { .. }
        | IR::NestedPlans => false,
        // Holds no expression: its children are the query's plan and the nested
        // plans, each visited as its own node. What a nested plan reads is not
        // lost either — the `ExprIR::NestedPlan` that calls it carries those
        // variables as children, so the expression holding the call reports
        // them. (`optimize` also strips this root before any pass runs.)
    }
}

/// Whether the rows the child `cur` of `parent` produces are handed to the
/// parent's other children. `Apply`-like operators run their body children once
/// per row of their input (first) child, and nothing flows from a body back to
/// the input. `Union`, `CartesianProduct` and the hash join run independent
/// children: a variable the other branch binds is a different variable even when
/// a fresh binder gave it the same `VarKey`.
const fn row_reaches_siblings(
    parent: &IR,
    is_input_child: bool,
) -> bool {
    match parent {
        IR::Union | IR::CartesianProduct | IR::ValueHashJoin { .. } => false,
        _ => is_input_child,
    }
}

/// Whether the operator binds the variable itself: a traverse's edge or path.
fn ir_binds_variable(
    ir: &IR,
    key: VarKey,
) -> bool {
    let is_it = |v: &Variable| v.key() == key;
    match ir {
        IR::CondTraverse { relationship, .. } | IR::ExpandInto { relationship, .. } => {
            is_it(&relationship.alias)
        }
        IR::CondVarLenTraverse {
            relationship,
            path_var,
            ..
        } => is_it(&relationship.alias) || path_var.as_ref().is_some_and(is_it),
        IR::AllShortestPaths(relationship) => is_it(&relationship.alias),
        _ => false,
    }
}

/// Whether any operator in the subtree rooted at `idx` reads the variable. The
/// branches of a nested `Union` are independent plans that a fresh binder gives
/// the same `VarKey` numbers, so a branch that binds the variable itself
/// reads its own, not the outer one, and is skipped.
fn subtree_reads_variable(
    plan: &DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
    key: VarKey,
) -> bool {
    // An explicit stack: plans nest as deep as the query's subqueries do.
    let mut pending = vec![idx];
    while let Some(idx) = pending.pop() {
        let node = plan.node(idx);
        if ir_references_variable(node.data(), key) {
            return true;
        }
        let is_union = matches!(node.data(), IR::Union);
        for child in node.children() {
            let own = is_union && child.walk::<Bfs>().any(|ir| ir_binds_variable(ir, key));
            if !own {
                pending.push(child.idx());
            }
        }
    }
    false
}

/// True when an operator outside the subtree rooted at `idx` reads the
/// variable `key` that the operator at `idx` binds.
///
/// Every row `idx` produces flows up through its ancestors, and each ancestor
/// hands it to its other branches too: `Apply`/`SemiApply`/`ForEach`/`Merge`
/// run their right-hand sub-plan once per input row, with the row as the
/// `Argument`. So the readers are not only the ancestors but every node outside
/// `idx`'s own subtree (whose nodes only produce its input): the body of
/// `CALL {}`, the right side of an `OPTIONAL MATCH` or pattern-predicate Apply,
/// a pattern comprehension sub-plan, a `FOREACH` body. Only the branches that
/// receive the row count (see [`row_reaches_siblings`]): the other side of a
/// `Union` or `CartesianProduct`, or an `Apply` input reached from its body,
/// never sees it.
pub(super) fn variable_read_outside(
    plan: &DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
    key: VarKey,
) -> bool {
    let mut cur = plan.node(idx);
    while let Some(parent) = cur.parent() {
        if ir_references_variable(parent.data(), key) {
            return true;
        }
        if row_reaches_siblings(parent.data(), parent.child(0).idx() == cur.idx()) {
            for sibling in parent.children() {
                if sibling.idx() != cur.idx() && subtree_reads_variable(plan, sibling.idx(), key) {
                    return true;
                }
            }
        }
        cur = parent;
    }
    false
}

/// Whether an index query's operands reference the variable. The operands are
/// the conjuncts `utilize_index` lifted out of a Filter, so they can name
/// runtime-bound values — `WHERE d.v > x` becomes a `Range` whose `min` reads
/// `x`.
pub(super) fn index_query_references_variable(
    query: &IndexQuery<QueryExpr<Variable>>,
    key: VarKey,
) -> bool {
    let refs = |e: &QueryExpr<Variable>| expr_references_variable(e, key);
    // An explicit stack rather than recursion: `And` and `Or` nest to whatever
    // depth the query's predicates do.
    let mut pending = vec![query];
    while let Some(q) = pending.pop() {
        let hit = match q {
            IndexQuery::Equal { value, .. } | IndexQuery::ArrayContains { value, .. } => {
                refs(value)
            }
            IndexQuery::InList { list, .. } => refs(list),
            IndexQuery::Range { min, max, .. } => {
                min.as_ref().is_some_and(&refs) || max.as_ref().is_some_and(&refs)
            }
            IndexQuery::Point { point, radius, .. } => refs(point) || refs(radius),
            IndexQuery::And(qs) | IndexQuery::Or(qs) => {
                pending.extend(qs);
                false
            }
        };
        if hit {
            return true;
        }
    }
    false
}

/// Whether a CREATE/MERGE pattern's inline attributes reference the variable.
fn query_graph_references_variable(
    pattern: &QueryGraph<Arc<String>, Arc<String>, Variable>,
    key: VarKey,
) -> bool {
    pattern
        .nodes()
        .iter()
        .any(|n| expr_references_variable(&n.attrs, key))
        || pattern
            .relationships()
            .iter()
            .any(|r| expr_references_variable(&r.attrs, key))
}

fn set_items_reference_variable(
    items: &[SetItem<Arc<String>, Variable>],
    key: VarKey,
) -> bool {
    items.iter().any(|item| match item {
        SetItem::Attribute { target, value, .. } => {
            expr_references_variable(target, key) || expr_references_variable(value, key)
        }
        SetItem::Label { var, .. } => var.key() == key,
    })
}

/// Whether an expression names the variable anywhere in it.
pub(super) fn expr_references_variable(
    expr: &DynTree<ExprIR<Variable>>,
    key: VarKey,
) -> bool {
    subtree_references_variable(&expr.root(), key)
}

/// The same question for one subtree of an expression — `utilize_node_by_id`
/// asks it of a single operand of a comparison.
pub(super) fn subtree_references_variable(
    node: &DynNode<ExprIR<Variable>>,
    key: VarKey,
) -> bool {
    node.walk::<Bfs>()
        .any(|e| matches!(e, ExprIR::Variable(v) if v.key() == key))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use orx_tree::DynTree;

    use super::index_query_references_variable;
    use crate::index::indexer::IndexQuery;
    use crate::parser::ast::{ExprIR, QueryExpr, VarKey, Variable};
    use crate::runtime::functions::Type;

    fn var(
        id: u32,
        scope_id: u32,
    ) -> QueryExpr<Variable> {
        Arc::new(DynTree::new(ExprIR::Variable(Variable {
            alias: None,
            id,
            scope_id,
            ty: Type::Any,
        })))
    }

    fn key(
        id: u32,
        scope_id: u32,
    ) -> VarKey {
        VarKey { id, scope_id }
    }

    fn eq(value: QueryExpr<Variable>) -> IndexQuery<QueryExpr<Variable>> {
        IndexQuery::Equal {
            key: Arc::new("k".to_string()),
            value,
        }
    }

    /// A reference buried under nested `And`/`Or` is found, at any depth and in
    /// any position among siblings, and only for the right `(id, scope)`.
    #[test]
    fn finds_a_reference_inside_nested_and_or() {
        let deep = IndexQuery::And(vec![
            eq(var(1, 0)),
            IndexQuery::Or(vec![
                eq(var(2, 0)),
                IndexQuery::And(vec![eq(var(3, 0)), eq(var(7, 1))]),
            ]),
        ]);
        assert!(index_query_references_variable(&deep, key(7, 1)));
        assert!(index_query_references_variable(&deep, key(1, 0)));
        assert!(
            !index_query_references_variable(&deep, key(7, 0)),
            "same id, other scope"
        );
        assert!(!index_query_references_variable(&deep, key(9, 0)));
    }
}
