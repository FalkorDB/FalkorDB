//! Reduces ExpandInto/CondTraverse `emit_relationship` to false when no
//! operator reads the edge variable.
//!
//! When `emit_relationship` is true, these operators produce one output row
//! per matching edge. When false, they collapse multi-edges into one row per
//! (src, dst) pair. The planner conservatively sets it to true for all
//! non-anonymous edges, but many queries name an edge variable without
//! actually consuming it (e.g. `MATCH (a)-[e]->(b) RETURN a, b`).
//!
//! The edge counts as read when any operator that can see the traverse's
//! output references it — see [`variable_read_outside`]. That is not only the
//! traverse's ancestors: the body of `CALL {}`, the right side of an
//! `OPTIONAL MATCH` or pattern-predicate Apply, and a `FOREACH` body are
//! sibling branches that receive every row as their argument.

use std::sync::Arc;

use orx_tree::{Bfs, Dyn, DynTree, NodeIdx, NodeRef};

use crate::index::IndexQuery;
use crate::parser::ast::{
    ExprIR, QueryExpr, QueryGraph, QueryNode, QueryRelationship, SetItem, Variable,
};

use super::super::IR;

type Node = QueryNode<Arc<String>, Variable>;
type Relationship = QueryRelationship<Arc<String>, Arc<String>, Variable>;

/// True when an operator outside the subtree rooted at `idx` reads the
/// variable `(var_id, scope_id)` that the operator at `idx` binds.
///
/// Every row `idx` produces flows up through its ancestors, and each ancestor
/// hands it to its other branches too — `Apply`/`SemiApply`/`ForEach`/`Merge`
/// run their right-hand sub-plan once per input row, with the row as the
/// `Argument`. So the readers are exactly the nodes outside `idx`'s own
/// subtree (whose nodes only produce its input). Nodes that cannot see the row
/// (e.g. the other side of a `CartesianProduct`) are included too: they can
/// only mention the variable if they bind it themselves, which makes the
/// answer conservative, never wrong.
pub(super) fn variable_read_outside(
    plan: &DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
    var_id: u32,
    scope_id: u32,
) -> bool {
    let mut cur = plan.node(idx);
    while let Some(parent) = cur.parent() {
        if ir_references_variable(parent.data(), var_id, scope_id) {
            return true;
        }
        for sibling in parent.children() {
            if sibling.idx() != cur.idx()
                && sibling
                    .walk::<Bfs>()
                    .any(|ir| ir_references_variable(ir, var_id, scope_id))
            {
                return true;
            }
        }
        cur = parent;
    }
    false
}

/// Check if any expression in an IR node references a variable with the
/// given (id, scope_id) pair.
///
/// Exhaustive on purpose: a new `IR` variant must decide here which of its
/// fields are reads, instead of silently answering "no".
#[allow(clippy::too_many_lines)]
pub(super) fn ir_references_variable(
    ir: &IR,
    var_id: u32,
    scope_id: u32,
) -> bool {
    let var = |v: &Variable| v.id == var_id && v.scope_id == scope_id;
    let expr = |e: &QueryExpr<Variable>| expr_references_variable(e, var_id, scope_id);
    let rel = |r: &Relationship| relationship_references_variable(r, var_id, scope_id);
    let index_query =
        |q: &IndexQuery<QueryExpr<Variable>>| index_query_references_variable(q, var_id, scope_id);
    match ir {
        IR::Project { exprs, copies } => {
            exprs.iter().any(|(_, e)| expr(e)) || copies.iter().any(|(v, _)| var(v))
        }
        IR::Filter(e) | IR::Skip(e) | IR::Limit(e) => expr(e),
        IR::Sort(exprs) => exprs.iter().any(|(e, _)| expr(e)),
        IR::Aggregate {
            keys,
            aggregations,
            projections,
            ..
        } => {
            keys.iter().any(|(_, e)| expr(e))
                || aggregations.iter().any(|(_, e)| expr(e))
                || projections.iter().any(|(v, _)| var(v))
        }
        IR::PathBuilder(paths) => paths.iter().any(|p| p.vars.iter().any(var)),
        IR::Unwind { expr: e, .. } | IR::ForEach { list: e, .. } => expr(e),
        IR::Delete { exprs, .. } | IR::Remove(exprs) => exprs.iter().any(expr),
        IR::Set(items) => set_items_reference_variable(items, var_id, scope_id),
        IR::Create(pattern) => pattern_references_variable(pattern, var_id, scope_id),
        IR::Merge {
            pattern,
            on_create,
            on_match,
        } => {
            pattern_references_variable(pattern, var_id, scope_id)
                || set_items_reference_variable(on_create, var_id, scope_id)
                || set_items_reference_variable(on_match, var_id, scope_id)
        }
        IR::ValueHashJoin { lhs_exp, rhs_exp } => expr(lhs_exp) || expr(rhs_exp),
        IR::ProcedureCall { args, .. } => args.iter().any(expr),
        IR::LoadCsv {
            file_path,
            delimiter,
            ..
        } => expr(file_path) || expr(delimiter),
        // A scan binds its node; only its inline properties are reads.
        IR::AllNodeScan(n) | IR::NodeByLabelScan { node: n } | IR::IncludePending { node: n } => {
            expr(&n.attrs)
        }
        IR::NodeByIndexScan { node: n, query, .. } => expr(&n.attrs) || index_query(query),
        IR::NodeByLabelAndIdScan { node: n, filter } | IR::NodeByIdSeek { node: n, filter } => {
            expr(&n.attrs) || filter.iter().any(|(e, _)| expr(e))
        }
        IR::NodeByFulltextScan { label, query, .. }
        | IR::EdgeByFulltextScan { label, query, .. } => expr(label) || expr(query),
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
        } => expr(label) || expr(attr) || expr(k) || expr(vector),
        // A traverse reads its bound endpoints, and a relationship alias that
        // is already bound (the same edge matched twice).
        IR::CondTraverse {
            relationship,
            chain,
            ..
        } => rel(relationship) || chain.iter().any(|r| rel(r)),
        IR::CondVarLenTraverse {
            relationship,
            edge_filter,
            ..
        } => rel(relationship) || edge_filter.as_ref().is_some_and(expr),
        IR::EdgeByIndexScan {
            relationship,
            query,
            ..
        } => rel(relationship) || index_query(query),
        IR::ExpandInto { relationship, .. } | IR::AllShortestPaths(relationship) => {
            rel(relationship)
        }
        IR::CreateIndex { options, .. } => options.as_ref().is_some_and(expr),
        // `Argument` / `Optional` list the variables their rows bind; they
        // read nothing.
        IR::Argument(_)
        | IR::Optional(_)
        | IR::CartesianProduct
        | IR::Apply
        | IR::SemiApply
        | IR::AntiSemiApply
        | IR::OrApplyMultiplexer(_)
        | IR::Distinct
        | IR::NestedPlans
        | IR::Union
        | IR::Commit
        | IR::DropIndex { .. } => false,
    }
}

fn node_references_variable(
    node: &Node,
    var_id: u32,
    scope_id: u32,
) -> bool {
    (node.alias.id == var_id && node.alias.scope_id == scope_id)
        || expr_references_variable(&node.attrs, var_id, scope_id)
}

fn relationship_references_variable(
    rel: &Relationship,
    var_id: u32,
    scope_id: u32,
) -> bool {
    (rel.alias.id == var_id && rel.alias.scope_id == scope_id)
        || expr_references_variable(&rel.attrs, var_id, scope_id)
        || node_references_variable(&rel.from, var_id, scope_id)
        || node_references_variable(&rel.to, var_id, scope_id)
}

/// A `CREATE`/`MERGE` pattern reads the variables its inline properties
/// mention. Its aliases are what the clause matches or creates, not reads —
/// a MERGE's own edge must not keep itself emitted.
fn pattern_references_variable(
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

fn index_query_references_variable(
    query: &IndexQuery<QueryExpr<Variable>>,
    var_id: u32,
    scope_id: u32,
) -> bool {
    let expr = |e: &QueryExpr<Variable>| expr_references_variable(e, var_id, scope_id);
    match query {
        IndexQuery::Equal { value, .. } | IndexQuery::ArrayContains { value, .. } => expr(value),
        IndexQuery::Range { min, max, .. } => {
            min.as_ref().is_some_and(expr) || max.as_ref().is_some_and(expr)
        }
        IndexQuery::Point { point, radius, .. } => expr(point) || expr(radius),
        IndexQuery::InList { list, .. } => expr(list),
        IndexQuery::And(children) | IndexQuery::Or(children) => children
            .iter()
            .any(|c| index_query_references_variable(c, var_id, scope_id)),
    }
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

fn expr_references_variable(
    expr: &DynTree<ExprIR<Variable>>,
    var_id: u32,
    scope_id: u32,
) -> bool {
    for idx in expr.root().indices::<Bfs>() {
        if let ExprIR::Variable(v) = expr.node(idx).data()
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

        if !variable_read_outside(plan, idx, edge_id, edge_scope_id) {
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

#[cfg(test)]
mod tests {
    use orx_tree::{Bfs, DynTree, NodeRef};

    use super::reduce_expand_into;
    use crate::parser::cypher::Parser;
    use crate::planner::{IR, Planner, binder::Binder};
    use crate::runtime::functions::init_functions;

    fn emitted_edges(query: &str) -> Vec<bool> {
        let _ = init_functions();
        let mut parser = Parser::new(query);
        parser.parse_parameters().expect("parse parameters");
        let raw = parser.parse().expect("parse");
        let (ir, scope_vars) = Binder::default().bind(raw).expect("bind");
        let mut plan: DynTree<IR> = Planner::new(scope_vars).plan(ir);
        reduce_expand_into(&mut plan);
        plan.root()
            .indices::<Bfs>()
            .filter_map(|idx| match plan.node(idx).data() {
                IR::CondTraverse {
                    emit_relationship, ..
                }
                | IR::ExpandInto {
                    emit_relationship, ..
                } => Some(*emit_relationship),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn unread_edge_collapses() {
        assert_eq!(
            emitted_edges("MATCH (a:A)-[r:R]->(b) RETURN count(*)"),
            vec![false]
        );
        // `r` is not imported, so the subquery cannot read it.
        assert_eq!(
            emitted_edges("MATCH (a:A)-[r:R]->(b) CALL { WITH a RETURN 1 AS one } RETURN count(*)"),
            vec![false]
        );
    }

    /// Each query reads `r` only from an operator that is not an ancestor of
    /// its traverse, or from an expression that used to be skipped.
    #[test]
    fn edge_read_outside_the_ancestors_is_emitted() {
        for query in [
            "MATCH (a:A)-[r:R]->(b) CALL { WITH r CREATE (:Q) }",
            "MATCH (a:A)-[r:R]->(b) CALL { WITH r RETURN 1 AS one } RETURN count(*)",
            "MATCH (a:A)-[r:R]->(b) CALL { WITH r DELETE r }",
            "MATCH (a:A)-[r:R]->(b) CALL { WITH r RETURN 1 AS x UNION ALL WITH r RETURN 2 AS x } RETURN count(*)",
            "MATCH (a:A)-[r:R]->(b) OPTIONAL MATCH (c:B) WHERE c.v <> id(r) RETURN count(*)",
            "MATCH (a:A)-[r:R]->(b) FOREACH (x IN [1] | CREATE (:Q {w: id(r)}))",
            "MATCH (a:A)-[r:R]->(b) FOREACH (x IN [r] | CREATE (:Q))",
            "MATCH (a:A)-[r:R]->(b) CREATE (:Q {w: id(r)})",
            "MATCH (a:A)-[r:R]->(b) MERGE (:Q {w: id(r)})",
            "MATCH (a:A)-[r:R]->(b) UNWIND [r] AS x RETURN count(*)",
        ] {
            let emitted = emitted_edges(query);
            assert_eq!(emitted.first(), Some(&true), "{query}: {emitted:?}");
        }
    }
}
