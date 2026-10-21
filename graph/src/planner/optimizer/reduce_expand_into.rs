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
//! `OPTIONAL MATCH` or pattern-predicate Apply, a pattern comprehension and a
//! `FOREACH` body are sibling branches that receive every row as their
//! argument. If the edge is never read, `emit_relationship` is set to false.

use orx_tree::{Bfs, DynTree, NodeRef};

use super::super::IR;
use super::references::variable_read_outside;

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
    /// its traverse: a sibling branch of an Apply (`CALL {}` body, OPTIONAL
    /// MATCH, pattern predicate / comprehension) or a `FOREACH` body.
    #[test]
    fn edge_read_outside_the_ancestors_is_emitted() {
        for query in [
            "MATCH (a:A)-[r:R]->(b) CALL { WITH r CREATE (:Q) }",
            "MATCH (a:A)-[r:R]->(b) CALL { WITH r RETURN 1 AS one } RETURN count(*)",
            "MATCH (a:A)-[r:R]->(b) CALL { WITH r DELETE r }",
            "MATCH (a:A)-[r:R]->(b) CALL { WITH r RETURN 1 AS x UNION ALL WITH r RETURN 2 AS x } RETURN count(*)",
            "MATCH (a:A)-[r:R]->(b) OPTIONAL MATCH (c:B) WHERE c.v <> id(r) RETURN count(*)",
            "MATCH (a:A)-[r:R]->(b) WITH [(a)-[:R]->(x:B) WHERE x.v <> id(r) | 1] AS l RETURN count(*)",
            "MATCH (a:A)-[r:R]->(b) FOREACH (x IN [1] | CREATE (:Q {w: id(r)}))",
        ] {
            let emitted = emitted_edges(query);
            assert_eq!(emitted.first(), Some(&true), "{query}: {emitted:?}");
        }
    }
}
