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

    /// Operators whose own arguments read `r`: a procedure call, `LOAD CSV`,
    /// and the scans that take their label / query / vector as expressions.
    #[test]
    fn edge_read_by_an_operators_arguments_is_emitted() {
        for query in [
            "MATCH (a:A)-[r:R]->(b) CALL db.idx.fulltext.queryNodes('A', toString(id(r))) YIELD node RETURN count(*)",
            "MATCH (a:A)-[r:R]->(b) CALL algo.BFS(a, id(r), 'R') YIELD nodes RETURN count(*)",
            "MATCH (a:A)-[r:R]->(b) LOAD CSV FROM toString(id(r)) AS row RETURN count(*)",
            "MATCH (a:A)-[r:R]->(b) LOAD CSV WITH HEADERS FROM toString(id(r)) AS row RETURN count(*)",
        ] {
            let emitted = emitted_edges(query);
            assert_eq!(emitted.first(), Some(&true), "{query}: {emitted:?}");
        }
    }

    /// `UNION` branches are independent plans, and a fresh binder gives each
    /// branch's `r` the same `(id, scope_id)`. A read of `r` in one branch must
    /// not keep the other branch's unread edge per-edge.
    #[test]
    fn edge_read_in_another_union_branch_is_not_a_read() {
        assert_eq!(
            emitted_edges(
                "MATCH (a:A)-[r:R]->(b) RETURN count(*) AS c UNION ALL MATCH (x:A)-[r:R]->(y) RETURN id(r) AS c"
            ),
            vec![false, true]
        );
        assert_eq!(
            emitted_edges(
                "MATCH (x:A)-[r:R]->(y) RETURN id(r) AS c UNION ALL MATCH (a:A)-[r:R]->(b) RETURN count(*) AS c"
            ),
            vec![true, false]
        );
    }

    /// The same, one level down: the `r` a nested `UNION` branch of a `CALL {}`
    /// body binds for itself is not a read of the outer `r` the body never
    /// imports.
    #[test]
    fn edge_bound_by_a_union_branch_in_a_call_body_is_not_a_read() {
        assert_eq!(
            emitted_edges(
                "MATCH (a:A)-[r:R]->(b) CALL { WITH a RETURN 1 AS x UNION ALL MATCH (x:A)-[r:R]->(y) RETURN id(r) AS x } RETURN count(*)"
            )
            .first(),
            Some(&false)
        );
        // The body that does import `r` still reads it.
        assert_eq!(
            emitted_edges(
                "MATCH (a:A)-[r:R]->(b) CALL { WITH r RETURN id(r) AS x UNION ALL MATCH (x:A)-[r2:R]->(y) RETURN id(r2) AS x } RETURN count(*)"
            )
            .first(),
            Some(&true)
        );
    }
    /// A body that imports `r` and also matches it again (a bound relationship)
    /// still reads the outer `r`, in any branch, even though that branch has a
    /// traverse with `r` as its alias.
    #[test]
    fn imported_and_rematched_edge_is_still_a_read() {
        for query in [
            "MATCH (a:A)-[r:R]->(b) CALL { WITH r MATCH ()-[r]->() RETURN 1 AS x UNION ALL WITH r MATCH ()-[r]->() RETURN 2 AS x } RETURN count(*)",
            "MATCH (a:A)-[r:R]->(b) CALL { WITH r MATCH ()-[r]->() RETURN 1 AS x UNION ALL WITH a RETURN 2 AS x } RETURN count(*)",
            "MATCH (a:A)-[r:R]->(b) CALL { WITH a RETURN 2 AS x UNION ALL WITH r MATCH ()-[r]->() RETURN 1 AS x } RETURN count(*)",
            "MATCH (a:A)-[r:R]->(b) CALL { WITH r MATCH ()-[r]->() RETURN count(*) AS x } RETURN count(*)",
        ] {
            let emitted = emitted_edges(query);
            assert_eq!(emitted.first(), Some(&true), "{query}: {emitted:?}");
        }
    }
}
