//! Reduces `CondTraverse`'s `bind_relationship` to false when the edge alias
//! is not consumed by any ancestor operator.
//!
//! A chain-less `CondTraverse` binds its edge alias to a representative edge
//! id, and finding one costs a tensor lookup per surviving row — up to three
//! GraphBLAS `extractElement` calls, since the effective value has to be read
//! through the `dp`/`dm`/`m` delta layers. That is dead work whenever the query
//! never reads the edge (`MATCH (a)-[:R]->(b) RETURN count(b)`).
//!
//! This is *not* what [`emit_relationship`] already decides. That flag governs
//! row multiplicity: false means one row per (src, dst) pair instead of one per
//! edge. An anonymous edge inside a named path gets `emit_relationship: false`
//! — one row per pair is right — and still has to be bound, because
//! `PathBuilder` reads the alias to assemble the path. So the two questions
//! ("how many rows" and "does anything read the edge") need separate answers,
//! and only the second one licenses skipping the lookup.
//!
//! Runs last in the pipeline, after every pass that rebuilds a `CondTraverse`
//! or changes its ancestors — including `fuse_optional_traverse`. Every
//! constructor sets `bind_relationship: true`, so a pass that rebuilds a node
//! can only ever be conservative, and this pass has the final say.
//!
//! [`emit_relationship`]: super::super::IR::CondTraverse

use orx_tree::{Bfs, DynTree, NodeRef};

use super::super::IR;
use super::reduce_expand_into::ir_references_variable;

pub(super) fn reduce_bound_edge(plan: &mut DynTree<IR>) {
    let indices: Vec<_> = plan.root().indices::<Bfs>().collect();
    for idx in indices {
        let (alias_id, alias_scope_id) = match plan.node(idx).data() {
            // A fused chain binds no edge at all, so there is nothing to
            // reduce; `chain` non-empty already suppresses the lookup.
            IR::CondTraverse {
                bind_relationship: true,
                relationship,
                chain,
                ..
            } if chain.is_empty() => (relationship.alias.id, relationship.alias.scope_id),
            _ => continue,
        };

        // Walk ancestors to check if the edge alias is referenced.
        let mut referenced = false;
        let mut cur = idx;
        while let Some(parent) = plan.node(cur).parent() {
            if ir_references_variable(parent.data(), alias_id, alias_scope_id) {
                referenced = true;
                break;
            }
            cur = parent.idx();
        }

        if !referenced
            && let IR::CondTraverse {
                bind_relationship, ..
            } = plan.node_mut(idx).data_mut()
        {
            *bind_relationship = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use orx_tree::{Bfs, DynTree, NodeRef};

    use super::super::fuse_optional_traverse::fuse_optional_traverse;
    use super::super::reduce_expand_into::reduce_expand_into;
    use super::reduce_bound_edge;
    use crate::parser::cypher::Parser;
    use crate::planner::{IR, Planner, binder::Binder};
    use crate::runtime::functions::init_functions;

    /// Compiles `query` through parse → bind → plan, then runs the Graph-free
    /// subset of the pipeline that governs `bind_relationship`, in pipeline
    /// order. The skipped passes (`select_scan_node`, `utilize_index`, ...)
    /// need a live GraphBLAS context and do not change which operators consume
    /// an edge alias.
    fn optimized_plan(query: &str) -> DynTree<IR> {
        // Aggregate calls make the planner resolve names against the function
        // registry, which is process-global and initialized once.
        let _ = init_functions();
        let mut parser = Parser::new(query);
        parser.parse_parameters().expect("parse parameters");
        let raw = parser.parse().expect("parse");
        let (ir, scope_vars) = Binder::default().bind(raw).expect("bind");
        let mut plan = Planner::new(scope_vars).plan(ir);

        reduce_expand_into(&mut plan);
        fuse_optional_traverse(&mut plan);
        reduce_bound_edge(&mut plan);
        plan
    }

    fn bound_edges(plan: &DynTree<IR>) -> Vec<bool> {
        plan.root()
            .indices::<Bfs>()
            .filter_map(|idx| match plan.node(idx).data() {
                IR::CondTraverse {
                    bind_relationship,
                    chain,
                    ..
                } if chain.is_empty() => Some(*bind_relationship),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn unread_edge_is_not_bound() {
        let plan = optimized_plan("MATCH (a:N)-[:R]->(b:N) RETURN count(b)");
        assert_eq!(bound_edges(&plan), vec![false]);
    }

    #[test]
    fn named_but_unread_edge_is_not_bound() {
        // Naming an edge is not reading it — the same case `reduce_expand_into`
        // already lowers `emit_relationship` for.
        let plan = optimized_plan("MATCH (a:N)-[r:R]->(b:N) RETURN a.v, b.v");
        assert_eq!(bound_edges(&plan), vec![false]);
    }

    #[test]
    fn returned_edge_is_bound() {
        let plan = optimized_plan("MATCH (a:N)-[r:R]->(b:N) RETURN r");
        assert_eq!(bound_edges(&plan), vec![true]);
    }

    #[test]
    fn filtered_edge_is_bound() {
        let plan = optimized_plan("MATCH (a:N)-[r:R]->(b:N) WHERE r.k > 1 RETURN b.v");
        assert_eq!(bound_edges(&plan), vec![true]);
    }

    #[test]
    fn deleted_edge_is_bound() {
        let plan = optimized_plan("MATCH (a:N)-[r:R]->(b:N) DELETE r");
        assert_eq!(bound_edges(&plan), vec![true]);
    }

    /// The case `emit_relationship` cannot express: an *anonymous* edge inside a
    /// named path collapses to one row per pair, yet `PathBuilder` still reads
    /// the alias to assemble the path. Skipping the lookup here would hand it
    /// an unbound column.
    #[test]
    fn anonymous_edge_in_a_named_path_is_bound() {
        let plan = optimized_plan("MATCH p = (a:N)-[:R]->(b:N) RETURN p");
        assert_eq!(bound_edges(&plan), vec![true]);
    }

    #[test]
    fn edge_read_after_a_with_barrier_is_bound() {
        let plan = optimized_plan("MATCH (a:N)-[r:R]->(b:N) WITH r AS e RETURN e");
        assert_eq!(bound_edges(&plan), vec![true]);
    }

    #[test]
    fn optional_traverse_keeps_the_distinction() {
        // `fuse_optional_traverse` rebuilds the CondTraverse; this pass runs
        // after it, so the rebuilt node still gets the right answer.
        let plan = optimized_plan("MATCH (a:N) OPTIONAL MATCH (a)-[:R]->(b:N) RETURN count(b)");
        assert_eq!(bound_edges(&plan), vec![false]);
        let plan = optimized_plan("MATCH (a:N) OPTIONAL MATCH (a)-[r:R]->(b:N) RETURN r");
        assert_eq!(bound_edges(&plan), vec![true]);
    }
}
