//! Reduces `CondVarLenTraverse`'s `emit_path` to false when the path /
//! relationship-list variable is not read by any operator.
//!
//! `CondVarLenTraverse` binds its relationship alias to a `Value::Path` (the
//! alternating `[Node, Rel, Node, ...]` element list). Materializing that path
//! means allocating and growing a `ThinVec` and wrapping it in an `Arc` for
//! every emitted row — wasted work when the query never reads the path
//! (e.g. `MATCH (a)-[:R*2..3]->(b) RETURN b.id`).
//!
//! This pass checks every operator that can see a `CondVarLenTraverse`'s
//! output (see `variable_read_outside`) for an expression referencing the
//! path alias. If it is never consumed, `emit_path` is set to false and the
//! operator skips path materialization.
//!
//! It runs twice. The first run (before the filter-movement passes) sees every
//! original path consumer (Project/Filter/Sort/Aggregate/PathBuilder/...). A
//! second run after `absorb_edge_filters_into_vlt` catches paths whose last
//! consumer was an edge-only filter that the absorption pass folded into the
//! traversal. The second run is safe because
//! `ir_references_variable` inspects `ValueHashJoin` keys — the only new path
//! consumer `replace_cartesian_with_hash_join` can introduce in between.

use orx_tree::{Bfs, DynTree, NodeRef};

use super::super::IR;
use super::reduce_expand_into::variable_read_outside;

pub(super) fn reduce_var_len_path(plan: &mut DynTree<IR>) {
    let indices: Vec<_> = plan.root().indices::<Bfs>().collect();
    for idx in indices {
        let (alias_id, alias_scope_id) = match plan.node(idx).data() {
            IR::CondVarLenTraverse {
                emit_path: true,
                relationship,
                ..
            } => (relationship.alias.id, relationship.alias.scope_id),
            _ => continue,
        };

        if !variable_read_outside(plan, idx, alias_id, alias_scope_id)
            && let IR::CondVarLenTraverse { emit_path, .. } = plan.node_mut(idx).data_mut()
        {
            *emit_path = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use orx_tree::{Bfs, DynTree, NodeRef};

    use super::super::absorb_edge_filters_into_vlt::absorb_edge_filters_into_vlt;
    use super::super::replace_cartesian_with_hash_join::replace_cartesian_with_hash_join;
    use super::reduce_var_len_path;
    use crate::parser::cypher::Parser;
    use crate::planner::{IR, Planner, binder::Binder};

    /// Compiles `query` through parse → bind → plan, then runs the Graph-free
    /// subset of the optimizer pipeline that governs `emit_path`, in pipeline
    /// order. The skipped passes (`reduce_count`, `select_scan_node`,
    /// `utilize_index`, ...) need a live GraphBLAS context and do not change
    /// which operators consume a path alias.
    fn optimized_varlen_plan(query: &str) -> DynTree<IR> {
        let mut parser = Parser::new(query);
        parser.parse_parameters().expect("parse parameters");
        let raw = parser.parse().expect("parse");
        let (ir, scope_vars) = Binder::default().bind(raw).expect("bind");
        let mut plan = Planner::new(scope_vars).plan(ir);

        reduce_var_len_path(&mut plan);
        replace_cartesian_with_hash_join(&mut plan);
        absorb_edge_filters_into_vlt(&mut plan);
        reduce_var_len_path(&mut plan);
        plan
    }

    fn emit_paths(plan: &DynTree<IR>) -> Vec<bool> {
        plan.root()
            .indices::<Bfs>()
            .filter_map(|idx| match plan.node(idx).data() {
                IR::CondVarLenTraverse { emit_path, .. } => Some(*emit_path),
                _ => None,
            })
            .collect()
    }

    fn has_value_hash_join(plan: &DynTree<IR>) -> bool {
        plan.root()
            .indices::<Bfs>()
            .any(|idx| matches!(plan.node(idx).data(), IR::ValueHashJoin { .. }))
    }

    #[test]
    fn absorbed_edge_filter_lets_unused_path_be_skipped() {
        // `e` is consumed only by the edge-only filter `e.w = 1`. Once
        // `absorb_edge_filters_into_vlt` folds that filter into the traversal,
        // the second pass run must reduce emit_path to false.
        let plan = optimized_varlen_plan("MATCH (a)-[e:R*1..2]->(b) WHERE e.w = 1 RETURN b.v");
        assert_eq!(emit_paths(&plan), vec![false]);
    }

    #[test]
    fn consumed_path_is_kept_despite_absorption() {
        // The relationship list `e` is also returned, so even though `e.w = 1`
        // is absorbable, emit_path must stay true.
        let plan = optimized_varlen_plan("MATCH (a)-[e:R*1..2]->(b) WHERE e.w = 1 RETURN e");
        assert!(emit_paths(&plan).into_iter().all(|kept| kept));
    }

    #[test]
    fn path_consumed_through_value_hash_join_is_kept() {
        // `e1` flows only into a ValueHashJoin key (`e1 = e2`, which the
        // cartesian-to-join pass rewrites). The re-run must still see that
        // consumer through the join and keep both paths materialized — this
        // guards the `ir_references_variable` ValueHashJoin arm.
        let plan = optimized_varlen_plan(
            "MATCH (a)-[e1:R*1..2]->(b), (c)-[e2:R*1..2]->(d) WHERE e1 = e2 RETURN b.v, d.v",
        );
        assert!(
            has_value_hash_join(&plan),
            "expected a ValueHashJoin in the plan"
        );
        let paths = emit_paths(&plan);
        assert_eq!(paths.len(), 2, "expected two CondVarLenTraverse nodes");
        assert!(paths.into_iter().all(|kept| kept));
    }

    #[test]
    fn path_read_outside_the_ancestors_is_kept() {
        for query in [
            "MATCH (a)-[p:R*1..2]->(b) CALL { WITH p RETURN length(p) AS s } RETURN s",
            "MATCH (a)-[p:R*1..2]->(b) OPTIONAL MATCH (c) WHERE length(p) = 1 RETURN count(c)",
            "MATCH (a)-[p:R*1..2]->(b) UNWIND p AS x RETURN count(x)",
            "MATCH (a)-[p:R*1..2]->(b) CREATE (:Q {n: length(p)})",
        ] {
            let paths = emit_paths(&optimized_varlen_plan(query));
            assert_eq!(paths, vec![true], "{query}");
        }
    }
}
