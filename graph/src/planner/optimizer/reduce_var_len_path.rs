//! Reduces `CondVarLenTraverse`'s `emit_path` to false when the path /
//! relationship-list variable is not consumed by any ancestor operator.
//!
//! `CondVarLenTraverse` binds its relationship alias to a `Value::Path` (the
//! alternating `[Node, Rel, Node, ...]` element list). Materializing that path
//! means allocating and growing a `ThinVec` and wrapping it in an `Arc` for
//! every emitted row — wasted work when the query never reads the path
//! (e.g. `MATCH (a)-[:R*2..3]->(b) RETURN b.id`).
//!
//! This pass walks ancestors of each `CondVarLenTraverse` to check whether any
//! expression references the path alias. If it is never consumed, `emit_path`
//! is set to false and the operator skips path materialization.
//!
//! It runs twice. The first run (before the filter-movement passes) sees every
//! original path consumer (Project/Filter/Sort/Aggregate/PathBuilder/...) as a
//! direct ancestor. The second run catches paths whose last consumer moved. The
//! second run is safe because `ir_references_variable` inspects `ValueHashJoin`
//! keys — the only new path consumer `replace_cartesian_with_hash_join` can
//! introduce in between.
//!
//! An edge-only `Filter` directly above the traverse is skipped either run: it
//! is fused into the walk when the operators are built, so it tests one edge at
//! a time and never reads the relationship list.

use orx_tree::{Bfs, DynTree, NodeRef};

use super::super::{IR, filter_is_fused_away};
use super::reduce_expand_into::ir_references_variable;

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

        // Walk ancestors to check if the path alias is referenced.
        //
        // A `Filter` that will be folded whole into this traverse when the
        // operators are built is not a consumer: its predicate runs per edge
        // inside the walk, against a single edge, and never reads the
        // materialized relationship list. Counting it would keep `emit_path`
        // true and make the walk build a `Value::Path` per row that nothing
        // reads — which is what `MATCH (a)-[e:R*1..2]->(b) WHERE e.w = 1
        // RETURN b.v` used to avoid back when the absorption happened as an IR
        // rewrite and deleted the node outright.
        let mut referenced = false;
        let mut cur = idx;
        while let Some(parent) = plan.node(cur).parent() {
            if !filter_is_fused_away(plan, parent.idx())
                && ir_references_variable(parent.data(), alias_id, alias_scope_id)
            {
                referenced = true;
                break;
            }
            cur = parent.idx();
        }

        if !referenced
            && let IR::CondVarLenTraverse { emit_path, .. } = plan.node_mut(idx).data_mut()
        {
            *emit_path = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use orx_tree::{Bfs, DynTree, NodeRef};

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
        // `e` is consumed only by the edge-only filter `e.w = 1`, which is
        // fused into the traversal when the operators are built, so it never
        // reads the relationship list: emit_path must be reduced to false.
        let plan = optimized_varlen_plan("MATCH (a)-[e:R*1..2]->(b) WHERE e.w = 1 RETURN b.v");
        assert_eq!(emit_paths(&plan), vec![false]);
    }

    #[test]
    fn consumed_path_is_kept_despite_absorption() {
        // The relationship list `e` is also returned, so even though `e.w = 1`
        // is fusable, emit_path must stay true.
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
}
