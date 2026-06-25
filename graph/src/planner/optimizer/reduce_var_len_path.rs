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
//! It runs before `replace_cartesian_with_hash_join` and the filter-movement
//! passes so every original path consumer (Project/Filter/Sort/Aggregate/
//! PathBuilder/...) is still a direct ancestor and `ValueHashJoin` nodes — which
//! `ir_references_variable` does not inspect — do not yet exist.

use orx_tree::{Bfs, DynTree, NodeRef};

use super::super::IR;
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
            && let IR::CondVarLenTraverse { emit_path, .. } = plan.node_mut(idx).data_mut()
        {
            *emit_path = false;
        }
    }
}
