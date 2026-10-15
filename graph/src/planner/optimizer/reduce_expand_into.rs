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

use super::super::IR;
use super::references::ir_references_variable;

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
            forget_sibling_edge(plan, edge_id, edge_scope_id);
        }
    }
}

/// A collapsed edge stands for every parallel edge of its (src, dst) pair, so
/// relationship uniqueness can no longer be checked against it: the one
/// representative it binds is arbitrary, and a sibling traverse rejecting that
/// edge would make the result depend on which parallel edge was picked
/// (`(a)-[r]->(x)<-[s]-(c)` over `a⇉x`). Drop it from every sibling list, so a
/// collapsed edge takes no part in uniqueness, as in C: the row count is then
/// the number of endpoint pairs, whichever edges they carry.
///
/// The collapsed edge keeps its *own* id: a non-empty list is what keeps its
/// traverse off the batched path, which binds no relationship column, and the
/// representative must stay bound for readers `ir_references_variable` does not
/// see (e.g. `UNWIND e`, `CREATE (...{p: e.p})`).
fn forget_sibling_edge(
    plan: &mut DynTree<IR>,
    edge_id: u32,
    edge_scope_id: u32,
) {
    let indices: Vec<_> = plan.root().indices::<Bfs>().collect();
    for idx in indices {
        if let IR::CondTraverse {
            relationship,
            sibling_edges,
            ..
        }
        | IR::ExpandInto {
            relationship,
            sibling_edges,
            ..
        } = plan.node_mut(idx).data_mut()
            && relationship.alias.scope_id == edge_scope_id
            && relationship.alias.id != edge_id
        {
            sibling_edges.retain(|&id| id != edge_id);
        }
    }
}
