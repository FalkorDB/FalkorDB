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

use orx_tree::{Bfs, Dyn, DynTree, NodeIdx, NodeRef};

use super::super::IR;
use super::references::ir_references_variable;
use crate::runtime::ops::{COLLAPSED_SIBLING, COLLAPSED_SIBLING_ANY_TYPE};

/// An edge collapsed by [`reduce_expand_into`]: its alias `(id, scope_id)`, the
/// flag its entry in the sibling lists carries (`None` when its types cannot be
/// told apart at run time), and the root of the branch it belongs to.
struct Collapsed {
    edge_id: u32,
    edge_scope_id: u32,
    flag: Option<u32>,
    branch: NodeIdx<Dyn<IR>>,
}

pub(super) fn reduce_expand_into(plan: &mut DynTree<IR>) {
    let indices: Vec<_> = plan.root().indices::<Bfs>().collect();
    let mut collapsed = vec![];
    for idx in indices {
        let (edge_id, edge_scope_id, flag) = match plan.node(idx).data() {
            IR::ExpandInto {
                emit_relationship: true,
                relationship,
                ..
            }
            | IR::CondTraverse {
                emit_relationship: true,
                relationship,
                ..
            } => (
                relationship.alias.id,
                relationship.alias.scope_id,
                match relationship.types.len() {
                    0 => Some(COLLAPSED_SIBLING_ANY_TYPE),
                    1 => Some(COLLAPSED_SIBLING),
                    _ => None,
                },
            ),
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
            collapsed.push(Collapsed {
                edge_id,
                edge_scope_id,
                flag,
                branch: branch_root(plan, idx),
            });
        }
    }
    flag_collapsed_siblings(plan, &collapsed);
}

/// The root of the plan branch containing `idx`: the child of the nearest
/// `Union` above it, else the plan root. Top-level `UNION` branches are bound
/// separately and reuse alias ids and scopes, so an edge only relates to the
/// operators of its own branch.
fn branch_root(
    plan: &DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
) -> NodeIdx<Dyn<IR>> {
    let mut cur = idx;
    while let Some(parent) = plan.node(cur).parent() {
        if matches!(parent.data(), IR::Union) {
            break;
        }
        cur = parent.idx();
    }
    cur
}

/// A collapsed edge stands for every parallel edge of its (src, dst) pair, so
/// relationship uniqueness cannot be checked against the one representative it
/// binds when that pair holds parallel edges: rejecting the representative
/// would make the result depend on which parallel edge was picked
/// (`(a)-[r]->(x)<-[s]-(c)` over `a⇉x`). Flag it in the sibling lists of its
/// branch, so the runtime ignores a collision with it only when an alternative
/// edge exists, and keeps checking it when the pair holds a single edge. One
/// pass per branch covers all the edges collapsed in it.
fn flag_collapsed_siblings(
    plan: &mut DynTree<IR>,
    collapsed: &[Collapsed],
) {
    let mut done: Vec<NodeIdx<Dyn<IR>>> = vec![];
    for first in collapsed {
        if done.contains(&first.branch) {
            continue;
        }
        done.push(first.branch);
        let indices: Vec<_> = plan.node(first.branch).indices::<Bfs>().collect();
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
            {
                for c in collapsed.iter().filter(|c| c.branch == first.branch) {
                    let Some(flag) = c.flag else { continue };
                    if relationship.alias.scope_id == c.edge_scope_id
                        && relationship.alias.id != c.edge_id
                    {
                        for id in sibling_edges.iter_mut().filter(|id| **id == c.edge_id) {
                            *id |= flag;
                        }
                    }
                }
            }
        }
    }
}
