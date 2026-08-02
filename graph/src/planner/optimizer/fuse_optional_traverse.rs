//! Fuses `Optional` wrappers whose sub-plan is a single-hop `CondTraverse`
//! fed directly by an `Argument` into one optional `CondTraverse`, mirroring
//! C FalkorDB's "Optional Conditional Traverse" operator.
//!
//! ## Plan shapes (before)
//!
//! ```text
//! Optional(vars = [e, b])          Apply
//!   <input>                          <input>
//!   CondTraverse{ (a)-[e]->(b) }     Optional(vars = [e, b])
//!     Argument                         CondTraverse{ (a)-[e]->(b) }
//!                                        Argument
//! ```
//!
//! ## Plan shape (after)
//!
//! ```text
//! CondTraverse{ (a)-[e]->(b), optional = true }
//!   <input>
//! ```
//!
//! The fused traverse null-pads the edge and destination columns for input
//! rows that produce no expansion, so the `Optional` NULL fallback machinery
//! (and its per-input-batch sub-plan instantiation) is skipped entirely.
//!
//! Fusion requires every optional variable to be introduced by the traverse
//! itself (its edge alias or its unbound endpoint) — anything else (named
//! paths, extra pattern parts, WHERE filters between the two nodes) keeps the
//! general `Optional` operator. This pass runs last, so no other pass has to
//! reason about optional traverses.

use orx_tree::{Bfs, DynNode, DynTree, NodeRef};

use super::super::IR;
use crate::parser::ast::Variable;

/// Checks whether `sub` is a fusable single-hop optional sub-plan:
/// `CondTraverse{ chain: [], optional: false }` over a single `Argument`,
/// with every var in `vars` introduced by the traverse itself.
fn fusable_traverse(
    sub: &DynNode<IR>,
    vars: &[Variable],
) -> bool {
    let IR::CondTraverse {
        relationship: rp,
        transposed,
        chain,
        optional: false,
        ..
    } = sub.data()
    else {
        return false;
    };
    if !chain.is_empty() {
        return false;
    }
    if sub.num_children() != 1 || !matches!(sub.child(0).data(), IR::Argument(_)) {
        return false;
    }
    // A self-loop pattern binds one alias via `from`; the null-pad
    // logic assumes distinct endpoint aliases.
    if rp.from.alias == rp.to.alias {
        return false;
    }
    let out_alias = if *transposed {
        &rp.from.alias
    } else {
        &rp.to.alias
    };
    // The destination must be one of the vars the Optional null-pads
    // (i.e. genuinely introduced by this clause), and no optional var
    // may come from anywhere other than the traverse itself.
    if !vars.contains(out_alias) {
        return false;
    }
    vars.iter().all(|v| *v == rp.alias || v == out_alias)
}

/// Clones `sub`'s CondTraverse data with `optional = true`.
fn fused_traverse(sub: &DynNode<IR>) -> IR {
    if let IR::CondTraverse {
        relationship,
        emit_relationship,
        sibling_edges,
        transposed,
        chain,
        ..
    } = sub.data()
    {
        IR::CondTraverse {
            relationship: relationship.clone(),
            emit_relationship: *emit_relationship,
            sibling_edges: sibling_edges.clone(),
            transposed: *transposed,
            chain: chain.clone(),
            optional: true,
        }
    } else {
        unreachable!();
    }
}

pub(super) fn fuse_optional_traverse(plan: &mut DynTree<IR>) {
    loop {
        // (fuse target idx, traverse node idx, intermediate Optional idx to
        // prune when the target is an Apply).
        let mut target = None;
        for idx in plan.root().indices::<Bfs>() {
            let node = plan.node(idx);
            match node.data() {
                // Standalone Optional: child 0 = input, child 1 = sub-plan.
                IR::Optional(vars) => {
                    if node.num_children() != 2 {
                        continue;
                    }
                    let sub = node.child(1);
                    if fusable_traverse(&sub, vars) {
                        target = Some((idx, sub.idx(), None));
                        break;
                    }
                }
                // Apply over Optional: child 0 = input, child 1 = Optional
                // whose single child is the sub-plan.
                IR::Apply => {
                    if node.num_children() != 2 {
                        continue;
                    }
                    let opt = node.child(1);
                    let IR::Optional(vars) = opt.data() else {
                        continue;
                    };
                    if opt.num_children() != 1 {
                        continue;
                    }
                    let sub = opt.child(0);
                    if fusable_traverse(&sub, vars) {
                        target = Some((idx, sub.idx(), Some(opt.idx())));
                        break;
                    }
                }
                _ => {}
            }
        }

        let Some((idx, sub_idx, opt_idx)) = target else {
            break;
        };

        let fused = fused_traverse(&plan.node(sub_idx));
        *plan.node_mut(idx).data_mut() = fused;
        // Prune last: it may invalidate other NodeIdx values, so the loop
        // restarts its BFS afterwards.
        plan.node_mut(opt_idx.unwrap_or(sub_idx)).prune();
    }
}
