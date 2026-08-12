//! Moves an edge-only `Filter` above a `CondTraverse` into the operator's
//! `edge_filter`, so the predicate prunes during edge iteration instead of
//! after the output batch has been built.
//!
//! ```text
//! Filter(e.weight = 1)              CondTraverse(a->b,
//!   |                          =>     edge_filter: e.weight = 1)
//! CondTraverse(a->b)                  |
//!   |                                 v
//!   v
//! ```
//!
//! The sibling of [`absorb_edge_filters_into_vlt`], and for the same reason,
//! but it has to run at the other end of the pipeline. That pass runs *before*
//! `utilize_index` because a var-length edge filter can never be served by an
//! index anyway; this one runs *after*, because a fixed-length traverse's edge
//! predicate very much can be — `MATCH ()-[e:R {v: 1}]->() RETURN e` becomes an
//! `EdgeByIndexScan`, and absorbing the Filter early would hide it from the
//! pass that does that.
//!
//! ## Why it is worth a pass of its own
//!
//! Without it, a traverse emits every candidate row and the `Filter` above
//! discards the failures — so the cost of a *selective* edge predicate is paid
//! in full, on rows that were never going to survive. Measured on
//! `MATCH (a:Person)-[:KNOWS {weight: 1}]->(b) RETURN count(b)` over a graph
//! where nothing matches, the traverse produced 10,000 rows for the Filter to
//! throw away. The more selective the predicate, the more work is wasted,
//! which is exactly backwards.
//!
//! It is a trade, not a free win: `FilterOp` evaluates simple predicates with
//! vectorized batch kernels, and absorbing gives that up for a scalar eval per
//! candidate edge. It pays off where the operator *expands* the row set, so
//! rejecting early avoids building rows — which is why only `CondTraverse`
//! qualifies and `ExpandInto`, which verifies between two bound endpoints,
//! does not.
//!
//! [`absorb_edge_filters_into_vlt`]: super::absorb_edge_filters_into_vlt

use orx_tree::{Bfs, DynTree, NodeRef};

use super::super::IR;
use super::collect_expr_variables;

/// Absorbs edge-only filters into the traverse below them.
pub(super) fn absorb_edge_filters_into_traverse(optimized_plan: &mut DynTree<IR>) {
    loop {
        let mut absorbed = false;
        for idx in optimized_plan.root().indices::<Bfs>().collect::<Vec<_>>() {
            let IR::Filter(filter) = optimized_plan.node(idx).data() else {
                continue;
            };
            if optimized_plan.node(idx).num_children() != 1 {
                continue;
            }
            let child = optimized_plan.node(idx).child(0);
            let alias = match child.data() {
                // A fused chain binds no per-hop edge, so there would be
                // nothing for the predicate to test. `fuse_anonymous_traverse`
                // refuses to fuse a hop whose edge is referenced by a Filter,
                // so this should not arise — decline rather than rely on it.
                IR::CondTraverse {
                    relationship,
                    chain,
                    ..
                } if chain.is_empty() => relationship.alias.clone(),
                // `ExpandInto` is deliberately excluded. It verifies an edge
                // between two already-bound endpoints rather than expanding the
                // row set, so few candidates reach it and there is little
                // wasted materialization to avoid — while `FilterOp` evaluates
                // simple predicates with vectorized batch kernels
                // (`ops/filter.rs`), which beats a scalar per-edge eval.
                // Measured on `expand-into inline rel attrs`, absorbing it cost
                // 20.6M -> 26.3M instructions, where absorbing the expanding
                // `CondTraverse` won 42.8M -> 28.6M.
                _ => continue,
            };

            // Only when the predicate talks about this edge and nothing else.
            // A filter mentioning any other variable may depend on bindings the
            // traverse has not produced yet at the point each edge is tested.
            let vars = collect_expr_variables(filter);
            if vars.len() != 1 || !vars.contains(&alias.id) {
                continue;
            }

            let filter = filter.clone();
            let child_idx = child.idx();
            match optimized_plan.node_mut(child_idx).data_mut() {
                // Matching only on `None` is what bounds the outer loop. The
                // loop repeats while anything was absorbed, so termination
                // would otherwise depend on the `take_out` below actually
                // consuming the Filter — remove that line and the same Filter
                // is rediscovered forever, wedging `optimize`. Ask for the
                // empty field instead, and the pass cannot loop regardless.
                IR::CondTraverse {
                    edge_filter: slot @ None,
                    ..
                } => *slot = Some(filter),
                _ => continue,
            }
            optimized_plan.node_mut(idx).take_out();
            absorbed = true;
            break;
        }
        if !absorbed {
            return;
        }
    }
}
