//! Fuses chains of anonymous-edge `CondTraverse` operators sharing
//! anonymous, unreferenced intermediate nodes into a single `CondTraverse`
//! whose runtime executes the chained traversal as a single F·A1·A2·…·An
//! matrix product, mirroring C FalkorDB's `_should_divide_expression`
//! decision (`src/arithmetic/algebraic_expression/algebraic_expression_construction.c`).
//!
//! ## Plan shape (before)
//!
//! ```text
//! CondTraverse{ rel = (b)-->(c), transposed=false, chain=[] }
//!   CondTraverse{ rel = (a)-->(b), transposed=false, chain=[] }
//!     <child>
//! ```
//!
//! ## Plan shape (after)
//!
//! ```text
//! CondTraverse{ rel = (a)-->(b),  // entry hop
//!               transposed = false,
//!               chain = [ (b)-->(c) ] }
//!   <child>
//! ```
//!
//! The intermediate variable `b` must be anonymous (`_anon*` prefix), have no
//! labels (the v1 runtime does not apply mid-chain label filters), have no
//! inline attribute predicates, and not be referenced by any ancestor of the
//! outer CT. Both edges must be anonymous, non-bidirectional, non-variable-
//! length, and have no inline attribute predicates. Both ops must have empty
//! `sibling_edges` and `transposed = false`. When any condition fails the
//! pair is left alone and the fast/slow paths run as before.
//!
//! ## Why fusion is restricted to existence tests
//!
//! The fused chain runs as a single boolean matrix product, so its result
//! says only *whether* a destination is reachable — not how many distinct
//! paths reach it. On a diamond (`a->b1->c`, `a->b2->c`) the two paths
//! collapse into one row, which is wrong for a plain `MATCH`: Cypher
//! requires one row per distinct pair of relationships. Fusing is therefore
//! only applied under a `SemiApply`/`AntiSemiApply` right branch, where the
//! rows are discarded and only their existence is consulted.

use orx_tree::{Bfs, DynTree, NodeRef};

use super::super::IR;
use super::reduce_expand_into;

fn is_anon(name_opt: Option<&std::sync::Arc<String>>) -> bool {
    name_opt.is_some_and(|n| n.starts_with("_anon"))
}

/// True when the QueryRelationship's inline-attrs tree is the empty `{}` map.
fn rel_attrs_empty(
    rel: &crate::parser::ast::QueryRelationship<
        std::sync::Arc<String>,
        std::sync::Arc<String>,
        crate::parser::ast::Variable,
    >
) -> bool {
    use crate::parser::ast::ExprIR;
    let root = rel.attrs.root();
    matches!(root.data(), ExprIR::Map) && root.children().next().is_none()
}

fn node_attrs_empty(
    node: &crate::parser::ast::QueryNode<std::sync::Arc<String>, crate::parser::ast::Variable>
) -> bool {
    use crate::parser::ast::ExprIR;
    let root = node.attrs.root();
    matches!(root.data(), ExprIR::Map) && root.children().next().is_none()
}

/// True when no ancestor of `idx` (excluding the parent CondTraverse if any)
/// references the variable `(var_id, scope_id)`. We reuse the per-IR-node
/// reference check from `reduce_expand_into`'s helper.
fn intermediate_unreferenced(
    plan: &DynTree<IR>,
    idx: orx_tree::NodeIdx<orx_tree::Dyn<IR>>,
    var_id: u32,
    scope_id: u32,
) -> bool {
    let mut cur = idx;
    while let Some(parent) = plan.node(cur).parent() {
        if reduce_expand_into::ir_references_variable(parent.data(), var_id, scope_id) {
            return false;
        }
        cur = parent.idx();
    }
    true
}

/// True when the rows this traverse produces only ever feed an existence
/// test, so collapsing distinct paths cannot change the query's answer.
///
/// `SemiApply`/`AntiSemiApply` run their right branch purely to learn
/// whether it yields *at least one* row and discard the rows themselves.
/// Everywhere else the row count is observable — it is the cardinality of
/// the result set, and it feeds `count(*)` and every other aggregate — so
/// fusing would silently drop the duplicate rows that distinct paths
/// through the intermediate node are supposed to produce.
fn feeds_existence_only(
    plan: &DynTree<IR>,
    idx: orx_tree::NodeIdx<orx_tree::Dyn<IR>>,
) -> bool {
    let mut cur = idx;
    while let Some(parent) = plan.node(cur).parent() {
        match parent.data() {
            // Row-preserving operators: they neither observe nor alter the
            // number of paths, so keep walking towards the consumer.
            IR::CondTraverse { .. } | IR::Filter(_) => {}
            IR::SemiApply | IR::AntiSemiApply => {
                // Child 0 is the driving stream, whose rows are emitted and
                // therefore counted; child 1 is the existence test.
                return parent.num_children() > 1 && parent.child(1).idx() == cur;
            }
            IR::OrApplyMultiplexer(_) => {
                // Same shape, one branch per disjunct: child 0 drives, and
                // children 1..N are existence branches whose rows only feed a
                // "did anything arrive" test before being discarded.
                return (1..parent.num_children()).any(|i| parent.child(i).idx() == cur);
            }
            _ => return false,
        }
        cur = parent.idx();
    }
    false
}

/// Returns true when `parent_ct` (outer) and `child_ct` (its only CT child)
/// can be fused into a single CondTraverse.
fn can_fuse(
    parent_ct: &IR,
    child_ct: &IR,
    plan: &DynTree<IR>,
    parent_idx: orx_tree::NodeIdx<orx_tree::Dyn<IR>>,
) -> bool {
    let (
        IR::CondTraverse {
            relationship: p_rel,
            emit_relationship: p_emit,
            sibling_edges: p_sib,
            transposed: p_trans,
            chain: _p_chain,
            optional: p_opt,
            bind_relationship: true,
        },
        IR::CondTraverse {
            relationship: c_rel,
            emit_relationship: c_emit,
            sibling_edges: c_sib,
            transposed: c_trans,
            chain: c_chain,
            optional: c_opt,
            bind_relationship: true,
        },
    ) = (parent_ct, child_ct)
    else {
        return false;
    };

    // Optional traverses null-pad unmatched rows; fusing them into a chain
    // would lose that per-hop semantics.
    if *p_opt || *c_opt {
        return false;
    }

    // Direction simplification: only fuse when both hops are storage-direction.
    if *p_trans || *c_trans {
        return false;
    }
    // Anonymous edges only (criteria for skipping edge binding mid-chain and
    // for matching C's `_should_populate_edge` gate).
    if !is_anon(p_rel.alias.name.as_ref()) || !is_anon(c_rel.alias.name.as_ref()) {
        return false;
    }
    // emit_relationship must be off on both — non-anonymous edges aren't
    // collapsible (would lose per-edge bindings). reduce_expand_into has
    // already turned off emit for unreferenced anonymous edges.
    if *p_emit || *c_emit {
        return false;
    }
    // Sibling edge uniqueness needs per-edge inspection — skip.
    if !p_sib.is_empty() || !c_sib.is_empty() {
        return false;
    }
    // No bidirectional or variable-length hops.
    if p_rel.bidirectional || c_rel.bidirectional {
        return false;
    }
    if p_rel.min_hops.is_some() || c_rel.min_hops.is_some() {
        return false;
    }
    // No inline attribute predicates on edges or endpoints.
    if !rel_attrs_empty(p_rel) || !rel_attrs_empty(c_rel) {
        return false;
    }
    // Intermediate node = parent.from = child.to. Must be the same alias.
    if p_rel.from.alias.id != c_rel.to.alias.id
        || p_rel.from.alias.scope_id != c_rel.to.alias.scope_id
    {
        return false;
    }
    // Intermediate node anonymous, no labels, no inline attrs.
    let intermediate = &p_rel.from;
    if !is_anon(intermediate.alias.name.as_ref()) {
        return false;
    }
    if !intermediate.labels.is_empty() {
        return false;
    }
    if !node_attrs_empty(intermediate) {
        return false;
    }
    // Endpoint inline attrs on entry hop's `from` and final `to` are
    // handled by surrounding Filter nodes (planner emits them above the CT),
    // so we only need to check the relationships and intermediate here.
    // The child's `from` and parent's `to` are external to the fused chain.
    // Already-fused chain entries on the child stay storage-direction by
    // construction (the pass only ever inserts non-transposed hops).
    let _ = c_chain;

    // The fused chain is evaluated as a single boolean matrix product, which
    // records only *whether* the destination is reachable, not how many
    // distinct paths reach it. That is only acceptable where the row count
    // is not observable.
    if !feeds_existence_only(plan, parent_idx) {
        return false;
    }

    // Intermediate must not be referenced by any ancestor of the parent CT,
    // since after fusion it disappears from the binding set. (Filters that
    // reference the intermediate sit between parent and grandparent — those
    // ancestors are skipped by intermediate_unreferenced's BFS walk, so we
    // explicitly check the parent's siblings/ancestors via the same helper.)
    if !intermediate_unreferenced(
        plan,
        parent_idx,
        intermediate.alias.id,
        intermediate.alias.scope_id,
    ) {
        return false;
    }
    true
}

pub(super) fn fuse_anonymous_traverse(plan: &mut DynTree<IR>) {
    loop {
        // Find one fusable pair (parent CT, child CT). Restart after each
        // mutation to avoid stale node indices.
        let mut fuse_target: Option<orx_tree::NodeIdx<orx_tree::Dyn<IR>>> = None;
        for idx in plan.root().indices::<Bfs>() {
            let parent_node = plan.node(idx);
            if !matches!(parent_node.data(), IR::CondTraverse { .. }) {
                continue;
            }
            if parent_node.num_children() != 1 {
                continue;
            }
            let child_node = parent_node.child(0);
            if !matches!(child_node.data(), IR::CondTraverse { .. }) {
                continue;
            }
            if can_fuse(parent_node.data(), child_node.data(), plan, idx) {
                fuse_target = Some(idx);
                break;
            }
        }

        let Some(parent_idx) = fuse_target else { break };

        // Extract data we need before mutating.
        let parent_node = plan.node(parent_idx);
        let child_node = parent_node.child(0);
        let child_idx = child_node.idx();

        let (parent_rel, parent_emit, parent_sib, parent_chain) = if let IR::CondTraverse {
            relationship,
            emit_relationship,
            sibling_edges,
            chain,
            ..
        } = parent_node.data()
        {
            (
                relationship.clone(),
                *emit_relationship,
                sibling_edges.clone(),
                chain.clone(),
            )
        } else {
            unreachable!();
        };
        let (child_rel, child_chain) = if let IR::CondTraverse {
            relationship,
            chain,
            ..
        } = child_node.data()
        {
            (relationship.clone(), chain.clone())
        } else {
            unreachable!();
        };

        // Build the merged chain: child's existing chain (entry-side hops),
        // then the parent's relationship, then parent's existing chain.
        let mut merged_chain = child_chain;
        merged_chain.reserve_exact(parent_chain.len() + 1);
        merged_chain.push(parent_rel);
        merged_chain.extend(parent_chain);

        // Detach grandchildren (child CT's subtree) so we can reattach them
        // under the merged op.
        let grandchild_indices: Vec<_> = plan.node(child_idx).children().map(|c| c.idx()).collect();
        let grandchild_trees: Vec<DynTree<IR>> = grandchild_indices
            .into_iter()
            .map(|g_idx| plan.node_mut(g_idx).clone_as_tree())
            .collect();

        // Replace parent's data with the merged CondTraverse, then attach
        // grandchildren and prune the original child CT last — pruning may
        // invalidate other NodeIdx values under orx-tree's default memory
        // policy, so it must be the final mutation that uses parent_idx.
        *plan.node_mut(parent_idx).data_mut() = IR::CondTraverse {
            relationship: child_rel,
            emit_relationship: parent_emit,
            sibling_edges: parent_sib,
            transposed: false,
            chain: merged_chain,
            optional: false,
            bind_relationship: true,
        };
        // Attach the original grandchildren under the merged op (now parent
        // has [child_CT, g1, g2, ...]).
        for g_tree in grandchild_trees {
            plan.node_mut(parent_idx).push_child_tree(g_tree);
        }
        // Prune the original child CT, leaving [g1, g2, ...].
        plan.node_mut(child_idx).prune();
    }
}

#[cfg(test)]
mod tests {
    use orx_tree::{Bfs, DynTree, NodeRef};

    use super::super::eliminate_true_filters::eliminate_true_filters;
    use super::super::push_filters_down::push_filters_down;
    use super::super::reduce_expand_into::reduce_expand_into;
    use super::super::reduce_var_len_path::reduce_var_len_path;
    use super::fuse_anonymous_traverse;
    use crate::parser::cypher::Parser;
    use crate::planner::{IR, Planner, binder::Binder};
    use crate::runtime::functions::init_functions;
    use std::collections::HashMap;

    /// Compiles `query` through parse → bind → plan, then runs the Graph-free
    /// prefix of the optimizer in pipeline order up to and including
    /// `fuse_anonymous_traverse`. The skipped passes (`reduce_count`,
    /// `select_scan_node`, `utilize_index`) need a live GraphBLAS context and
    /// do not affect whether a CondTraverse pair is fusable.
    fn optimized_plan(query: &str) -> DynTree<IR> {
        let _ = init_functions();
        let mut parser = Parser::new(query);
        parser.parse_parameters().expect("parse parameters");
        let raw = parser.parse().expect("parse");
        let (ir, scope_vars) = Binder::default().bind(raw).expect("bind");
        let mut plan = Planner::new(scope_vars).plan(ir);

        reduce_expand_into(&mut plan);
        reduce_var_len_path(&mut plan);
        eliminate_true_filters(&mut plan, &HashMap::new());
        push_filters_down(&mut plan);
        fuse_anonymous_traverse(&mut plan);
        plan
    }

    /// Number of CondTraverse ops carrying at least one fused hop.
    fn fused_ops(plan: &DynTree<IR>) -> usize {
        plan.root()
            .indices::<Bfs>()
            .filter(|idx| {
                matches!(plan.node(*idx).data(),
                    IR::CondTraverse { chain, .. } if !chain.is_empty())
            })
            .count()
    }

    #[test]
    fn plain_match_does_not_fuse() {
        // The row count is observable here, and a boolean matrix product
        // would lose one row per extra path through the intermediate.
        let plan = optimized_plan("MATCH (a:A)-[:R]->()-[:R]->(c:C) RETURN count(*)");
        assert_eq!(fused_ops(&plan), 0);
    }

    #[test]
    fn named_intermediate_does_not_fuse() {
        let plan = optimized_plan("MATCH (a:A)-[:R]->(m)-[:R]->(c:C) RETURN m");
        assert_eq!(fused_ops(&plan), 0);
    }

    #[test]
    fn pattern_predicate_fuses() {
        // SemiApply consults only whether a row arrives, so collapsing the
        // paths cannot change the answer.
        let plan = optimized_plan("MATCH (a:A) WHERE (a)-[:R]->()-[:R]->() RETURN a");
        assert_eq!(fused_ops(&plan), 1);
    }

    #[test]
    fn negated_pattern_predicate_fuses() {
        let plan = optimized_plan("MATCH (a:A) WHERE NOT (a)-[:R]->()-[:R]->() RETURN a");
        assert_eq!(fused_ops(&plan), 1);
    }

    #[test]
    fn or_of_pattern_predicates_fuses_every_branch() {
        // Or Apply Multiplexer branches are existence tests too.
        let plan = optimized_plan(
            "MATCH (a:A) WHERE (a)-[:R]->()-[:R]->() OR (a)-[:Q]->()-[:Q]->() RETURN a",
        );
        assert_eq!(fused_ops(&plan), 2);
    }
}
