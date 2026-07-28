//! Scan node selection optimizer pass.
//!
//! Selects the optimal starting endpoint for chains of `CondTraverse`
//! operators and inserts (or replaces) the leaf scan accordingly. If the
//! best endpoint is on the opposite side of the chain from the current leaf,
//! the entire chain is reversed and each `CondTraverse` is marked
//! `transposed = true` so the runtime knows to transpose the relationship
//! matrix scan.
//!
//! ## Endpoint Scoring
//!
//! Each candidate endpoint is scored by (highest priority first):
//!
//! 1. **Bound** (score 3) -- already provided by a child operator (e.g.
//!    Project, Aggregate, Argument from an outer Apply)
//! 2. **Filtered** (score 2) -- referenced by a Filter ancestor above the
//!    chain, or has inline property attributes ({name: 'Alice'})
//! 3. **Labeled** (score 1) -- has at least one label
//! 4. **Cardinality** (tiebreaker) -- label with fewer nodes wins
//!
//! ## Single CondTraverse
//!
//! ```text
//! Before (to is better):          After (swap + transposed):
//!
//! CondTraverse (a)->(b:Person)    CondTraverse (b:Person)->(a)
//!                                   transposed = true
//!                                   |
//!                                   v
//!                                 NodeByLabelScan(:Person)
//! ```
//!
//! ## Chain Reversal
//!
//! For chains of CondTraverse operators (CT_0 -> CT_1 -> ... -> CT_n), if
//! the best endpoint is at the top of the chain, the entire chain order is
//! reversed and each relationship's from/to is swapped:
//!
//! ```text
//! Before:                          After:
//!
//! CT_2: (c)->(d)                   CT_0': (d)->(c)  [transposed]
//!   |                                |
//!   v                                v
//! CT_1: (b)->(c)                   CT_1': (c)->(b)  [transposed]
//!   |                                |
//!   v                                v
//! CT_0: (a)->(b)                   CT_2': (b)->(a)  [transposed]
//!                                    |
//!                                    v
//!                                  NodeByLabelScan(:D)
//! ```
//!
//! Inter-chain Filter nodes (inline attribute filters on intermediate
//! destination nodes) are preserved and reattached after reversal.

use std::collections::HashSet;
use std::sync::Arc;

use orx_tree::{Bfs, Dyn, DynTree, NodeIdx, NodeRef};

use crate::{
    graph::graph::Graph,
    parser::ast::{ExprIR, QueryNode, QueryRelationship, Variable},
    tree,
};

use super::super::{IR, inline_attrs_to_filter};

/// Scores a candidate scan endpoint for the scan node selection optimizer.
///
/// Higher score = better starting point. Priority:
/// - Bound variable (provided by child operator): score 3
/// - Filtered variable (referenced by a Filter ancestor): score 2
/// - Labeled variable: score 1
/// - Neither: score 0
///
/// When scores are equal, the endpoint with fewer label nodes is preferred.
fn score_endpoint(
    node: &Arc<QueryNode<Arc<String>, Variable>>,
    filtered_vars: &HashSet<u32>,
    bound_vars: &HashSet<u32>,
    graph: &Graph,
) -> (u32, u64) {
    let mut score = 0u32;
    if bound_vars.contains(&node.alias.id) {
        score += 3;
    }
    if filtered_vars.contains(&node.alias.id) {
        score += 2;
    }
    // Node attribute filters (e.g. {name: "Nicolas Cage"}) also count as filters.
    if node.attrs.root().num_children() > 0 {
        score += 2;
    }
    if !node.labels.is_empty() {
        score += 1;
    }
    // Cardinality: minimum label node count (lower is better).
    // For nodes with no labels, use u64::MAX so labeled nodes win ties.
    let cardinality = if node.labels.is_empty() {
        u64::MAX
    } else {
        node.labels
            .iter()
            .map(|l| graph.label_node_count(l))
            .min()
            .unwrap_or(u64::MAX)
    };
    (score, cardinality)
}

/// Collects variable IDs referenced by Filter nodes that are ancestors of
/// the given node index, up to the first non-Filter/non-CondTraverse ancestor.
fn collect_filtered_vars(
    plan: &DynTree<IR>,
    start_idx: NodeIdx<Dyn<IR>>,
) -> HashSet<u32> {
    let mut vars = HashSet::new();
    let mut current = start_idx;
    while let Some(parent) = plan.node(current).parent() {
        match parent.data() {
            IR::Filter(filter) => {
                for idx in filter.root().indices::<Bfs>() {
                    if let ExprIR::Variable(v) = filter.node(idx).data() {
                        vars.insert(v.id);
                    }
                }
            }
            // Walk through transparent operators to find filters higher up
            IR::CondTraverse { .. } | IR::CondVarLenTraverse { .. } | IR::PathBuilder(_) => {}
            _ => break,
        }
        current = parent.idx();
    }
    // Also check the node at current if it has a parent that is a filter
    // (the loop above moves through parents)
    vars
}

/// Creates a scan subtree for the given node, with an optional inline attr
/// filter. Shape: `[Filter →] [IncludePending →] Scan [→ Argument]`.
///
/// `include_pending` wraps the scan with `IncludePending`, required inside
/// MERGE match branches so the scan sees in-flight mutations. `argument`
/// attaches an `Argument` leaf below the scan, so correlated rows keep
/// flowing when the scan replaces a bare `Argument` child.
fn make_scan_subtree(
    node: &Arc<QueryNode<Arc<String>, Variable>>,
    include_pending: bool,
    argument: Option<IR>,
) -> DynTree<IR> {
    let attr_filter = inline_attrs_to_filter(&node.alias, &node.attrs);
    let mut scan = if node.labels.is_empty() {
        DynTree::new(IR::AllNodeScan(node.clone()))
    } else {
        DynTree::new(IR::NodeByLabelScan { node: node.clone() })
    };
    if let Some(arg) = argument {
        let root_idx = scan.root().idx();
        scan.node_mut(root_idx).push_child(arg);
    }
    if include_pending {
        scan = tree!(IR::IncludePending { node: node.clone() }, scan);
    }
    if let Some(filter_expr) = attr_filter {
        scan = tree!(IR::Filter(Arc::new(filter_expr)), scan);
    }
    scan
}

/// Returns true if `idx` sits inside the match branch (last child) of the
/// nearest enclosing `Merge`. Scans inserted there must be wrapped with
/// `IncludePending`, mirroring what the planner does for its own scans.
fn in_merge_match_branch(
    plan: &DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
) -> bool {
    let mut current = plan.node(idx);
    while let Some(parent) = current.parent() {
        if matches!(parent.data(), IR::Merge { .. }) {
            return current.sibling_idx() == parent.num_children() - 1;
        }
        current = parent;
    }
    false
}

/// Returns true if the subtree rooted at `idx` is a planner-added scan: zero
/// or more single-child `Filter` / `IncludePending` wrappers terminating in an
/// `AllNodeScan` or `NodeByLabelScan` — the exact shape [`make_scan_subtree`]
/// (plus `IncludePending` wrapping) produces. A `Filter` wrapping an
/// outer-context operator (e.g. `ExpandInto`, another traversal) must NOT
/// match: pruning it would drop part of the query.
fn is_planner_scan_subtree(
    plan: &DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
) -> bool {
    let mut node = plan.node(idx);
    loop {
        match node.data() {
            IR::AllNodeScan(_) | IR::NodeByLabelScan { .. } => return true,
            IR::Filter(_) | IR::IncludePending { .. } if node.num_children() == 1 => {
                node = node.child(0);
            }
            _ => return false,
        }
    }
}

/// Creates a new `QueryRelationship` with from and to swapped.
fn swap_relationship(
    rel: &Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>,
    new_from: Arc<QueryNode<Arc<String>, Variable>>,
    new_to: Arc<QueryNode<Arc<String>, Variable>>,
) -> Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>> {
    let mut swapped = QueryRelationship::new(
        rel.alias.clone(),
        rel.types.clone(),
        rel.attrs.clone(),
        new_from,
        new_to,
        rel.bidirectional,
        rel.min_hops,
        rel.max_hops,
    );
    swapped.all_shortest_paths = rel.all_shortest_paths;
    Arc::new(swapped)
}

/// Collects output variable alias IDs from an IR node.
/// Used to detect which variables a child operator provides.
fn collect_output_aliases(ir: &IR) -> HashSet<u32> {
    let mut aliases = HashSet::new();
    match ir {
        IR::AllNodeScan(n) | IR::NodeByLabelScan { node: n, .. } => {
            aliases.insert(n.alias.id);
        }
        IR::NodeByIndexScan { node, .. }
        | IR::NodeByLabelAndIdScan { node, .. }
        | IR::NodeByIdSeek { node, .. } => {
            aliases.insert(node.alias.id);
        }
        IR::NodeByFulltextScan { node, score, .. } => {
            aliases.insert(node.id);
            if let Some(s) = score {
                aliases.insert(s.id);
            }
        }
        IR::Project { exprs, copies } => {
            for (var, _) in exprs {
                aliases.insert(var.id);
            }
            for (var, _) in copies {
                aliases.insert(var.id);
            }
        }
        IR::Aggregate { names, .. } => {
            for var in names {
                aliases.insert(var.id);
            }
        }
        IR::Unwind { var, .. } => {
            aliases.insert(var.id);
        }
        // Argument with known bound vars: the incoming rows bind exactly
        // these variables. `Argument(None)` stays opaque (conservative).
        // Only the id is kept: this set feeds `score_endpoint`, which is a
        // preference heuristic over bare ids (as `filtered_vars` already is)
        // and never decides plan validity. The scope-sensitive decision —
        // whether the Argument is transparent — compares full pairs.
        IR::Argument(Some(vars)) => {
            aliases.extend(vars.iter().map(|(id, _)| *id));
        }
        _ => {}
    }
    aliases
}

/// Computes the structural path (sequence of sibling indices from the root) to
/// the node at `idx`. Unlike a [`NodeIdx`], a path is immune to the tree's
/// `Auto` memory-reclaim reorganization that can occur after node removals, so
/// it can be used to re-resolve a node after the tree has been mutated.
fn node_path(
    plan: &DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
) -> Vec<usize> {
    let mut path = Vec::new();
    let mut current = plan.node(idx);
    while let Some(parent) = current.parent() {
        path.push(current.sibling_idx());
        current = parent;
    }
    path.reverse();
    path
}

/// Re-resolves a structural path produced by [`node_path`] back into a fresh
/// [`NodeIdx`] for the current tree. Returns `None` if the path no longer
/// corresponds to an existing node (e.g. the structure changed beneath it).
fn resolve_path(
    plan: &DynTree<IR>,
    path: &[usize],
) -> Option<NodeIdx<Dyn<IR>>> {
    let mut node = plan.root();
    for &pos in path {
        node = node.get_child(pos)?;
    }
    Some(node.idx())
}

/// Selects the optimal scan node for leaf `CondTraverse` operators.
///
/// For each bottom-of-chain CondTraverse (leaf or with a non-CT child),
/// determines the best endpoint to scan from based on: (1) bound variables
/// from child, (2) filter presence, (3) label presence, (4) label cardinality.
/// Adds a `NodeByLabelScan` or `AllNodeScan` child for leaf chains, and
/// optionally swaps from/to with `transposed=true` if the better endpoint is
/// at the other end.
///
/// For chains of CondTraverse operators, walks up to find the best endpoint
/// across the entire chain. If the best endpoint is not at the bottom, reverses
/// the chain direction.
pub(super) fn select_scan_node(
    optimized_plan: &mut DynTree<IR>,
    graph: &Graph,
) {
    // Collect all bottom-of-chain CondTraverse nodes as structural paths.
    // A "bottom CT" is a CT that either has no children (leaf) or whose
    // only child is not a CT (e.g., Project, AllNodeScan).
    //
    // We record paths rather than `NodeIdx`es because processing one chain
    // mutates the tree (prune / push_child_tree). Those removals can trigger
    // orx-tree's `Auto` memory-reclaim reorganization, which invalidates any
    // previously-collected `NodeIdx` for nodes that are still in the tree.
    // A structural path is re-resolved against the live tree just before use.
    //
    // Paths are processed deepest-first so that a processed chain's local
    // mutations (which only ever affect its own subtree) never shift the path
    // of a not-yet-processed, shallower chain.
    let mut bottom_ct_paths = Vec::new();
    for idx in optimized_plan.root().indices::<Bfs>() {
        let node = optimized_plan.node(idx);
        if !matches!(node.data(), IR::CondTraverse { .. }) {
            continue;
        }
        if node.num_children() == 0 {
            bottom_ct_paths.push(node_path(optimized_plan, idx));
            continue;
        }
        if node.num_children() != 1 {
            continue;
        }
        // Walk through single-child Filter nodes to find the real child.
        let mut child = node.child(0);
        while matches!(child.data(), IR::Filter(_)) && child.num_children() == 1 {
            child = child.child(0);
        }
        if !matches!(child.data(), IR::CondTraverse { .. }) {
            bottom_ct_paths.push(node_path(optimized_plan, idx));
        }
    }
    bottom_ct_paths.sort_by_key(|p| std::cmp::Reverse(p.len()));

    for path in bottom_ct_paths {
        // Re-resolve the path against the (possibly mutated) live tree. Skip if
        // the node no longer exists or is no longer a CondTraverse.
        let Some(bottom_idx) = resolve_path(optimized_plan, &path) else {
            continue;
        };
        if !matches!(
            optimized_plan.node(bottom_idx).data(),
            IR::CondTraverse { .. }
        ) {
            continue;
        }
        let is_leaf = optimized_plan.node(bottom_idx).num_children() == 0;
        // Detect if the child is a planner-added scan (not an outer-context op).
        let has_planner_scan = !is_leaf && {
            let child_idx = optimized_plan.node(bottom_idx).child(0).idx();
            is_planner_scan_subtree(optimized_plan, child_idx)
        };
        // Treat CTs with planner-added scans like leaf CTs for scan selection.
        let effectively_leaf = is_leaf || has_planner_scan;

        // Walk up the chain of CondTraverse nodes to collect all endpoints.
        // The walk skips single-child Filter nodes between CTs (these are
        // inline attribute filters on destination nodes).
        let mut chain: Vec<NodeIdx<Dyn<IR>>> = vec![bottom_idx];
        {
            let mut current = bottom_idx;
            while let Some(parent) = optimized_plan.node(current).parent() {
                if matches!(parent.data(), IR::CondTraverse { .. }) {
                    chain.push(parent.idx());
                    current = parent.idx();
                } else if matches!(parent.data(), IR::Filter(_)) && parent.num_children() == 1 {
                    // Skip single-child Filter between CTs, but check if
                    // its parent is a CT to continue the chain.
                    let filter_idx = parent.idx();
                    if let Some(grandparent) = optimized_plan.node(filter_idx).parent()
                        && matches!(grandparent.data(), IR::CondTraverse { .. })
                    {
                        chain.push(grandparent.idx());
                        current = grandparent.idx();
                        continue;
                    }
                    break;
                } else {
                    break;
                }
            }
        }

        // Collect filtered variables from Filter ancestors above the chain.
        let top_of_chain = *chain.last().unwrap();
        let filtered_vars = collect_filtered_vars(optimized_plan, top_of_chain);

        // For non-leaf chains, detect bound variables from the child.
        // Only consider vars as "bound" if they come from an outer context
        // (Project, Aggregate, Argument, etc.), NOT from scan children
        // added by the planner — those just provide starting nodes.
        let bound_vars = if is_leaf {
            HashSet::new()
        } else {
            let child_idx = optimized_plan.node(bottom_idx).child(0).idx();
            let child_data = optimized_plan.node(child_idx).data();
            match child_data {
                IR::AllNodeScan(_)
                | IR::NodeByLabelScan { .. }
                | IR::NodeByIndexScan { .. }
                | IR::NodeByLabelAndIdScan { .. }
                | IR::Filter(_) => HashSet::new(),
                _ => collect_output_aliases(child_data),
            }
        };

        // If the child is an Argument with a known bound-var set, capture the
        // set: when it binds none of the chain's endpoints, the Argument only
        // replays outer rows for correlation/multiplicity and a scan can be
        // inserted between the CT and the Argument.
        let child_argument_vars: Option<Vec<(u32, u32)>> = if is_leaf {
            None
        } else {
            match optimized_plan.node(bottom_idx).child(0).data() {
                IR::Argument(Some(vars)) => Some(vars.clone()),
                _ => None,
            }
        };

        // Collect all candidate endpoints from the chain.
        // Each endpoint is (node, chain_position, is_from).
        // chain_position 0 = bottom, higher = closer to root.
        let mut candidates: Vec<(Arc<QueryNode<Arc<String>, Variable>>, usize, bool)> = vec![];

        // Track which alias IDs we've already seen to avoid duplicates
        // (a node that is the `to` of one CT and `from` of the next).
        let mut seen_aliases = HashSet::new();

        for (pos, &ct_idx) in chain.iter().enumerate() {
            if let IR::CondTraverse { relationship, .. } = optimized_plan.node(ct_idx).data() {
                if seen_aliases.insert(relationship.from.alias.id) {
                    candidates.push((relationship.from.clone(), pos, true));
                }
                if seen_aliases.insert(relationship.to.alias.id) {
                    candidates.push((relationship.to.clone(), pos, false));
                }
            }
        }

        // The Argument is "transparent" when it provably binds none of the
        // chain's endpoint variables. Treat the CT like a leaf then: a scan
        // will be inserted with the Argument re-attached beneath it.
        let arg_transparent = child_argument_vars.as_ref().is_some_and(|vars| {
            candidates
                .iter()
                .all(|(node, _, _)| !vars.contains(&(node.alias.id, node.alias.scope_id)))
        });
        let effectively_leaf = effectively_leaf || arg_transparent;
        // Any scan this pass builds inside a MERGE match branch must see
        // in-flight mutations, mirroring the planner's
        // set_include_pending_on_scans. This depends only on where the
        // traversal sits, never on why we are rebuilding the scan: the
        // non-transparent paths below also replace planner-added scan
        // subtrees (which may already carry IncludePending), and rebuilding
        // one without the wrapper would silently stop it observing pending
        // mutations.
        let in_merge = in_merge_match_branch(optimized_plan, bottom_idx);
        // The Argument leaf to re-attach beneath any scan that replaces it.
        let make_argument = || IR::Argument(child_argument_vars.clone());

        // Score each candidate and find the best.
        let best = candidates.iter().max_by(|a, b| {
            let (score_a, card_a) = score_endpoint(&a.0, &filtered_vars, &bound_vars, graph);
            let (score_b, card_b) = score_endpoint(&b.0, &filtered_vars, &bound_vars, graph);
            score_a
                .cmp(&score_b)
                .then_with(|| card_b.cmp(&card_a)) // lower cardinality = better
                // Prefer leaf position (0) and `from` side to preserve
                // the original traversal direction when all else is equal.
                .then_with(|| b.1.cmp(&a.1)) // lower chain pos = better
                .then_with(|| {
                    // Prefer is_from=true (original `from` node)
                    a.2.cmp(&b.2)
                })
        });

        let Some((best_node, best_pos, best_is_from)) = best.cloned() else {
            continue;
        };

        // Determine if we need to reverse the chain.
        // The best endpoint should become the `from` of the bottom CT.
        let need_swap = if best_pos == 0 {
            // Best is at the bottom CT. Swap only if it's the `to`.
            !best_is_from
        } else {
            // Best is at a parent CT. Need to reverse the chain.
            true
        };

        if need_swap && (chain.len() == 1 || best_pos == 0) {
            // Best endpoint is the `to` of the bottom CT.  Swap the bottom
            // CT only — upper CTs in the chain remain unchanged because the
            // shared node (best_node) is still produced into env by the
            // transposed bottom CT and consumed by the next CT's `from`.
            let ct_idx = chain[0];
            if let IR::CondTraverse {
                relationship,
                emit_relationship,
                sibling_edges,
                ..
            } = optimized_plan.node(ct_idx).data()
            {
                let new_from = relationship.to.clone();
                let new_to = relationship.from.clone();
                let new_rel = swap_relationship(relationship, new_from, new_to);
                let emit = *emit_relationship;
                let edges = sibling_edges.clone();
                let scan_node = relationship.to.clone();

                // Check if child is a planner-added scan before mutating
                let child_is_planner_scan = if is_leaf {
                    false
                } else {
                    let child_idx = optimized_plan.node(ct_idx).child(0).idx();
                    is_planner_scan_subtree(optimized_plan, child_idx)
                };

                // Remove the old child if it was a planner-added scan or a
                // transparent Argument (re-attached beneath the new scan).
                // `prune` can trigger `Auto` memory reclaim, invalidating
                // every NodeIdx — re-resolve the CT via its structural path
                // (`path`, which points at chain[0] and is unaffected by
                // removing its own child).
                let ct_idx = if child_is_planner_scan || arg_transparent {
                    let child_idx = optimized_plan.node(ct_idx).child(0).idx();
                    optimized_plan.node_mut(child_idx).prune();
                    resolve_path(optimized_plan, &path)
                        .expect("pruning a child never changes the parent's path")
                } else {
                    ct_idx
                };

                // Build scan subtree before taking mutable borrow
                let scan_subtree =
                    make_scan_subtree(&scan_node, in_merge, arg_transparent.then(make_argument));

                let mut op = optimized_plan.node_mut(ct_idx);
                *op.data_mut() = IR::CondTraverse {
                    relationship: new_rel,
                    emit_relationship: emit,
                    sibling_edges: edges,
                    transposed: true,
                    chain: Vec::new(),
                };

                if is_leaf || child_is_planner_scan || arg_transparent {
                    // Add scan subtree (with optional attr filter) as child.
                    op.push_child_tree(scan_subtree);
                }
                // else: child is from outer context, keep it.
            }
        } else if need_swap && chain.len() > 1 {
            // Best is at a parent CT (best_pos > 0). Reverse the chain.

            // Collect relationship data from each CT in the chain (bottom to root).
            let mut rels: Vec<(
                Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>,
                bool,
                Vec<u32>,
            )> = Vec::new();
            // Also collect Filter nodes between CTs (keyed by destination alias).
            // These are inline attribute filters on destination nodes.
            let mut inter_ct_filters: Vec<(usize, DynTree<IR>)> = Vec::new();
            for (i, &ct_idx) in chain.iter().enumerate() {
                if let IR::CondTraverse {
                    relationship,
                    emit_relationship,
                    sibling_edges,
                    ..
                } = optimized_plan.node(ct_idx).data()
                {
                    rels.push((
                        relationship.clone(),
                        *emit_relationship,
                        sibling_edges.clone(),
                    ));
                }
                // Collect Filter nodes between this CT and the next CT in chain.
                if i < chain.len() - 1 {
                    let next_ct_idx = chain[i + 1];
                    // Walk from next_ct -> ... -> current_ct, collect Filters.
                    let mut walk = optimized_plan.node(next_ct_idx).child(0).idx();
                    while walk != ct_idx {
                        let walk_data = optimized_plan.node(walk).data();
                        if matches!(walk_data, IR::Filter(_)) {
                            // Clone just the Filter node (without its children)
                            let filter_expr = match walk_data {
                                IR::Filter(expr) => expr.clone(),
                                _ => unreachable!(),
                            };
                            inter_ct_filters.push((i, tree!(IR::Filter(filter_expr))));
                        }
                        if optimized_plan.node(walk).num_children() > 0 {
                            walk = optimized_plan.node(walk).child(0).idx();
                        } else {
                            break;
                        }
                    }
                }
            }

            // Detach existing child of the bottom CT (if non-leaf) for reattachment,
            // but only if it's NOT a planner-added scan or a transparent
            // Argument (those get replaced by a new scan for best_node).
            let existing_child = if is_leaf {
                None
            } else {
                let child_idx = optimized_plan.node(bottom_idx).child(0).idx();
                let child_is_planner_scan = is_planner_scan_subtree(optimized_plan, child_idx);
                if child_is_planner_scan || arg_transparent {
                    None // Will create a new scan for best_node instead
                } else {
                    Some(optimized_plan.node_mut(child_idx).clone_as_tree())
                }
            };

            // Reverse the chain and swap from/to on each relationship.
            rels.reverse();
            let mut new_rels: Vec<(
                Arc<QueryRelationship<Arc<String>, Arc<String>, Variable>>,
                bool,
                Vec<u32>,
                bool, // transposed
            )> = Vec::new();

            for (rel, emit, edges) in &rels {
                let new_from = rel.to.clone();
                let new_to = rel.from.clone();
                let new_rel = swap_relationship(rel, new_from, new_to);
                new_rels.push((new_rel, *emit, edges.clone(), true));
            }

            // Build the new subtree bottom-up, inserting inter-CT filters at
            // the correct hop.  `new_rels.into_iter().rev()` yields hops
            // corresponding to original chain positions 0, 1, …, n-1.
            // A filter collected at original position `i` should be inserted
            // right after the hop for original chain[i] is wrapped around the
            // subtree (and before the next hop wraps it).
            let mut subtree = existing_child.unwrap_or_else(|| {
                make_scan_subtree(&best_node, in_merge, arg_transparent.then(make_argument))
            });
            for (step, (rel, emit, edges, transposed)) in new_rels.into_iter().rev().enumerate() {
                subtree = tree!(
                    IR::CondTraverse {
                        relationship: rel,
                        emit_relationship: emit,
                        sibling_edges: edges,
                        transposed,
                        chain: Vec::new(),
                    },
                    subtree
                );
                // The original chain position for this step is `step`.
                // Apply any inter-CT filters that were between chain[step]
                // and chain[step+1] in the original (pre-reversal) chain.
                for (orig_pos, filter_tree) in &inter_ct_filters {
                    if *orig_pos == step {
                        let filter_data = filter_tree.root().data().clone();
                        subtree = tree!(filter_data, subtree);
                    }
                }
            }

            // Replace the chain in the plan. Each `prune` below can trigger
            // `Auto` memory reclaim, invalidating every NodeIdx, so the top
            // CT is re-resolved via its structural path after each removal
            // (its own path is unaffected by removing its descendants).
            let top_path = node_path(optimized_plan, *chain.last().unwrap());

            // Detach all children of the top CT (the old chain below it).
            let top_idx = loop {
                let top_idx = resolve_path(optimized_plan, &top_path)
                    .expect("pruning a descendant never changes the ancestor's path");
                if optimized_plan.node(top_idx).num_children() == 0 {
                    break top_idx;
                }
                let child_idx = optimized_plan.node(top_idx).child(0).idx();
                optimized_plan.node_mut(child_idx).prune();
            };

            // Replace the top CT with the root of the new subtree.
            let new_root = subtree.root();
            *optimized_plan.node_mut(top_idx).data_mut() = new_root.data().clone();

            // Add children of the new subtree root to the top CT node.
            for child in new_root.children() {
                let child_tree: DynTree<IR> = child.clone_as_tree();
                optimized_plan.node_mut(top_idx).push_child_tree(child_tree);
            }
        } else if effectively_leaf {
            // No swap needed. Add/replace a scan for the current `from` node.
            let ct_idx = chain[0];
            if let IR::CondTraverse { relationship, .. } = optimized_plan.node(ct_idx).data() {
                let scan_node = relationship.from.clone();
                let from_node = relationship.from.clone();
                let to_node = relationship.to.clone();
                let new_rel = swap_relationship(relationship, from_node, to_node);

                if let IR::CondTraverse {
                    emit_relationship,
                    sibling_edges,
                    transposed,
                    ..
                } = optimized_plan.node(ct_idx).data()
                {
                    let emit = *emit_relationship;
                    let edges = sibling_edges.clone();
                    let trans = *transposed;

                    // Remove the old child if it was a planner-added scan or
                    // a transparent Argument (re-attached beneath the new
                    // scan). `prune` can trigger `Auto` memory reclaim,
                    // invalidating every NodeIdx — re-resolve the CT via its
                    // structural path (`path`, which points at chain[0] and
                    // is unaffected by removing its own child).
                    let ct_idx = if has_planner_scan || arg_transparent {
                        let child_idx = optimized_plan.node(ct_idx).child(0).idx();
                        optimized_plan.node_mut(child_idx).prune();
                        resolve_path(optimized_plan, &path)
                            .expect("pruning a child never changes the parent's path")
                    } else {
                        ct_idx
                    };

                    // Build scan subtree with optional attr filter
                    let scan_subtree = make_scan_subtree(
                        &scan_node,
                        in_merge,
                        arg_transparent.then(make_argument),
                    );

                    let mut op = optimized_plan.node_mut(ct_idx);
                    *op.data_mut() = IR::CondTraverse {
                        relationship: new_rel,
                        emit_relationship: emit,
                        sibling_edges: edges,
                        transposed: trans,
                        chain: Vec::new(),
                    };

                    op.push_child_tree(scan_subtree);
                }
            }
        }
        // else: no swap needed and non-leaf — nothing to do, child already attached.
    }
}
