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
//! 2. **Filtered** (score 2) -- referenced by a Filter around the chain
//!    (inline pattern attributes are lowered to Filters by the planner)
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
    parser::ast::{AllShortestPaths, ExprIR, QueryNode, QueryRelationship, Variable},
    tree,
};

use super::super::IR;

/// Scores a candidate scan endpoint for the scan node selection optimizer.
///
/// Returns `(score, filter_runs_late, cardinality)`.
///
/// Higher score = better starting point. Priority:
/// - Bound variable (provided by child operator): score 3
/// - Filtered variable (referenced by any Filter around the chain): score 2
/// - Labeled variable: score 1
/// - Neither: score 0
///
/// `filter_runs_late` breaks ties between equally-constrained endpoints: it is
/// true when the endpoint's predicate sits *above* the chain, so it currently
/// runs only after the whole traversal. Scanning from there moves it to the
/// front, which is the larger win — `MATCH (A:L {v:1})-->(B)-->(C), (B)-->(D:L
/// {v:1})` should reach `D` before expanding to the unconstrained `C`. An
/// endpoint whose Filter already sits mid-chain gains much less from being
/// scanned first, and without this it would win on the chain-position tiebreak
/// purely for being nearer the leaf.
///
/// When those are equal, the endpoint with fewer label nodes is preferred.
fn score_endpoint(
    node: &Arc<QueryNode<Arc<String>, Variable>>,
    filtered_vars: &FilteredVars,
    bound_vars: &HashSet<u32>,
    graph: &Graph,
) -> (u32, bool, u64) {
    let mut score = 0u32;
    if bound_vars.contains(&node.alias.id) {
        score += 3;
    }
    // Inline attrs (e.g. `{name: 'Nicolas Cage'}`) are lowered to Filters by
    // the planner and stripped from the pattern, so they arrive here through
    // `filtered_vars` like any other predicate rather than being counted
    // separately — which used to score such an endpoint twice.
    if filtered_vars.all.contains(&node.alias.id) {
        score += 2;
    }
    let filter_runs_late = filtered_vars.above.contains(&node.alias.id);
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
    (score, filter_runs_late, cardinality)
}

/// Collects variable IDs referenced by Filter nodes around the chain at
/// `start_idx`: ancestors above it, and the inter-operator Filters within the
/// chain below it.
///
/// The downward half matters because the planner emits an endpoint's
/// inline-attr Filter directly above the operator that binds it, so in
/// `MATCH (a)-[]->(b {x:1})-[]->(c:C)` the Filter on `b` sits *between* the two
/// traverses — below `start_idx`, which is the top of the chain. Looking only
/// upwards misses it, and `b` then scores as if it had no predicate at all.
/// Variables constrained by Filters around a chain, split by where the Filter
/// sits — see [`score_endpoint`], which ranks the two differently.
struct FilteredVars {
    /// Referenced by a Filter *above* the chain: the predicate runs only after
    /// the whole traversal.
    above: HashSet<u32>,
    /// `above` plus the variables referenced by Filters between the chain's
    /// own operators, whose predicates already run mid-traversal.
    all: HashSet<u32>,
}

fn collect_filtered_vars(
    plan: &DynTree<IR>,
    start_idx: NodeIdx<Dyn<IR>>,
) -> FilteredVars {
    fn collect(
        filter: &crate::parser::ast::QueryExpr<Variable>,
        vars: &mut HashSet<u32>,
    ) {
        for idx in filter.root().indices::<Bfs>() {
            if let ExprIR::Variable(v) = filter.node(idx).data() {
                vars.insert(v.id);
            }
        }
    }

    let mut above = HashSet::new();
    let mut current = start_idx;
    while let Some(parent) = plan.node(current).parent() {
        match parent.data() {
            IR::Filter(filter) => collect(filter, &mut above),
            // Walk through transparent operators to find filters higher up
            IR::CondTraverse { .. } | IR::CondVarLenTraverse { .. } | IR::PathBuilder(_) => {}
            _ => break,
        }
        current = parent.idx();
    }

    // Descend the chain's single-child spine for the Filters the planner
    // parked between operators — an endpoint bound mid-chain has its
    // inline-attr Filter emitted directly above the operator that binds it, so
    // in `MATCH (a)-[]->(b {x:1})-[]->(c:C)` the Filter on `b` is below
    // `start_idx` and invisible to the upward walk. Without it `b` would score
    // as though nothing constrained it.
    let mut all = above.clone();
    let mut node = plan.node(start_idx);
    loop {
        match node.data() {
            IR::Filter(filter) => collect(filter, &mut all),
            IR::CondTraverse { .. } | IR::CondVarLenTraverse { .. } | IR::PathBuilder(_) => {}
            _ => break,
        }
        if node.num_children() != 1 {
            break;
        }
        node = node.child(0);
    }
    FilteredVars { above, all }
}

/// Creates a scan subtree for the given node. Shape:
/// `[Filter →] [IncludePending →] Scan [→ Argument]`.
///
/// `include_pending` wraps the scan with `IncludePending`, required inside
/// MERGE match branches so the scan sees in-flight mutations. `argument`
/// attaches an `Argument` leaf below the scan, so correlated rows keep
/// flowing when the scan replaces a bare `Argument` child. `filters` are the
/// `Filter` operators salvaged from the subtree this one replaces — see
/// [`filters_of`].
fn make_scan_subtree(
    node: &Arc<QueryNode<Arc<String>, Variable>>,
    include_pending: bool,
    argument: Option<IR>,
    filters: Vec<IR>,
) -> DynTree<IR> {
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
    // Innermost first, so the original nesting order is preserved.
    for filter in filters.into_iter().rev() {
        scan = tree!(filter, scan);
    }
    scan
}

/// Returns the `Filter` operators on a scan subtree's single-child spine,
/// outermost first.
///
/// The sibling of [`argument_leaf_of`]: whatever the spine carried has to come
/// along when the subtree is pruned and rebuilt. This pass used to re-derive an
/// inline-attr filter from the rebuilt node's own `attrs`, but a MATCH
/// pattern's attrs are stripped once the planner has lowered them, so these
/// `Filter` nodes are now the predicate's only representation. Dropping one
/// drops the predicate — `MATCH (a:A {x:1}) MATCH (a)-[:R]->(b)` stitches
/// clause 1's `Filter → NodeByLabelScan` in as the traverse's child, and this
/// pass prunes and rebuilds it.
fn filters_of(
    plan: &DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
) -> Vec<IR> {
    let mut filters = vec![];
    let mut node = plan.node(idx);
    loop {
        if matches!(node.data(), IR::Filter(_)) {
            filters.push(node.data().clone());
        }
        if node.num_children() != 1 {
            return filters;
        }
        node = node.child(0);
    }
}

/// Returns the `Argument` leaf of a scan subtree, if it has one.
///
/// `add_argument_to_leaves` attaches an `Argument` *below* the leaf scan of a
/// correlated sub-plan, and this pass builds the same shape itself, so a
/// planner-scan subtree can look like
/// `Filter → IncludePending → Scan → Argument`. Whenever such a subtree is
/// pruned and rebuilt the leaf has to come along: without it the rebuilt scan
/// stops replaying outer rows and the sub-plan loses its correlation (and
/// row multiplicity).
fn argument_leaf_of(
    plan: &DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
) -> Option<IR> {
    let mut node = plan.node(idx);
    loop {
        if let IR::Argument(_) = node.data() {
            return Some(node.data().clone());
        }
        if node.num_children() != 1 {
            return None;
        }
        node = node.child(0);
    }
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

/// Returns the alias id of the scan at the bottom of a planner-added scan
/// subtree (the shape [`is_planner_scan_subtree`] accepts), or `None` when
/// `idx` does not root such a subtree.
fn planner_scan_alias(
    plan: &DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
) -> Option<u32> {
    let mut node = plan.node(idx);
    loop {
        match node.data() {
            IR::AllNodeScan(n) | IR::NodeByLabelScan { node: n, .. } => return Some(n.alias.id),
            IR::Filter(_) | IR::IncludePending { .. } if node.num_children() == 1 => {
                node = node.child(0);
            }
            _ => return None,
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

/// Picks which endpoint of a leaf `CondVarLenTraverse` to scan.
///
/// The planner always scans the pattern's `from` endpoint. When `to` is the
/// more selective one (labeled, or filtered by inline attributes / a `WHERE`
/// predicate) scanning it instead is cheaper: `CondVarLenTraverseOp` sees a
/// bound `to` with an unbound `from` and walks the relationship backwards, so
/// the pattern still binds exactly the same rows. This mirrors what the
/// `CondTraverse` chain logic does, and matches FalkorDB C — which likewise
/// only reverses on selectivity, never on label cardinality.
fn select_var_len_scan_node(
    optimized_plan: &mut DynTree<IR>,
    graph: &Graph,
) {
    // Paths rather than `NodeIdx`es, and deepest-first, for the same reason
    // `select_scan_node` uses them: `prune` can reorganize the tree.
    let mut vlt_paths = Vec::new();
    for idx in optimized_plan.root().indices::<Bfs>() {
        if matches!(
            optimized_plan.node(idx).data(),
            IR::CondVarLenTraverse { .. }
        ) {
            vlt_paths.push(node_path(optimized_plan, idx));
        }
    }
    vlt_paths.sort_by_key(|p| std::cmp::Reverse(p.len()));

    for path in vlt_paths {
        let Some(idx) = resolve_path(optimized_plan, &path) else {
            continue;
        };
        let node = optimized_plan.node(idx);
        let IR::CondVarLenTraverse {
            relationship,
            expand_into,
            ..
        } = node.data()
        else {
            continue;
        };
        // Nothing to choose between: both endpoints are already bound, both
        // ends are the same variable, or the runtime has no backwards walk for
        // this pattern (bidirectional / allShortestPaths).
        if *expand_into
            || relationship.bidirectional
            || relationship.all_shortest_paths != AllShortestPaths::No
            || relationship.from.alias.id == relationship.to.alias.id
        {
            continue;
        }
        // Only rewrite when the planner put its own scan for `from` here.
        if node.num_children() != 1 {
            continue;
        }
        let child_idx = node.child(0).idx();
        if planner_scan_alias(optimized_plan, child_idx) != Some(relationship.from.alias.id) {
            continue;
        }
        let from = relationship.from.clone();
        let to = relationship.to.clone();

        // A correlated sub-plan may already bind `to` from its outer context;
        // scanning it here would rebind it. `Argument(None)` is opaque about
        // what it carries, so treat it as binding everything.
        let argument = argument_leaf_of(optimized_plan, child_idx);
        match &argument {
            Some(IR::Argument(None)) => continue,
            Some(IR::Argument(Some(vars))) if vars.contains(&(to.alias.id, to.alias.scope_id)) => {
                continue;
            }
            _ => {}
        }

        let filtered_vars = collect_filtered_vars(optimized_plan, idx);
        // This used to refuse the reversal unless every endpoint carrying
        // inline attrs also had a Filter above the traverse — a cross-check
        // that the two representations agreed. There is only one
        // representation now: the planner lowers the attrs and strips them, so
        // the check could only ever be vacuously true. Replacing the scan
        // subtree still cannot lose the predicate, because it lives in a Filter
        // above this traverse and `push_filters_down` lands it back on the new
        // scan.
        let bound = HashSet::new();
        let (from_score, _, _) = score_endpoint(&from, &filtered_vars, &bound, graph);
        let (to_score, _, _) = score_endpoint(&to, &filtered_vars, &bound, graph);
        // A tie keeps the pattern's own direction.
        if to_score <= from_score {
            continue;
        }

        let in_merge = in_merge_match_branch(optimized_plan, idx);
        let mut scan = if to.labels.is_empty() {
            DynTree::new(IR::AllNodeScan(to.clone()))
        } else {
            DynTree::new(IR::NodeByLabelScan { node: to.clone() })
        };
        if let Some(arg) = argument {
            let root_idx = scan.root().idx();
            scan.node_mut(root_idx).push_child(arg);
        }
        if in_merge {
            scan = tree!(IR::IncludePending { node: to.clone() }, scan);
        }

        optimized_plan.node_mut(child_idx).prune();
        let idx = resolve_path(optimized_plan, &path)
            .expect("pruning a child never changes the parent's path");
        optimized_plan.node_mut(idx).push_child_tree(scan);
    }
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
            let (score_a, late_a, card_a) =
                score_endpoint(&a.0, &filtered_vars, &bound_vars, graph);
            let (score_b, late_b, card_b) =
                score_endpoint(&b.0, &filtered_vars, &bound_vars, graph);
            score_a
                .cmp(&score_b)
                // Equally constrained: prefer the one whose predicate runs last
                // today, since scanning from it moves that filter to the front.
                .then_with(|| late_a.cmp(&late_b))
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

                // Check if child is a planner-added scan before mutating, and
                // capture any Argument leaf it carries so the rebuilt scan
                // keeps replaying outer rows.
                let (child_is_planner_scan, preserved_argument) = if is_leaf {
                    (false, None)
                } else {
                    let child_idx = optimized_plan.node(ct_idx).child(0).idx();
                    let is_scan = is_planner_scan_subtree(optimized_plan, child_idx);
                    let arg = if arg_transparent {
                        Some(make_argument())
                    } else if is_scan {
                        argument_leaf_of(optimized_plan, child_idx)
                    } else {
                        None
                    };
                    (is_scan, arg)
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

                // Chain reversal: the pruned subtree's Filters constrain the
                // *old* scan endpoint, which this traverse now binds instead of
                // scanning, so they cannot move onto the new scan. The planner's
                // copy above the operator still enforces them.
                let scan_subtree =
                    make_scan_subtree(&scan_node, in_merge, preserved_argument, vec![]);

                let mut op = optimized_plan.node_mut(ct_idx);
                *op.data_mut() = IR::CondTraverse {
                    relationship: new_rel,
                    emit_relationship: emit,
                    sibling_edges: edges,
                    transposed: true,
                    chain: Vec::new(),
                    optional: false,
                    bind_relationship: true,
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
            // Argument (those get replaced by a new scan for best_node). When
            // it is replaced, carry over any Argument leaf it held.
            let mut preserved_argument = None;
            let existing_child = if is_leaf {
                None
            } else {
                let child_idx = optimized_plan.node(bottom_idx).child(0).idx();
                let child_is_planner_scan = is_planner_scan_subtree(optimized_plan, child_idx);
                if child_is_planner_scan || arg_transparent {
                    preserved_argument = if arg_transparent {
                        Some(make_argument())
                    } else {
                        argument_leaf_of(optimized_plan, child_idx)
                    };
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
                make_scan_subtree(&best_node, in_merge, preserved_argument, vec![])
            });
            for (step, (rel, emit, edges, transposed)) in new_rels.into_iter().rev().enumerate() {
                subtree = tree!(
                    IR::CondTraverse {
                        relationship: rel,
                        emit_relationship: emit,
                        sibling_edges: edges,
                        transposed,
                        chain: Vec::new(),
                        optional: false,
                        bind_relationship: true,
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
                    // Capture the pruned subtree's Argument leaf (if any)
                    // before it is dropped, so the rebuilt scan keeps it.
                    let preserved_argument = if arg_transparent {
                        Some(make_argument())
                    } else if has_planner_scan {
                        let child_idx = optimized_plan.node(ct_idx).child(0).idx();
                        argument_leaf_of(optimized_plan, child_idx)
                    } else {
                        None
                    };
                    // Not a swap: the rebuilt scan is for the same node that
                    // was pruned, so its Filters belong on it and are now the
                    // only copy of any inline-attr predicate.
                    let preserved_filters = if has_planner_scan {
                        let child_idx = optimized_plan.node(ct_idx).child(0).idx();
                        filters_of(optimized_plan, child_idx)
                    } else {
                        vec![]
                    };

                    let ct_idx = if has_planner_scan || arg_transparent {
                        let child_idx = optimized_plan.node(ct_idx).child(0).idx();
                        optimized_plan.node_mut(child_idx).prune();
                        resolve_path(optimized_plan, &path)
                            .expect("pruning a child never changes the parent's path")
                    } else {
                        ct_idx
                    };

                    let scan_subtree = make_scan_subtree(
                        &scan_node,
                        in_merge,
                        preserved_argument,
                        preserved_filters,
                    );

                    let mut op = optimized_plan.node_mut(ct_idx);
                    *op.data_mut() = IR::CondTraverse {
                        relationship: new_rel,
                        emit_relationship: emit,
                        sibling_edges: edges,
                        transposed: trans,
                        chain: Vec::new(),
                        optional: false,
                        bind_relationship: true,
                    };

                    op.push_child_tree(scan_subtree);
                }
            }
        }
        // else: no swap needed and non-leaf — nothing to do, child already attached.
    }

    select_var_len_scan_node(optimized_plan, graph);
}
