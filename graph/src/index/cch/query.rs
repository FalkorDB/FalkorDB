//! Phase 3: rank-pruned bidirectional search and path unpacking (#3210).
//!
//! The hierarchy is queried entirely in rank space. Both halves of the search only ever
//! *climb* rank over the upward graph; they differ only in which per-arc weight they read —
//! `up_w` forward (from the source), `dn_w` backward (from the target). They meet at the
//! shared rank minimising the two distances, and the shortcut arcs on the resulting route
//! are then expanded, through each arc's recorded mid, back into original edges.
//!
//! That this is correct — that climbing only upward from both ends finds the true shortest
//! path — is what Phase 2 bought: an arc holds the cheapest path between its endpoints
//! through strictly lower-ranked nodes, so any descent a real shortest path makes is
//! already folded into some arc above it. See [`customize`](super::customize) for the
//! layout, and `src/procedures/proc_cch_idx_query.c` on `master` for the C original.
//!
//! The search is stateless and reads nothing mutable, so any number of queries can run
//! against one hierarchy in parallel.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use rustc_hash::FxHashMap;

use super::Cch;
use super::order::log_warning;

/// What one half of the search knows about a rank it has reached.
#[derive(Clone, Copy)]
struct Reached {
    dist: f64,
    /// Previous rank on the best route found so far; `None` only at the start.
    pred: Option<u32>,
    /// Popped with its final distance — later, longer offers for it are stale.
    finalized: bool,
}

/// A heap entry ordered so that `BinaryHeap`, which pops the *greatest*, pops the nearest
/// rank. The rank breaks ties, so an equal-distance frontier is explored in a fixed order.
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    dist: f64,
    rank: u32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        // reversed on both fields: nearest first, lowest rank first
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.rank.cmp(&self.rank))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Which direction of each arc a search half reads.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// From the source: arc `lo -> hi`, i.e. `up_w`.
    Forward,
    /// From the target, walked backwards: arc `hi -> lo`, i.e. `dn_w`.
    Backward,
}

/// Dijkstra from `start` restricted to the upward graph.
///
/// Returns every rank reached. "Rank-pruned" is the whole point: a node's upward
/// neighbourhood is a small fraction of its real one, so neither half explores the graph —
/// they explore two thin cones that meet near the top of the hierarchy.
fn search(
    cch: &Cch,
    start: u32,
    direction: Direction,
) -> FxHashMap<u32, Reached> {
    let topo = cch.topology();
    let mut reached: FxHashMap<u32, Reached> = FxHashMap::default();
    reached.insert(
        start,
        Reached {
            dist: 0.0,
            pred: None,
            finalized: false,
        },
    );

    let mut heap = BinaryHeap::new();
    heap.push(Candidate {
        dist: 0.0,
        rank: start,
    });

    while let Some(Candidate { rank: cur, .. }) = heap.pop() {
        let cur_dist = {
            let rec = reached.get_mut(&cur).expect("popped an unreached rank");
            if rec.finalized {
                continue; // a stale duplicate offer
            }
            rec.finalized = true;
            rec.dist
        };

        let range = topo.up_range(cur);
        for (i, &next) in topo.up(cur).iter().enumerate() {
            let slot = range.start + i;
            let w = match direction {
                Direction::Forward => cch.up_weight(slot),
                Direction::Backward => cch.dn_weight(slot),
            };
            if !w.is_finite() {
                // The arc exists, but carries nothing in this direction. Pruning here is
                // an optimization, not a correctness guard -- `nd < rec.dist` below would
                // reject an infinite offer anyway -- but without it every structurally
                // adjacent rank would still be entered into the map at infinity, which is
                // exactly the blow-up rank pruning exists to avoid.
                continue;
            }

            let nd = cur_dist + w;
            let rec = reached.entry(next).or_insert(Reached {
                dist: f64::INFINITY,
                pred: None,
                finalized: false,
            });
            if !rec.finalized && nd < rec.dist {
                rec.dist = nd;
                rec.pred = Some(cur);
                heap.push(Candidate {
                    dist: nd,
                    rank: next,
                });
            }
        }
    }

    reached
}

/// The customized weight and mid of the arc directed `from -> to`, or `None` if the two
/// ranks are not adjacent in the chordal supergraph.
///
/// An arc is stored once, at its lower-ranked endpoint; the requested direction is what
/// picks `up_*` from `dn_*`.
fn arc(
    cch: &Cch,
    from: u32,
    to: u32,
) -> Option<(f64, Option<u32>)> {
    let (lo, hi) = if from < to { (from, to) } else { (to, from) };
    let slot = cch.topology().arc_slot(lo, hi)?;
    Some(if from < to {
        (cch.up_weight(slot), cch.up_mid(slot))
    } else {
        (cch.dn_weight(slot), cch.dn_mid(slot))
    })
}

/// Expand the arc `from -> to` into the ranks it passes through, appending everything after
/// `from` to `out` — so a caller that has already pushed the route's first rank ends up
/// with a complete sequence.
///
/// An arc with no mid is an original edge and contributes just its far end. An arc with a
/// mid `m` expands to `from -> m -> to`, each half expanded the same way. A mid is always
/// strictly below both endpoints, so the expansion terminates.
///
/// Iterative rather than recursive (the C original recurses): the chain of mids is as deep
/// as the hierarchy, and a query runs on a Redis worker thread whose stack is not ours to
/// gamble with.
///
/// Returns `false` if an arc on the route is missing, which would mean the hierarchy no
/// longer matches the graph it was built from.
fn unpack(
    cch: &Cch,
    from: u32,
    to: u32,
    out: &mut Vec<u32>,
) -> bool {
    let mut stack = vec![(from, to)];
    while let Some((a, b)) = stack.pop() {
        let Some((_, mid)) = arc(cch, a, b) else {
            return false;
        };
        match mid {
            None => out.push(b),
            Some(m) => {
                debug_assert!(m < a.min(b), "a mid must be below both endpoints");
                // b-half first on the stack so the a-half is popped first
                stack.push((m, b));
                stack.push((a, m));
            }
        }
    }
    true
}

/// The shortcut-level route as a sequence of rank arcs, read off the two searches'
/// predecessor chains: `start ... meet` forward, then `meet ... end` backward.
fn route(
    fwd: &FxHashMap<u32, Reached>,
    bwd: &FxHashMap<u32, Reached>,
    meet: u32,
) -> Vec<(u32, u32)> {
    let mut arcs = Vec::new();

    // forward half: walk predecessors back from the meeting point, then reverse
    let mut cur = meet;
    while let Some(pred) = fwd[&cur].pred {
        arcs.push((pred, cur));
        cur = pred;
    }
    arcs.reverse();

    // backward half: the target's search ran backwards, so its predecessor chain from the
    // meeting point already reads in travel order
    cur = meet;
    while let Some(pred) = bwd[&cur].pred {
        arcs.push((cur, pred));
        cur = pred;
    }

    arcs
}

/// Phase 3. See [`Cch::query`].
pub(super) fn query(
    cch: &Cch,
    src: u32,
    dst: u32,
) -> Option<(f64, Vec<u32>)> {
    let n = cch.node_count();
    if src as usize >= n || dst as usize >= n {
        return None;
    }
    if src == dst {
        return Some((0.0, vec![src]));
    }

    let topo = cch.topology();
    let fwd = search(cch, topo.rank_of(src), Direction::Forward);
    let bwd = search(cch, topo.rank_of(dst), Direction::Backward);

    // meet wherever the two cones overlap most cheaply. Iterating the smaller side keeps
    // this proportional to the search, not to the graph.
    let (small, large) = if fwd.len() <= bwd.len() {
        (&fwd, &bwd)
    } else {
        (&bwd, &fwd)
    };
    let mut best = f64::INFINITY;
    let mut meet = None;
    for (&rank, here) in small {
        let Some(there) = large.get(&rank) else {
            continue;
        };
        let total = here.dist + there.dist;
        // on a tie keep the lowest rank: the map's iteration order is not stable, and the
        // same query must always return the same path
        if total < best || (total == best && Some(rank) < meet) {
            best = total;
            meet = Some(rank);
        }
    }

    let meet = meet?;
    if !best.is_finite() {
        return None;
    }

    let mut ranks = vec![topo.rank_of(src)];
    for (a, b) in route(&fwd, &bwd, meet) {
        if !unpack(cch, a, b, &mut ranks) {
            // Only reachable if the hierarchy and the graph have drifted apart, which the
            // maintenance path is supposed to prevent. Say so rather than return a route
            // with a hole in it.
            // Once per process: a drifted index would otherwise log on every query.
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, AtomicOrdering::Relaxed) {
                log_warning(
                    "CCH: query hit a missing arc while unpacking a shortcut; the index \
                     no longer matches the graph. Drop and recreate it. (Logged once.)",
                );
            }
            debug_assert!(false, "unpacking hit a missing arc");
            return None;
        }
    }

    erase_loops(&mut ranks);
    Some((best, ranks.into_iter().map(|r| topo.node_of(r)).collect()))
}

/// Cut every cycle out of a route, keeping its first visit to each node.
///
/// Each half of the route unpacks to a valid walk, but with zero-weight edges the two
/// halves can pass through the same node, so the joined walk can repeat one. A cycle on a
/// *shortest* walk must weigh exactly zero (weights are non-negative, and a positive one
/// could be cut to make the walk shorter), so removing it leaves a walk of the same weight
/// whose hops are all still real edges. The C original does not do this.
///
/// This only ever fires when the graph has a cycle of zero-weight edges: on a shortest walk a
/// repeated node means a cycle of weight exactly zero, and a cycle weighs zero only if every
/// edge in it does. Weights are required to be finite and `>= 0`, not `> 0` (`algo.SPpaths`
/// likewise assumes only non-negative weights), so such cycles are legal input: two nodes
/// joined in both directions by weight-0 edges is the smallest.
fn erase_loops(route: &mut Vec<u32>) {
    let mut at: FxHashMap<u32, usize> = FxHashMap::default();
    let mut out = 0;
    for i in 0..route.len() {
        let r = route[i];
        if let Some(&first) = at.get(&r) {
            // drop everything visited since the first time we stood on `r`
            for &gone in &route[first + 1..out] {
                at.remove(&gone);
            }
            out = first + 1;
        } else {
            at.insert(r, out);
            route[out] = r;
            out += 1;
        }
    }
    route.truncate(out);
}

#[cfg(test)]
mod tests {
    use crate::index::cch::Cch;
    use crate::index::cch::testkit::{
        EdgeList, all_pairs, assert_path_valid, disconnected_components, empty, grid, path,
        random_sparse, single_node, star, tied_paths, to_input, with_isolated_nodes,
        with_parallel_edges, with_self_loops, with_zero_weight_edge, zero_weight_cycle,
    };

    /// Every pair, against the kit's Dijkstra for the weight and its path checker for the
    /// witness. `assert_path_valid` deliberately does not check that a *cheaper* path
    /// exists, so the two halves are both needed: `all_pairs` says the weight is optimal,
    /// `assert_path_valid` says the returned nodes actually realise it.
    fn check_all_pairs(edges: &EdgeList) {
        let cch = Cch::build(&to_input(edges)).expect("build");
        let truth = all_pairs(edges);

        for s in 0..edges.n as u32 {
            for t in 0..edges.n as u32 {
                let expected = truth[s as usize][t as usize];
                match cch.query(s, t) {
                    Some((weight, nodes)) => {
                        assert!(
                            expected.is_finite(),
                            "{s}->{t}: answered {weight} where nothing is reachable"
                        );
                        assert_eq!(weight, expected, "{s}->{t} weight");
                        assert_path_valid(edges, s, t, weight, &nodes);
                        assert_simple(&nodes, s, t);
                    }
                    None => assert!(
                        expected.is_infinite(),
                        "{s}->{t}: answered unreachable, but the oracle says {expected}"
                    ),
                }
            }
        }
    }

    /// `assert_path_valid` checks hops and the sum, not that the path is simple. A shortest
    /// path never needs to repeat a node, and a caller turning hops into edges (`db.idx.cch.query`, #3215) must
    /// not be handed a loop.
    fn assert_simple(
        nodes: &[u32],
        s: u32,
        t: u32,
    ) {
        let mut seen = std::collections::HashSet::new();
        for &v in nodes {
            assert!(seen.insert(v), "{s}->{t}: path {nodes:?} repeats node {v}");
        }
    }

    /// Half the edges at weight zero: plenty of zero-cost cycles, which is where the two
    /// unpacked halves of a route can both pass through the same node.
    #[test]
    fn paths_are_simple_even_with_many_zero_weight_edges() {
        for seed in 0..40u64 {
            let base = random_sparse(12, 30, 3, seed);
            let mut edges = EdgeList::new(base.n);
            for (i, &(u, v, w)) in base.edges.iter().enumerate() {
                edges.push(u, v, if i % 2 == 0 { 0.0 } else { w });
            }
            check_all_pairs(&edges);
        }
    }

    #[test]
    fn queries_match_the_oracle_on_a_grid() {
        check_all_pairs(&grid(5, 6));
    }

    #[test]
    fn queries_match_the_oracle_on_random_graphs() {
        for seed in 0..8u64 {
            check_all_pairs(&random_sparse(40, 120, 9, seed));
        }
    }

    /// Directed and sparse enough that most pairs are genuinely unreachable.
    #[test]
    fn queries_match_the_oracle_on_a_sparse_one_way_graph() {
        check_all_pairs(&random_sparse(30, 35, 5, 99));
    }

    #[test]
    fn queries_match_the_oracle_with_parallel_edges() {
        let pairs: Vec<(u32, u32)> = (0..14u32).map(|u| (u, (u + 1) % 15)).collect();
        check_all_pairs(&with_parallel_edges(15, &pairs, 4, 9, 11));
    }

    #[test]
    fn queries_match_the_oracle_with_self_loops() {
        check_all_pairs(&with_self_loops(grid(3, 5), &[0, 7, 14], 3.0));
    }

    #[test]
    fn queries_match_the_oracle_across_disconnected_components() {
        check_all_pairs(&disconnected_components(3, 9, 18, 7, 0xD15C));
    }

    #[test]
    fn queries_match_the_oracle_with_zero_weights() {
        check_all_pairs(&with_zero_weight_edge(random_sparse(22, 60, 5, 3), 0, 1));
        check_all_pairs(&zero_weight_cycle(7));
    }

    #[test]
    fn queries_match_the_oracle_with_isolated_nodes() {
        check_all_pairs(&with_isolated_nodes(random_sparse(18, 50, 6, 2), 7));
    }

    #[test]
    fn queries_match_the_oracle_on_stars_paths_and_ties() {
        check_all_pairs(&star(7, 2.5));
        check_all_pairs(&path(11));
        check_all_pairs(&tied_paths(5, 3.0));
    }

    #[test]
    fn queries_match_the_oracle_on_degenerate_graphs() {
        check_all_pairs(&single_node());
        check_all_pairs(&empty(4));
        check_all_pairs(&empty(0));
    }

    #[test]
    fn erase_loops_keeps_first_visits_and_drops_the_cycles_between() {
        let erase = |v: &[u32]| {
            let mut v = v.to_vec();
            super::erase_loops(&mut v);
            v
        };
        assert_eq!(erase(&[1, 2, 3]), vec![1, 2, 3]);
        assert_eq!(erase(&[6, 3, 0, 1, 0]), vec![6, 3, 0]);
        assert_eq!(erase(&[10, 0, 10, 8, 3]), vec![10, 8, 3]);
        assert_eq!(erase(&[1, 2, 3, 2, 4, 1, 5]), vec![1, 5]);
        assert_eq!(erase(&[1, 2, 1, 3, 4, 3, 5]), vec![1, 3, 5]);
        assert_eq!(erase(&[7]), vec![7]);
    }

    #[test]
    fn a_node_outside_the_hierarchy_has_no_path() {
        let cch = Cch::build(&to_input(&grid(2, 3))).expect("build");
        assert_eq!(cch.node_count(), 6);
        assert!(cch.query(6, 0).is_none(), "src past the last node");
        assert!(cch.query(0, 6).is_none(), "dst past the last node");
        assert!(cch.query(u32::MAX, 0).is_none());
    }

    #[test]
    fn src_equals_dst_is_the_trivial_path() {
        let cch = Cch::build(&to_input(&grid(3, 3))).expect("build");
        assert_eq!(cch.query(4, 4), Some((0.0, vec![4])));

        // including a node with no edges at all
        let cch = Cch::build(&to_input(&empty(3))).expect("build");
        assert_eq!(cch.query(2, 2), Some((0.0, vec![2])));
    }

    /// Hand-checked: a one-way detour. 0 -> 1 -> 2 costs 3; the direct 0 -> 2 costs 10; and
    /// nothing leads back, so 2 -> 0 is unreachable.
    #[test]
    fn a_one_way_detour_is_taken_and_not_reversed() {
        let mut edges = EdgeList::new(3);
        for (u, v, w) in [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 10.0)] {
            edges.push(u, v, w);
        }
        let cch = Cch::build(&to_input(&edges)).expect("build");

        assert_eq!(cch.query(0, 2), Some((3.0, vec![0, 1, 2])));
        assert_eq!(cch.query(0, 1), Some((1.0, vec![0, 1])));
        assert_eq!(cch.query(2, 0), None);
        assert_eq!(cch.query(1, 0), None);
    }

    /// A long path forces a chain of shortcuts whose expansion is many levels deep: the
    /// test that the unpacking is complete, not just that the weight is right.
    #[test]
    fn a_long_path_unpacks_to_every_intermediate_node() {
        let edges = path(40);
        let cch = Cch::build(&to_input(&edges)).expect("build");
        let (weight, nodes) = cch.query(0, 39).expect("the path is connected");
        assert_eq!(weight, 39.0);
        assert_eq!(nodes, (0..40u32).collect::<Vec<_>>());
        assert_path_valid(&edges, 0, 39, weight, &nodes);
    }

    #[test]
    fn the_same_query_always_returns_the_same_path() {
        let edges = tied_paths(6, 2.0);
        let cch = Cch::build(&to_input(&edges)).expect("build");
        let first = cch.query(0, 7).expect("reachable");
        for _ in 0..16 {
            assert_eq!(cch.query(0, 7), Some(first.clone()));
        }
    }

    /// A recustomization must change what the query answers, not just what the arrays hold.
    #[test]
    fn a_recustomized_hierarchy_answers_with_the_new_metric() {
        let mut cheap_detour = EdgeList::new(3);
        for (u, v, w) in [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 10.0)] {
            cheap_detour.push(u, v, w);
        }
        let mut cheap_direct = EdgeList::new(3);
        for (u, v, w) in [(0, 1, 9.0), (1, 2, 9.0), (0, 2, 1.0)] {
            cheap_direct.push(u, v, w);
        }

        let mut cch = Cch::build(&to_input(&cheap_detour)).expect("build");
        assert_eq!(cch.query(0, 2), Some((2.0, vec![0, 1, 2])));

        let second = to_input(&cheap_direct);
        cch.customize(&second.weights).expect("recustomize");
        assert_eq!(cch.query(0, 2), Some((1.0, vec![0, 2])));
    }
}
