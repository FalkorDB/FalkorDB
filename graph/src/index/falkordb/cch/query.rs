//! Phase 3: an upward walk of the elimination tree from both ends, then path unpacking
//! (#3210).
//!
//! The hierarchy is queried entirely in rank space. Both halves of the search only ever
//! *climb* rank over the upward graph; they differ only in which per-arc weight they read —
//! `upward_weight` forward (from the source), `downward_weight` backward (from the target). They meet at the
//! shared rank minimising the two distances, and the shortcut arcs on the resulting route
//! are then expanded, through each arc's recorded middle, back into original edges.
//!
//! That this is correct — that climbing only upward from both ends finds the true shortest
//! path — is what Phase 2 bought: an arc holds the cheapest path between its endpoints
//! through strictly lower-ranked nodes, so any descent a real shortest path makes is
//! already folded into some arc above it. See [`customize`](super::customize) for the
//! layout, and `src/procedures/proc_cch_idx_query.c` on `master` for the C original.
//!
//! Neither half needs a priority queue or a map. Every rank above `r` in the upward graph
//! is an ancestor of `r` in Phase 1's elimination tree, whose parent of `r` is `up(r)[0]`,
//! its lowest upper neighbour. So a climb from `r` can only reach `r`'s ancestor chain,
//! which is already in rank order: each half relaxes the upward arcs of the ranks on its
//! chain from the bottom up, and by the time it gets to a rank every arc into it has been
//! relaxed. The two halves can only meet on the ancestors their chains share, from the
//! lowest common ancestor to the root. This is the elimination-tree query of Dibbelt,
//! Strasser and Wagner, "Customizable Contraction Hierarchies" (ACM JEA, 2016). The C
//! original runs two Dijkstra searches instead; the weight is the same, and where two
//! paths tie the one returned can differ.
//!
//! The search is stateless and reads nothing mutable, so any number of queries can run
//! against one hierarchy in parallel.

use rustc_hash::FxHashMap;

use super::Cch;
use super::order::Rank;
use crate::graph::graph::NodeId;

/// Which direction of each arc a search half reads.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// From the source: arc `lower -> higher`, i.e. `upward_weight`.
    Forward,
    /// From the target, walked backwards: arc `higher -> lower`, i.e. `downward_weight`.
    Backward,
}

/// What one half of the search knows about a rank on its chain.
#[derive(Clone, Copy)]
struct Reached {
    /// Cheapest distance found so far from the half's start; infinite until reached.
    dist: f64,
    /// Chain position of the previous rank on that route; `None` at the start, and at a
    /// rank not reached yet.
    pred: Option<usize>,
}

/// One half of the search: its start rank's ancestor chain, and what it knows about each
/// rank on it.
struct Chain {
    direction: Direction,
    /// `ranks[0]` is the start, and `ranks[k + 1]` is the elimination-tree parent of
    /// `ranks[k]`, so the ranks ascend and the last one is the root of the start's tree.
    ranks: Vec<Rank>,
    /// `reached[k]` is about `ranks[k]`.
    reached: Vec<Reached>,
}

impl Chain {
    fn new(
        cch: &Cch,
        start: Rank,
        direction: Direction,
    ) -> Self {
        let topology = cch.topology();
        let mut ranks = vec![start];
        let mut rank = start;
        // a root has no upper neighbour
        while let Some(&parent) = topology.up(rank).first() {
            ranks.push(parent);
            rank = parent;
        }
        let mut reached = vec![
            Reached {
                dist: f64::INFINITY,
                pred: None,
            };
            ranks.len()
        ];
        reached[0].dist = 0.0;
        Self {
            direction,
            ranks,
            reached,
        }
    }

    /// Offer every upward arc of the rank at chain position `at` to the rank it leads to.
    ///
    /// Must be called for positions in ascending order: then every arc into `at` has
    /// already been offered, so its distance is final.
    fn relax_from(
        &mut self,
        cch: &Cch,
        at: usize,
    ) {
        let dist = self.reached[at].dist;
        if dist.is_infinite() {
            return; // not reached: nothing to offer
        }
        let topology = cch.topology();
        let rank = self.ranks[at];
        // Every upper neighbour is an ancestor, so it is on the chain above `at`. Both
        // lists ascend, so one cursor moving forward finds them all.
        let mut position = at + 1;
        for (slot, &above) in topology.up_range(rank).zip(topology.up(rank)) {
            position = gallop_to(&self.ranks, position, above);
            debug_assert_eq!(
                self.ranks.get(position),
                Some(&above),
                "an upper neighbour must be an ancestor"
            );
            let weight = match self.direction {
                Direction::Forward => cch.upward_weight(slot),
                Direction::Backward => cch.downward_weight(slot),
            };
            // an arc closed in this direction (`+inf`) offers `+inf`, which never wins
            let offer = dist + weight;
            let entry = &mut self.reached[position];
            if offer < entry.dist {
                entry.dist = offer;
                entry.pred = Some(at);
            }
        }
    }

    /// The arcs from the start to the rank at chain position `at`, in travel order for a
    /// forward half: `(lower position's rank, higher position's rank)` pairs, lowest first.
    fn arcs_up_to(
        &self,
        mut at: usize,
    ) -> Vec<(Rank, Rank)> {
        let mut arcs = Vec::new();
        while let Some(pred) = self.reached[at].pred {
            arcs.push((self.ranks[pred], self.ranks[at]));
            at = pred;
        }
        arcs.reverse();
        arcs
    }
}

/// The first position at or after `from` whose rank is at least `rank`, in an ascending
/// `ranks`; `ranks.len()` if there is none.
///
/// Gallops forward 1, 2, 4, ... positions, then binary-searches the last step, so a rank a
/// few positions ahead costs a few comparisons and a far one stays logarithmic. Upper
/// neighbours sit close together on the chain, so most calls are the short case.
fn gallop_to(
    ranks: &[Rank],
    from: usize,
    rank: Rank,
) -> usize {
    // every position below `low` holds a smaller rank; `high` is the next one to probe
    let mut low = from;
    let mut high = from;
    let mut step = 1;
    while high < ranks.len() && ranks[high] < rank {
        low = high + 1;
        high += step;
        step *= 2;
    }
    let high = high.min(ranks.len());
    low + ranks[low..high].partition_point(|&r| r < rank)
}

/// The customized weight and middle of the arc directed `from -> to`, or `None` if the two
/// ranks are not adjacent in the chordal supergraph.
///
/// An arc is stored once, at its lower-ranked endpoint; the requested direction is what
/// picks `upward_*` from `downward_*`.
fn arc(
    cch: &Cch,
    from: Rank,
    to: Rank,
) -> Option<(f64, Option<Rank>)> {
    let slot = cch.topology().arc_slot(from, to)?;
    Some(if from < to {
        (cch.upward_weight(slot), cch.upward_middle(slot))
    } else {
        (cch.downward_weight(slot), cch.downward_middle(slot))
    })
}

/// Expand the arc `from -> to` into the ranks it passes through, appending everything after
/// `from` to `out` — so a caller that has already pushed the route's first rank ends up
/// with a complete sequence.
///
/// An arc with no middle is an original edge and contributes just its far end. An arc with
/// a middle `m` expands to `from -> m -> to`, each half expanded the same way. A middle is
/// always strictly below both endpoints, so the expansion terminates.
///
/// Iterative rather than recursive (the C original recurses): the chain of middles is as deep
/// as the hierarchy, and a query runs on a Redis worker thread whose stack is not ours to
/// gamble with.
///
/// Returns `false` if an arc on the route is missing, which would mean the hierarchy no
/// longer matches the graph it was built from.
fn unpack(
    cch: &Cch,
    from: Rank,
    to: Rank,
    out: &mut Vec<Rank>,
) -> bool {
    let mut stack = vec![(from, to)];
    while let Some((a, b)) = stack.pop() {
        let Some((_, middle)) = arc(cch, a, b) else {
            return false;
        };
        match middle {
            None => out.push(b),
            Some(m) => {
                debug_assert!(m < a.min(b), "a middle must be below both endpoints");
                // b-half first on the stack so the a-half is popped first
                stack.push((m, b));
                stack.push((a, m));
            }
        }
    }
    true
}

/// Phase 3. See [`Cch::query`].
pub(super) fn query(
    cch: &Cch,
    src: u64,
    dst: u64,
) -> Option<(f64, Vec<u64>)> {
    let node_count = cch.node_count();
    let in_hierarchy = |id: u64| usize::try_from(id).is_ok_and(|index| index < node_count);
    if !in_hierarchy(src) || !in_hierarchy(dst) {
        return None;
    }
    if src == dst {
        return Some((0.0, vec![src]));
    }

    let topology = cch.topology();
    let src_rank = topology.rank_of(NodeId::from(src));
    let mut forward = Chain::new(cch, src_rank, Direction::Forward);
    let mut backward = Chain::new(
        cch,
        topology.rank_of(NodeId::from(dst)),
        Direction::Backward,
    );

    // The ancestors both chains share are their common tail, from the lowest common ancestor
    // to the root. No shared root means two trees, i.e. two components: no path.
    let shared = forward
        .ranks
        .iter()
        .rev()
        .zip(backward.ranks.iter().rev())
        .take_while(|(forward_rank, backward_rank)| forward_rank == backward_rank)
        .count();
    if shared == 0 {
        return None;
    }
    let forward_first_shared = forward.ranks.len() - shared;
    let backward_first_shared = backward.ranks.len() - shared;

    // Below the lowest common ancestor the halves cannot meet: relax all of it.
    for at in 0..forward_first_shared {
        forward.relax_from(cch, at);
    }
    for at in 0..backward_first_shared {
        backward.relax_from(cch, at);
    }

    // Then climb the shared ranks, deciding the meeting on the way: each rank's two
    // distances are final when the climb gets to it.
    let mut best = f64::INFINITY;
    let mut meet = None; // offset into the shared ranks
    for offset in 0..shared {
        let forward_at = forward_first_shared + offset;
        let backward_at = backward_first_shared + offset;
        let (forward_dist, backward_dist) = (
            forward.reached[forward_at].dist,
            backward.reached[backward_at].dist,
        );
        // strictly less: on a tie the lower rank, met first, stays, so the same query always
        // returns the same path. An infinite total never passes, so `meet` stays `None`
        // exactly when no rank is reachable from both ends.
        if forward_dist + backward_dist < best {
            best = forward_dist + backward_dist;
            meet = Some(offset);
        }
        // Weights are `>= 0`, so whatever a half offers from here costs at least its
        // distance here. At `best` or past it, that cannot lead to a cheaper meeting higher
        // up, and a tie would not move the meeting (strictly less, above), so the half
        // stops offering.
        if forward_dist < best {
            forward.relax_from(cch, forward_at);
        }
        if backward_dist < best {
            backward.relax_from(cch, backward_at);
        }
    }
    let meet = meet?;

    // forward half from the source up to the meeting rank, then the backward half from the
    // meeting rank down to the target: the backward search ran against travel, so its
    // arcs are flipped back
    let mut route = forward.arcs_up_to(forward_first_shared + meet);
    route.extend(
        backward
            .arcs_up_to(backward_first_shared + meet)
            .into_iter()
            .rev()
            .map(|(lower, higher)| (higher, lower)),
    );

    let mut ranks = vec![src_rank];
    for (a, b) in route {
        if !unpack(cch, a, b, &mut ranks) {
            // Only reachable if the hierarchy and the graph have drifted apart, which the
            // maintenance path is supposed to prevent. Return nothing rather than a route
            // with a hole in it.
            debug_assert!(false, "unpacking hit a missing arc");
            return None;
        }
    }

    let mut nodes: Vec<u64> = ranks
        .into_iter()
        .map(|rank| u64::from(topology.node_of(rank)))
        .collect();
    erase_loops(&mut nodes);
    Some((best, nodes))
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
/// edge in it does. Weights are required to be `>= 0`, not `> 0` (`algo.SPpaths`
/// likewise assumes only non-negative weights), so such cycles are legal input: two nodes
/// joined in both directions by weight-0 edges is the smallest.
fn erase_loops(route: &mut Vec<u64>) {
    let mut at: FxHashMap<u64, usize> = FxHashMap::default();
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
    use crate::graph::graphblas::matrix::Matrix;
    use crate::index::falkordb::cch::Cch;
    use crate::index::falkordb::cch::order::Rank;
    use crate::index::falkordb::cch::testkit::{
        all_pairs, disconnected_components, empty, grid, path, random_sparse, reweight,
        single_node, star, tied_paths, to_input, weights, with_isolated_nodes,
        with_zero_weight_edge, zero_weight_cycle,
    };
    use crate::testkit::check_index_against_oracle;

    /// Every ordered pair against the shared oracle ([`check_index_against_oracle`]): the
    /// kit's Dijkstra for reachability and weight, compared with a tolerance (the hierarchy
    /// sums an arc's weight in another order than Dijkstra, so a fractional weight can
    /// differ in the last bits), and every returned path checked to realise that weight and
    /// to repeat no node (a caller turning hops into edges, `db.idx.cch.query`, must not be
    /// handed a loop).
    fn check_all_pairs(edges: &Matrix<f64>) {
        let cch = Cch::build(&to_input(edges)).expect("build");
        check_index_against_oracle(|s, t| cch.query(s, t), edges);
    }

    /// Half the edges at weight zero: plenty of zero-cost cycles, which is where the two
    /// unpacked halves of a route can both pass through the same node.
    #[test]
    fn paths_are_simple_even_with_many_zero_weight_edges() {
        for seed in 0..40u64 {
            let base = random_sparse(12, 30, 3, seed);
            check_all_pairs(&reweight(&base, |i, w| if i % 2 == 0 { 0.0 } else { w }));
        }
    }

    /// Every pair against the oracle on the kit's graph families: a grid, random graphs, a
    /// sparse one-way graph where most pairs are unreachable, disconnected components, a
    /// zero-weight edge and cycle, isolated nodes, a star, a long path (a chain of
    /// shortcuts many levels deep, so the unpacking must be complete), tied paths, and the
    /// degenerate graphs.
    #[test]
    fn queries_match_the_oracle_on_kit_graphs() {
        let mut graphs = vec![
            grid(5, 6),
            random_sparse(30, 35, 5, 99),
            disconnected_components(3, 9, 18, 7, 0xD15C),
            with_zero_weight_edge(random_sparse(22, 60, 5, 3), 0, 1),
            zero_weight_cycle(7),
            with_isolated_nodes(random_sparse(18, 50, 6, 2), 7),
            star(7, 2.5),
            path(40),
            tied_paths(5, 3.0),
            single_node(),
            empty(4),
            empty(0),
        ];
        graphs.extend((0..8u64).map(|seed| random_sparse(40, 120, 9, seed)));
        for edges in &graphs {
            check_all_pairs(edges);
        }
    }

    #[test]
    fn erase_loops_keeps_first_visits_and_drops_the_cycles_between() {
        let erase = |v: &[u64]| {
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
        // 2 is erased with the cycle 1 2 3 1; the 3 after it must not find the erased one
        assert_eq!(erase(&[1, 2, 3, 1, 3]), vec![1, 3]);
    }

    /// Weights far apart in magnitude: the hierarchy sums this path in another order than
    /// Dijkstra does, so the two can differ in the last bits of `1e16 + 2`. The oracle's
    /// tolerance accepts that; an exact comparison would fail on it.
    #[test]
    fn weights_summed_in_another_order_agree_within_the_tolerance() {
        check_all_pairs(&weights(
            4,
            &[
                (3, 2, 1e16),
                (2, 1, 1.0),
                (1, 0, 1.0),
                (0, 1, 1.0),
                (1, 2, 1.0),
                (2, 3, 1e16),
            ],
        ));
    }

    #[test]
    fn gallop_to_finds_the_first_position_at_or_above_a_rank() {
        let ranks: Vec<Rank> = [2, 3, 5, 8, 13, 21, 34, 55, 89].map(Rank::new).to_vec();
        for from in 0..=ranks.len() {
            for rank in (0..100).map(Rank::new) {
                let expected = from + ranks[from..].partition_point(|&r| r < rank);
                assert_eq!(
                    super::gallop_to(&ranks, from, rank),
                    expected,
                    "from {from}, rank {rank:?}"
                );
            }
        }
        assert_eq!(super::gallop_to(&[], 0, Rank::new(7)), 0);
    }

    #[test]
    fn a_node_outside_the_hierarchy_has_no_path() {
        let cch = Cch::build(&to_input(&grid(2, 3))).expect("build");
        assert_eq!(cch.node_count(), 6);
        assert!(cch.query(6, 0).is_none(), "src past the last node");
        assert!(cch.query(0, 6).is_none(), "dst past the last node");
        assert!(cch.query(u64::MAX, 0).is_none());
    }

    /// Hand-checked: a one-way detour. 0 -> 1 -> 2 costs 3; the direct 0 -> 2 costs 10; and
    /// nothing leads back, so 2 -> 0 is unreachable.
    #[test]
    fn a_one_way_detour_is_taken_and_not_reversed() {
        let edges = weights(3, &[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 10.0)]);
        let cch = Cch::build(&to_input(&edges)).expect("build");

        assert_eq!(cch.query(0, 2), Some((3.0, vec![0, 1, 2])));
        assert_eq!(cch.query(0, 1), Some((1.0, vec![0, 1])));
        assert_eq!(cch.query(2, 0), None);
        assert_eq!(cch.query(1, 0), None);
    }

    /// Closing an edge answers like removing it, against the oracle, on random graphs
    /// with every third edge closed, and on a recustomization that closes and reopens.
    #[test]
    fn queries_match_the_oracle_with_closed_edges() {
        for seed in 0..8u64 {
            let closed = seed as usize % 3;
            let edges = reweight(&random_sparse(30, 90, 7, seed), |k, w| {
                if k % 3 == closed { f64::INFINITY } else { w }
            });
            check_all_pairs(&edges);
        }

        let open = grid(4, 4);
        let closed = reweight(&open, |k, w| if k % 2 == 0 { f64::INFINITY } else { w });
        let cch = Cch::build(&to_input(&open)).expect("build");
        let closed_cch = cch.customized(&to_input(&closed).weights).expect("close");
        let truth = all_pairs(&closed);
        for s in 0..16 {
            for t in 0..16 {
                let got = closed_cch.query(s, t).map_or(f64::INFINITY, |(w, _)| w);
                assert_eq!(got, truth[s as usize][t as usize], "closed {s}->{t}");
            }
        }
        let reopened = closed_cch
            .customized(&to_input(&open).weights)
            .expect("reopen");
        let truth = all_pairs(&open);
        for s in 0..16 {
            for t in 0..16 {
                let got = reopened.query(s, t).map_or(f64::INFINITY, |(w, _)| w);
                assert_eq!(got, truth[s as usize][t as usize], "reopened {s}->{t}");
            }
        }
    }
}
