//! Fiduccia-Mattheyses refinement.
//!
//! Two refinement kernels:
//!
//! - `refine_bisection`: classic FM with best-balanced rollback.
//!   Picks the highest-gain boundary vertex, flips its partition,
//!   updates neighbor gains, locks it, and repeats. Tracks both
//!   "current cut" and "best cut subject to balance" separately; at
//!   the end of the pass rolls back to the best balanced state.
//! - `refine_separator`: greedy node-separator reduction. For each
//!   separator vertex it computes the gain (weight saved by pulling
//!   the vertex out of the separator, minus weight of neighbors on
//!   the far side that would need to enter the separator). Accepts
//!   only positive-gain moves that respect the balance constraint.
//!   A full two-sided FM with negative-gain acceptance is deferred
//!   until a concrete quality gap motivates it.
//!
//! Priority queue: a lazy `BinaryHeap<(gain, Reverse(v))>` rather
//! than METIS's bucket array. Correct; the O(log n) overhead per
//! operation is acceptable at the graph sizes FERAL targets
//! (≤ 100k vertices, FM rarely dominates runtime).

use feral_ordering_core::Idx;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graph::Graph;
use crate::initial_partition::{cut_size, part_weight, PART_A, PART_B};

pub const PART_SEP: u8 = 2;

/// Refine an edge bisection with FM, tracking the best balanced cut.
///
/// `labels[v] ∈ {PART_A, PART_B}`. Modifies `labels` in place.
/// Returns the final edge cut.
pub fn refine_bisection(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    max_passes: u32,
) -> Idx {
    let n = graph.nvtxs as usize;
    if n < 2 {
        return cut_size(graph, labels);
    }
    let total: i64 = graph.vwgt.iter().copied().sum();
    let max_side = ((1.0 + max_imbalance) * total as f64 / 2.0).ceil() as i64;

    let mut pass_cut = cut_size(graph, labels);

    for _pass in 0..max_passes {
        let before_pass = pass_cut;
        let mut gain: Vec<Idx> = vec![0; n];
        compute_gains(graph, labels, &mut gain);

        let mut locked: Vec<bool> = vec![false; n];
        let mut heap: BinaryHeap<(Idx, Reverse<Idx>, Idx)> = BinaryHeap::new();
        // (gain, Reverse(vertex), stamp): stamp is the gain snapshot
        // stored alongside the entry so stale entries can be skipped.
        //
        // NOTE: every vertex is seeded here, not just the current
        // boundary. METIS seeds only boundary vertices and lazily
        // re-inserts a vertex the first time a neighbour move makes it
        // a boundary vertex, costing Ω(boundary) per pass rather than
        // the Ω(n log n) below. Interior vertices have
        // gain = -internal_degree ≤ 0, so they sit at the bottom of
        // this max-heap and are only popped after the positive-gain
        // boundary moves that actually reduce the cut. Seeding all n is
        // a deliberate simplicity-over-speed trade at FERAL's target
        // sizes (≤ 100k vertices, where FM rarely dominates runtime);
        // see dev/tried-and-rejected.md (O11).
        for (v, &g) in gain.iter().enumerate().take(n) {
            heap.push((g, Reverse(v as Idx), g));
        }

        let mut cur_cut = pass_cut;
        let mut moves: Vec<Idx> = Vec::new();
        let mut a_w = part_weight(graph, labels, PART_A);
        let mut b_w = total - a_w;
        // If the pass starts imbalanced, best_prefix = None means
        // "no balanced state seen yet" — rollback to prefix 0 would
        // keep the imbalanced start, which defeats the purpose of
        // FM on imbalanced input. The first balanced state the
        // trajectory reaches is unconditionally recorded.
        let starts_balanced = a_w.max(b_w) <= max_side;
        let mut best_prefix: Option<usize> = if starts_balanced { Some(0) } else { None };
        let mut best_cut = pass_cut;
        let mut best_a_w = a_w;
        let mut best_b_w = b_w;
        let mut no_improve: u32 = 0;

        while let Some((_, Reverse(v), stamp)) = heap.pop() {
            let vu = v as usize;
            if locked[vu] {
                continue;
            }
            if stamp != gain[vu] {
                // Stale — re-push if we haven't moved it yet.
                continue;
            }
            // Tentatively move v to the other side.
            let from = labels[vu];
            let to = if from == PART_A { PART_B } else { PART_A };
            let (new_a_w, new_b_w) = if to == PART_A {
                (a_w + graph.vwgt[vu], b_w - graph.vwgt[vu])
            } else {
                (a_w - graph.vwgt[vu], b_w + graph.vwgt[vu])
            };
            let side_max = new_a_w.max(new_b_w);
            // Always allow the move so FM can climb out of poor local
            // configurations, but only consider it for "best" if the
            // resulting partition is balanced.
            labels[vu] = to;
            a_w = new_a_w;
            b_w = new_b_w;
            cur_cut -= gain[vu];
            locked[vu] = true;
            moves.push(v);

            // Update neighbor gains.
            let lo = graph.xadj[vu] as usize;
            let hi = graph.xadj[vu + 1] as usize;
            for k in lo..hi {
                let u = graph.adjncy[k] as usize;
                if locked[u] {
                    continue;
                }
                // gain = ed - id. If neighbour u shared v's old side
                // (`from`): edge (u,v) was internal, now crosses →
                // u's ed +w, id -w → Δgain = +2w. If u shares v's
                // new side (`to`): edge was crossing, now internal
                // → Δgain = -2w.
                let w = graph.adjwgt[k];
                if labels[u] == from {
                    gain[u] += 2 * w;
                } else {
                    gain[u] -= 2 * w;
                }
                heap.push((gain[u], Reverse(u as Idx), gain[u]));
            }

            if side_max <= max_side {
                // Record as best iff (a) this is the first balanced
                // state seen this pass, or (b) it improves on the
                // previous best balanced cut.
                let is_first_balanced = best_prefix.is_none();
                if is_first_balanced || cur_cut < best_cut {
                    best_cut = cur_cut;
                    best_prefix = Some(moves.len());
                    best_a_w = a_w;
                    best_b_w = b_w;
                    no_improve = 0;
                } else {
                    no_improve += 1;
                }
            } else {
                no_improve += 1;
            }
            if no_improve >= 50 {
                break;
            }
        }

        // Roll back moves after best_prefix. If the pass started
        // imbalanced and never reached a balanced state, best_prefix
        // is None — keep the full move sequence (labels at the
        // post-last-move state) so the next pass starts from the
        // closest-to-balance configuration FM found, rather than
        // rolling all the way back to the imbalanced start.
        match best_prefix {
            Some(prefix) => {
                for &v in moves.iter().skip(prefix) {
                    let vu = v as usize;
                    labels[vu] = if labels[vu] == PART_A { PART_B } else { PART_A };
                }
                a_w = best_a_w;
                b_w = best_b_w;
                pass_cut = best_cut;
            }
            None => {
                // Labels already at post-last-move state; keep them.
                pass_cut = cur_cut;
            }
        }
        let _ = (a_w, b_w); // kept for post-pass debug assertions
        if pass_cut == before_pass {
            break;
        }
    }
    pass_cut
}

/// Compute per-vertex gain = (edges to other side) - (edges to own side).
fn compute_gains(graph: &Graph, labels: &[u8], gain: &mut [Idx]) {
    for (v, &lv) in labels.iter().enumerate() {
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        let mut ed: Idx = 0;
        let mut id: Idx = 0;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            let w = graph.adjwgt[k];
            if labels[u] == lv {
                id = id.saturating_add(w);
            } else {
                ed = ed.saturating_add(w);
            }
        }
        gain[v] = ed - id;
    }
}

/// Greedy node-separator refinement. Accepts positive-gain moves that
/// respect the balance constraint. Returns the final separator weight.
pub fn refine_separator(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    max_passes: u32,
) -> i64 {
    let n = graph.nvtxs as usize;
    let total: i64 = graph.vwgt.iter().copied().sum();
    let max_side = ((1.0 + max_imbalance) * total as f64 / 2.0).ceil() as i64;

    for _pass in 0..max_passes {
        let mut changed = false;
        let mut a_w = part_weight(graph, labels, PART_A);
        let mut b_w = part_weight(graph, labels, PART_B);
        for v in 0..n {
            if labels[v] != PART_SEP {
                continue;
            }
            // Compute cost of pulling v to side A or side B.
            let (cost_to_a, cost_to_b) = separator_pull_costs(graph, labels, v);
            let vwgt_v = graph.vwgt[v];
            // Net separator change = cost_to_side - vwgt_v.
            // Move is beneficial if cost_to_side < vwgt_v (gain > 0).
            let gain_a = vwgt_v - cost_to_a;
            let gain_b = vwgt_v - cost_to_b;
            let (best_gain, best_side) = if gain_a >= gain_b {
                (gain_a, PART_A)
            } else {
                (gain_b, PART_B)
            };
            if best_gain <= 0 {
                continue;
            }
            // Balance check: after the move, side gains vwgt_v; the
            // other side gains zero (only separator weight changes).
            let new_a = if best_side == PART_A {
                a_w + vwgt_v
            } else {
                a_w
            };
            let new_b = if best_side == PART_B {
                b_w + vwgt_v
            } else {
                b_w
            };
            if new_a.max(new_b) > max_side {
                continue;
            }
            // Apply move: v → best_side; far-side neighbors → SEP.
            labels[v] = best_side;
            if best_side == PART_A {
                a_w = new_a;
            } else {
                b_w = new_b;
            }
            let lo = graph.xadj[v] as usize;
            let hi = graph.xadj[v + 1] as usize;
            let far = if best_side == PART_A { PART_B } else { PART_A };
            for k in lo..hi {
                let u = graph.adjncy[k] as usize;
                if labels[u] == far {
                    labels[u] = PART_SEP;
                    if far == PART_A {
                        a_w -= graph.vwgt[u];
                    } else {
                        b_w -= graph.vwgt[u];
                    }
                }
            }
            changed = true;
        }
        if !changed {
            break;
        }
    }
    separator_weight(graph, labels)
}

/// Fiduccia-Mattheyses refinement of a **node separator**.
///
/// `labels[v] ∈ {PART_A, PART_B, PART_SEP}` on entry and on exit, and
/// the input must already be a valid separator. Returns the final
/// separator weight, which is never larger than the input's.
///
/// This is METIS's `FM_2WayNodeRefine1Sided` (`libmetis/sfm.c`):
/// passes alternate the target side, so one priority queue suffices.
/// Within a pass the highest-gain separator vertex is pulled into the
/// target side — which forces its neighbours on the far side into the
/// separator — and the pass keeps going through **negative**-gain
/// moves, rolling back at the end to the lightest separator it saw.
/// That hill-climbing is the whole point: [`refine_separator`] accepts
/// only positive gains and therefore stops at the first local minimum.
///
/// Gain model, for a separator vertex `v` moved to side `to`:
///
/// ```text
///   gain(v) = vwgt[v] - sum of vwgt[u] over neighbours u with labels[u] == other
/// ```
///
/// since `v` leaves the separator and every far-side neighbour joins it.
///
/// Balance is a hard bound: a move that would push `pwgts[to]` past
/// `(1 + max_imbalance) * total / 2` ends the pass, as in METIS.
pub fn refine_separator_fm(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    max_passes: u32,
) -> i64 {
    let n = graph.nvtxs as usize;
    debug_assert_eq!(labels.len(), n);
    let total: i64 = graph.vwgt.iter().copied().sum();
    let max_side = ((1.0 + max_imbalance) * total as f64 / 2.0).ceil() as i64;

    let mut pwgt = [
        part_weight(graph, labels, PART_A),
        part_weight(graph, labels, PART_B),
    ];
    let mut sep = separator_weight(graph, labels);

    // `edeg[v][s]` is the weight of v's neighbours on side s; only
    // maintained while v is in the separator.
    let mut edeg: Vec<[i64; 2]> = vec![[0, 0]; n];
    // Undo log: `moves[m]` is the vertex pulled out of the separator by
    // move m, and `pulled[m]` the far-side neighbours it dragged in.
    let mut moves: Vec<usize> = Vec::new();
    let mut pulled: Vec<Vec<usize>> = Vec::new();

    for pass in 0..max_passes {
        let to = if pass % 2 == 0 { PART_A } else { PART_B };
        let other = if to == PART_A { PART_B } else { PART_A };
        let to_i = if to == PART_A { 0 } else { 1 };
        let other_i = 1 - to_i;

        for v in 0..n {
            if labels[v] == PART_SEP {
                edeg[v] = side_degrees(graph, labels, v);
            }
        }
        let mut heap: BinaryHeap<(i64, Reverse<usize>)> = BinaryHeap::new();
        let mut n_sep = 0usize;
        for v in 0..n {
            if labels[v] == PART_SEP {
                n_sep += 1;
                heap.push((graph.vwgt[v] - edeg[v][other_i], Reverse(v)));
            }
        }
        if n_sep == 0 {
            break;
        }
        // METIS's tolerance for how far past the best separator a pass
        // is allowed to wander before giving up.
        let limit = (3 * (n_sep + 1)).min(300);

        moves.clear();
        pulled.clear();
        let mut best_sep = sep;
        let mut best_at = 0usize; // number of moves kept at the best state
        let mut moved = vec![false; n];

        while let Some((key, Reverse(v))) = heap.pop() {
            if labels[v] != PART_SEP || moved[v] {
                continue;
            }
            let cur = graph.vwgt[v] - edeg[v][other_i];
            if key != cur {
                // Stale entry; a fresher one is in the heap.
                continue;
            }
            if pwgt[to_i] + graph.vwgt[v] > max_side {
                break;
            }
            // Apply the move.
            moved[v] = true;
            labels[v] = to;
            pwgt[to_i] += graph.vwgt[v];
            sep -= graph.vwgt[v];
            let mut dragged: Vec<usize> = Vec::new();
            for k in graph.xadj[v] as usize..graph.xadj[v + 1] as usize {
                let u = graph.adjncy[k] as usize;
                match labels[u] {
                    l if l == other => {
                        labels[u] = PART_SEP;
                        pwgt[other_i] -= graph.vwgt[u];
                        sep += graph.vwgt[u];
                        edeg[u] = side_degrees(graph, labels, u);
                        dragged.push(u);
                        if !moved[u] {
                            heap.push((graph.vwgt[u] - edeg[u][other_i], Reverse(u)));
                        }
                    }
                    PART_SEP => {
                        edeg[u][to_i] += graph.vwgt[v];
                        if !moved[u] {
                            heap.push((graph.vwgt[u] - edeg[u][other_i], Reverse(u)));
                        }
                    }
                    _ => {}
                }
            }
            // A vertex dragged in changes its neighbours' `edeg[other]`.
            for &u in &dragged {
                for k in graph.xadj[u] as usize..graph.xadj[u + 1] as usize {
                    let w = graph.adjncy[k] as usize;
                    if labels[w] == PART_SEP && w != u {
                        edeg[w][other_i] -= graph.vwgt[u];
                        if !moved[w] {
                            heap.push((graph.vwgt[w] - edeg[w][other_i], Reverse(w)));
                        }
                    }
                }
            }
            moves.push(v);
            pulled.push(dragged);

            if sep < best_sep {
                best_sep = sep;
                best_at = moves.len();
            } else if moves.len() - best_at > limit {
                break;
            }
        }

        // Roll back to the lightest separator this pass saw.
        while moves.len() > best_at {
            let v = moves.pop().unwrap_or(0);
            let dragged = pulled.pop().unwrap_or_default();
            labels[v] = PART_SEP;
            pwgt[to_i] -= graph.vwgt[v];
            sep += graph.vwgt[v];
            for u in dragged {
                labels[u] = other;
                pwgt[other_i] += graph.vwgt[u];
                sep -= graph.vwgt[u];
            }
        }
        debug_assert_eq!(sep, best_sep);
        if best_at == 0 && pass > 0 {
            // Neither side found anything; further passes would repeat.
            break;
        }
    }
    sep
}

/// `[weight of v's PART_A neighbours, weight of v's PART_B neighbours]`.
fn side_degrees(graph: &Graph, labels: &[u8], v: usize) -> [i64; 2] {
    let mut d = [0i64; 2];
    for k in graph.xadj[v] as usize..graph.xadj[v + 1] as usize {
        let u = graph.adjncy[k] as usize;
        if labels[u] == PART_A {
            d[0] += graph.vwgt[u];
        } else if labels[u] == PART_B {
            d[1] += graph.vwgt[u];
        }
    }
    d
}

/// Sum of `vwgt[u]` over neighbors u of v whose label is the given side.
fn separator_pull_costs(graph: &Graph, labels: &[u8], v: usize) -> (i64, i64) {
    let lo = graph.xadj[v] as usize;
    let hi = graph.xadj[v + 1] as usize;
    let mut cost_far_a: i64 = 0; // cost of pulling v to A = weight of neighbors currently on B
    let mut cost_far_b: i64 = 0;
    for k in lo..hi {
        let u = graph.adjncy[k] as usize;
        let wu = graph.vwgt[u];
        match labels[u] {
            PART_A => cost_far_b += wu,
            PART_B => cost_far_a += wu,
            _ => {}
        }
    }
    (cost_far_a, cost_far_b)
}

/// Total vertex weight of `PART_SEP` vertices.
pub fn separator_weight(graph: &Graph, labels: &[u8]) -> i64 {
    let mut s: i64 = 0;
    for (v, &l) in labels.iter().enumerate() {
        if l == PART_SEP {
            s += graph.vwgt[v];
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initial_partition::initial_bisect_ggp;
    use crate::rng::SplitMix;
    use crate::separator::construct_separator as construct;
    use feral_ordering_core::CscPattern;
    use std::collections::BTreeSet;

    fn csc_from_triples(n: usize, triples: &[(usize, usize)]) -> (Vec<Idx>, Vec<Idx>) {
        let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
        for &(i, j) in triples {
            set.insert((i, j));
            set.insert((j, i));
        }
        let mut cols: Vec<Vec<Idx>> = vec![Vec::new(); n];
        for &(r, c) in &set {
            cols[c].push(r as Idx);
        }
        for col in &mut cols {
            col.sort();
        }
        let mut col_ptr: Vec<Idx> = vec![0];
        let mut row_idx: Vec<Idx> = Vec::new();
        for col in &cols {
            for &r in col {
                row_idx.push(r);
            }
            col_ptr.push(row_idx.len() as Idx);
        }
        (col_ptr, row_idx)
    }

    fn grid(m: usize, n: usize) -> Graph {
        let idx = |r: usize, c: usize| r * n + c;
        let total = m * n;
        let mut t = Vec::new();
        for r in 0..m {
            for c in 0..n {
                let k = idx(r, c);
                t.push((k, k));
                if r + 1 < m {
                    t.push((k, idx(r + 1, c)));
                }
                if c + 1 < n {
                    t.push((k, idx(r, c + 1)));
                }
            }
        }
        let (cp, ri) = csc_from_triples(total, &t);
        let pat = CscPattern::new(total, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    /// Regression test for the FM neighbour-update sign bug fixed
    /// alongside this test (see `dev/research/metis-fm-sign-bug.md`).
    ///
    /// The bug flipped the signs at the `gain[u] ± 2w` neighbour
    /// update, so on a graph where FM actually had to move vertices,
    /// `cur_cut` drifted into negative impossible territory and FM
    /// rolled every move back. Existing tests missed it because they
    /// either started from already-optimal cuts (`initial_bisect_ggp`
    /// on grid is at the optimum), let the balance guard block every
    /// move, or only checked permutation validity.
    ///
    /// Two assertions matter here:
    ///
    /// 1. **I1 (bookkeeping consistency).** `returned_cut` must equal
    ///    `cut_size(graph, labels)` recomputed from scratch. This is
    ///    the assertion the bug *cannot* survive.
    /// 2. **Cut actually drops.** Path P_10 with alternating ABAB
    ///    labels has cut = 9 and balanced optimum cut = 1. FM with
    ///    correct bookkeeping must reduce the cut.
    #[test]
    fn fm_sign_invariant_on_alternating_path() {
        // Path 0-1-...-9.
        let mut t = Vec::new();
        for i in 0..10 {
            t.push((i, i));
        }
        for i in 0..9 {
            t.push((i, i + 1));
        }
        let (cp, ri) = csc_from_triples(10, &t);
        let pat = CscPattern::new(10, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();

        let mut labels: Vec<u8> = (0..10u8)
            .map(|k| if k % 2 == 0 { PART_A } else { PART_B })
            .collect();
        let before = cut_size(&g, &labels);
        assert_eq!(before, 9, "alternating path P_10 has cut 9");

        let after = refine_bisection(&g, &mut labels, 0.20, 32);

        // I1 — the assertion that catches the sign bug directly.
        assert_eq!(
            after,
            cut_size(&g, &labels),
            "returned cut must equal cut_size(labels) recomputed from scratch"
        );
        // Quality — FM must actually move at least one vertex on this
        // adversarial input.
        assert!(
            after < before,
            "FM must reduce cut from {} on alternating P_10, got {}",
            before,
            after
        );
        assert!(after >= 0, "cut size is non-negative, got {}", after);
    }

    #[test]
    fn refine_bisection_does_not_increase_cut() {
        let g = grid(8, 8);
        let total: i64 = g.vwgt.iter().copied().sum();
        let mut rng = SplitMix::new(17);
        let mut labels = initial_bisect_ggp(&g, &mut rng, total / 2);
        let initial = cut_size(&g, &labels);
        let final_cut = refine_bisection(&g, &mut labels, 0.20, 5);
        assert_eq!(
            final_cut,
            cut_size(&g, &labels),
            "reported cut matches labels"
        );
        assert!(
            final_cut <= initial,
            "cut must not increase (before={}, after={})",
            initial,
            final_cut
        );
    }

    #[test]
    fn refine_bisection_balance_respected() {
        let g = grid(6, 6);
        let total: i64 = g.vwgt.iter().copied().sum();
        let mut rng = SplitMix::new(9);
        let mut labels = initial_bisect_ggp(&g, &mut rng, total / 2);
        let returned = refine_bisection(&g, &mut labels, 0.20, 5);
        // I1 (bookkeeping consistency): returned cut equals cut
        // recomputed from labels.
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        let a = part_weight(&g, &labels, PART_A);
        let b = part_weight(&g, &labels, PART_B);
        let max_allowed = ((1.20_f64) * total as f64 / 2.0).ceil() as i64;
        assert!(a.max(b) <= max_allowed, "balance: a={} b={}", a, b);
        assert!(a > 0 && b > 0);
    }

    #[test]
    fn refine_bisection_bad_init_improves() {
        // Start from an adversarial labeling (all on one side, one
        // vertex on the other) — FM must rebalance. I2 (cut never
        // grows) does not apply across the imbalanced → balanced
        // transition: the starting cut of 2 is achievable only
        // because 15 vertices are on one side.
        let g = grid(4, 4);
        let mut labels = vec![PART_A; 16];
        labels[0] = PART_B;
        let returned = refine_bisection(&g, &mut labels, 0.20, 10);
        let after = cut_size(&g, &labels);
        assert_eq!(returned, after, "I1: bookkeeping");
        let total: i64 = g.vwgt.iter().copied().sum();
        let a = part_weight(&g, &labels, PART_A);
        let b = total - a;
        let max_side = ((1.20_f64) * total as f64 / 2.0).ceil() as i64;
        assert!(
            a.max(b) <= max_side,
            "I4: FM rebalanced from imbalanced start (a={}, b={}, max={})",
            a,
            b,
            max_side
        );
    }

    #[test]
    fn separator_weight_accounting() {
        let g = grid(3, 3);
        // Set up an explicit separator: middle row is SEP.
        let labels: Vec<u8> = (0..9u8)
            .map(|k| {
                let r = k / 3;
                match r {
                    0 => PART_A,
                    1 => PART_SEP,
                    _ => PART_B,
                }
            })
            .collect();
        assert_eq!(separator_weight(&g, &labels), 3);
    }

    // Adversarial set A1–A10 from
    // dev/research/metis-fm-sign-bug.md §5. Standing regression tests
    // enforcing I1 (bookkeeping consistency) and, where meaningful,
    // I2 (cut never grows), I4 (balance respected at exit),
    // I6 (determinism).
    //
    // A1 is covered by `fm_sign_invariant_on_alternating_path` above.
    // A2–A10 follow. Each constructs `initial cut` and (where
    // applicable) `optimum` by hand — never by running the solver
    // under test — per CLAUDE.md's oracle-independence rule.

    fn path(n: usize) -> Graph {
        let mut t: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
        for i in 0..n.saturating_sub(1) {
            t.push((i, i + 1));
        }
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    fn cycle(n: usize) -> Graph {
        let mut t: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
        for i in 0..n {
            t.push((i, (i + 1) % n));
        }
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    fn complete_bipartite(m: usize, k: usize) -> Graph {
        let n = m + k;
        let mut t: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
        for i in 0..m {
            for j in m..n {
                t.push((i, j));
            }
        }
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    #[test]
    fn a2_path_p20_alternating() {
        // P_20 with alternating ABABAB…: every edge crosses → cut 19.
        // Optimum balanced cut is 1 (single middle edge).
        let g = path(20);
        let mut labels: Vec<u8> = (0..20u8)
            .map(|k| if k % 2 == 0 { PART_A } else { PART_B })
            .collect();
        assert_eq!(cut_size(&g, &labels), 19, "construction: alternating P_20");
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(
            returned < 19,
            "FM must improve alternating P_20 cut, got {}",
            returned
        );
        assert!(returned >= 0);
    }

    #[test]
    fn a3_cycle_c12_alternating() {
        // Even cycle C_12 with alternating labels: every edge crosses
        // → cut 12. Optimum balanced cut is 2 (two "cut points" on
        // the cycle).
        let g = cycle(12);
        let mut labels: Vec<u8> = (0..12u8)
            .map(|k| if k % 2 == 0 { PART_A } else { PART_B })
            .collect();
        assert_eq!(cut_size(&g, &labels), 12, "construction: alternating C_12");
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(
            returned <= 12,
            "I2: cut never grows (before=12, after={})",
            returned
        );
        assert!(returned >= 2, "cycle C_12 balanced cut is ≥ 2");
    }

    #[test]
    fn a4_grid_4x4_checkerboard() {
        // 4×4 grid with checkerboard labels (r+c) mod 2: every edge
        // crosses → cut 24. Balanced optimum is 4 (one row/column).
        let g = grid(4, 4);
        let mut labels: Vec<u8> = (0..16u8)
            .map(|k| {
                let r = (k as usize) / 4;
                let c = (k as usize) % 4;
                if (r + c).is_multiple_of(2) {
                    PART_A
                } else {
                    PART_B
                }
            })
            .collect();
        assert_eq!(cut_size(&g, &labels), 24, "construction: 4x4 checkerboard");
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(
            returned < 24,
            "FM must improve 4x4 checkerboard cut, got {}",
            returned
        );
        assert!(returned >= 4, "4x4 grid balanced cut is ≥ 4");
    }

    #[test]
    fn a5_grid_6x6_mixed_boundary() {
        // 6×6 grid with a mixed but deterministic label assignment
        // (SplitMix-seeded). I1 and I2 are the load-bearing assertions;
        // the exact optimum depends on the initial labeling so we only
        // check cut never grows.
        let g = grid(6, 6);
        let mut rng = SplitMix::new(0xA5A5_A5A5);
        let mut labels: Vec<u8> = (0..36)
            .map(|_| {
                if rng.next_u64() & 1 == 0 {
                    PART_A
                } else {
                    PART_B
                }
            })
            .collect();
        let before = cut_size(&g, &labels);
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(
            returned <= before,
            "I2: cut never grows (before={}, after={})",
            before,
            returned
        );
    }

    #[test]
    fn a6_k44_unbalanced_init_rebalances() {
        // K_{4,4}, labels = [A, B, B, B, B, B, B, B]. Structure is
        // bipartite between {0..4} and {4..8}, so vertex 0 connects
        // to 4,5,6,7 → starting cut 4 with a=1, b=7.
        //
        // max_imbalance=0.50, total=8 → max_side=6. The 1-7 start
        // violates balance. Post-fix, refine_bisection treats an
        // imbalanced pass-start as "no valid rollback target" via a
        // None sentinel on best_prefix and unconditionally records
        // the first balanced state the FM trajectory reaches.
        //
        // The I2 (cut never grows) property does not apply across
        // the imbalanced→balanced transition: the starting cut of 4
        // is cheap only because the partition is violating.
        let g = complete_bipartite(4, 4);
        let mut labels: Vec<u8> = vec![PART_B; 8];
        labels[0] = PART_A;
        assert_eq!(cut_size(&g, &labels), 4, "construction: K_{{4,4}} 1-7");
        let returned = refine_bisection(&g, &mut labels, 0.50, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        let total: i64 = g.vwgt.iter().copied().sum();
        let a = part_weight(&g, &labels, PART_A);
        let b = total - a;
        let max_side = ((1.50_f64) * total as f64 / 2.0).ceil() as i64;
        assert!(
            a.max(b) <= max_side,
            "I4: FM rebalanced (a={}, b={}, max={})",
            a,
            b,
            max_side
        );
    }

    #[test]
    fn a7_two_k4_bridge_no_spurious_moves() {
        // Two K_4 blocks (0..4 and 4..8) joined by a single bridge
        // edge (3,4). Labels: all A. Cut = 0 (bridge is internal).
        // FM should not grow the cut; I2 is the gate.
        let n = 8;
        let mut t: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect();
        for i in 0..4 {
            for j in (i + 1)..4 {
                t.push((i, j));
            }
        }
        for i in 4..8 {
            for j in (i + 1)..8 {
                t.push((i, j));
            }
        }
        t.push((3, 4));
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let mut labels = vec![PART_A; n];
        assert_eq!(cut_size(&g, &labels), 0, "construction: all-A cut is 0");
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(returned >= 0);
    }

    #[test]
    fn a8_path_p10_all_a_empty_side() {
        // P_10 with all A labels. Cut = 0. Degenerate empty B side —
        // FM must not panic and must not grow the cut.
        let g = path(10);
        let mut labels = vec![PART_A; 10];
        assert_eq!(cut_size(&g, &labels), 0);
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(returned >= 0);
    }

    #[test]
    fn a9_single_vertex_returns_zero() {
        // n=1 — the n<2 short-circuit must fire.
        let cp: Vec<Idx> = vec![0, 0];
        let ri: Vec<Idx> = vec![];
        let pat = CscPattern::new(1, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let mut labels = vec![PART_A];
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, 0);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
    }

    #[test]
    fn a10_empty_edge_set_no_moves() {
        // n=8, no edges, half A / half B. No gains → no moves.
        let cp: Vec<Idx> = vec![0; 9];
        let ri: Vec<Idx> = vec![];
        let pat = CscPattern::new(8, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let mut labels: Vec<u8> = (0..8u8)
            .map(|k| if k < 4 { PART_A } else { PART_B })
            .collect();
        assert_eq!(cut_size(&g, &labels), 0);
        let returned = refine_bisection(&g, &mut labels, 0.20, 32);
        assert_eq!(returned, 0);
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        // I6 determinism: second run on fresh labels yields same result.
        let mut labels2: Vec<u8> = (0..8u8)
            .map(|k| if k < 4 { PART_A } else { PART_B })
            .collect();
        let returned2 = refine_bisection(&g, &mut labels2, 0.20, 32);
        assert_eq!(returned, returned2, "I6: determinism");
        assert_eq!(labels, labels2, "I6: determinism (labels)");
    }

    #[test]
    fn refine_separator_reduces_weight_on_padded_case() {
        // Construct a 3x3 grid with middle row SEP and the middle
        // vertex "padded" — add an extra SEP vertex adjacent only to
        // A-side. Refinement should pull it out to A.
        let g = grid(3, 3);
        let mut labels: Vec<u8> = (0..9u8)
            .map(|k| {
                let r = k / 3;
                match r {
                    0 => PART_A,
                    1 => PART_SEP,
                    _ => PART_B,
                }
            })
            .collect();
        // Make index 3 (row 1, col 0) isolated from B side by
        // relabelling its row-2 neighbor as SEP too.
        labels[6] = PART_SEP;
        let before = separator_weight(&g, &labels);
        let after = refine_separator(&g, &mut labels, 0.50, 10);
        // I1 (bookkeeping consistency): returned separator weight
        // matches separator_weight(labels) recomputed from scratch.
        assert_eq!(after, separator_weight(&g, &labels), "I1: bookkeeping");
        assert!(
            after <= before,
            "separator weight must not grow (before={}, after={})",
            before,
            after
        );
    }

    // ---- node-separator FM (`refine_separator_fm`) ------------------
    //
    // Oracles here are hand-computed minimum separators on graphs
    // whose answer is obvious by inspection, not outputs of this
    // crate: a path has a 1-vertex separator, a `k x k` grid has a
    // `k`-vertex straight cut, and two cliques joined by an edge are
    // separated by either endpoint of that edge.

    fn path_graph(n: usize) -> Graph {
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i));
            if i + 1 < n {
                t.push((i, i + 1));
            }
        }
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    /// Two `k`-cliques joined by the single edge `(k-1, k)`.
    fn barbell(k: usize) -> Graph {
        let n = 2 * k;
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i));
        }
        for a in 0..k {
            for b in (a + 1)..k {
                t.push((a, b));
                t.push((a + k, b + k));
            }
        }
        t.push((k - 1, k));
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    /// Every A-vertex and B-vertex pair must be non-adjacent.
    fn valid_sep(graph: &Graph, labels: &[u8]) -> bool {
        for v in 0..graph.nvtxs as usize {
            let lv = labels[v];
            if lv == PART_SEP {
                continue;
            }
            for k in graph.xadj[v] as usize..graph.xadj[v + 1] as usize {
                let lu = labels[graph.adjncy[k] as usize];
                if lu != PART_SEP && lu != lv {
                    return false;
                }
            }
        }
        true
    }

    /// P_21 cut at the middle with a deliberately fat 5-vertex
    /// separator. The minimum balanced separator of a path is one
    /// vertex, and FM must find it: pulling a separator endpoint into
    /// its own side costs nothing, because its only far-side neighbour
    /// is already in the separator.
    #[test]
    fn node_fm_shrinks_a_fat_path_separator() {
        let g = path_graph(21);
        let mut labels = vec![PART_A; 21];
        for (v, l) in labels.iter_mut().enumerate() {
            *l = match v {
                0..=7 => PART_A,
                8..=12 => PART_SEP,
                _ => PART_B,
            };
        }
        let before = separator_weight(&g, &labels);
        assert_eq!(before, 5);
        let after = refine_separator_fm(&g, &mut labels, 0.2, 10);
        assert!(valid_sep(&g, &labels), "not a separator: {labels:?}");
        assert_eq!(after, separator_weight(&g, &labels), "bookkeeping drift");
        assert_eq!(after, 1, "path separator should shrink to 1, got {after}");
    }

    /// A ragged separator on a 9x9 grid must come down to at most one
    /// full column (9 vertices), the obvious straight cut.
    #[test]
    fn node_fm_reaches_a_straight_grid_cut() {
        let k = 9;
        let g = grid(k, k);
        let mut labels = vec![PART_A; k * k];
        for r in 0..k {
            for c in 0..k {
                // Ragged three-column separator around the middle.
                let width = if r % 2 == 0 { 3 } else { 2 };
                let lo = 4 - width / 2;
                labels[r * k + c] = if c < lo {
                    PART_A
                } else if c < lo + width {
                    PART_SEP
                } else {
                    PART_B
                };
            }
        }
        let before = separator_weight(&g, &labels);
        let after = refine_separator_fm(&g, &mut labels, 0.2, 10);
        assert!(valid_sep(&g, &labels));
        assert_eq!(after, separator_weight(&g, &labels), "bookkeeping drift");
        assert!(
            after <= k as i64,
            "grid separator {after} should be <= {k} (was {before})"
        );
    }

    /// Two cliques joined by one edge: the minimum separator is one of
    /// that edge's endpoints. Starting from a whole clique in the
    /// separator, FM must get to 1.
    #[test]
    fn node_fm_finds_the_barbell_cut_vertex() {
        let k = 8;
        let g = barbell(k);
        let mut labels = vec![PART_B; 2 * k];
        for l in labels.iter_mut().take(k) {
            *l = PART_SEP;
        }
        let after = refine_separator_fm(&g, &mut labels, 0.4, 10);
        assert!(valid_sep(&g, &labels));
        assert_eq!(after, 1, "barbell separator should be 1, got {after}");
    }

    /// The rollback makes the pass monotone: whatever it is handed, it
    /// never returns something worse.
    #[test]
    fn node_fm_never_worsens_the_separator() {
        let g = grid(12, 12);
        let mut rng = SplitMix::new(7);
        for trial in 0..8u32 {
            let mut labels = initial_bisect_ggp(&g, &mut rng, 72);
            construct(&g, &mut labels);
            let before = separator_weight(&g, &labels);
            let after = refine_separator_fm(&g, &mut labels, 0.2, 10);
            assert!(valid_sep(&g, &labels), "trial {trial}");
            assert!(
                after <= before,
                "trial {trial}: {before} -> {after} got worse"
            );
        }
    }

    /// Balance is a hard constraint, at every tolerance.
    #[test]
    fn node_fm_respects_the_balance_bound() {
        let g = grid(10, 10);
        for &imb in &[0.0f64, 0.03, 0.2, 0.4] {
            let mut labels = vec![PART_A; 100];
            for (v, l) in labels.iter_mut().enumerate() {
                *l = match v % 10 {
                    0..=3 => PART_A,
                    4..=5 => PART_SEP,
                    _ => PART_B,
                };
            }
            refine_separator_fm(&g, &mut labels, imb, 10);
            let total: i64 = g.vwgt.iter().copied().sum();
            let max_side = ((1.0 + imb) * total as f64 / 2.0).ceil() as i64;
            let a = part_weight(&g, &labels, PART_A);
            let b = part_weight(&g, &labels, PART_B);
            assert!(
                a <= max_side && b <= max_side,
                "imb {imb}: a={a} b={b} max={max_side}"
            );
        }
    }

    /// Same input, same output — the crate contract requires it.
    #[test]
    fn node_fm_is_deterministic() {
        let g = grid(11, 11);
        let mut rng = SplitMix::new(3);
        let mut labels = initial_bisect_ggp(&g, &mut rng, 60);
        construct(&g, &mut labels);
        let mut a = labels.clone();
        let mut b = labels.clone();
        let wa = refine_separator_fm(&g, &mut a, 0.2, 10);
        let wb = refine_separator_fm(&g, &mut b, 0.2, 10);
        assert_eq!(wa, wb);
        assert_eq!(a, b);
    }
}
