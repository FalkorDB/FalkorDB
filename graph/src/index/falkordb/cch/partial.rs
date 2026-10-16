//! Scoped recustomization (#3217): rewrite only the arcs a weight change can reach.
//!
//! Ported from `CCH_RecustomizeScoped` / `CCHIndex_Recustomize` in the C implementation
//! (`src/algorithms/cch.c`, `src/index/cch_index.c`). The layout it works on is the one
//! [`customize`](super::customize) documents: arcs are slots of the upward rows, and an arc
//! `(lo, hi)` holds the cheapest path between its endpoints through strictly lower ranks.
//!
//! # Why a queue ordered by lower rank is enough
//!
//! Recomputing arc `(lo, hi)` reads only arcs `(x, lo)` and `(x, hi)` with `x < lo`: the
//! lower triangles it closes. When it changes, the arcs that read *it* are the top arcs
//! `(min(hi, w), max(hi, w))` of the triangles with apex `lo`, for `w` in `up(lo)` — and
//! every one of those has a lower endpoint above `lo`. So popping arcs in increasing
//! lower-endpoint rank finalizes each arc after everything it reads, which is the same
//! argument that makes the full sweep a single pass, restricted to the affected cone.
//!
//! The recomputation of one arc is the full sweep's, term for term: the same seed, the
//! same candidates `dn(x, lo) + up(x, hi)` and `dn(x, hi) + up(x, lo)`, visited in the
//! same increasing `x`, kept under the same strict `<`. So the result is bit-identical to
//! a full [`customize`](super::customize::customize), mids included — which is what the
//! tests check.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use rustc_hash::FxHashSet;

use super::customize::customize;
use super::order::Rank;
use super::{Cch, CchError, Recustomized, WeightCsr, index_id};

/// The stored weight of the directed edge `u -> v`, or `None` when `weights` has no such
/// entry. Rows are sorted (the [`WeightCsr`] contract), so this is a binary search.
fn edge_entry(
    weights: &WeightCsr,
    u: u64,
    v: u64,
) -> Option<f64> {
    let csr = &weights.csr;
    let start = csr.row_offsets[u as usize] as usize;
    csr.row(u as usize)
        .binary_search(&v)
        .ok()
        .map(|k| weights.values[start + k])
}

/// The seed an arc direction gets: its edge's weight, or infinity for "no edge" — exactly
/// as a full customization seeds it.
fn seed_of(entry: Option<f64>) -> f64 {
    entry.unwrap_or(f64::INFINITY)
}

/// A stored weight a customization would refuse: negative (`-inf` included) or NaN, as
/// `customize` refuses. An explicit `+inf` entry is legal, a closed edge, and seeds its arc
/// with infinity exactly as absence ("no edge") does.
fn bad_entry(entry: Option<f64>) -> bool {
    // A negative weight, `-inf` and NaN are refused; `+inf` and `-0.0` are not.
    entry.is_some_and(|w| w.is_nan() || w < 0.0)
}

impl Cch {
    /// Recompute the metric after the edges in `changed` changed weight, touching only the
    /// arcs those changes can reach.
    ///
    /// `changed` lists node-space `(u, v)` pairs whose cheapest weight may differ from the
    /// last customization — raised, lowered, removed, or added where the hierarchy already
    /// has an arc. Direction does not matter (an arc carries both), duplicates are fine,
    /// and self-loops are ignored. `weights` is the **complete** new metric, the same
    /// [`WeightCsr`] a full [`Cch::customized`] would take:
    /// - a removed edge is simply absent from it, and seeds its arc with infinity, as an
    ///   explicit `+inf` (a closed edge) does;
    /// - every pair whose weight differs from the previous metric must be in `changed`.
    ///   A pair left out keeps its old arc weights, so the result is wrong, not refused.
    ///
    /// # Preconditions
    /// `weights` came from `CchInput::build`, which has already
    /// refused negative and NaN weights. This path re-checks only the seeds of
    /// the arcs in `changed` (before writing anything), not the whole metric: doing the
    /// latter would cost O(edges) and make the scoped path no cheaper than the full one.
    ///
    /// # Errors
    /// [`CchError::InvalidWeight`] if a changed pair's new weight is negative or NaN, and
    /// [`CchError::Internal`] if `weights` is malformed (a bug). Nothing is written in
    /// either case, nor on [`Recustomized::NeedsRebuild`].
    pub fn recustomize_scoped(
        &mut self,
        changed: &[(u64, u64)],
        weights: &WeightCsr,
    ) -> Result<Recustomized, CchError> {
        let topology = Arc::clone(&self.topology);
        let n = topology.node_count();

        // Weights for a different node count mean the graph gained or lost nodes, which
        // only a rebuild handles. Arrays of inconsistent lengths are a malformed CSR, a bug.
        // The same cheap tests `customize` starts with, minus its O(edges) scans: this path
        // only ever reads the rows of changed nodes, each bound-checked below.
        let csr = &weights.csr;
        if csr.node_count != n {
            return Ok(Recustomized::NeedsRebuild);
        }
        if csr.row_offsets.len() != n + 1
            || csr.row_offsets.last().map(|&offset| offset as usize) != Some(csr.neighbors.len())
            || weights.values.len() != csr.neighbors.len()
        {
            return Err(CchError::Internal("the weight CSR is malformed"));
        }

        // Resolve every changed pair to its arc before writing anything, so both
        // `NeedsRebuild` and `InvalidWeight` leave the metric untouched.
        let mut front: Vec<(Rank, usize)> = Vec::with_capacity(changed.len());
        for &(u, v) in changed {
            if u == v {
                continue;
            }
            if u >= index_id(n) || v >= index_id(n) {
                return Ok(Recustomized::NeedsRebuild);
            }
            let (ru, rv) = (topology.rank_of(u.into()), topology.rank_of(v.into()));
            let (uv, vu) = (edge_entry(weights, u, v), edge_entry(weights, v, u));
            if bad_entry(uv) || bad_entry(vu) {
                return Err(CchError::InvalidWeight);
            }
            match topology.arc_slot(ru, rv) {
                Some(slot) => front.push((ru.min(rv), slot)),
                // No arc and no edge either way: an edge that was never there was
                // "removed". Nothing in the hierarchy reads it.
                None if uv.is_none() && vu.is_none() => {}
                None => return Ok(Recustomized::NeedsRebuild),
            }
        }
        front.sort_unstable();
        front.dedup();

        // C's bulk-update line: past n/8 changed arcs the cones overlap so much that one
        // full sweep is simpler and no slower.
        if front.len() > n / 8 {
            // `customize` refuses with `NeedsRebuild` when `weights` holds an edge the
            // hierarchy has no arc for (a new adjacency nobody listed in `changed`), and,
            // like this API's `NeedsRebuild`, nothing has been written.
            return match customize(&topology, weights) {
                Ok(metric) => {
                    self.metric = metric;
                    Ok(Recustomized::Full)
                }
                Err(CchError::NeedsRebuild) => Ok(Recustomized::NeedsRebuild),
                Err(e) => Err(e),
            };
        }

        let mut queued: FxHashSet<usize> = front.iter().map(|&(_, slot)| slot).collect();
        let mut heap: BinaryHeap<Reverse<(Rank, usize)>> = front.into_iter().map(Reverse).collect();
        let metric = &mut self.metric;
        let mut arcs = 0;

        while let Some(Reverse((lo, slot))) = heap.pop() {
            arcs += 1;
            let hi = topology.up(lo)[slot - topology.up_range(lo).start];
            let (u, v) = (
                u64::from(topology.node_of(lo)),
                u64::from(topology.node_of(hi)),
            );

            // Reseed from the original edges, then relax over every lower common neighbour
            // x of lo and hi: x in down(lo) and in down(hi). Both rows ascend, so a merge.
            let (uv, vu) = (edge_entry(weights, u, v), edge_entry(weights, v, u));
            let mut best_up = seed_of(uv);
            let mut best_dn = seed_of(vu);
            debug_assert!(!bad_entry(uv) && !bad_entry(vu), "see # Preconditions");
            let mut mid_up = None;
            let mut mid_dn = None;

            let (dlo, dhi) = (topology.down(lo), topology.down(hi));
            let (mut a, mut b) = (0, 0);
            while a < dlo.len() && b < dhi.len() {
                let (xa, xb) = (dlo[a], dhi[b]);
                if xa < xb {
                    a += 1;
                    continue;
                }
                if xa > xb {
                    b += 1;
                    continue;
                }
                let x = xa;
                // x in down(lo) <=> lo in up(x), likewise for hi: both arcs exist.
                let x_lo = topology.arc_slot(x, lo).expect("down/up are inverses");
                let x_hi = topology.arc_slot(x, hi).expect("down/up are inverses");

                // lo -> x -> hi
                let cand_up = metric.downward_weight[x_lo] + metric.upward_weight[x_hi];
                if cand_up < best_up {
                    best_up = cand_up;
                    mid_up = Some(x);
                }
                // hi -> x -> lo
                let cand_dn = metric.downward_weight[x_hi] + metric.upward_weight[x_lo];
                if cand_dn < best_dn {
                    best_dn = cand_dn;
                    mid_dn = Some(x);
                }
                a += 1;
                b += 1;
            }

            let changed_weight =
                best_up != metric.upward_weight[slot] || best_dn != metric.downward_weight[slot];
            metric.upward_weight[slot] = best_up;
            metric.downward_weight[slot] = best_dn;
            metric.upward_middle[slot] = mid_up;
            metric.downward_middle[slot] = mid_dn;

            // A changed arc invalidates the top arc of every triangle it is the bottom of:
            // apex lo, other bottom arc (lo, w). All of those sit above lo, so they pop
            // after everything they read is final.
            if changed_weight {
                for &w in topology.up(lo) {
                    if w == hi {
                        continue;
                    }
                    let d_slot = topology
                        .arc_slot(hi, w)
                        .expect("chordal closure: upper neighbours of lo are adjacent");
                    if queued.insert(d_slot) {
                        heap.push(Reverse((hi.min(w), d_slot)));
                    }
                }
            }
        }

        Ok(Recustomized::Scoped { arcs })
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::graph::graphblas::matrix::{Dup, Matrix};
    use crate::index::falkordb::cch::Recustomized;
    use crate::index::falkordb::cch::customize::{Metric, customize};
    use crate::index::falkordb::cch::order::{Rank, Topology};
    use crate::index::falkordb::cch::testkit::{
        entries, grid, lower_rank_distances, node_count, random_sparse, to_input,
    };
    use crate::index::falkordb::cch::{Cch, CchError, WeightCsr};

    /// The metric as plain vectors, for byte-for-byte comparison.
    type Snapshot = (Vec<f64>, Vec<f64>, Vec<Option<Rank>>, Vec<Option<Rank>>);

    fn snapshot(m: &Metric) -> Snapshot {
        (
            m.upward_weight.to_vec(),
            m.downward_weight.to_vec(),
            m.upward_middle.to_vec(),
            m.downward_middle.to_vec(),
        )
    }

    /// What a full customization of the same topology gives for `weights`: the oracle
    /// every scoped result must match exactly, mids included.
    fn full(
        cch: &Cch,
        weights: &WeightCsr,
    ) -> Snapshot {
        snapshot(&customize(&cch.topology, weights).expect("full customize"))
    }

    /// Also check the weights against the kit's `lower_rank_distances`, which derives
    /// the lower-restricted distance from the weight matrix and the order alone.
    fn check_against_oracle(
        cch: &Cch,
        g: &Matrix<f64>,
    ) {
        let topology = &cch.topology;
        let expected = lower_rank_distances(g, &topology.rank_to_node_ids());
        for lo in topology.ranks() {
            let start = topology.up_range(lo).start;
            for (i, &hi) in topology.up(lo).iter().enumerate() {
                let (u, v) = (
                    u64::from(topology.node_of(lo)),
                    u64::from(topology.node_of(hi)),
                );
                assert_eq!(
                    cch.metric.upward_weight[start + i],
                    expected[u as usize][v as usize]
                );
                assert_eq!(
                    cch.metric.downward_weight[start + i],
                    expected[v as usize][u as usize]
                );
            }
        }
    }

    /// Some pair of distinct nodes with no chordal arc between them.
    fn non_adjacent_pair(topology: &Topology) -> (u64, u64) {
        let n = topology.node_count() as u64;
        (0..n)
            .flat_map(|a| (0..n).map(move |b| (a, b)))
            .find(|&(a, b)| {
                let (ra, rb) = (topology.rank_of(a.into()), topology.rank_of(b.into()));
                a != b && topology.arc_slot(ra, rb).is_none()
            })
            .expect("a sparse graph has a non-adjacent pair")
    }

    /// One random edit to `g`: raise or lower a weight, remove an edge, or set an edge
    /// between nodes that already share a chordal arc. Returns the pair touched.
    fn random_edit(
        rng: &mut SmallRng,
        cch: &Cch,
        g: &mut Matrix<f64>,
    ) -> (u64, u64) {
        let topology = &cch.topology;
        let edges = entries(g);
        match rng.random_range(0..4) {
            0 | 1 if !edges.is_empty() => {
                let (u, v, w) = edges[rng.random_range(0..edges.len())];
                // Integer weights so ties are common: ties are where a scoped mid
                // could most easily drift from the full sweep's.
                let w = if rng.random_range(0..2) == 0 {
                    w + 1.0 + rng.random_range(0..5) as f64
                } else {
                    (w - 1.0 - rng.random_range(0..5) as f64).max(0.0)
                };
                g.set(u, v, w);
                (u, v)
            }
            2 if !edges.is_empty() => {
                let (u, v, _) = edges[rng.random_range(0..edges.len())];
                g.remove(u, v);
                (u, v)
            }
            _ => loop {
                let lo = Rank::new(rng.random_range(0..node_count(g)));
                let up = topology.up(lo);
                if up.is_empty() {
                    continue;
                }
                let hi = up[rng.random_range(0..up.len())];
                let (mut u, mut v) = (
                    u64::from(topology.node_of(lo)),
                    u64::from(topology.node_of(hi)),
                );
                if rng.random_range(0..2) == 0 {
                    std::mem::swap(&mut u, &mut v);
                }
                g.set(u, v, 1.0 + rng.random_range(0..20) as f64);
                break (u, v);
            },
        }
    }

    /// Close an open edge of `pairs` (the graph's edges as first built), or reopen a closed
    /// one at its first weight. Returns the pair touched.
    fn close_or_reopen(
        rng: &mut SmallRng,
        pairs: &[(u64, u64, f64)],
        g: &mut Matrix<f64>,
    ) -> (u64, u64) {
        let (u, v, first) = pairs[rng.random_range(0..pairs.len())];
        let closed = g.get(u, v).is_some_and(f64::is_infinite);
        g.set(u, v, if closed { first } else { f64::INFINITY });
        (u, v)
    }

    /// The main oracle: after every batch of random edits the scoped result is
    /// bit-identical to a full customization of the same topology — weights *and* mids
    /// (exact mid equality, not "the detour reproduces the weight": the scoped
    /// recomputation visits candidates in the same order under the same strict `<`) —
    /// and matches the kit's `lower_rank_distances`. Every other batch closes open edges
    /// (`+inf`, a legal weight) and reopens closed ones instead.
    #[test]
    fn scoped_equals_full_on_random_edit_batches() {
        let mut scoped_runs = 0;
        for seed in 1..=20u64 {
            let mut g = random_sparse(240, 700, 20, seed);
            let pairs = entries(&g);
            let mut cch = Cch::build(&to_input(&g)).expect("build");
            let mut rng = SmallRng::seed_from_u64(seed);

            for batch in 0..8 {
                let size = 1 + rng.random_range(0..8);
                let changed: Vec<(u64, u64)> = (0..size)
                    .map(|_| {
                        if batch % 2 == 0 {
                            random_edit(&mut rng, &cch, &mut g)
                        } else {
                            close_or_reopen(&mut rng, &pairs, &mut g)
                        }
                    })
                    .collect();
                let input = to_input(&g);

                let outcome = cch
                    .recustomize_scoped(&changed, &input.weights)
                    .expect("recustomize");
                assert!(
                    matches!(outcome, Recustomized::Scoped { .. }),
                    "seed {seed} batch {batch}: expected the scoped path, got {outcome:?}"
                );
                scoped_runs += 1;
                assert_eq!(
                    snapshot(&cch.metric),
                    full(&cch, &input.weights),
                    "seed {seed} batch {batch}: scoped != full after {changed:?}"
                );
                check_against_oracle(&cch, &g);
            }
        }
        assert_eq!(scoped_runs, 160);
    }

    /// A local change stays local: one weight on a 30x30 grid touches a small fraction of
    /// the arcs. This is what proves the scoped path is scoped and not a full sweep in
    /// disguise.
    #[test]
    fn a_local_edit_recomputes_a_small_cone() {
        let mut g = grid(30, 30);
        let mut cch = Cch::build(&to_input(&g)).expect("build");
        let total = cch.topology.arc_count();

        assert!(g.get(465, 466).is_some());
        g.set(465, 466, 5.0);
        let input = to_input(&g);
        let outcome = cch
            .recustomize_scoped(&[(465, 466)], &input.weights)
            .unwrap();

        let Recustomized::Scoped { arcs } = outcome else {
            panic!("expected scoped, got {outcome:?}");
        };
        assert!(arcs >= 1);
        assert!(
            arcs * 10 < total,
            "one edit recomputed {arcs} of {total} arcs: not scoped"
        );
        assert_eq!(snapshot(&cch.metric), full(&cch, &input.weights));
        eprintln!("grid 30x30: one edit recomputed {arcs} of {total} arcs");
    }

    /// A pair with no chordal arc cannot be recustomized into existence. The call must say
    /// so and leave the metric exactly as it was — checked before any write, so a valid
    /// change listed *first* is not applied either.
    #[test]
    fn a_new_adjacency_needs_a_rebuild_and_writes_nothing() {
        let g = random_sparse(240, 500, 20, 7);
        let mut cch = Cch::build(&to_input(&g)).unwrap();
        let topology = &cch.topology;
        let (u, v) = non_adjacent_pair(topology);
        let (eu, ev, ew) = entries(&g)[0];

        let mut edited = g.dup();
        edited.set(eu, ev, ew + 7.0); // a valid change, listed first
        edited.set(u, v, 3.0); // the impossible one
        let input = to_input(&edited);

        let before = snapshot(&cch.metric);
        let outcome = cch
            .recustomize_scoped(&[(eu, ev), (u, v)], &input.weights)
            .unwrap();
        assert_eq!(outcome, Recustomized::NeedsRebuild);
        assert_eq!(
            snapshot(&cch.metric),
            before,
            "NeedsRebuild must write nothing"
        );

        // A node the hierarchy does not have is the same case.
        let outcome = cch.recustomize_scoped(&[(0, 240)], &input.weights).unwrap();
        assert_eq!(outcome, Recustomized::NeedsRebuild);
        assert_eq!(snapshot(&cch.metric), before);
    }

    /// A malformed weight CSR is a bug, refused as [`CchError::Internal`] before anything is
    /// written, not reported as a rebuild.
    #[test]
    fn a_malformed_weight_csr_is_an_internal_error_and_writes_nothing() {
        let g = random_sparse(240, 500, 20, 7);
        let mut cch = Cch::build(&to_input(&g)).unwrap();
        let before = snapshot(&cch.metric);
        let (u, v, _) = entries(&g)[0];
        let mut short = to_input(&g);
        short.weights.values = short.weights.values[1..].into();
        assert!(matches!(
            cch.recustomize_scoped(&[(u, v)], &short.weights),
            Err(CchError::Internal(_))
        ));
        assert_eq!(snapshot(&cch.metric), before);
    }

    /// A negative new weight is refused before anything is written, whether or not its pair
    /// has an arc.
    #[test]
    fn a_bad_new_weight_is_refused_and_writes_nothing() {
        let g = random_sparse(240, 500, 20, 11);
        let mut cch = Cch::build(&to_input(&g)).unwrap();
        let before = snapshot(&cch.metric);
        let (u, v, _) = entries(&g)[3];
        let mut input = to_input(&g);
        // Corrupt the one entry, bypassing to_input's own resolution.
        let row = input.weights.csr.row_offsets[u as usize] as usize;
        let k = row + input.weights.csr.row(u as usize).binary_search(&v).unwrap();
        // Refused as `customize` refuses them; `+inf` is legal (a closed edge).
        for bad in [-1.0, f64::NAN, f64::NEG_INFINITY] {
            input.weights.values[k] = bad;
            assert_eq!(
                cch.recustomize_scoped(&[(u, v)], &input.weights).err(),
                Some(CchError::InvalidWeight),
                "{bad}"
            );
            assert_eq!(
                snapshot(&cch.metric),
                before,
                "{bad}: nothing may be written"
            );
        }

        // On a pair with no arc it is still a bad weight, not a missing arc.
        let (u, v) = non_adjacent_pair(&cch.topology);
        let mut edited = g.dup();
        edited.set(u, v, 3.0);
        let mut input = to_input(&edited);
        let row = input.weights.csr.row_offsets[u as usize] as usize;
        let k = row + input.weights.csr.row(u as usize).binary_search(&v).unwrap();
        input.weights.values[k] = -1.0;
        assert_eq!(
            cch.recustomize_scoped(&[(u, v)], &input.weights).err(),
            Some(CchError::InvalidWeight)
        );
        assert_eq!(snapshot(&cch.metric), before);
    }

    /// Past `n / 8` distinct changed arcs the full customization runs instead — and the
    /// result is the same either way.
    #[test]
    fn the_n_over_8_valve_switches_to_a_full_customize() {
        let n = 240;
        let mut g = random_sparse(n, 900, 20, 5);
        let mut cch = Cch::build(&to_input(&g)).unwrap();
        let edges = entries(&g);

        // Exactly n/8 distinct arcs stays scoped; one more goes full. Distinct arcs, not
        // pairs: (u, v) and (v, u) share one arc, so collect by arc.
        let topology = cch.topology.clone();
        let mut seen = std::collections::HashSet::new();
        let mut picks = Vec::new();
        for (k, &(u, v, _)) in edges.iter().enumerate() {
            let (ru, rv) = (topology.rank_of(u.into()), topology.rank_of(v.into()));
            let slot = topology.arc_slot(ru, rv).unwrap();
            if seen.insert(slot) {
                picks.push(k);
            }
            if picks.len() == n / 8 + 1 {
                break;
            }
        }
        assert_eq!(picks.len(), n / 8 + 1);

        for (count, want_full) in [(n / 8, false), (n / 8 + 1, true)] {
            let mut changed = Vec::new();
            for &k in &picks[..count] {
                let (u, v, _) = edges[k];
                g.set(u, v, g.get(u, v).unwrap() + 1.0);
                changed.push((u, v));
            }
            let input = to_input(&g);
            let outcome = cch.recustomize_scoped(&changed, &input.weights).unwrap();
            assert_eq!(
                outcome == Recustomized::Full,
                want_full,
                "{count} distinct arcs gave {outcome:?}"
            );
            assert_eq!(snapshot(&cch.metric), full(&cch, &input.weights));
        }
    }

    /// Duplicates, both directions of one arc, and self-loops in `changed` are harmless.
    #[test]
    fn changed_may_repeat_and_include_self_loops() {
        let mut g = random_sparse(240, 600, 20, 3);
        let mut cch = Cch::build(&to_input(&g)).unwrap();
        let (u, v, w) = entries(&g)[0];
        g.set(u, v, w + 3.0);
        let input = to_input(&g);
        let outcome = cch
            .recustomize_scoped(&[(u, v), (v, u), (u, v), (u, u)], &input.weights)
            .unwrap();
        assert!(matches!(outcome, Recustomized::Scoped { .. }));
        assert_eq!(snapshot(&cch.metric), full(&cch, &input.weights));
    }
}
