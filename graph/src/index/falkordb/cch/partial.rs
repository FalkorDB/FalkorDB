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
//!
//! # Where the seeds come from
//!
//! An arc's seed is the cheapest current edge of its pair. [`Cch::recustomize_seeded`] asks
//! a [`SeedFn`] for it, once per direction of each arc it recomputes, as C's `_scoped_seed`
//! callback does, so a write that changed a handful of pairs never builds the O(edges)
//! weight matrix a full customization takes. Past the `n / 8` valve the full sweep needs
//! every seed, so it answers [`Recustomized::NeedsFull`] and leaves building the matrix to
//! the caller, as C's `CCHIndex_Recustomize` does.
//!
//! Only arcs whose value changes are written, so on a copy-on-write metric
//! ([`WeightArr`](super::customize::WeightArr)) a cone arc that keeps its value keeps its
//! chunk shared with the previous version.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use super::order::Rank;
use super::{Cch, CchError, Recustomized, index_id};

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

/// The current weight of the directed node pair `(u, v)` as a full customization would seed
/// it: `Some(w)` for the cheapest edge the index covers (`+inf` for a closed one), `None`
/// when the pair has no edge. Only ever asked with `u != v`, both below the hierarchy's node
/// count.
///
/// [`Cch::recustomize_seeded`] takes it from the caller, so a write never has to build the
/// O(edges) weight matrix (in production it reads the graph pair by pair:
/// [`PairWeights`](super::input::PairWeights)).
pub(super) type SeedFn<'a> = dyn Fn(u64, u64) -> Option<f64> + 'a;

impl Cch {
    /// Recompute the metric after the edges in `changed` changed weight, touching only the
    /// arcs those changes can reach, each recomputed arc seeded by `seed` (the current
    /// cheapest weight of a directed pair, `None` for no edge): what a write calls, so it
    /// reads only the pairs the cone reaches, never the O(edges) weight matrix.
    ///
    /// `changed` lists node-space `(u, v)` pairs whose cheapest weight may differ from the
    /// last customization — raised, lowered, removed, or added where the hierarchy already
    /// has an arc. Direction does not matter (an arc carries both), duplicates are fine,
    /// and self-loops are ignored. A removed edge seeds its arc with infinity, as an
    /// explicit `+inf` (a closed edge) does. Every pair whose weight differs from the
    /// previous metric must be in `changed`: a pair left out keeps its old arc weights, so
    /// the result is wrong, not refused.
    ///
    /// The result is bit-identical to a full customization. Past `n / 8` distinct changed
    /// arcs it answers [`Recustomized::NeedsFull`] rather than running the full sweep, which
    /// needs every seed: the caller builds the [`WeightCsr`] and calls [`Self::customized`].
    ///
    /// # Preconditions
    /// `seed` gives what `CchInput::build` would, which has already refused negative and NaN
    /// weights. This path re-checks only the seeds of the arcs in `changed` (before writing
    /// anything), not the whole metric: doing the latter would cost O(edges) and make the
    /// scoped path no cheaper than the full one.
    ///
    /// # Errors
    /// [`CchError::InvalidWeight`] if a changed pair's seed is negative or NaN, below the
    /// `n / 8` valve. Past it no seed is read: the answer is [`Recustomized::NeedsFull`], and
    /// the caller's full customization refuses such a weight instead. Nothing is written on
    /// an error, nor on [`Recustomized::NeedsRebuild`] or [`Recustomized::NeedsFull`].
    pub(crate) fn recustomize_seeded(
        &mut self,
        changed: &[(u64, u64)],
        seed: &SeedFn,
    ) -> Result<Recustomized, CchError> {
        let topology = Arc::clone(&self.topology);
        let n = topology.node_count();

        // Resolve every changed pair to its arc before writing anything, so both
        // `NeedsRebuild` and `InvalidWeight` leave the metric untouched. Slots first, which
        // reads no seed, so a set past the valve costs no seed at all.
        let mut front: Vec<(Rank, usize)> = Vec::with_capacity(changed.len());
        let mut no_arc: Vec<(u64, u64)> = Vec::new();
        for &(u, v) in changed {
            if u == v {
                continue;
            }
            if u >= index_id(n) || v >= index_id(n) {
                return Ok(Recustomized::NeedsRebuild);
            }
            let (ru, rv) = (topology.rank_of(u.into()), topology.rank_of(v.into()));
            let (lo, hi) = if ru < rv { (ru, rv) } else { (rv, ru) };
            match topology.arc_slot(lo, hi) {
                Some(slot) => front.push((lo, slot)),
                None => no_arc.push((u, v)),
            }
        }
        front.sort_unstable();
        front.dedup();

        // C's bulk-update line (`CCHIndex_Recustomize`): past n/8 changed arcs the cones
        // overlap so much that one full sweep is simpler and no slower. (A pair with no arc
        // and an edge is then caught by the full sweep's shape check.)
        if front.len() > n / 8 {
            return Ok(Recustomized::NeedsFull);
        }
        // No arc and no edge either way: an edge that was never there was "removed".
        // Nothing in the hierarchy reads it. No arc but an edge: a new adjacency.
        // A bad weight is refused as one either way.
        for &(u, v) in &no_arc {
            let (uv, vu) = (seed(u, v), seed(v, u));
            if bad_entry(uv) || bad_entry(vu) {
                return Err(CchError::InvalidWeight);
            }
            if uv.is_some() || vu.is_some() {
                return Ok(Recustomized::NeedsRebuild);
            }
        }
        // The seeds of the changed arcs, checked, and kept for when each arc pops, so a
        // changed pair is looked up once. In arc orientation: `up` is node_of(lo) ->
        // node_of(hi).
        let mut known: FxHashMap<usize, (Option<f64>, Option<f64>)> = FxHashMap::default();
        for &(lo, slot) in &front {
            let hi = topology.up(lo)[slot - topology.up_range(lo).start];
            let (a, b) = (
                u64::from(topology.node_of(lo)),
                u64::from(topology.node_of(hi)),
            );
            let (ab, ba) = (seed(a, b), seed(b, a));
            if bad_entry(ab) || bad_entry(ba) {
                return Err(CchError::InvalidWeight);
            }
            known.insert(slot, (ab, ba));
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
            let (uv, vu) = known
                .get(&slot)
                .copied()
                .unwrap_or_else(|| (seed(u, v), seed(v, u)));
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
            // Written only where they differ: an arc the cone reaches but does not change
            // leaves its chunk shared with the previous version.
            if best_up != metric.upward_weight[slot] {
                metric.upward_weight.set(slot, best_up);
            }
            if best_dn != metric.downward_weight[slot] {
                metric.downward_weight.set(slot, best_dn);
            }
            if mid_up != metric.upward_middle[slot] {
                metric.upward_middle.set(slot, mid_up);
            }
            if mid_dn != metric.downward_middle[slot] {
                metric.downward_middle.set(slot, mid_dn);
            }

            // A changed arc invalidates the top arc of every triangle it is the bottom of:
            // apex lo, other bottom arc (lo, w). All of those sit above lo, so they pop
            // after everything they read is final.
            if changed_weight {
                for &w in topology.up(lo) {
                    if w == hi {
                        continue;
                    }
                    let (d_lo, d_hi) = if hi < w { (hi, w) } else { (w, hi) };
                    let d_slot = topology
                        .arc_slot(d_lo, d_hi)
                        .expect("chordal closure: upper neighbours of lo are adjacent");
                    if queued.insert(d_slot) {
                        heap.push(Reverse((d_lo, d_slot)));
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
        entries, grid, lower_rank_distances, node_count, random_sparse, reweight, to_input,
    };
    use crate::index::falkordb::cch::{Cch, CchError, WeightCsr};

    /// The stored weight of the directed edge `u -> v`, or `None` when `weights` has no such
    /// entry: the seed a full customization of `weights` gives the pair. Rows are sorted
    /// (the [`WeightCsr`] contract), so this is a binary search.
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

    /// [`Cch::recustomize_seeded`], every pair seeded from the complete `weights`.
    fn seeded(
        cch: &mut Cch,
        changed: &[(u64, u64)],
        weights: &WeightCsr,
    ) -> Result<Recustomized, CchError> {
        cch.recustomize_seeded(changed, &|u, v| edge_entry(weights, u, v))
    }

    /// The metric as plain vectors, for byte-for-byte comparison.
    type Snapshot = (Vec<f64>, Vec<f64>, Vec<Option<Rank>>, Vec<Option<Rank>>);

    fn snapshot(m: &Metric) -> Snapshot {
        (
            m.upward_weight.iter().collect(),
            m.downward_weight.iter().collect(),
            m.upward_middle.iter().collect(),
            m.downward_middle.iter().collect(),
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
                a != b && topology.arc_slot(ra.min(rb), ra.max(rb)).is_none()
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

    /// A local change stays local: one weight on a 30x30 grid touches a small fraction of
    /// the arcs. This is what proves the scoped path is scoped and not a full sweep in
    /// disguise.
    #[test]
    fn a_local_edit_recomputes_a_small_cone() {
        let mut g = grid(30, 30);
        let base = Cch::build(&to_input(&g)).expect("build");
        let mut cch = base.new_version();
        let total = cch.topology.arc_count();

        assert!(g.get(465, 466).is_some());
        g.set(465, 466, 5.0);
        let input = to_input(&g);
        let outcome = seeded(&mut cch, &[(465, 466)], &input.weights).unwrap();

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

        // Only the arcs whose value changed are written: the version owns exactly the chunks
        // those land in, and a cone arc that keeps its value keeps its chunk shared.
        let mut changed_only = base.new_version();
        let (m, out) = (&cch.metric, &mut changed_only.metric);
        for slot in 0..total {
            if m.upward_weight[slot] != base.metric.upward_weight[slot] {
                out.upward_weight.set(slot, m.upward_weight[slot]);
            }
            if m.downward_weight[slot] != base.metric.downward_weight[slot] {
                out.downward_weight.set(slot, m.downward_weight[slot]);
            }
            if m.upward_middle[slot] != base.metric.upward_middle[slot] {
                out.upward_middle.set(slot, m.upward_middle[slot]);
            }
            if m.downward_middle[slot] != base.metric.downward_middle[slot] {
                out.downward_middle.set(slot, m.downward_middle[slot]);
            }
        }
        assert_eq!(
            cch.metric_bytes_over(&base),
            changed_only.metric_bytes_over(&base)
        );
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
                seeded(&mut cch, &[(u, v)], &input.weights).err(),
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
            seeded(&mut cch, &[(u, v)], &input.weights).err(),
            Some(CchError::InvalidWeight)
        );
        assert_eq!(snapshot(&cch.metric), before);
    }

    /// Exactly `n / 8` distinct changed arcs stay scoped; one more asks for the full
    /// customization instead, and writes nothing.
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
            let slot = topology.arc_slot(ru.min(rv), ru.max(rv)).unwrap();
            if seen.insert(slot) {
                picks.push(k);
            }
            if picks.len() == n / 8 + 1 {
                break;
            }
        }
        assert_eq!(picks.len(), n / 8 + 1);

        for (count, want_full) in [(n / 8, false), (n / 8 + 1, true)] {
            let before = snapshot(&cch.metric);
            let mut changed = Vec::new();
            for &k in &picks[..count] {
                let (u, v, _) = edges[k];
                g.set(u, v, g.get(u, v).unwrap() + 1.0);
                changed.push((u, v));
            }
            let input = to_input(&g);
            let outcome = seeded(&mut cch, &changed, &input.weights).unwrap();
            if want_full {
                assert_eq!(outcome, Recustomized::NeedsFull, "{count} distinct arcs");
                assert_eq!(snapshot(&cch.metric), before, "NeedsFull writes nothing");
            } else {
                assert!(
                    matches!(outcome, Recustomized::Scoped { .. }),
                    "{count} distinct arcs gave {outcome:?}"
                );
                assert_eq!(snapshot(&cch.metric), full(&cch, &input.weights));
            }
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
        let outcome = seeded(&mut cch, &[(u, v), (v, u), (u, v), (u, u)], &input.weights).unwrap();
        assert!(matches!(outcome, Recustomized::Scoped { .. }));
        assert_eq!(snapshot(&cch.metric), full(&cch, &input.weights));
    }

    /// The main oracle: after every batch of random edits the scoped result is
    /// bit-identical to a full customization of the same topology — weights *and* mids
    /// (exact mid equality, not "the detour reproduces the weight": the scoped
    /// recomputation visits candidates in the same order under the same strict `<`) —
    /// and matches the kit's `lower_rank_distances`. Every other batch closes open edges
    /// (`+inf`, a legal weight) and reopens closed ones instead. Seeded pair by pair, it
    /// reads only the pairs of the arcs it recomputes, never the whole metric.
    #[test]
    fn seeded_equals_full_and_reads_only_the_cone() {
        let mut scoped_runs = 0;
        for seed in 1..=20u64 {
            let mut edges = random_sparse(240, 700, 20, seed);
            let pairs = entries(&edges);
            let mut cch = Cch::build(&to_input(&edges)).expect("build");
            let mut rng = SmallRng::seed_from_u64(seed);
            for batch in 0..8 {
                let changed: Vec<(u64, u64)> = (0..1 + rng.random_range(0..8))
                    .map(|_| {
                        if batch % 2 == 0 {
                            random_edit(&mut rng, &cch, &mut edges)
                        } else {
                            close_or_reopen(&mut rng, &pairs, &mut edges)
                        }
                    })
                    .collect();
                let input = to_input(&edges);
                let asked = std::cell::Cell::new(0usize);
                let seed_fn = |u: u64, v: u64| {
                    asked.set(asked.get() + 1);
                    edge_entry(&input.weights, u, v)
                };
                let outcome = cch.recustomize_seeded(&changed, &seed_fn).expect("seeded");
                let Recustomized::Scoped { arcs } = outcome else {
                    panic!("seed {seed} batch {batch}: {outcome:?}");
                };
                assert!(
                    asked.get() <= 2 * arcs,
                    "{} seeds for {arcs} arcs",
                    asked.get()
                );
                scoped_runs += 1;
                assert_eq!(
                    snapshot(&cch.metric),
                    full(&cch, &input.weights),
                    "seed {seed} batch {batch}: scoped != full after {changed:?}"
                );
                check_against_oracle(&cch, &edges);
            }
        }
        assert_eq!(scoped_runs, 160);
    }

    /// Past `n / 8` distinct arcs the seeded entry point asks for a full customization and
    /// writes nothing. A pair with no chordal arc and an edge cannot be recustomized into
    /// existence: it needs a rebuild, checked before any write, so a valid change listed
    /// *first* is not applied either; a node the hierarchy does not have is the same case.
    #[test]
    fn seeded_hands_back_past_the_valve_and_on_a_new_adjacency() {
        let n = 240;
        let g = random_sparse(n, 900, 20, 5);
        let mut cch = Cch::build(&to_input(&g)).unwrap();
        let before = snapshot(&cch.metric);
        let changed: Vec<(u64, u64)> = entries(&g).iter().map(|&(u, v, _)| (u, v)).collect();
        let input = to_input(&reweight(&g, |_, w| w + 1.0));
        let seed_fn = |u: u64, v: u64| edge_entry(&input.weights, u, v);
        assert_eq!(
            cch.recustomize_seeded(&changed, &seed_fn).unwrap(),
            Recustomized::NeedsFull
        );
        assert_eq!(snapshot(&cch.metric), before, "NeedsFull writes nothing");

        let (a, b) = non_adjacent_pair(&cch.topology);
        let seed_fn = |u: u64, v: u64| {
            if (u, v) == (a, b) {
                Some(1.0)
            } else {
                edge_entry(&input.weights, u, v)
            }
        };
        let (eu, ev, _) = entries(&g)[0];
        for changed in [vec![(a, b)], vec![(eu, ev), (a, b)], vec![(0, n as u64)]] {
            assert_eq!(
                cch.recustomize_seeded(&changed, &seed_fn).unwrap(),
                Recustomized::NeedsRebuild,
                "{changed:?}"
            );
            assert_eq!(
                snapshot(&cch.metric),
                before,
                "{changed:?}: nothing written"
            );
        }
        // A negative seed for a changed pair is refused before anything is written.
        let (u, v, _) = entries(&g)[0];
        let seed_fn = |x: u64, y: u64| {
            if (x, y) == (u, v) {
                Some(-1.0)
            } else {
                edge_entry(&input.weights, x, y)
            }
        };
        assert_eq!(
            cch.recustomize_seeded(&[(u, v)], &seed_fn).err(),
            Some(CchError::InvalidWeight)
        );
        assert_eq!(snapshot(&cch.metric), before);
    }
}
