//! Scoped recustomization (#3217): rewrite only the arcs a weight change can reach.
//!
//! Ported from `CCH_RecustomizeScoped` / `CCHIndex_Recustomize` in the C implementation
//! (`src/algorithms/cch.c`, `src/index/cch_index.c`). The layout it works on is the one
//! [`customize`](super::customize) documents: arcs are slots of the upward CSR, and an arc
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

use super::customize::{NO_MID, customize};
use super::{Cch, CchError, WeightCsr};

/// What [`Cch::recustomize_scoped`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recustomized {
    /// Only the affected cone was recomputed; `arcs` is how many arcs that was.
    Scoped { arcs: usize },
    /// Enough arcs changed (more than `n / 8`) that a full customization ran instead.
    Full,
    /// A changed pair has no arc in the hierarchy (a new adjacency), names a node the
    /// hierarchy does not have, or `weights` is shaped for a different node count.
    /// Recustomizing cannot represent it: the structure has to be rebuilt. **Nothing was
    /// written.** On the `Full` path this is also what `customize`'s
    /// `CchError::Internal` shape refusal becomes, so the two paths answer alike.
    NeedsRebuild,
}

/// The stored weight of the directed edge `u -> v`, or `None` when `weights` has no such
/// entry. Rows are sorted (the [`WeightCsr`] contract), so this is a binary search.
fn edge_entry(
    weights: &WeightCsr,
    u: u32,
    v: u32,
) -> Option<f64> {
    let csr = &weights.csr;
    let start = csr.p[u as usize] as usize;
    csr.row(u as usize)
        .binary_search(&v)
        .ok()
        .map(|k| weights.w[start + k])
}

/// The seed an arc direction gets: its edge's weight, or infinity for "no edge" — exactly
/// as a full customization seeds it.
fn seed_of(entry: Option<f64>) -> f64 {
    entry.unwrap_or(f64::INFINITY)
}

/// A stored weight a customization would refuse: negative, NaN or infinite. Absence is
/// how "no edge" is spelled; an explicit `+inf` entry is refused, as `customize` does.
fn bad_entry(entry: Option<f64>) -> bool {
    entry.is_some_and(|w| !w.is_finite() || w < 0.0)
}

impl Cch {
    /// Recompute the metric after the edges in `changed` changed weight, touching only the
    /// arcs those changes can reach.
    ///
    /// `changed` lists node-space `(u, v)` pairs whose cheapest weight may differ from the
    /// last customization — raised, lowered, removed, or added where the hierarchy already
    /// has an arc. Direction does not matter (an arc carries both), duplicates are fine,
    /// and self-loops are ignored. `weights` is the **complete** new metric, the same
    /// [`WeightCsr`] a full [`Cch::customize`] would take:
    /// - a removed edge is simply absent from it, and seeds its arc with infinity;
    /// - every pair whose weight differs from the previous metric must be in `changed`.
    ///   A pair left out keeps its old arc weights, so the result is wrong, not refused.
    ///
    /// # Preconditions
    /// `weights` came from [`CchInput::build`](super::CchInput::build), which has already
    /// refused negative, NaN and infinite weights. This path re-checks only the seeds of
    /// the arcs in `changed` (before writing anything), not the whole metric: doing the
    /// latter would cost O(edges) and make the scoped path no cheaper than the full one.
    ///
    /// # Errors
    /// [`CchError::InvalidWeight`] if a changed pair's new weight is negative, NaN or
    /// infinite. Nothing is written in that case, nor on [`Recustomized::NeedsRebuild`].
    pub fn recustomize_scoped(
        &mut self,
        changed: &[(u32, u32)],
        weights: &WeightCsr,
    ) -> Result<Recustomized, CchError> {
        let topo = Arc::clone(&self.topo);
        let n = topo.node_count();

        // Weights shaped for a different node count mean the graph gained or lost nodes,
        // which only a rebuild handles. The same cheap shape test `customize` starts with,
        // minus its O(edges) column scan: this path only ever reads the rows of changed
        // nodes, each bound-checked below.
        let csr = &weights.csr;
        if csr.n != n
            || csr.p.len() != n + 1
            || csr.p.last().map(|&p| p as usize) != Some(csr.i.len())
            || weights.w.len() != csr.i.len()
        {
            return Ok(Recustomized::NeedsRebuild);
        }

        // Resolve every changed pair to its arc before writing anything, so both
        // `NeedsRebuild` and `InvalidWeight` leave the metric untouched.
        let mut front: Vec<(u32, usize)> = Vec::with_capacity(changed.len());
        for &(u, v) in changed {
            if u == v {
                continue;
            }
            if u as usize >= n || v as usize >= n {
                return Ok(Recustomized::NeedsRebuild);
            }
            let (ru, rv) = (topo.rank_of(u), topo.rank_of(v));
            let (lo, hi) = if ru < rv { (ru, rv) } else { (rv, ru) };
            let (uv, vu) = (edge_entry(weights, u, v), edge_entry(weights, v, u));
            match topo.arc_slot(lo, hi) {
                Some(slot) => {
                    if bad_entry(uv) || bad_entry(vu) {
                        return Err(CchError::InvalidWeight);
                    }
                    front.push((lo, slot));
                }
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
            // `customize` refuses with `Internal` exactly when `weights` holds an edge the
            // hierarchy has no arc for — a new adjacency nobody listed in `changed`. That
            // is this API's `NeedsRebuild`, and like it, nothing has been written.
            return match customize(&topo, weights) {
                Ok(metric) => {
                    self.metric = metric;
                    Ok(Recustomized::Full)
                }
                Err(CchError::Internal(_)) => Ok(Recustomized::NeedsRebuild),
                Err(e) => Err(e),
            };
        }

        let mut queued: FxHashSet<usize> = front.iter().map(|&(_, slot)| slot).collect();
        let mut heap: BinaryHeap<Reverse<(u32, usize)>> = front.into_iter().map(Reverse).collect();
        let metric = &mut self.metric;
        let mut arcs = 0;

        while let Some(Reverse((lo, slot))) = heap.pop() {
            arcs += 1;
            let hi = topo.up(lo)[slot - topo.up_range(lo).start];
            let (u, v) = (topo.node_of(lo), topo.node_of(hi));

            // Reseed from the original edges, then relax over every lower common neighbour
            // x of lo and hi: x in down(lo) and in down(hi). Both rows ascend, so a merge.
            let (uv, vu) = (edge_entry(weights, u, v), edge_entry(weights, v, u));
            let mut best_up = seed_of(uv);
            let mut best_dn = seed_of(vu);
            debug_assert!(!bad_entry(uv) && !bad_entry(vu), "see # Preconditions");
            let mut mid_up = NO_MID;
            let mut mid_dn = NO_MID;

            let (dlo, dhi) = (topo.down(lo), topo.down(hi));
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
                let x_lo = topo.arc_slot(x, lo).expect("down/up are inverses");
                let x_hi = topo.arc_slot(x, hi).expect("down/up are inverses");

                // lo -> x -> hi
                let cand_up = metric.dn_w.get(x_lo) + metric.up_w.get(x_hi);
                if cand_up < best_up {
                    best_up = cand_up;
                    mid_up = x;
                }
                // hi -> x -> lo
                let cand_dn = metric.dn_w.get(x_hi) + metric.up_w.get(x_lo);
                if cand_dn < best_dn {
                    best_dn = cand_dn;
                    mid_dn = x;
                }
                a += 1;
                b += 1;
            }

            let changed_weight =
                best_up != metric.up_w.get(slot) || best_dn != metric.dn_w.get(slot);
            metric.up_w.set(slot, best_up);
            metric.dn_w.set(slot, best_dn);
            metric.up_mid.set(slot, mid_up);
            metric.dn_mid.set(slot, mid_dn);

            // A changed arc invalidates the top arc of every triangle it is the bottom of:
            // apex lo, other bottom arc (lo, w). All of those sit above lo, so they pop
            // after everything they read is final.
            if changed_weight {
                for &w in topo.up(lo) {
                    if w == hi {
                        continue;
                    }
                    let (d_lo, d_hi) = if hi < w { (hi, w) } else { (w, hi) };
                    let d_slot = topo
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
    use super::Recustomized;
    use crate::index::cch::customize::{Metric, customize};
    use crate::index::cch::order::Topology;
    use crate::index::cch::testkit::{
        EdgeList, grid, lower_rank_distances, random_sparse, to_input,
    };
    use crate::index::cch::{Cch, CchError, WeightCsr};

    /// The metric as plain vectors, for byte-for-byte comparison.
    type Snapshot = (Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>);

    fn snapshot(m: &Metric) -> Snapshot {
        (
            m.up_w.iter().collect(),
            m.dn_w.iter().collect(),
            m.up_mid.iter().collect(),
            m.dn_mid.iter().collect(),
        )
    }

    /// What a full customization of the same topology gives for `weights`: the oracle
    /// every scoped result must match exactly, mids included.
    fn full(
        cch: &Cch,
        weights: &WeightCsr,
    ) -> Snapshot {
        snapshot(&customize(&cch.topo, weights).expect("full customize"))
    }

    /// Also check the weights against the kit's `lower_rank_distances`, which derives
    /// the lower-restricted distance from the edge list and the order alone.
    fn check_against_oracle(
        cch: &Cch,
        edges: &EdgeList,
    ) {
        let topo = &cch.topo;
        let expected = lower_rank_distances(edges, topo.perm());
        for lo in 0..edges.n as u32 {
            let start = topo.up_range(lo).start;
            for (i, &hi) in topo.up(lo).iter().enumerate() {
                let (u, v) = (topo.node_of(lo), topo.node_of(hi));
                assert_eq!(
                    cch.metric.up_w.get(start + i),
                    expected[u as usize][v as usize]
                );
                assert_eq!(
                    cch.metric.dn_w.get(start + i),
                    expected[v as usize][u as usize]
                );
            }
        }
    }

    /// Some pair of distinct nodes with no chordal arc between them.
    fn non_adjacent_pair(topo: &Topology) -> (u32, u32) {
        let n = topo.node_count() as u32;
        (0..n)
            .flat_map(|a| (0..n).map(move |b| (a, b)))
            .find(|&(a, b)| {
                let (ra, rb) = (topo.rank_of(a), topo.rank_of(b));
                a != b && topo.arc_slot(ra.min(rb), ra.max(rb)).is_none()
            })
            .expect("a sparse graph has a non-adjacent pair")
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(
            &mut self,
            n: usize,
        ) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// One random edit to `edges`: raise or lower a weight, remove an edge, or add an
    /// edge between nodes that already share a chordal arc. Returns the pair touched.
    fn random_edit(
        rng: &mut Rng,
        cch: &Cch,
        edges: &mut EdgeList,
    ) -> (u32, u32) {
        let topo = &cch.topo;
        match rng.below(4) {
            0 | 1 if !edges.edges.is_empty() => {
                let k = rng.below(edges.edges.len());
                let e = &mut edges.edges[k];
                // Integer weights so ties are common: ties are where a scoped mid
                // could most easily drift from the full sweep's.
                e.2 = if rng.below(2) == 0 {
                    e.2 + 1.0 + rng.below(5) as f64
                } else {
                    (e.2 - 1.0 - rng.below(5) as f64).max(0.0)
                };
                (e.0, e.1)
            }
            2 if !edges.edges.is_empty() => {
                let (u, v, _) = edges.edges.swap_remove(rng.below(edges.edges.len()));
                (u, v)
            }
            _ => loop {
                let lo = rng.below(edges.n) as u32;
                let up = topo.up(lo);
                if up.is_empty() {
                    continue;
                }
                let hi = up[rng.below(up.len())];
                let (mut u, mut v) = (topo.node_of(lo), topo.node_of(hi));
                if rng.below(2) == 0 {
                    std::mem::swap(&mut u, &mut v);
                }
                edges.push(u, v, 1.0 + rng.below(20) as f64);
                break (u, v);
            },
        }
    }

    /// The main oracle: after every batch of random edits the scoped result is
    /// bit-identical to a full customization of the same topology — weights *and* mids
    /// (exact mid equality, not "the detour reproduces the weight": the scoped
    /// recomputation visits candidates in the same order under the same strict `<`) —
    /// and matches the kit's `lower_rank_distances`.
    #[test]
    fn scoped_equals_full_on_random_edit_batches() {
        let mut scoped_runs = 0;
        for seed in 1..=20u64 {
            let mut edges = random_sparse(240, 700, 20, seed);
            let mut cch = Cch::build(&to_input(&edges)).expect("build");
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);

            for batch in 0..8 {
                let size = 1 + rng.below(8);
                let changed: Vec<(u32, u32)> = (0..size)
                    .map(|_| random_edit(&mut rng, &cch, &mut edges))
                    .collect();
                let input = to_input(&edges);

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
                check_against_oracle(&cch, &edges);
            }
        }
        assert_eq!(scoped_runs, 160);
    }

    /// A local change stays local: one weight on a 30x30 grid touches a small fraction of
    /// the arcs. This is what proves the scoped path is scoped and not a full sweep in
    /// disguise.
    #[test]
    fn a_local_edit_recomputes_a_small_cone() {
        let mut edges = grid(30, 30);
        let mut cch = Cch::build(&to_input(&edges)).expect("build");
        let total = cch.topo.arc_count();

        let k = edges
            .edges
            .iter()
            .position(|&(u, v, _)| (u, v) == (465, 466))
            .unwrap();
        edges.edges[k].2 = 5.0;
        let input = to_input(&edges);
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

    /// Removing an edge is spelled as its absence from `weights`, and its arc reseeds to
    /// infinity — no explicit `+inf` anywhere.
    #[test]
    fn a_removed_edge_reseeds_its_arc_to_infinity() {
        // 0 - 1 - 2 path, both directions; remove 1 -> 2 only.
        let mut edges = EdgeList::new(3);
        for (u, v) in [(0, 1), (1, 0), (1, 2), (2, 1)] {
            edges.push(u, v, 1.0);
        }
        let mut cch = Cch::build(&to_input(&edges)).unwrap();
        edges.edges.retain(|&(u, v, _)| (u, v) != (1, 2));
        let input = to_input(&edges);
        assert!(input.weights.w.iter().all(|w| w.is_finite()));
        cch.recustomize_scoped(&[(1, 2)], &input.weights).unwrap();
        assert_eq!(snapshot(&cch.metric), full(&cch, &input.weights));
        check_against_oracle(&cch, &edges);
        assert_eq!(cch.query(1, 2), None, "1 -> 2 must now be unreachable");
        assert!(cch.query(2, 1).is_some(), "2 -> 1 is untouched");
    }

    /// A pair with no chordal arc cannot be recustomized into existence. The call must say
    /// so and leave the metric exactly as it was — checked before any write, so a valid
    /// change listed *first* is not applied either.
    #[test]
    fn a_new_adjacency_needs_a_rebuild_and_writes_nothing() {
        let edges = random_sparse(240, 500, 20, 7);
        let mut cch = Cch::build(&to_input(&edges)).unwrap();
        let topo = &cch.topo;
        let (u, v) = non_adjacent_pair(topo);
        let (eu, ev, ew) = edges.edges[0];

        let mut edited = edges.clone();
        edited.edges[0].2 = ew + 7.0; // a valid change, listed first
        edited.push(u, v, 3.0); // the impossible one
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

    /// A negative new weight is refused before anything is written.
    #[test]
    fn a_bad_new_weight_is_refused_and_writes_nothing() {
        let edges = random_sparse(240, 500, 20, 11);
        let mut cch = Cch::build(&to_input(&edges)).unwrap();
        let before = snapshot(&cch.metric);
        let (u, v, _) = edges.edges[3];
        let mut input = to_input(&edges);
        // Corrupt the one entry, bypassing to_input's own resolution.
        let row = input.weights.csr.p[u as usize] as usize;
        let k = row + input.weights.csr.row(u as usize).binary_search(&v).unwrap();
        // An explicit +inf entry is refused, as `customize` refuses it; only absence
        // means "no edge".
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            input.weights.w[k] = bad;
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
    }

    /// Past `n / 8` distinct changed arcs the full customization runs instead — and the
    /// result is the same either way.
    #[test]
    fn the_n_over_8_valve_switches_to_a_full_customize() {
        let n = 240;
        let mut edges = random_sparse(n, 900, 20, 5);
        let mut cch = Cch::build(&to_input(&edges)).unwrap();

        // Exactly n/8 distinct arcs stays scoped; one more goes full. Distinct arcs, not
        // pairs: (u, v) and (v, u) share one arc, so collect by arc.
        let topo = cch.topo.clone();
        let mut seen = std::collections::HashSet::new();
        let mut picks = Vec::new();
        for (k, &(u, v, _)) in edges.edges.iter().enumerate() {
            let (ru, rv) = (topo.rank_of(u), topo.rank_of(v));
            let slot = topo.arc_slot(ru.min(rv), ru.max(rv)).unwrap();
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
                edges.edges[k].2 += 1.0;
                changed.push((edges.edges[k].0, edges.edges[k].1));
            }
            let input = to_input(&edges);
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
        let mut edges = random_sparse(240, 600, 20, 3);
        let mut cch = Cch::build(&to_input(&edges)).unwrap();
        let (u, v, _) = edges.edges[0];
        edges.edges[0].2 += 3.0;
        let input = to_input(&edges);
        let outcome = cch
            .recustomize_scoped(&[(u, v), (v, u), (u, v), (u, u)], &input.weights)
            .unwrap();
        assert!(matches!(outcome, Recustomized::Scoped { .. }));
        assert_eq!(snapshot(&cch.metric), full(&cch, &input.weights));
    }

    /// Both paths answer a wrong shape the same way, writing nothing: weights for a
    /// different node count up front, and — on the `Full` path, which reads every edge —
    /// a new adjacency the caller did not list in `changed`.
    #[test]
    fn shape_mismatches_need_a_rebuild_on_both_paths() {
        let n = 240;
        let edges = random_sparse(n, 900, 20, 9);
        let mut cch = Cch::build(&to_input(&edges)).unwrap();
        let before = snapshot(&cch.metric);

        // A node more than the hierarchy has.
        let mut bigger = edges.clone();
        bigger.n = n + 1;
        let input = to_input(&bigger);
        let (u, v, _) = edges.edges[0];
        assert_eq!(
            cch.recustomize_scoped(&[(u, v)], &input.weights).unwrap(),
            Recustomized::NeedsRebuild
        );
        assert_eq!(snapshot(&cch.metric), before);

        // A CSR whose last offset disagrees with its column array: refused, not a panic.
        let mut malformed = to_input(&edges);
        *malformed.weights.csr.p.last_mut().unwrap() += 1;
        assert_eq!(
            cch.recustomize_scoped(&[(u, v)], &malformed.weights)
                .unwrap(),
            Recustomized::NeedsRebuild
        );
        assert_eq!(snapshot(&cch.metric), before);

        // Enough listed changes to take the Full path, plus one unlisted new adjacency.
        let topo = cch.topo.clone();
        let (a, b) = non_adjacent_pair(&topo);
        let mut edited = edges.clone();
        let changed: Vec<(u32, u32)> = edited.edges.iter().map(|e| (e.0, e.1)).collect();
        for e in &mut edited.edges {
            e.2 += 1.0;
        }
        edited.push(a, b, 1.0);
        let input = to_input(&edited);
        assert_eq!(
            cch.recustomize_scoped(&changed, &input.weights).unwrap(),
            Recustomized::NeedsRebuild,
            "the Full path must answer an unlisted new adjacency like the scoped one"
        );
        assert_eq!(snapshot(&cch.metric), before);
    }

    /// Not a test: the scoped path against a full customization on a 200x200 grid, one
    /// edit at a time. `cargo test -p graph --lib scoped_vs_full_timing -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore = "timing, not a check"]
    fn scoped_vs_full_timing() {
        use std::time::Instant;
        let mut edges = grid(200, 200);
        let mut cch = Cch::build(&to_input(&edges)).unwrap();
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let (mut scoped, mut fullt, mut arcs_total) = (0u128, 0u128, 0usize);
        let rounds = 50;
        for _ in 0..rounds {
            let k = rng.below(edges.edges.len());
            edges.edges[k].2 = 1.0 + rng.below(9) as f64;
            let (u, v, _) = edges.edges[k];
            let input = to_input(&edges);

            let t = Instant::now();
            let out = cch.recustomize_scoped(&[(u, v)], &input.weights).unwrap();
            scoped += t.elapsed().as_micros();
            let Recustomized::Scoped { arcs } = out else {
                panic!("{out:?}")
            };
            arcs_total += arcs;

            let t = Instant::now();
            let reference = customize(&cch.topo, &input.weights).unwrap();
            fullt += t.elapsed().as_micros();
            assert_eq!(snapshot(&cch.metric), snapshot(&reference));
        }
        eprintln!(
            "grid 200x200, {} arcs, {rounds} single-edge edits: scoped mean {} us ({} arcs \
             recomputed on average), full customize mean {} us",
            cch.topo.arc_count(),
            scoped / rounds as u128,
            arcs_total / rounds,
            fullt / rounds as u128
        );
    }
}
