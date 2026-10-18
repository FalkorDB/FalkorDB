//! Phase 2: customization and [`WeightArr`] (#3209, #3217).
//!
//! Phase 2 writes one concrete metric into the structure Phase 1 built. It is the only
//! phase that looks at a weight, and the only one that runs again when the metric changes:
//! [`Topology`] is reused untouched.
//!
//! # Internal layout
//!
//! Scoped recustomization (#3217) and any later change to how the weight arrays are stored
//! both code against what follows, so it is spelled out rather than left to be read off the
//! code.
//!
//! ## Two spaces
//!
//! * **node space** — ids as the rest of the engine knows them, `0..n`. [`CchInput`]'s
//!   `skeleton` and `weights` are both in node space.
//! * **rank space** — elimination order, `0..n`. Everything inside [`Cch`] except `perm`
//!   and `iperm` themselves is in rank space.
//!
//! `perm[rank] = node` ([`Topology::node_of`]) and `iperm[node] = rank`
//! ([`Topology::rank_of`]) translate between them.
//!
//! ## Arc ids
//!
//! The chordal supergraph is held as the upward CSR `cch_p`/`cch_i` over ranks. An **arc
//! id** (called a *slot* in the code) is simply an index into `cch_i`:
//!
//! ```text
//! slot k, with cch_p[lo] <= k < cch_p[lo + 1]   <->   the arc between ranks lo and cch_i[k]
//! ```
//!
//! so `lo` is recoverable from `k` by a binary search over `cch_p`, and `hi = cch_i[k]`
//! directly. `lo < hi` always: a rank only lists its *upper* neighbours.
//! [`Topology::arc_slot(lo, hi)`](Topology::arc_slot) is the reverse lookup, and
//! [`Topology::up_range(lo)`](Topology::up_range) is the half-open slot range of one rank's
//! arcs — `up(lo)[i]` is slot `up_range(lo).start + i`.
//!
//! Slots are stable for the life of a [`Topology`]: they change only when the hierarchy is
//! rebuilt, never when it is recustomized.
//!
//! ## The four arrays
//!
//! All four are indexed by arc id and so all have length `cch_i.len()`
//! ([`Topology::arc_count`]). For slot `k` joining ranks `lo` and `hi = cch_i[k]`:
//!
//! | array | meaning |
//! |---|---|
//! | `up_w[k]`   | cheapest `lo -> hi` weight through nodes of rank `< lo` |
//! | `dn_w[k]`   | cheapest `hi -> lo` weight through nodes of rank `< lo` |
//! | `up_mid[k]` | rank of the apex `up_w[k]` routes through, or [`NO_MID`] |
//! | `dn_mid[k]` | rank of the apex `dn_w[k]` routes through, or [`NO_MID`] |
//!
//! "through nodes of rank `< lo`" is the whole invariant: an arc holds the shortest path
//! between its endpoints that uses *only strictly lower-ranked* intermediates. A weight of
//! `f64::INFINITY` means no such path exists — including for the arcs that exist purely as
//! fill-in and never acquire one.
//!
//! `up_mid[k] == NO_MID` means the arc's weight is its own original edge, with nothing to
//! unpack; otherwise the arc expands to `lo -> up_mid[k] -> hi`, each half of which is
//! itself an arc to expand recursively. A mid rank is always strictly below `lo`.
//!
//! ## Why one forward sweep is enough
//!
//! For each rank `x` taken in increasing order, every pair `{y, z}` of its upper neighbours
//! forms a triangle `x < y < z` whose top arc `(y, z)` can be improved by detouring through
//! `x`. The chordal property guarantees that top arc exists, which is exactly what Phase 1
//! bought. Processing `x` in increasing order means the two lower arcs `(x, y)` and `(x, z)`
//! are already final when read: any improvement to them would come from a triangle with an
//! even lower apex, all of which were handled in an earlier iteration. So no iteration to a
//! fixed point is needed.

use std::sync::Arc;

use super::order::Topology;
use super::{Cch, CchError, CchInput, WeightCsr, check_size};

/// `up_mid`/`dn_mid` value for "this arc is its own original edge; nothing to unpack".
///
/// A sentinel rather than `Option<u32>`: the mid arrays are one entry per chordal arc, and
/// `Option<u32>` would cost 8 bytes per arc instead of 4. No real rank can collide with it,
/// because the ordering backend caps the node count at `i32::MAX` (see
/// [`OrderError`](super::order::OrderError)).
pub(crate) const NO_MID: u32 = u32::MAX;

/// A versioned array of per-arc metric values.
///
/// The four weight arrays are the only mutable state in a [`Cch`]; everything structural is
/// shared by `Arc`. They go through this wrapper so that the storage can later be swapped for a
/// copy-on-write array (a shared base plus a small overlay of changed slots) without a
/// single caller changing: the surface below — `get`, `set`, `len`, `is_empty`,
/// `new_version`, `iter`, `memory_usage` — is deliberately the surface such an array needs.
///
/// This version is backed by a plain `Vec` and copies it on [`Self::new_version`], which is
/// correct but O(n) per version. That is the cost a copy-on-write array would remove.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WeightArr<T: Copy> {
    values: Vec<T>,
}

// The whole surface is fixed up front, as the interface a copy-on-write array needs, so the
// methods marked `dead_code` have no caller until the storage is swapped.
impl<T: Copy> WeightArr<T> {
    /// `len` slots, all holding `value`.
    pub(crate) fn filled(
        len: usize,
        value: T,
    ) -> Self {
        Self {
            values: vec![value; len],
        }
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// # Panics
    /// If `slot` is out of range — an arc id that does not exist is a bug, not a miss.
    pub(crate) fn get(
        &self,
        slot: usize,
    ) -> T {
        self.values[slot]
    }

    /// # Panics
    /// If `slot` is out of range.
    pub(crate) fn set(
        &mut self,
        slot: usize,
        value: T,
    ) {
        self.values[slot] = value;
    }

    /// A copy that can be written without disturbing this one. The seam a copy-on-write array would make cheap.
    #[allow(dead_code)]
    pub(crate) fn new_version(&self) -> Self {
        self.clone()
    }

    #[allow(dead_code)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.values.iter().copied()
    }

    pub(crate) fn memory_usage(&self) -> usize {
        size_of::<Self>() + self.values.capacity() * size_of::<T>()
    }
}

/// The four per-arc arrays a customization produces, parallel to `cch_i`.
pub(crate) struct Metric {
    pub(crate) up_w: WeightArr<f64>,
    pub(crate) dn_w: WeightArr<f64>,
    pub(crate) up_mid: WeightArr<u32>,
    pub(crate) dn_mid: WeightArr<u32>,
}

impl Metric {
    pub(crate) fn memory_usage(&self) -> usize {
        self.up_w.memory_usage()
            + self.dn_w.memory_usage()
            + self.up_mid.memory_usage()
            + self.dn_mid.memory_usage()
    }
}

/// Message for weights shaped for some other graph than the one the hierarchy was built
/// for — almost always a `customize` call that should have been a rebuild.
const SHAPE_MISMATCH: &str = "weights do not match the hierarchy (rebuild, do not recustomize)";

/// Reject a metric this index cannot be correct for, before any of it is written.
///
/// Dijkstra-style relaxation needs non-negative weights, and `+inf`/NaN would make a
/// seeded arc indistinguishable from an absent one. The C implementation does not check;
/// refusing at build time is the contract decision recorded for this port.
///
/// The shape is checked too, because `customize` is public: weights built for a graph that
/// has since gained a node would otherwise index past the end of `iperm`, and short arrays
/// would index past the end of `w`. Both are caller errors, refused rather than panicked.
fn validate(
    topo: &Topology,
    weights: &WeightCsr,
) -> Result<(), CchError> {
    let csr = &weights.csr;
    let n = topo.node_count();
    if csr.n != n
        || csr.p.len() != n + 1
        || csr.p.first().copied() != Some(0)
        || csr.p.windows(2).any(|pair| pair[0] > pair[1])
        || csr.p.last().copied() != Some(csr.i.len() as u64)
        || weights.w.len() != csr.i.len()
        || csr.i.iter().any(|&v| v as usize >= n)
    {
        return Err(CchError::Internal(SHAPE_MISMATCH));
    }
    if weights.w.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(CchError::InvalidWeight);
    }
    Ok(())
}

impl WeightCsr {
    /// Every stored entry as `(row, column, weight)`, row by row, columns ascending within a
    /// row. Assumes the offsets are valid (`n + 1` of them, non-decreasing, the last at most
    /// `i.len()`) and that `w` has one weight per entry; see [`validate`].
    pub(crate) fn entries(&self) -> impl Iterator<Item = (u32, u32, f64)> + '_ {
        (0..self.csr.n).flat_map(move |u| {
            let (lo, hi) = (self.csr.p[u] as usize, self.csr.p[u + 1] as usize);
            #[allow(clippy::cast_possible_truncation)]
            self.csr.i[lo..hi]
                .iter()
                .zip(&self.w[lo..hi])
                .map(move |(&v, &w)| (u as u32, v, w))
        })
    }
}

/// Phase 2: seed every chordal arc from `weights`, then relax every lower triangle.
///
/// # Errors
/// [`CchError::InvalidWeight`] if any weight is negative, infinite or NaN, and
/// [`CchError::Internal`] if `weights` is not shaped for this topology — including an edge
/// between two nodes the hierarchy has no arc for, which only a rebuild can add.
pub(crate) fn customize(
    topo: &Topology,
    weights: &WeightCsr,
) -> Result<Metric, CchError> {
    validate(topo, weights)?;

    let mut metric = seed(topo, weights)?;
    relax(topo, &mut metric);
    Ok(metric)
}

/// Original edges take their weight from `weights`; every other arc starts at infinity.
///
/// C reads this the other way round — one `extractElement(W, u, v)` per arc — which costs a
/// lookup for each of the many arcs that are pure fill-in and will miss. Scattering the
/// edges into an infinity-filled array instead touches each *edge* once, and the chordal
/// supergraph has far more arcs than the graph has edges.
fn seed(
    topo: &Topology,
    weights: &WeightCsr,
) -> Result<Metric, CchError> {
    let arcs = topo.arc_count();
    let mut metric = Metric {
        up_w: WeightArr::filled(arcs, f64::INFINITY),
        dn_w: WeightArr::filled(arcs, f64::INFINITY),
        up_mid: WeightArr::filled(arcs, NO_MID),
        dn_mid: WeightArr::filled(arcs, NO_MID),
    };

    // Parallelisable, but not done: every stored entry (u, v) writes exactly one slot of
    // `up_w` (if `u` ranks below `v`) or `dn_w` (if above), and the two directions of a pair
    // go to different arrays, so with duplicate-free rows no two entries write the same
    // location and the rows are independent. What stops a one-line change is that the
    // writes go through `&mut WeightArr`, which would have to be split by row or slot range.
    for (u, v, w) in weights.entries() {
        if u == v {
            continue; // a self-loop is never on a shortest path
        }
        let (rank_u, rank_v) = (topo.rank_of(u), topo.rank_of(v));

        let (lo, hi) = if rank_u < rank_v {
            (rank_u, rank_v)
        } else {
            (rank_v, rank_u)
        };
        // Every directed weight entry (u, v) has u-v in the skeleton, so for weights
        // built from the same graph the arc is there. A miss means the graph has
        // gained an adjacency since the hierarchy was built; dropping the weight would
        // make queries silently too long.
        let Some(slot) = topo.arc_slot(lo, hi) else {
            return Err(CchError::Internal(SHAPE_MISMATCH));
        };

        if rank_u < rank_v {
            metric.up_w.set(slot, w);
        } else {
            metric.dn_w.set(slot, w);
        }
    }

    Ok(metric)
}

/// Basic customization: one forward sweep over the lower triangles.
///
/// See the module doc for why a single pass in increasing rank order suffices.
///
/// Parallelism, not done: iteration `x` only reads arcs whose lower end is `x`, which are
/// written only by iterations of lower ranks `x'` that `x` is an upper neighbour of, so it
/// has to run after those. Two iterations are therefore independent only if neither feeds
/// the other, but independence in that sense is not enough: iterations of different `x`
/// can also write the *same* arc `(y, z)` (any two common lower neighbours of `y` and `z`),
/// so running them concurrently would race on its weight and, on a tie, change which `x`
/// is recorded as the mid. A parallel version has to change the loop (for example, have
/// each arc pull the minimum over its own lower triangles, so each is written once) and
/// keep the lowest-`x` tie-break. That has not been implemented or measured here.
fn relax(
    topo: &Topology,
    metric: &mut Metric,
) {
    for x in 0..topo.node_count() as u32 {
        let up = topo.up(x);
        let base = topo.up_range(x).start;

        for (i, &y) in up.iter().enumerate() {
            // x -> y and y -> x. Both are final: the only writes below land on arcs whose
            // lower endpoint is y or above, never on x's own row.
            let w_xy_up = metric.up_w.get(base + i);
            let w_xy_dn = metric.dn_w.get(base + i);

            // `up` is ascending, so everything past i is the z > y half of each pair, and
            // each unordered pair {y, z} is visited exactly once.
            for (j, &z) in up.iter().enumerate().skip(i + 1) {
                let w_xz_up = metric.up_w.get(base + j);
                let w_xz_dn = metric.dn_w.get(base + j);

                let Some(slot) = topo.arc_slot(y, z) else {
                    debug_assert!(false, "chordal closure broken: no arc between {y} and {z}");
                    continue;
                };

                // y -> x -> z improves the upward arc
                let cand_up = w_xy_dn + w_xz_up;
                if cand_up < metric.up_w.get(slot) {
                    metric.up_w.set(slot, cand_up);
                    metric.up_mid.set(slot, x);
                }

                // z -> x -> y improves the downward arc
                let cand_dn = w_xz_dn + w_xy_up;
                if cand_dn < metric.dn_w.get(slot) {
                    metric.dn_w.set(slot, cand_dn);
                    metric.dn_mid.set(slot, x);
                }
            }
        }
    }
}

impl Cch {
    /// Phase 1 then Phase 2.
    pub(super) fn build_impl(input: &CchInput) -> Result<Self, CchError> {
        debug_assert_eq!(input.skeleton.n, input.n);
        debug_assert_eq!(input.weights.csr.n, input.n);

        check_size(input.n.max(input.skeleton.n), input.skeleton.i.len() as u64)?;
        let topo = Arc::new(Topology::build(&input.skeleton)?);
        let metric = customize(&topo, &input.weights)?;
        Ok(Self { topo, metric })
    }

    /// Phase 2 again, same structure, new metric. The topology is shared, not rebuilt.
    pub(super) fn customize_impl(
        &mut self,
        weights: &WeightCsr,
    ) -> Result<(), CchError> {
        self.metric = customize(&self.topo, weights)?;
        Ok(())
    }

    #[allow(dead_code)] // the query change (#3210) is the first reader
    pub(crate) fn topology(&self) -> &Topology {
        &self.topo
    }

    /// Cheapest `lo -> hi` weight of arc `slot`, through strictly lower-ranked nodes only.
    #[allow(dead_code)] // the query change (#3210) is the first reader
    pub(crate) fn up_weight(
        &self,
        slot: usize,
    ) -> f64 {
        self.metric.up_w.get(slot)
    }

    /// Cheapest `hi -> lo` weight of arc `slot`.
    #[allow(dead_code)] // the query change (#3210) is the first reader
    pub(crate) fn dn_weight(
        &self,
        slot: usize,
    ) -> f64 {
        self.metric.dn_w.get(slot)
    }

    /// The apex rank `up_weight(slot)` routes through, or `None` if the arc is just its
    /// original edge.
    #[allow(dead_code)] // the query change (#3210) is the first reader
    pub(crate) fn up_mid(
        &self,
        slot: usize,
    ) -> Option<u32> {
        match self.metric.up_mid.get(slot) {
            NO_MID => None,
            mid => Some(mid),
        }
    }

    /// The apex rank `dn_weight(slot)` routes through, or `None`.
    #[allow(dead_code)] // the query change (#3210) is the first reader
    pub(crate) fn dn_mid(
        &self,
        slot: usize,
    ) -> Option<u32> {
        match self.metric.dn_mid.get(slot) {
            NO_MID => None,
            mid => Some(mid),
        }
    }

    pub(super) fn node_count_impl(&self) -> usize {
        self.topo.node_count()
    }

    pub(super) fn memory_usage_impl(&self) -> usize {
        size_of::<Self>() + self.topo.memory_usage() + self.metric.memory_usage()
    }
}

#[cfg(test)]
mod tests {
    use super::{NO_MID, WeightArr};
    use crate::index::cch::order::{IdentityOrdering, Topology};
    use crate::index::cch::testkit::{
        EdgeList, all_pairs, assert_input_invariants, disconnected_components, empty, grid,
        lower_rank_distances, path, random_sparse, single_node, star, tied_paths, to_input,
        with_isolated_nodes, with_parallel_edges, with_self_loops, with_zero_weight_edge,
        zero_weight_cycle,
    };
    use crate::index::cch::{Cch, CchError, CchInput, Csr, MAX_NODES, WeightCsr};

    /// The weight of the original `u -> v` edge in node space, or infinity if there is
    /// none. What an arc must hold when it records no mid.
    fn original(
        input: &CchInput,
        u: u32,
        v: u32,
    ) -> f64 {
        let csr = &input.weights.csr;
        let row = csr.p[u as usize] as usize..csr.p[u as usize + 1] as usize;
        row.clone()
            .find(|&k| csr.i[k] == v)
            .map_or(f64::INFINITY, |k| input.weights.w[k])
    }

    /// Every arc holds exactly the lower-restricted shortest path between its endpoints —
    /// checked against the kit's `lower_rank_distances`, which derives that quantity from
    /// the edge list and the elimination order alone and shares no code with the triangle
    /// sweep it is checking.
    fn check_metric(
        cch: &Cch,
        edges: &EdgeList,
        input: &CchInput,
    ) {
        let topo = cch.topology();
        let expected = lower_rank_distances(edges, topo.perm());

        for lo in 0..edges.n as u32 {
            let range = topo.up_range(lo);
            for (i, &hi) in topo.up(lo).iter().enumerate() {
                let slot = range.start + i;
                let (u, v) = (topo.node_of(lo), topo.node_of(hi));

                assert_eq!(
                    cch.up_weight(slot),
                    expected[u as usize][v as usize],
                    "arc {u}->{v} (ranks {lo}->{hi}, slot {slot}) upward weight"
                );
                assert_eq!(
                    cch.dn_weight(slot),
                    expected[v as usize][u as usize],
                    "arc {v}->{u} (ranks {hi}->{lo}, slot {slot}) downward weight"
                );

                // The mid is what the query change (#3210) unpacks a shortcut with, so it has to be more than
                // plausible: the recorded detour must reproduce the arc's own weight.
                if let Some(m) = cch.up_mid(slot) {
                    assert!(
                        m < lo,
                        "arc {lo}-{hi} routes through rank {m}, not below it"
                    );
                    let to_mid = topo.arc_slot(m, lo).expect("mid is adjacent to lo");
                    let from_mid = topo.arc_slot(m, hi).expect("mid is adjacent to hi");
                    assert_eq!(
                        cch.up_weight(slot),
                        cch.dn_weight(to_mid) + cch.up_weight(from_mid),
                        "arc {lo}->{hi} does not equal its recorded detour via {m}"
                    );
                } else {
                    assert_eq!(
                        cch.up_weight(slot),
                        original(input, u, v),
                        "arc {lo}->{hi} records no mid, so it must be its own original edge"
                    );
                }

                if let Some(m) = cch.dn_mid(slot) {
                    assert!(
                        m < lo,
                        "arc {hi}-{lo} routes through rank {m}, not below it"
                    );
                    let to_mid = topo.arc_slot(m, hi).expect("mid is adjacent to hi");
                    let from_mid = topo.arc_slot(m, lo).expect("mid is adjacent to lo");
                    assert_eq!(
                        cch.dn_weight(slot),
                        cch.dn_weight(to_mid) + cch.up_weight(from_mid),
                        "arc {hi}->{lo} does not equal its recorded detour via {m}"
                    );
                } else {
                    assert_eq!(
                        cch.dn_weight(slot),
                        original(input, v, u),
                        "arc {hi}->{lo} records no mid, so it must be its own original edge"
                    );
                }
            }
        }
    }

    /// Adding every chordal arc to the graph, in both directions with its customized
    /// weight, must not make any pair cheaper to reach: a shortcut has to stand for a real
    /// path. Catches under-estimation, which `check_metric` would also catch — but through
    /// a different oracle (`all_pairs`), so a shared misreading of the rank restriction
    /// cannot hide behind both.
    fn check_no_shortcut_beats_reality(
        cch: &Cch,
        edges: &EdgeList,
    ) {
        let truth = all_pairs(edges);
        let topo = cch.topology();

        let mut augmented = edges.clone();
        for lo in 0..edges.n as u32 {
            let range = topo.up_range(lo);
            for (i, &hi) in topo.up(lo).iter().enumerate() {
                let slot = range.start + i;
                let (u, v) = (topo.node_of(lo), topo.node_of(hi));
                if cch.up_weight(slot).is_finite() {
                    augmented.push(u, v, cch.up_weight(slot));
                }
                if cch.dn_weight(slot).is_finite() {
                    augmented.push(v, u, cch.dn_weight(slot));
                }
            }
        }

        let with_shortcuts = all_pairs(&augmented);
        for s in 0..edges.n {
            for t in 0..edges.n {
                assert_eq!(
                    with_shortcuts[s][t], truth[s][t],
                    "shortcuts changed the {s}->{t} distance"
                );
            }
        }
    }

    fn check(edges: &EdgeList) {
        let input = to_input(edges);
        assert_input_invariants(&input);
        let cch = Cch::build(&input).expect("build");
        assert_eq!(cch.node_count(), edges.n);
        check_metric(&cch, edges, &input);
        check_no_shortcut_beats_reality(&cch, edges);
    }

    #[test]
    fn arcs_match_the_oracle_on_a_grid() {
        check(&grid(5, 6));
    }

    #[test]
    fn arcs_match_the_oracle_on_random_graphs() {
        for seed in 0..8u64 {
            check(&random_sparse(45, 140, 9, seed));
        }
    }

    #[test]
    fn arcs_match_the_oracle_with_parallel_edges() {
        let pairs: Vec<(u32, u32)> = (0..14u32).map(|u| (u, (u + 1) % 15)).collect();
        check(&with_parallel_edges(15, &pairs, 4, 9, 11));
    }

    #[test]
    fn arcs_match_the_oracle_with_self_loops() {
        check(&with_self_loops(grid(3, 5), &[0, 7, 14], 3.0));
    }

    #[test]
    fn arcs_match_the_oracle_on_a_disconnected_graph() {
        check(&disconnected_components(3, 10, 20, 7, 0xBEEF));
    }

    #[test]
    fn arcs_match_the_oracle_with_zero_weights() {
        check(&with_zero_weight_edge(random_sparse(25, 70, 5, 3), 0, 1));
        check(&zero_weight_cycle(8));
    }

    #[test]
    fn arcs_match_the_oracle_with_isolated_nodes() {
        check(&with_isolated_nodes(random_sparse(20, 55, 6, 2), 9));
    }

    #[test]
    fn arcs_match_the_oracle_on_stars_paths_and_ties() {
        check(&star(8, 2.5));
        check(&path(12));
        check(&tied_paths(4, 3.0));
    }

    /// The one-way case: an arc's two directions are independent, and a one-way street
    /// leaves the opposite direction at infinity unless a detour supplies it.
    #[test]
    fn a_one_way_edge_seeds_only_its_own_direction() {
        let mut edges = EdgeList::new(2);
        edges.push(0, 1, 4.0);
        let input = to_input(&edges);
        let cch = Cch::build(&input).expect("build");
        let topo = cch.topology();

        assert_eq!(topo.arc_count(), 1, "one edge, one arc");
        let slot = topo.arc_slot(0, 1).expect("the arc exists");
        let forward = topo.node_of(0) == 0;
        assert_eq!(
            cch.up_weight(slot),
            if forward { 4.0 } else { f64::INFINITY }
        );
        assert_eq!(
            cch.dn_weight(slot),
            if forward { f64::INFINITY } else { 4.0 }
        );
        assert_eq!(cch.up_mid(slot), None);
        assert_eq!(cch.dn_mid(slot), None);
        check_metric(&cch, &edges, &input);
    }

    /// Hand-checked: a 3-node path through a hub, eliminated hub-first so the shortcut is
    /// forced. 0 is the apex; 1-2 have no direct edge.
    #[test]
    fn a_shortcut_records_the_apex_it_routes_through() {
        let mut edges = EdgeList::new(3);
        for (u, v, w) in [(0, 1, 2.0), (1, 0, 2.0), (0, 2, 5.0), (2, 0, 5.0)] {
            edges.push(u, v, w);
        }
        let input = to_input(&edges);
        let topo = Topology::build_with(&input.skeleton, &IdentityOrdering).expect("phase 1");
        let metric = super::customize(&topo, &input.weights).expect("phase 2");

        // ranks are node ids under IdentityOrdering; eliminating 0 joins 1 and 2
        let slot = topo.arc_slot(1, 2).expect("fill-in arc 1-2 exists");
        assert_eq!(metric.up_w.get(slot), 7.0, "1 -> 0 -> 2");
        assert_eq!(metric.dn_w.get(slot), 7.0, "2 -> 0 -> 1");
        assert_eq!(metric.up_mid.get(slot), 0, "apex is rank 0");
        assert_eq!(metric.dn_mid.get(slot), 0);

        // the two original arcs keep their own weight and have nothing to unpack
        let a01 = topo.arc_slot(0, 1).expect("arc 0-1");
        assert_eq!(metric.up_w.get(a01), 2.0);
        assert_eq!(metric.up_mid.get(a01), NO_MID);
    }

    /// The node ceiling is checked before anything is allocated, so it can be tested with
    /// deliberately truncated arrays rather than 2^28 real nodes.
    #[test]
    fn a_graph_above_max_nodes_is_refused() {
        let huge = |n: usize| CchInput {
            n,
            skeleton: Csr {
                n,
                p: vec![0],
                i: Vec::new(),
            },
            weights: WeightCsr {
                csr: Csr {
                    n,
                    p: vec![0],
                    i: Vec::new(),
                },
                w: Vec::new(),
            },
        };
        assert_eq!(
            Cch::build(&huge(MAX_NODES + 1)).err(),
            Some(CchError::GraphTooLarge)
        );
        assert_eq!(
            Cch::build(&huge(u32::MAX as usize + 1)).err(),
            Some(CchError::GraphTooLarge)
        );
    }

    #[test]
    fn a_negative_weight_is_refused() {
        let mut edges = EdgeList::new(3);
        edges.push(0, 1, 1.0);
        edges.push(1, 2, -1.0);
        assert_eq!(
            Cch::build(&to_input(&edges)).err(),
            Some(CchError::InvalidWeight)
        );
    }

    #[test]
    fn a_nan_weight_is_refused() {
        let mut edges = EdgeList::new(2);
        edges.push(0, 1, f64::NAN);
        assert_eq!(
            Cch::build(&to_input(&edges)).err(),
            Some(CchError::InvalidWeight)
        );
    }

    #[test]
    fn an_infinite_weight_is_refused() {
        for w in [f64::INFINITY, f64::NEG_INFINITY] {
            let mut edges = EdgeList::new(2);
            edges.push(0, 1, w);
            assert_eq!(
                Cch::build(&to_input(&edges)).err(),
                Some(CchError::InvalidWeight)
            );
        }
    }

    /// `customize` is public, so weights for a different graph must be refused, never
    /// half-applied, never a panic. The last case is the dangerous one: a brand-new
    /// adjacency the hierarchy has no arc for, which only a rebuild can absorb.
    #[test]
    fn weights_shaped_for_another_graph_are_refused() {
        let mut cch = Cch::build(&to_input(&path(4))).expect("build");
        let before: Vec<f64> = cch.metric.up_w.iter().collect();
        let refused = Some(CchError::Internal(super::SHAPE_MISMATCH));

        let more_nodes = to_input(&path(5));
        assert_eq!(cch.customize(&more_nodes.weights).err(), refused);

        let mut short = to_input(&path(4));
        short.weights.w.pop();
        assert_eq!(cch.customize(&short.weights).err(), refused);

        let mut new_adjacency = path(4);
        new_adjacency.push(0, 3, 1.0);
        let new_adjacency = to_input(&new_adjacency);
        assert_eq!(cch.customize(&new_adjacency.weights).err(), refused);

        // offsets that go backwards would panic while slicing a row
        let mut backwards = to_input(&path(4));
        backwards.weights.csr.p[1..3].copy_from_slice(&[2, 1]);
        assert_eq!(cch.customize(&backwards.weights).err(), refused);

        let after: Vec<f64> = cch.metric.up_w.iter().collect();
        assert_eq!(before, after, "nothing was written");
    }

    /// A self-loop in `weights` is ignored, not asserted on.
    #[test]
    fn a_self_loop_in_the_weights_is_ignored() {
        let edges = grid(2, 3);
        let input = to_input(&edges);
        let mut with_loop = to_input(&edges);
        // splice a 0 -> 0 entry into row 0 of the weights, keeping it sorted
        let csr = &mut with_loop.weights.csr;
        csr.i.insert(0, 0);
        with_loop.weights.w.insert(0, 0.5);
        for p in &mut csr.p[1..] {
            *p += 1;
        }
        let topo = Topology::build(&input.skeleton).expect("phase 1");
        let plain = super::customize(&topo, &input.weights).expect("plain");
        let looped = super::customize(&topo, &with_loop.weights).expect("with a loop");
        assert_eq!(plain.up_w, looped.up_w);
        assert_eq!(plain.dn_w, looped.dn_w);
    }

    /// A refused metric must not leave a half-written one behind.
    #[test]
    fn a_refused_recustomization_leaves_the_old_metric_intact() {
        let good = to_input(&grid(3, 4));
        let mut cch = Cch::build(&good).expect("build");
        let before: Vec<f64> = cch.metric.up_w.iter().collect();

        let mut bad_edges = grid(3, 4);
        bad_edges.push(0, 1, -5.0);
        let bad = to_input(&bad_edges);
        assert_eq!(cch.customize(&bad.weights), Err(CchError::InvalidWeight));

        let after: Vec<f64> = cch.metric.up_w.iter().collect();
        assert_eq!(before, after);
    }

    #[test]
    fn the_empty_and_single_node_graphs_customize_to_nothing() {
        for edges in [empty(0), single_node(), empty(6)] {
            let input = to_input(&edges);
            let cch = Cch::build(&input).expect("build");
            assert_eq!(cch.node_count(), edges.n);
            assert_eq!(cch.topology().arc_count(), 0);
            assert!(cch.metric.up_w.is_empty());
            check_metric(&cch, &edges, &input);
        }
    }

    /// Recustomizing must land exactly where a fresh build on the new metric would. The
    /// two metrics share an edge *set*, so they share a skeleton, hence a topology, hence
    /// an arc-slot layout — which is what makes the arrays directly comparable.
    #[test]
    fn recustomizing_equals_a_fresh_build_with_the_new_metric() {
        let base = random_sparse(30, 90, 9, 17);
        let mut rescaled = EdgeList::new(base.n);
        for (i, &(u, v, w)) in base.edges.iter().enumerate() {
            rescaled.push(u, v, w * 2.0 + (i % 3) as f64);
        }

        let first = to_input(&base);
        let second = to_input(&rescaled);
        let mut cch = Cch::build(&first).expect("build");
        cch.customize(&second.weights).expect("recustomize");
        check_metric(&cch, &rescaled, &second);

        let fresh = Cch::build(&second).expect("fresh build");
        assert_eq!(cch.metric.up_w, fresh.metric.up_w);
        assert_eq!(cch.metric.dn_w, fresh.metric.dn_w);
        assert_eq!(cch.metric.up_mid, fresh.metric.up_mid);
        assert_eq!(cch.metric.dn_mid, fresh.metric.dn_mid);
    }

    #[test]
    fn the_same_input_always_produces_the_same_metric() {
        let input = to_input(&grid(4, 7));
        let a = Cch::build(&input).expect("a");
        let b = Cch::build(&input).expect("b");
        assert_eq!(a.metric.up_w, b.metric.up_w);
        assert_eq!(a.metric.dn_w, b.metric.dn_w);
        assert_eq!(a.metric.up_mid, b.metric.up_mid);
        assert_eq!(a.metric.dn_mid, b.metric.dn_mid);
    }

    #[test]
    fn memory_usage_counts_the_arrays_and_grows_with_the_graph() {
        let small = Cch::build(&to_input(&grid(3, 3))).expect("small");
        let big = Cch::build(&to_input(&grid(12, 12))).expect("big");
        assert!(small.memory_usage() > 0);
        assert!(big.memory_usage() > small.memory_usage());
    }

    #[test]
    fn weight_arr_round_trips_and_versions_independently() {
        let mut a = WeightArr::filled(4, 1.0_f64);
        assert_eq!(a.len(), 4);
        assert!(!a.is_empty());
        a.set(2, 9.0);
        assert_eq!(a.get(2), 9.0);

        let mut b = a.new_version();
        b.set(2, 3.0);
        assert_eq!(a.get(2), 9.0, "a version must not see the other's writes");
        assert_eq!(b.get(2), 3.0);
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![1.0, 1.0, 9.0, 1.0]);
        assert!(WeightArr::<f64>::filled(0, 0.0).is_empty());
    }
}
