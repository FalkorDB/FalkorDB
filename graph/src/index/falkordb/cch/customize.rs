//! Phase 2: customization (#3209, #3217).
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
//! * **rank space** — elimination order, [`Rank`]s `0..n`. Everything inside [`Cch`] except
//!   the order itself is in rank space.
//!
//! [`Topology::node_of`] and [`Topology::rank_of`] translate between them.
//!
//! ## Arc ids
//!
//! The chordal supergraph is held as upward rows over ranks ([`Topology::up`]). An **arc
//! id** (called a *slot* in the code) numbers the arcs row by row: rank `lower`'s arcs are
//! the slots [`Topology::up_range(lower)`](Topology::up_range), and
//!
//! ```text
//! slot up_range(lower).start + i   <->   the arc between ranks lower and up(lower)[i]
//! ```
//!
//! `lower < higher` always: a rank only lists its *upper* neighbors.
//! [`Topology::arc_slot`] is the reverse lookup: given an arc's two ends, in either order,
//! its slot.
//!
//! Slots are stable for the life of a [`Topology`]: they change only when the hierarchy is
//! rebuilt, never when it is recustomized.
//!
//! ## The four arrays
//!
//! All four are indexed by arc id and so all have [`Topology::arc_count`] entries. For slot
//! `k` joining ranks `lower` and `higher`:
//!
//! | array | meaning |
//! |---|---|
//! | `upward_weight[k]` | cheapest `lower -> higher` weight through nodes of rank `< lower` |
//! | `downward_weight[k]` | cheapest `higher -> lower` weight through nodes of rank `< lower` |
//! | `upward_middle[k]` | rank of the apex `upward_weight[k]` routes through, if any |
//! | `downward_middle[k]` | rank of the apex `downward_weight[k]` routes through, if any |
//!
//! "through nodes of rank `< lower`" is the whole invariant: an arc holds the shortest path
//! between its endpoints that uses *only strictly lower-ranked* intermediates. A weight of
//! `f64::INFINITY` means no such path exists — including for the arcs that exist purely as
//! fill-in and never acquire one, and for an edge whose own weight is `+inf`: a closed edge
//! keeps its arc but seeds it with "no path", exactly as an absent edge does. Sums stay in
//! `[0, +inf]` (`+inf + x = +inf`), and nothing here subtracts weights, so no NaN can
//! arise.
//!
//! `upward_middle[k] == None` means the arc's weight is its own original edge, with nothing
//! to unpack; otherwise the arc expands to `lower -> upward_middle[k] -> higher`, each half of
//! which is itself an arc to expand recursively. A middle rank is always strictly below
//! `lower`.
//! `Option<Rank>` is the size of a [`Rank`], so the middle arrays cost no more than a
//! sentinel would.
//!
//! ## Ties
//!
//! Every improvement is a strict `<`, so a tie never moves an arc:
//! - an original edge that a detour only ties keeps its own weight and records no middle
//!   rank;
//! - among detours that tie, the lowest apex (the first one the sweep reaches) is the middle rank.
//!
//! The result is therefore one exact set of arrays for a given metric and order, which is
//! what lets a scoped recustomization be compared bit for bit with a full one. Any other
//! version of the sweep (parallel, scoped) has to keep both rules.
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

#[cfg(test)]
use std::fmt::{self, Debug, Formatter};
use std::ops::Index;
use std::sync::Arc;

use super::order::{Rank, Topology};
use super::{Cch, CchError, CchInput, WeightCsr, index_id};
use crate::graph::graph::NodeId;

/// A versioned array of per-arc metric values: the storage of the four arrays of a
/// [`Metric`].
///
/// The four arrays are the only mutable state in a [`Cch`]; everything structural is shared
/// by `Arc`. A version must be cheap to take, because a writer takes one per write that
/// recustomizes, and cheap to hold, because readers keep old versions alive.
///
/// The slots live in fixed chunks of [`CHUNK_LEN`], each behind its own `Arc`, and the
/// chunk table is behind an `Arc` too (copy-on-write at two levels). Taking a version is
/// one `Arc` bump. The first write to a version clones the table, one `Arc` bump per
/// chunk; the first write to a chunk copies that chunk only. So a version owns the table
/// plus the chunks it wrote, a read is two indexed loads, and there is nothing to compact.
///
/// A read indexes it like a slice (`array[slot]`); a write goes through [`Self::set`],
/// which is where the copy happens.
#[derive(Clone)]
pub(crate) struct WeightArr<T: Copy> {
    chunks: Arc<[Arc<[T]>]>,
    len: usize,
}

/// Slots per chunk, as a power of two. A version's first write clones the table (one `Arc`
/// per chunk) and copies every chunk it touches, so the choice trades table size against
/// chunk size. Swept from 128 to 16,384 slots on the California road network (5.93M arcs):
/// below 1,024 the per-version table dominates (a single-edge SET cost 1.9 ms at 128 against
/// 0.23 ms at 1,024); on Linux 2,048 is a third faster than 1,024 (0.19 against 0.29 ms
/// server p50) and halves the median bytes a version owns (0.23 MB against 0.42 MB) and the
/// table (181 KB against 362 KB for the four arrays). An `f64` chunk (16 KB) still fits any
/// L1D. Larger chunks buy little latency (0.16 ms at 8,192) and raise the worst case a
/// version owns (2.8 MB at 8,192, 4.0 MB at 16,384, against 2.0 MB).
const CHUNK_BITS: usize = 11;
const CHUNK_LEN: usize = 1 << CHUNK_BITS;

/// Lists the slots; only the tests print one.
#[cfg(test)]
impl<T: Copy + Debug> Debug for WeightArr<T> {
    fn fmt(
        &self,
        formatter: &mut Formatter<'_>,
    ) -> fmt::Result {
        formatter.debug_list().entries(self.iter()).finish()
    }
}

/// Equal when every slot reads the same, however the chunks are shared. Only the tests
/// compare two.
#[cfg(test)]
impl<T: Copy + PartialEq> PartialEq for WeightArr<T> {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}

impl<T: Copy> Index<usize> for WeightArr<T> {
    type Output = T;

    /// # Panics
    /// If `slot` is out of range — an arc id that does not exist is a bug, not a miss.
    #[inline]
    fn index(
        &self,
        slot: usize,
    ) -> &T {
        &self.chunks[slot >> CHUNK_BITS][slot & (CHUNK_LEN - 1)]
    }
}

impl<T: Copy> WeightArr<T> {
    /// Cut a fully built array into chunks. The `Vec` is freed on return, so converting
    /// arrays one at a time holds at most one of them twice.
    fn from_vec(values: Vec<T>) -> Self {
        Self {
            chunks: values.chunks(CHUNK_LEN).map(Arc::from).collect(),
            len: values.len(),
        }
    }

    /// Writes into this version only: the table and the chunk holding `slot` are copied
    /// first if any other version shares them.
    ///
    /// # Panics
    /// If `slot` is out of range.
    #[inline]
    pub(crate) fn set(
        &mut self,
        slot: usize,
        value: T,
    ) {
        let chunk = &mut Arc::make_mut(&mut self.chunks)[slot >> CHUNK_BITS];
        Arc::make_mut(chunk)[slot & (CHUNK_LEN - 1)] = value;
    }

    /// A version that shares every chunk until either side writes.
    fn new_version(&self) -> Self {
        self.clone()
    }

    /// Heap bytes of the slots, the chunk table, and the two reference counts every `Arc`
    /// allocation carries. Shared chunks are counted in full, as every version can reach all
    /// of them. The struct itself is counted by whoever holds it.
    fn memory_usage(&self) -> usize {
        let counts = 2 * size_of::<usize>();
        self.len * size_of::<T>() + self.chunks.len() * (size_of::<Arc<[T]>>() + counts) + counts
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = T> + '_ {
        self.chunks.iter().flat_map(|chunk| chunk.iter().copied())
    }

    /// Heap bytes this version holds that `base` does not share with it: its own table
    /// (unless the two share one) and every chunk that is not the one `base` holds at the
    /// same position. What taking a version from `base` and writing it has cost in memory.
    #[cfg(test)]
    fn owned_bytes_over(
        &self,
        base: &Self,
    ) -> usize {
        let counts = 2 * size_of::<usize>();
        if Arc::ptr_eq(&self.chunks, &base.chunks) {
            return 0;
        }
        let table = self.chunks.len() * size_of::<Arc<[T]>>() + counts;
        let chunks: usize = self
            .chunks
            .iter()
            .enumerate()
            .filter(|&(position, chunk)| {
                base.chunks
                    .get(position)
                    .is_none_or(|base_chunk| !Arc::ptr_eq(base_chunk, chunk))
            })
            .map(|(_, chunk)| chunk.len() * size_of::<T>() + counts)
            .sum();
        table + chunks
    }
}

/// The four per-arc arrays a customization produces, indexed by arc id.
pub(crate) struct Metric {
    pub(crate) upward_weight: WeightArr<f64>,
    pub(crate) downward_weight: WeightArr<f64>,
    pub(crate) upward_middle: WeightArr<Option<Rank>>,
    pub(crate) downward_middle: WeightArr<Option<Rank>>,
}

impl Metric {
    /// A version whose four arrays share this one's chunks until either side writes.
    fn new_version(&self) -> Self {
        Self {
            upward_weight: self.upward_weight.new_version(),
            downward_weight: self.downward_weight.new_version(),
            upward_middle: self.upward_middle.new_version(),
            downward_middle: self.downward_middle.new_version(),
        }
    }

    /// Heap bytes of the four arrays; the inline headers are counted by the owning `Cch`.
    pub(crate) fn memory_usage(&self) -> usize {
        self.upward_weight.memory_usage()
            + self.downward_weight.memory_usage()
            + self.upward_middle.memory_usage()
            + self.downward_middle.memory_usage()
    }

    /// Heap bytes of the four arrays this metric does not share with `base`
    /// ([`WeightArr::owned_bytes_over`]).
    #[cfg(test)]
    pub(crate) fn owned_bytes_over(
        &self,
        base: &Self,
    ) -> usize {
        self.upward_weight.owned_bytes_over(&base.upward_weight)
            + self.downward_weight.owned_bytes_over(&base.downward_weight)
            + self.upward_middle.owned_bytes_over(&base.upward_middle)
            + self.downward_middle.owned_bytes_over(&base.downward_middle)
    }
}

/// The four arrays as flat boxed slices while a full customization rewrites every slot.
///
/// The relaxation sweep is bound by the latency of its scattered reads, and the chunk
/// lookup of a [`WeightArr`] adds a dependent load to each (measured on `cch/p14`: a
/// quarter slower on a 2.8M-arc grid, half slower through [`WeightArr::set`]). Building
/// flat and cutting into chunks at the end costs one copy of each array, taken one array at
/// a time.
struct Buffers {
    upward_weight: Box<[f64]>,
    downward_weight: Box<[f64]>,
    upward_middle: Box<[Option<Rank>]>,
    downward_middle: Box<[Option<Rank>]>,
}

impl Buffers {
    fn into_metric(self) -> Metric {
        Metric {
            upward_weight: WeightArr::from_vec(self.upward_weight.into_vec()),
            downward_weight: WeightArr::from_vec(self.downward_weight.into_vec()),
            upward_middle: WeightArr::from_vec(self.upward_middle.into_vec()),
            downward_middle: WeightArr::from_vec(self.downward_middle.into_vec()),
        }
    }
}

/// Whether `weights` has the shape every [`WeightCsr`] out of
/// [`CchInput::build`](super::CchInput) has: a well-formed CSR
/// ([`Csr::is_well_formed`](super::Csr::is_well_formed)), one weight per entry, and rows
/// strictly ascending (one entry per node pair). Never panics, whatever `weights` holds.
fn well_formed(weights: &WeightCsr) -> bool {
    let csr = &weights.csr;
    // Rows are sliced only once the offsets are known to be in range.
    csr.is_well_formed()
        && weights.values.len() == csr.neighbors.len()
        && (0..csr.node_count).all(|node| csr.row(node).windows(2).all(|pair| pair[0] < pair[1]))
}

/// Whether `weights` has the shape of weights for `topology`'s graph: steps 1 to 3 of
/// [`validate`], which looks at no weight value.
fn check_shape(
    topology: &Topology,
    weights: &WeightCsr,
) -> Result<(), CchError> {
    let csr = &weights.csr;
    if csr.node_count != topology.node_count() {
        return Err(CchError::NeedsRebuild);
    }
    if !well_formed(weights) {
        return Err(CchError::Internal("the weight CSR is malformed"));
    }
    let every_entry_has_an_arc = (0..csr.node_count).all(|row| {
        let u = index_id(row);
        let rank_u = topology.rank_of(NodeId::from(u));
        csr.row(row).iter().all(|&v| {
            // a self-loop is never seeded, so it needs no arc
            v == u
                || topology
                    .arc_slot(rank_u, topology.rank_of(NodeId::from(v)))
                    .is_some()
        })
    });
    if every_entry_has_an_arc {
        Ok(())
    } else {
        Err(CchError::NeedsRebuild)
    }
}

/// Reject a metric this index cannot be correct for, before any of it is written.
///
/// Dijkstra-style relaxation needs non-negative weights, and a NaN compares false with
/// everything, so it would silently drop out of every `<`. `+inf` is legal: it is a closed
/// edge, seeded exactly like an absent one ("no path"). The C implementation does not
/// check; refusing at build time is the contract decision recorded for this port.
///
/// The checks run in this order, each before anything reads what it guards, so none of
/// them panics, whatever `weights` holds. Steps 1 to 3 are [`check_shape`]:
/// 1. **The node count.** [`Cch::customized`] takes weights read from the graph as it is
///    now, which can be a later version than the one the hierarchy was built for. A graph
///    that has since gained nodes gives weights for more nodes:
///    [`CchError::NeedsRebuild`], expected on a live graph.
/// 2. **The shape** ([`well_formed`]). Every [`WeightCsr`] comes out of
///    [`CchInput::build`](super::CchInput) with it, so one without it is a bug,
///    [`CchError::Internal`], refused rather than indexed past an array's end. A repeated
///    pair would be seeded twice, the last weight silently winning.
/// 3. **An arc for every entry**: a new adjacency, an edge between two nodes the
///    hierarchy does not join, is the other "later graph" case, [`CchError::NeedsRebuild`].
///    It is checked before the weights, so weights with both a new adjacency and an invalid
///    weight ask for the rebuild, which then refuses that weight.
/// 4. **The weights:** [`CchError::InvalidWeight`].
fn validate(
    topology: &Topology,
    weights: &WeightCsr,
) -> Result<(), CchError> {
    check_shape(topology, weights)?;
    // A negative weight, `-inf` and NaN are refused; `+inf` and `-0.0` are not.
    if weights
        .values
        .iter()
        .any(|weight| weight.is_nan() || *weight < 0.0)
    {
        return Err(CchError::InvalidWeight);
    }
    Ok(())
}

impl WeightCsr {
    /// Every stored entry as `(row, column, weight)`, row by row, columns ascending within a
    /// row. Assumes the offsets are valid (`node_count + 1` of them, non-decreasing, the last
    /// at most `neighbors.len()`) and that `values` has one weight per entry; see
    /// [`validate`].
    pub(crate) fn entries(&self) -> impl Iterator<Item = (u64, u64, f64)> + '_ {
        (0..self.csr.node_count).flat_map(move |row| {
            let entries = self.csr.row_range(row);
            let u = index_id(row);
            self.csr.neighbors[entries.clone()]
                .iter()
                .zip(&self.values[entries])
                .map(move |(&v, &weight)| (u, v, weight))
        })
    }
}

/// Phase 2: seed every chordal arc from `weights`, then relax every lower triangle.
///
/// # Errors
/// [`CchError::InvalidWeight`] if any weight is negative or NaN (`+inf` is legal),
/// [`CchError::NeedsRebuild`] if `weights` come from a later graph (more nodes, or an edge
/// between two nodes the hierarchy has no arc for, which only a rebuild can add), and
/// [`CchError::Internal`] if `weights` is malformed (a bug). See [`validate`].
pub(crate) fn customize(
    topology: &Topology,
    weights: &WeightCsr,
) -> Result<Metric, CchError> {
    validate(topology, weights)?;

    let mut buffers = seed(topology, weights);
    relax(topology, &mut buffers);
    Ok(buffers.into_metric())
}

/// Original edges take their weight from `weights`; every other arc starts at infinity.
///
/// C reads this the other way round — two `extractElement`s per arc, `W(u, v)` and
/// `W(v, u)` — which costs two lookups for each of the many arcs that are pure fill-in and
/// will miss. Scattering the edges into infinity-filled arrays instead touches each *edge*
/// once, and the chordal supergraph has far more arcs than the graph has edges.
///
/// `weights` must have passed [`validate`], which found an arc for every entry.
fn seed(
    topology: &Topology,
    weights: &WeightCsr,
) -> Buffers {
    let arc_count = topology.arc_count();
    let mut metric = Buffers {
        upward_weight: vec![f64::INFINITY; arc_count].into_boxed_slice(),
        downward_weight: vec![f64::INFINITY; arc_count].into_boxed_slice(),
        upward_middle: vec![None; arc_count].into_boxed_slice(),
        downward_middle: vec![None; arc_count].into_boxed_slice(),
    };

    // Parallelisable, but not done: every stored entry (u, v) writes exactly one slot of
    // `upward_weight` (if `u` ranks below `v`) or `downward_weight` (if above), and the two
    // directions of a pair go to different arrays, so with duplicate-free rows no two entries
    // write the same location and the rows are independent. What stops a one-line change is
    // that the writes go through `&mut` arrays, which would have to be split by row or slot
    // range.
    for (u, v, weight) in weights.entries() {
        if u == v {
            continue; // a self-loop is never on a shortest path
        }
        let (rank_u, rank_v) = (topology.rank_of(u.into()), topology.rank_of(v.into()));
        let Some(slot) = topology.arc_slot(rank_u, rank_v) else {
            unreachable!("check_shape found an arc for every entry other than a self-loop");
        };

        if rank_u < rank_v {
            metric.upward_weight[slot] = weight;
        } else {
            metric.downward_weight[slot] = weight;
        }
    }

    metric
}

/// Basic customization: one forward sweep over the lower triangles.
///
/// See the module doc for why a single pass in increasing rank order suffices, and for the
/// tie rules the strict `<` below implements.
///
/// The top arc `(y, z)` of each triangle is found without a search: everything in `up(x)`
/// above `y` is also in `up(y)` (eliminating `x` joined them), and both rows are ascending,
/// so one cursor walks `up(y)` forward as `z` walks the rest of `up(x)`.
///
/// Parallelism, not done: iteration `x` only reads arcs whose lower end is `x`, which are
/// written only by iterations of lower ranks `x'` that `x` is an upper neighbour of, so it
/// has to run after those. Two iterations are therefore independent only if neither feeds
/// the other, but independence in that sense is not enough: iterations of different `x`
/// can also write the *same* arc `(y, z)` (any two common lower neighbours of `y` and `z`),
/// so running them concurrently would race on its weight and, on a tie, change which `x`
/// is recorded as the middle rank. A parallel version has to change the loop (for example, have
/// each arc pull the minimum over its own lower triangles, so each is written once) and
/// keep the tie rules. That has not been implemented or measured here.
///
/// # Panics
/// If the topology is not chordal (a `z` missing from `up(y)`), which [`Topology`]
/// guarantees by construction: a bug, never data.
fn relax(
    topology: &Topology,
    metric: &mut Buffers,
) {
    for x in topology.ranks() {
        let above_x = topology.up(x);
        let x_slots_start = topology.up_range(x).start;

        for (i, &y) in above_x.iter().enumerate() {
            // x -> y and y -> x. Both are final: the only writes below land on arcs whose
            // lower endpoint is y or above, never on x's own row.
            let x_to_y = metric.upward_weight[x_slots_start + i];
            let y_to_x = metric.downward_weight[x_slots_start + i];

            let above_y = topology.up(y);
            let y_slots_start = topology.up_range(y).start;
            let mut cursor = 0;

            // `above_x` is ascending, so everything past i is the z > y half of each pair,
            // and each unordered pair {y, z} is visited exactly once.
            for (j, &z) in above_x.iter().enumerate().skip(i + 1) {
                let x_to_z = metric.upward_weight[x_slots_start + j];
                let z_to_x = metric.downward_weight[x_slots_start + j];

                cursor += above_y[cursor..]
                    .iter()
                    .position(|&above| above == z)
                    .unwrap_or_else(|| {
                        panic!("chordal closure broken: no arc between {y:?} and {z:?}")
                    });
                let slot = y_slots_start + cursor;

                // y -> x -> z improves the upward arc
                let y_to_z_through_x = y_to_x + x_to_z;
                if y_to_z_through_x < metric.upward_weight[slot] {
                    metric.upward_weight[slot] = y_to_z_through_x;
                    metric.upward_middle[slot] = Some(x);
                }

                // z -> x -> y improves the downward arc
                let z_to_y_through_x = z_to_x + x_to_y;
                if z_to_y_through_x < metric.downward_weight[slot] {
                    metric.downward_weight[slot] = z_to_y_through_x;
                    metric.downward_middle[slot] = Some(x);
                }
            }
        }
    }
}

impl Cch {
    /// Phase 1 + Phase 2.
    ///
    /// # Errors
    /// [`CchError::InvalidWeight`] for a negative or NaN weight,
    /// [`CchError::Ordering`] if no elimination order can be produced, and
    /// [`CchError::Internal`] if `input` contradicts itself: a skeleton or weights for
    /// another node count than `input`'s, weights with an edge its skeleton lacks, or a
    /// malformed CSR. [`CchInput::build`] makes the weights and the skeleton from one graph,
    /// so each is a bug; never [`CchError::NeedsRebuild`], since this is the rebuild.
    pub(crate) fn build(input: &CchInput) -> Result<Self, CchError> {
        if input.skeleton.node_count != input.n || input.weights.csr.node_count != input.n {
            return Err(CchError::Internal(
                "the input's skeleton or weights are for another node count",
            ));
        }

        let topology = Arc::new(Topology::build(&input.skeleton)?);
        let metric = customize(&topology, &input.weights).map_err(|error| match error {
            CchError::NeedsRebuild => {
                CchError::Internal("the input's weights do not match its own skeleton")
            }
            other => other,
        })?;
        Ok(Self { topology, metric })
    }

    /// Phase 2 only: a hierarchy with the same structure, sharing this one's [`Topology`],
    /// and a new metric. This one is not touched, so nothing is written on error.
    ///
    /// The new metric is built while this one is still held: a customization briefly holds
    /// two metrics.
    ///
    /// # Errors
    /// [`CchError::InvalidWeight`] for a negative or NaN weight,
    /// [`CchError::NeedsRebuild`] if `weights` come from a later graph than the hierarchy — a
    /// different node count, or an edge between two nodes it has no arc for — and
    /// [`CchError::Internal`] if `weights` is malformed (a bug).
    pub fn customized(
        &self,
        weights: &WeightCsr,
    ) -> Result<Self, CchError> {
        Ok(Self {
            topology: Arc::clone(&self.topology),
            metric: customize(&self.topology, weights)?,
        })
    }

    /// A copy for a new graph version: the [`Topology`] is shared by `Arc`, and the metric
    /// through [`WeightArr::new_version`], which copies a chunk only when the copy writes
    /// it. Recustomizing the copy in place ([`Self::recustomize_scoped`]) leaves this one
    /// untouched, and costs the chunks it writes, not the metric.
    #[must_use]
    pub fn new_version(&self) -> Self {
        Self {
            topology: Arc::clone(&self.topology),
            metric: self.metric.new_version(),
        }
    }

    /// Heap bytes of the four weight arrays that this hierarchy holds and `base` does not
    /// share: zero for a fresh [`Self::new_version`] of `base`, then the chunk table and the
    /// chunks each write copied. The [`Topology`] is not counted (versions share it). For
    /// two unrelated hierarchies it is the whole metric. Only the tests call it.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn metric_bytes_over(
        &self,
        base: &Self,
    ) -> usize {
        self.metric.owned_bytes_over(&base.metric)
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.topology.node_count()
    }

    /// Bytes held by this hierarchy: the struct itself, the shared [`Topology`], and the
    /// four weight arrays. Two versions sharing one `Topology` each report its bytes, as
    /// [`VersionedIndex::memory_usage`](super::super::versioned::VersionedIndex::memory_usage)
    /// counts a shared page.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        // `size_of::<Self>()` already covers `Metric`'s four inline headers
        size_of::<Self>() + self.topology.memory_usage() + self.metric.memory_usage()
    }

    pub(crate) fn topology(&self) -> &Topology {
        &self.topology
    }

    /// Cheapest `lower -> higher` weight of arc `slot`, through strictly lower-ranked nodes only.
    pub(crate) fn upward_weight(
        &self,
        slot: usize,
    ) -> f64 {
        self.metric.upward_weight[slot]
    }

    /// Cheapest `higher -> lower` weight of arc `slot`.
    pub(crate) fn downward_weight(
        &self,
        slot: usize,
    ) -> f64 {
        self.metric.downward_weight[slot]
    }

    /// The apex rank `upward_weight(slot)` routes through, or `None` if the arc is just its
    /// original edge.
    pub(crate) fn upward_middle(
        &self,
        slot: usize,
    ) -> Option<Rank> {
        self.metric.upward_middle[slot]
    }

    /// The apex rank `downward_weight(slot)` routes through, or `None`.
    pub(crate) fn downward_middle(
        &self,
        slot: usize,
    ) -> Option<Rank> {
        self.metric.downward_middle[slot]
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::graph::NodeId;
    use crate::index::falkordb::cch::order::{Rank, Topology};
    use std::collections::HashMap;
    use std::fmt::Debug;
    use std::sync::Arc;

    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::{CHUNK_LEN, WeightArr};

    use crate::graph::graphblas::matrix::Matrix;
    use crate::index::falkordb::cch::testkit::{
        all_pairs, assert_input_invariants, disconnected_components, empty, entries, grid,
        insert_weight_entry, lower_rank_distances, node_count, path, random_sparse, reweight,
        single_node, star, tied_paths, to_input, weights, with_isolated_nodes,
        with_zero_weight_edge, zero_weight_cycle,
    };
    use crate::index::falkordb::cch::{Cch, CchError, CchInput};

    /// Node-id order, `rank_to_node[rank] = rank`, over `node_count` nodes.
    fn node_id_order(node_count: u64) -> Vec<NodeId> {
        (0..node_count).map(NodeId::from).collect()
    }

    /// The weight of the original `u -> v` edge in node space, or infinity if there is
    /// none. What an arc must hold when it records no middle rank.
    fn original(
        input: &CchInput,
        u: u64,
        v: u64,
    ) -> f64 {
        let csr = &input.weights.csr;
        let row = csr.row_offsets[u as usize] as usize..csr.row_offsets[u as usize + 1] as usize;
        row.clone()
            .find(|&k| csr.neighbors[k] == v)
            .map_or(f64::INFINITY, |k| input.weights.values[k])
    }

    /// The weight of each detour `from -> x -> to` through a common lower neighbor `x` of
    /// `from` and `to`, apex first, lowest apex first.
    fn detours(
        cch: &Cch,
        from: Rank,
        to: Rank,
    ) -> Vec<(Rank, f64)> {
        let topology = cch.topology();
        let below_to = topology.down(to);
        topology
            .down(from)
            .iter()
            .filter(|x| below_to.binary_search(x).is_ok())
            .map(|&x| {
                let into_x = topology.arc_slot(x, from).expect("x is below from");
                let out_of_x = topology.arc_slot(x, to).expect("x is below to");
                (x, cch.downward_weight(into_x) + cch.upward_weight(out_of_x))
            })
            .collect()
    }

    /// The tie rules (module doc, "Ties") for one arc direction: a recorded middle rank strictly
    /// beats the original edge, and no lower apex ties it.
    fn check_tie_rules(
        cch: &Cch,
        (from, to): (Rank, Rank),
        weight: f64,
        middle: Option<Rank>,
        original: f64,
    ) {
        let Some(middle) = middle else { return };
        assert!(
            weight < original,
            "arc {from:?}->{to:?} routes through {middle:?} for {weight}, which only ties its own \
             edge ({original}): the edge must win a tie"
        );
        for (x, through_x) in detours(cch, from, to) {
            assert!(
                x >= middle || through_x > weight,
                "arc {from:?}->{to:?} routes through {middle:?}, but the lower apex {x:?} ties it"
            );
        }
    }

    /// Every arc holds exactly the lower-restricted shortest path between its endpoints —
    /// checked against the kit's `lower_rank_distances`, which derives that quantity from
    /// the edge list and the elimination order alone and shares no code with the triangle
    /// sweep it is checking. Each middle rank obeys the tie rules: the result is one exact set of
    /// arrays, not any of several that answer alike.
    fn check_metric(
        cch: &Cch,
        edges: &Matrix<f64>,
        input: &CchInput,
    ) {
        let topology = cch.topology();
        let expected = lower_rank_distances(edges, &topology.rank_to_node_ids());

        for lower in topology.ranks() {
            let range = topology.up_range(lower);
            for (i, &higher) in topology.up(lower).iter().enumerate() {
                let slot = range.start + i;
                let (u, v) = (
                    u64::from(topology.node_of(lower)),
                    u64::from(topology.node_of(higher)),
                );

                assert_eq!(
                    cch.upward_weight(slot),
                    expected[u as usize][v as usize],
                    "arc {u}->{v} (ranks {lower:?}->{higher:?}, slot {slot}) upward weight"
                );
                assert_eq!(
                    cch.downward_weight(slot),
                    expected[v as usize][u as usize],
                    "arc {v}->{u} (ranks {higher:?}->{lower:?}, slot {slot}) downward weight"
                );

                // The middle rank is what the query change (#3210) unpacks a shortcut with, so
                // it has to be more than plausible: the recorded detour must reproduce the
                // arc's own weight.
                if let Some(middle) = cch.upward_middle(slot) {
                    assert!(
                        middle < lower,
                        "arc {lower:?}-{higher:?} routes through rank {middle:?}, not below it"
                    );
                    let to_mid = topology
                        .arc_slot(middle, lower)
                        .expect("middle is adjacent to lower");
                    let from_mid = topology
                        .arc_slot(middle, higher)
                        .expect("middle is adjacent to higher");
                    assert_eq!(
                        cch.upward_weight(slot),
                        cch.downward_weight(to_mid) + cch.upward_weight(from_mid),
                        "arc {lower:?}->{higher:?} does not equal its recorded detour via {middle:?}"
                    );
                } else {
                    assert_eq!(
                        cch.upward_weight(slot),
                        original(input, u, v),
                        "arc {lower:?}->{higher:?} has no middle rank, so it must be its own edge"
                    );
                }
                check_tie_rules(
                    cch,
                    (lower, higher),
                    cch.upward_weight(slot),
                    cch.upward_middle(slot),
                    original(input, u, v),
                );

                if let Some(middle) = cch.downward_middle(slot) {
                    assert!(
                        middle < lower,
                        "arc {higher:?}-{lower:?} routes through rank {middle:?}, not below it"
                    );
                    let to_mid = topology
                        .arc_slot(middle, higher)
                        .expect("middle is adjacent to higher");
                    let from_mid = topology
                        .arc_slot(middle, lower)
                        .expect("middle is adjacent to lower");
                    assert_eq!(
                        cch.downward_weight(slot),
                        cch.downward_weight(to_mid) + cch.upward_weight(from_mid),
                        "arc {higher:?}->{lower:?} does not equal its recorded detour via {middle:?}"
                    );
                } else {
                    assert_eq!(
                        cch.downward_weight(slot),
                        original(input, v, u),
                        "arc {higher:?}->{lower:?} has no middle rank, so it must be its own edge"
                    );
                }
                check_tie_rules(
                    cch,
                    (higher, lower),
                    cch.downward_weight(slot),
                    cch.downward_middle(slot),
                    original(input, v, u),
                );
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
        edges: &Matrix<f64>,
    ) {
        let truth = all_pairs(edges);
        let topology = cch.topology();

        // Each shortcut as an edge of its own, next to the original one: the cheaper of the
        // two is the pair's weight.
        let mut augmented: HashMap<(u64, u64), f64> = entries(edges)
            .into_iter()
            .map(|(u, v, w)| ((u, v), w))
            .collect();
        let mut add = |u: u64, v: u64, w: f64| {
            let cur = augmented.entry((u, v)).or_insert(f64::INFINITY);
            *cur = cur.min(w);
        };
        for lower in topology.ranks() {
            let range = topology.up_range(lower);
            for (i, &higher) in topology.up(lower).iter().enumerate() {
                let slot = range.start + i;
                let (u, v) = (
                    u64::from(topology.node_of(lower)),
                    u64::from(topology.node_of(higher)),
                );
                if cch.upward_weight(slot).is_finite() {
                    add(u, v, cch.upward_weight(slot));
                }
                if cch.downward_weight(slot).is_finite() {
                    add(v, u, cch.downward_weight(slot));
                }
            }
        }

        let n = node_count(edges);
        let augmented: Vec<_> = augmented.into_iter().map(|((u, v), w)| (u, v, w)).collect();
        let with_shortcuts = all_pairs(&weights(n, &augmented));
        for s in 0..n {
            for t in 0..n {
                assert_eq!(
                    with_shortcuts[s][t], truth[s][t],
                    "shortcuts changed the {s}->{t} distance"
                );
            }
        }
    }

    fn check(edges: &Matrix<f64>) {
        let input = to_input(edges);
        assert_input_invariants(&input);
        let cch = Cch::build(&input).expect("build");
        assert_eq!(cch.node_count(), node_count(edges));
        check_metric(&cch, edges, &input);
        check_no_shortcut_beats_reality(&cch, edges);
    }

    /// Every arc against the oracle on the kit's graph families: a grid, random graphs, a
    /// disconnected graph, a zero-weight edge and a zero-weight cycle, isolated nodes, a
    /// star, a path and tied paths.
    #[test]
    fn arcs_match_the_oracle_on_kit_graphs() {
        let mut graphs = vec![
            grid(5, 6),
            disconnected_components(3, 10, 20, 7, 0xBEEF),
            with_zero_weight_edge(random_sparse(25, 70, 5, 3), 0, 1),
            zero_weight_cycle(8),
            with_isolated_nodes(random_sparse(20, 55, 6, 2), 9),
            star(8, 2.5),
            path(12),
            tied_paths(4, 3.0),
        ];
        graphs.extend((0..8u64).map(|seed| random_sparse(45, 140, 9, seed)));
        for edges in &graphs {
            check(edges);
        }
    }

    /// Hand-checked: a 3-node path through a hub, eliminated hub-first so the shortcut is
    /// forced. 0 is the apex; 1-2 have no direct edge.
    #[test]
    fn a_shortcut_records_the_apex_it_routes_through() {
        let edges = weights(3, &[(0, 1, 2.0), (1, 0, 2.0), (0, 2, 5.0), (2, 0, 5.0)]);
        let input = to_input(&edges);
        let topology = Topology::from_order(&input.skeleton, node_id_order(3)).expect("phase 1");
        let metric = super::customize(&topology, &input.weights).expect("phase 2");
        let rank = Rank::new;

        // ranks are node ids in node-id order; eliminating 0 joins 1 and 2
        let slot = topology
            .arc_slot(rank(1), rank(2))
            .expect("fill-in arc 1-2 exists");
        assert_eq!(metric.upward_weight[slot], 7.0, "1 -> 0 -> 2");
        assert_eq!(metric.downward_weight[slot], 7.0, "2 -> 0 -> 1");
        assert_eq!(metric.upward_middle[slot], Some(rank(0)), "apex is rank 0");
        assert_eq!(metric.downward_middle[slot], Some(rank(0)));

        // the two original arcs keep their own weight and have nothing to unpack
        let a01 = topology.arc_slot(rank(0), rank(1)).expect("arc 0-1");
        assert_eq!(metric.upward_weight[a01], 2.0);
        assert_eq!(metric.upward_middle[a01], None);
    }

    /// Hand-checked tie: 2 and 3 have no edge, and are joined at weight 2 through apex 0
    /// and through apex 1 alike. The lowest apex stays the middle rank, the rule `relax` says a
    /// parallel version has to keep.
    #[test]
    fn a_tie_keeps_the_lowest_apex_as_the_mid() {
        let mut both_ways = Vec::new();
        for (u, v) in [(0, 2), (0, 3), (1, 2), (1, 3)] {
            both_ways.extend([(u, v, 1.0), (v, u, 1.0)]);
        }
        let input = to_input(&weights(4, &both_ways));
        let topology = Topology::from_order(&input.skeleton, node_id_order(4)).expect("phase 1");
        let metric = super::customize(&topology, &input.weights).expect("phase 2");
        let rank = Rank::new;

        let slot = topology
            .arc_slot(rank(2), rank(3))
            .expect("fill-in arc 2-3 exists");
        assert_eq!(metric.upward_weight[slot], 2.0);
        assert_eq!(metric.downward_weight[slot], 2.0);
        assert_eq!(
            metric.upward_middle[slot],
            Some(rank(0)),
            "2 -> 3 ties via 0 and 1: the lowest apex"
        );
        assert_eq!(
            metric.downward_middle[slot],
            Some(rank(0)),
            "3 -> 2 ties via 0 and 1: the lowest apex"
        );
    }

    /// Hand-checked tie with the original edge: 1-2 weighs 2 directly and 2 through apex 0
    /// (0-1 and 0-2 weigh 1 each). The edge wins: the arc keeps weight 2 and records no
    /// middle rank, the rule `relax`'s strict `<` implements and a scoped or parallel version
    /// has to keep.
    #[test]
    fn a_tie_with_the_original_edge_keeps_the_edge() {
        let mut both_ways = Vec::new();
        for (u, v, w) in [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 2.0)] {
            both_ways.extend([(u, v, w), (v, u, w)]);
        }
        let input = to_input(&weights(3, &both_ways));
        let topology = Topology::from_order(&input.skeleton, node_id_order(3)).expect("phase 1");
        let metric = super::customize(&topology, &input.weights).expect("phase 2");
        let rank = Rank::new;

        let slot = topology.arc_slot(rank(1), rank(2)).expect("arc 1-2");
        assert_eq!(metric.upward_weight[slot], 2.0);
        assert_eq!(metric.downward_weight[slot], 2.0);
        assert_eq!(
            metric.upward_middle[slot], None,
            "1 -> 2: the edge wins the tie"
        );
        assert_eq!(
            metric.downward_middle[slot], None,
            "2 -> 1: the edge wins the tie"
        );
    }

    /// A negative, NaN or -inf weight is refused. `-0.0` is `>= 0`, so it is a weight like
    /// `0.0`.
    #[test]
    fn bad_weights_are_refused_and_negative_zero_is_a_zero_weight() {
        for (case, w) in [
            ("negative", -1.0),
            ("NaN", f64::NAN),
            ("-inf", f64::NEG_INFINITY),
        ] {
            let edges = weights(3, &[(0, 1, 1.0), (1, 2, w)]);
            assert_eq!(
                Cch::build(&to_input(&edges)).err(),
                Some(CchError::InvalidWeight),
                "{case}"
            );
        }

        let edges = weights(2, &[(0, 1, -0.0)]);
        let input = to_input(&edges);
        let cch = Cch::build(&input).expect("-0.0 is legal");
        check_metric(&cch, &edges, &input);
    }

    /// Closing an edge is the same metric as removing it, on the kit's graph families,
    /// every arc compared, middle ranks included: closing every edge of a random third of the
    /// pairs, then customizing that same topology without those edges.
    #[test]
    fn closing_an_edge_gives_the_metric_of_removing_it() {
        for (idx, open) in [grid(4, 5), random_sparse(40, 120, 9, 3), tied_paths(4, 2.0)]
            .into_iter()
            .enumerate()
        {
            let n = node_count(&open);
            let closed = reweight(&open, |k, w| if k % 3 == 0 { f64::INFINITY } else { w });
            let kept: Vec<_> = entries(&open)
                .into_iter()
                .enumerate()
                .filter(|&(k, _)| k % 3 != 0)
                .map(|(_, e)| e)
                .collect();
            let removed = weights(n, &kept);
            let closed_input = to_input(&closed);
            let topology = Topology::build(&closed_input.skeleton).expect("phase 1");
            let with_closed = super::customize(&topology, &closed_input.weights).expect("closed");
            let cch = Cch {
                topology: topology.into(),
                metric: with_closed,
            };
            check_metric(&cch, &closed, &closed_input);

            // the oracle over the graph without the closed edges, on the same order
            let expected = lower_rank_distances(&removed, &cch.topology().rank_to_node_ids());
            let topology = cch.topology();
            for lower in topology.ranks() {
                let range = topology.up_range(lower);
                for (i, &higher) in topology.up(lower).iter().enumerate() {
                    let (u, v) = (
                        u64::from(topology.node_of(lower)),
                        u64::from(topology.node_of(higher)),
                    );
                    assert_eq!(
                        cch.upward_weight(range.start + i),
                        expected[u as usize][v as usize],
                        "graph #{idx}: arc {u}->{v}"
                    );
                    assert_eq!(
                        cch.downward_weight(range.start + i),
                        expected[v as usize][u as usize],
                        "graph #{idx}: arc {v}->{u}"
                    );
                }
            }
        }
    }

    /// An input that contradicts its own node count is a bug in whatever built it: `build`
    /// refuses it as [`CchError::Internal`], in debug and release builds alike, before
    /// anything is ordered or indexed. Never [`CchError::NeedsRebuild`]: this is the rebuild.
    #[test]
    fn build_refuses_an_input_for_another_node_count() {
        let internal = |what: &str, input: &CchInput| {
            assert!(
                matches!(Cch::build(input), Err(CchError::Internal(_))),
                "{what}"
            );
        };

        let mut weights_too_wide = to_input(&path(4));
        weights_too_wide.weights = to_input(&path(5)).weights;
        internal("weights for 5 nodes, input for 4", &weights_too_wide);

        let mut skeleton_too_wide = to_input(&path(4));
        skeleton_too_wide.skeleton = to_input(&path(5)).skeleton;
        internal("a skeleton for 5 nodes, input for 4", &skeleton_too_wide);

        let mut count_too_small = to_input(&path(4));
        count_too_small.n = 3;
        internal("both for 4 nodes, input for 3", &count_too_small);
    }

    /// The degenerate graphs: no node at all, and one node with no edge. Both build and
    /// customize to no arcs.
    #[test]
    fn an_empty_and_a_single_node_graph_customize_to_no_arcs() {
        for (what, edges) in [("no nodes", empty(0)), ("a single node", single_node())] {
            let input = to_input(&edges);
            let cch = Cch::build(&input).expect(what);
            assert_eq!(cch.topology().arc_count(), 0, "{what}");
            let again = cch.customized(&input.weights).expect(what);
            assert_eq!(again.node_count(), node_count(&edges), "{what}");
        }
    }

    /// A self-loop in `weights` is ignored, not asserted on.
    #[test]
    fn a_self_loop_in_the_weights_is_ignored() {
        let edges = grid(2, 3);
        let input = to_input(&edges);
        let mut with_loop = to_input(&edges);
        insert_weight_entry(&mut with_loop, 0, 0, 0.5);
        let topology = Topology::build(&input.skeleton).expect("phase 1");
        let plain = super::customize(&topology, &input.weights).expect("plain");
        let looped = super::customize(&topology, &with_loop.weights).expect("with a loop");
        assert_eq!(plain.upward_weight, looped.upward_weight);
        assert_eq!(plain.downward_weight, looped.downward_weight);
    }

    /// Recustomizing must land exactly where a fresh build on the new metric would. The
    /// two metrics share an edge *set*, so they share a skeleton, hence a topology, hence
    /// an arc-slot layout — which is what makes the arrays directly comparable. The new
    /// hierarchy shares the topology, and the one it came from keeps its own metric.
    #[test]
    fn recustomizing_equals_a_fresh_build_with_the_new_metric() {
        let base = random_sparse(30, 90, 9, 17);
        let rescaled = reweight(&base, |i, w| w * 2.0 + (i % 3) as f64);

        let first = to_input(&base);
        let second = to_input(&rescaled);
        let built = Cch::build(&first).expect("build");
        let cch = built.customized(&second.weights).expect("recustomize");
        assert!(
            Arc::ptr_eq(&cch.topology, &built.topology),
            "the topology is shared, not rebuilt"
        );
        check_metric(&cch, &rescaled, &second);

        let fresh = Cch::build(&second).expect("fresh build");
        assert_eq!(cch.metric.upward_weight, fresh.metric.upward_weight);
        assert_eq!(cch.metric.downward_weight, fresh.metric.downward_weight);
        assert_eq!(cch.metric.upward_middle, fresh.metric.upward_middle);
        assert_eq!(cch.metric.downward_middle, fresh.metric.downward_middle);
        // ... and the arrays did change, while the hierarchy it came from kept its own
        assert_ne!(cch.metric.upward_weight, built.metric.upward_weight);
        check_metric(&built, &base, &first);
    }

    /// What `customized` says for each kind of weights over the hierarchy of `path(4)`: any
    /// values over the same adjacency are taken (or refused for a bad value only), weights
    /// from a later graph ask for a rebuild, never half-applied, and a malformed CSR is a bug
    /// (each malformed case breaks one rule of the shape), never a rebuild request, a panic
    /// or a silently wrong metric. The shape is checked before the values, so a new
    /// adjacency with a bad weight asks for the rebuild.
    #[test]
    fn customized_answers_each_kind_of_weights_with_its_own_outcome() {
        let cch = Cch::build(&to_input(&path(4))).expect("build");

        let own = to_input(&path(4));
        let reweighted = to_input(&reweight(&path(4), |_, w| w + 3.0));
        let mut negative = path(4);
        let (u, v, _) = entries(&negative)[0];
        negative.set(u, v, -1.0);
        let negative = to_input(&negative);
        // `to_input` drops self-loops, so this one goes in by hand
        let mut looped = to_input(&path(4));
        insert_weight_entry(&mut looped, 2, 2, 1.0);

        let more_nodes = to_input(&path(5));
        let with_new_adjacency = |weight_of_1_2: f64| {
            let mut edges = path(4);
            edges.set(0, 3, 1.0);
            edges.set(1, 2, weight_of_1_2);
            to_input(&edges)
        };
        let new_adjacency = with_new_adjacency(1.0);
        let new_adjacency_and_negative = with_new_adjacency(-1.0);

        let mut short = to_input(&path(4));
        short.weights.values = short.weights.values[1..].into();
        let mut bad_offsets = to_input(&path(4));
        bad_offsets.weights.csr.row_offsets.swap(1, 2);
        // row 1 is [0, 2]: repeating its 0 is a second weight for the pair 1 -> 0
        // the first entry would be skipped, its weight silently dropped
        let mut late_start = to_input(&path(4));
        late_start.weights.csr.row_offsets[0] = 1;
        // node 4 does not exist: seeding it would index past the end of `node_to_rank`
        let mut past_the_end = to_input(&path(4));
        let last = past_the_end.weights.csr.neighbors.len() - 1;
        past_the_end.weights.csr.neighbors[last] = 4;
        let mut repeated = to_input(&path(4));
        insert_weight_entry(&mut repeated, 1, 0, 7.0);

        const REBUILD: Option<CchError> = Some(CchError::NeedsRebuild);
        const INTERNAL: Option<CchError> = Some(CchError::Internal("the weight CSR is malformed"));
        for (what, input, expected) in [
            ("its own weights", &own, None),
            ("new values, same edges", &reweighted, None),
            ("a self-loop needs no arc", &looped, None),
            ("a negative value", &negative, Some(CchError::InvalidWeight)),
            ("one node more", &more_nodes, REBUILD),
            ("an adjacency with no arc", &new_adjacency, REBUILD),
            (
                "a new adjacency and a negative value",
                &new_adjacency_and_negative,
                REBUILD,
            ),
            ("a short weight array", &short, INTERNAL),
            ("decreasing offsets", &bad_offsets, INTERNAL),
            ("offsets that start past 0", &late_start, INTERNAL),
            ("a neighbor past the node count", &past_the_end, INTERNAL),
            ("a repeated pair", &repeated, INTERNAL),
        ] {
            assert_eq!(cch.customized(&input.weights).err(), expected, "{what}");
        }
    }

    /// A version is a plain `Vec` model: every slot must read as the model says.
    fn assert_reads<T: Copy + PartialEq + Debug>(
        array: &WeightArr<T>,
        model: &[T],
    ) {
        assert_eq!(array.len(), model.len());
        assert_eq!(array.iter().collect::<Vec<_>>(), model);
        for (slot, &value) in model.iter().enumerate() {
            assert_eq!(array[slot], value, "slot {slot}");
        }
    }

    /// Slots on both sides of every chunk boundary, plus the first and the last.
    fn edge_slots(len: usize) -> Vec<usize> {
        let mut slots: Vec<usize> = (0..=len / CHUNK_LEN)
            .flat_map(|chunk| [chunk * CHUNK_LEN, (chunk * CHUNK_LEN).wrapping_sub(1)])
            .chain([0, len.wrapping_sub(1)])
            .filter(|&slot| slot < len)
            .collect();
        slots.sort_unstable();
        slots.dedup();
        slots
    }

    /// `from_vec` cuts at every chunk boundary and keeps a partial last chunk; writes on
    /// both sides of every boundary land where they should.
    #[test]
    fn weight_arr_handles_lengths_that_are_not_a_multiple_of_the_chunk() {
        for len in [
            0,
            1,
            CHUNK_LEN - 1,
            CHUNK_LEN,
            CHUNK_LEN + 1,
            3 * CHUNK_LEN,
            3 * CHUNK_LEN + 7,
        ] {
            let mut model: Vec<Option<Rank>> = vec![None; len];
            let mut array = WeightArr::from_vec(model.clone());
            assert_reads(&array, &model);
            assert_eq!(array.chunks.len(), len.div_ceil(CHUNK_LEN), "len {len}");
            for slot in edge_slots(len) {
                array.set(slot, Some(Rank::new(slot)));
                model[slot] = Some(Rank::new(slot));
            }
            assert_reads(&array, &model);
        }
    }

    /// The point of the chunks: a write copies the chunk it lands in and nothing else, so
    /// a version owns the chunks it wrote, not the array.
    #[test]
    fn a_write_copies_only_the_chunk_it_lands_in() {
        let len = 4 * CHUNK_LEN + 9;
        let base = WeightArr::from_vec((0..len).map(|slot| slot as f64).collect());
        let mut version = base.new_version();
        assert!(
            Arc::ptr_eq(&base.chunks, &version.chunks),
            "taking a version copies nothing"
        );

        version.set(2 * CHUNK_LEN + 1, -1.0);
        version.set(2 * CHUNK_LEN + 2, -2.0); // same chunk: already the version's own
        let shared: Vec<bool> = base
            .chunks
            .iter()
            .zip(version.chunks.iter())
            .map(|(left, right)| Arc::ptr_eq(left, right))
            .collect();
        assert_eq!(shared, [true, true, false, true, true]);
        assert_eq!(base[2 * CHUNK_LEN + 1], (2 * CHUNK_LEN + 1) as f64);
        assert_eq!(version[2 * CHUNK_LEN + 1], -1.0);
    }

    /// Random writes into a random family of versions, each checked against its own `Vec`.
    #[test]
    fn random_writes_across_versions_match_a_vec_model() {
        let len = 3 * CHUNK_LEN + 11;
        let mut rng = SmallRng::seed_from_u64(1);
        let mut versions = vec![(WeightArr::from_vec(vec![0_u64; len]), vec![0_u64; len])];
        for step in 0..4000 {
            let pick = rng.random_range(0..versions.len());
            if step % 50 == 0 {
                let (array, model) = &versions[pick];
                let fork = (array.new_version(), model.clone());
                versions.push(fork);
                continue;
            }
            let slot = rng.random_range(0..len);
            let (array, model) = &mut versions[pick];
            array.set(slot, step);
            model[slot] = step;
        }
        for (array, model) in &versions {
            assert_reads(array, model);
        }
    }

    /// The same through a whole hierarchy: a `new_version` owns none of the metric until it
    /// is written, and a fresh build owns all of it.
    #[test]
    fn a_new_version_owns_no_metric_until_it_is_written() {
        let edges = grid(40, 40);
        let input = to_input(&edges);
        let cch = Cch::build(&input).expect("build");
        let copy = cch.new_version();
        assert_eq!(copy.metric_bytes_over(&cch), 0);
        let other = Cch::build(&input).expect("another build");
        assert_eq!(other.metric_bytes_over(&cch), cch.metric.memory_usage());

        let mut written = cch.new_version();
        written.metric.upward_weight.set(0, 0.5);
        assert!(written.metric_bytes_over(&cch) > 0);
        assert_eq!(cch.new_version().metric_bytes_over(&cch), 0);
    }
}
