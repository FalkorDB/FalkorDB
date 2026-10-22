//! Customizable Contraction Hierarchies (CCH) path index.
//!
//! A graph version holds its indexes as `Arc<Cch>`, in its native index state
//! ([`FalkorDbIndex`]); the `Graph` methods in `graph.rs` create, drop and list them.
//!
//! Layout (one file per area, so changes to different areas do not touch each other's files):
//! - `order.rs`     Phase 1: nested-dissection ordering and triangulation (#3208)
//! - `nd.rs`        the nested-dissection call behind Phase 1's ordering
//! - `customize.rs` Phase 2: customization (#3209), and `WeightArr`, the metric's storage
//! - `partial.rs`   scoped recustomization (#3217)
//! - `query.rs`     bidirectional upward search and path unpacking (#3210)
//! - `input.rs`     `Graph` -> [`CchInput`] (#3211)
//! - `maint.rs`     keeping the indexes correct after writes (#3214), and the write-time weight check
//! - `procedure.rs` `db.idx.cch.query` and the CCH rows of `db.indexes()`
//! - `fallback.rs`  the graph search a query falls back to while the hierarchy cannot answer
//! - `testkit.rs`   test fixtures and oracles (#3207), in test builds only
//!
//! The types themselves live here in `mod.rs` because every part needs them; the other
//! files are where the `impl` blocks land.

mod customize;
pub(crate) mod fallback;
mod input;
pub(crate) mod maint;
mod nd;
mod order;
mod partial;
pub(crate) mod procedure;
mod query;
#[cfg(test)]
pub(crate) mod testkit;

pub(crate) use maint::CchDirty;

use std::ops::Range;
use std::sync::Arc;

use roaring::RoaringTreemap;

use super::falkordb_index::FalkorDbIndex;
use super::online_build;
use super::versioned::{Absorbed, BuildState, IndexEntry, VersionedIndex};
use crate::graph::graph::{Graph, TypeId};
use crate::graph::graphblas::matrix::CsrExport;
use crate::graph::graphblas::tensor::GrB_INDEX_MAX;
use customize::Metric;
use order::Topology;
use thiserror::Error;

/// The largest node count (`CchInput::n`, the graph's node capacity) a hierarchy is built
/// for: the engine's own graph limit, `GrB_INDEX_MAX` (`2^60 - 1`). The id space refuses an
/// id at or past it (`IdSpace::create`) and `node_cap`, the dimension of every node matrix,
/// never grows past it, so no graph reaches [`Cch::build`] with more. Every node id and
/// rank is therefore below `usize::MAX`, which is what lets `Option<Rank>` be the size of a
/// rank.
///
/// A limit, not a promise: the hierarchy holds several dense arrays of `n + 1` entries, so in
/// practice memory stops a graph well before this.
#[allow(clippy::cast_possible_truncation)] // `2^60 - 1`; `try_from` is not usable in a `const`
const MAX_NODES: usize = GrB_INDEX_MAX as usize;

/// A `u64` node id or offset as an index.
///
/// # Panics
/// If `id` does not fit a `usize`, which only a target narrower than 64 bits allows: every
/// id and offset here is below [`MAX_NODES`] or a `Vec`'s length.
fn id_index(id: u64) -> usize {
    usize::try_from(id).expect("a node id or offset fits a usize")
}

/// An index as a `u64` node id or offset.
///
/// # Panics
/// Never on a supported target: `usize` is at most 64 bits wide.
fn index_id(index: usize) -> u64 {
    u64::try_from(index).expect("an index fits a u64")
}

/// Compressed sparse row over `node_count` nodes, ids `u64`. Its arrays are sized once,
/// when it is built, and never grow; they are `Vec`s as the GraphBLAS export hands them over.
///
/// Every `Csr` here holds, by construction:
/// * `row_offsets` has `node_count + 1` offsets, starting at 0, never decreasing, ending at
///   `neighbors.len()`;
/// * every id in `neighbors` is below `node_count`;
/// * every row is sorted ascending and duplicate-free.
///
/// A `Csr` exported from a GraphBLAS matrix gets all three from the export (`n + 1` row
/// pointers, column indices below the dimension, sorted and unique within a row); the ones
/// this module builds itself keep them.
pub(crate) struct Csr {
    node_count: usize,
    /// `node_count + 1` offsets into `neighbors`.
    row_offsets: Vec<u64>,
    /// Each row's node ids, sorted and duplicate-free within the row.
    neighbors: Vec<u64>,
}

impl Csr {
    /// The neighbors of `node`: `neighbors[row_offsets[node]..row_offsets[node + 1]]`, sorted
    /// and duplicate-free.
    ///
    /// # Panics
    /// If `node >= node_count`, or an offset is out of range of `neighbors`.
    #[must_use]
    fn row(
        &self,
        node: usize,
    ) -> &[u64] {
        &self.neighbors[self.row_range(node)]
    }

    /// Where `node`'s row sits in `neighbors` (and in a [`WeightCsr`]'s `values`):
    /// `row_offsets[node]..row_offsets[node + 1]`.
    ///
    /// # Panics
    /// If `node >= node_count`.
    fn row_range(
        &self,
        node: usize,
    ) -> Range<usize> {
        id_index(self.row_offsets[node])..id_index(self.row_offsets[node + 1])
    }

    /// Whether `row_offsets` is shaped as the type requires (`node_count + 1` offsets from 0 to
    /// `neighbors.len()`, never decreasing) and every id is below `node_count`.
    /// `O(node_count + neighbors)`, no allocation: the check that keeps the ordering backend
    /// reading this `Csr` within its arrays.
    #[must_use]
    fn is_well_formed(&self) -> bool {
        let node_count = self.node_count;
        // `len - 1`, not `node_count + 1`: `node_count` may be anything, `usize::MAX` included.
        self.row_offsets.len().checked_sub(1) == Some(node_count)
            && self.row_offsets[0] == 0
            && usize::try_from(self.row_offsets[node_count]) == Ok(self.neighbors.len())
            && self
                .row_offsets
                .windows(2)
                .all(|offsets| offsets[0] <= offsets[1])
            && self
                .neighbors
                .iter()
                .all(|&neighbor| usize::try_from(neighbor).is_ok_and(|index| index < node_count))
    }

    /// Whether this is the adjacency of a simple undirected graph: well formed, rows
    /// sorted and duplicate-free, no self-loop, and every entry mirrored (`b` in row `a`
    /// iff `a` in row `b`). What [`CchInput::skeleton`] holds by construction; checked
    /// again before the ordering backend reads it.
    #[must_use]
    fn is_symmetric_simple(&self) -> bool {
        self.is_well_formed()
            && (0..self.node_count).all(|node| {
                let row = self.row(node);
                row.windows(2).all(|pair| pair[0] < pair[1])
                    && row.iter().all(|&neighbor| {
                        id_index(neighbor) != node
                            && self
                                .row(id_index(neighbor))
                                .binary_search(&index_id(node))
                                .is_ok()
                    })
            })
    }
}

/// Directed weights: same shape as a [`Csr`], with one weight per entry.
pub struct WeightCsr {
    csr: Csr,
    /// One weight per entry of `csr.neighbors`.
    values: Vec<f64>,
}

impl Csr {
    /// The `Csr` over a GraphBLAS export's pattern arrays, taken as they are (the export
    /// guarantees the invariants above), and the export's values.
    fn from_export<V>(export: CsrExport<V>) -> (Self, Vec<V>) {
        let csr = Self {
            node_count: export.row_count(),
            row_offsets: export.row_offsets,
            neighbors: export.column_indices,
        };
        (csr, export.values)
    }
}

/// A pattern export: the skeleton.
impl From<CsrExport<()>> for Csr {
    fn from(export: CsrExport<()>) -> Self {
        Self::from_export(export).0
    }
}

/// A weighted export: pattern and values, unchanged.
impl From<CsrExport<f64>> for WeightCsr {
    fn from(export: CsrExport<f64>) -> Self {
        let (csr, values) = Csr::from_export(export);
        Self { csr, values }
    }
}

/// What `Graph` hands the hierarchy builder (built by #3211, consumed by #3208 and #3209).
pub(crate) struct CchInput {
    n: usize,
    /// Symmetric, no self-loops ([`Csr::is_symmetric_simple`]): `CchInput::build` takes it
    /// as the pattern union of the weights and their transpose after dropping the diagonal.
    /// Phase 1 only.
    skeleton: Csr,
    /// Directed. One entry per node pair: the cheapest weight over all parallel edges and all
    /// indexed relationship types. A missing or non-numeric attribute counts as 1.0.
    /// An absent entry means "no edge". Weights must be `>= 0`; `+inf` is legal and means a
    /// closed edge: it exists, but no path can use it.
    weights: WeightCsr,
}

/// Identity of an index. The key is the whole definition.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CchIndexDef {
    rel_types: Vec<TypeId>,
    weight_attr: u16,
}

impl CchIndexDef {
    /// Sorts and deduplicates `rel_types`, so equal sets are equal indexes.
    #[must_use]
    pub(crate) fn new(
        mut rel_types: Vec<TypeId>,
        weight_attr: u16,
    ) -> Self {
        rel_types.sort_unstable();
        rel_types.dedup();
        // The definition is kept for the index's life and counted by its length.
        rel_types.shrink_to_fit();
        Self {
            rel_types,
            weight_attr,
        }
    }

    /// Sorted, duplicate-free — normalised by [`Self::new`], which is the only constructor.
    #[must_use]
    pub(crate) fn rel_types(&self) -> &[TypeId] {
        &self.rel_types
    }
}

#[derive(Debug, PartialEq, Eq, Error)]
pub enum CchError {
    #[error("a CCH index with this definition already exists")]
    AlreadyExists,
    #[error("no such CCH index")]
    NotFound,
    #[error("unknown relationship type")]
    UnknownRelType,
    #[error("unknown attribute")]
    UnknownAttribute,
    /// A weight was negative (`-inf` included) or NaN: weights must be `>= 0`. `+inf` is
    /// legal (a closed edge).
    #[error("a weight is negative or NaN")]
    InvalidWeight,
    /// The weights are not shaped for the hierarchy: a different node count, or an edge
    /// between two nodes it has no arc for. They come from a later version of the graph than
    /// the one the hierarchy was built for, and only a rebuild can take them. Expected on a
    /// live graph, not a bug. A malformed weight CSR is a bug, [`Self::Internal`], and a
    /// build never returns this.
    #[error("the weights do not match the CCH hierarchy: it needs a rebuild")]
    NeedsRebuild,
    /// A bug, not bad input. The message says which.
    #[error("internal CCH error: {0}")]
    Internal(&'static str),
    /// A bug, not bad input: no elimination order could be produced. The message is the
    /// ordering's own error, the backend's included, as text: the backend is a dependency
    /// of this module, not of its callers.
    #[error("internal CCH error: {0}")]
    Ordering(String),
    /// The build gave up because the index builder is stopping (the server is shutting down).
    /// Never adopted: a batch given up is not published.
    #[error("the CCH build was cancelled: the index builder is stopping")]
    Cancelled,
}

impl From<order::OrderError> for CchError {
    fn from(error: order::OrderError) -> Self {
        Self::Ordering(error.to_string())
    }
}

impl CchError {
    /// The query error for a failed `CREATE`/`DROP CCH INDEX`, worded like C's where C has a
    /// message.
    ///
    /// Not the `Display` text: that is the error's own wording, and the C wording differs
    /// (`AlreadyExists`/`NotFound` say what the index is over, `Internal` drops "internal").
    #[must_use]
    pub(crate) fn query_message(&self) -> String {
        match self {
            Self::AlreadyExists => String::from(
                "a CCH index already exists over these relationship types and weight attribute",
            ),
            Self::NotFound => {
                String::from("no CCH index over these relationship types and weight attribute")
            }
            Self::UnknownRelType => String::from("CCH index error: unknown relationship type"),
            Self::UnknownAttribute => String::from("CCH index error: unknown weight attribute"),
            Self::InvalidWeight => String::from(
                "CCH index error: edge weights must be >= 0 (+inf marks a closed edge)",
            ),
            Self::NeedsRebuild => format!("CCH index error: {self}"),
            Self::Internal(message) => format!("CCH index error: {message}"),
            Self::Ordering(error) => format!("CCH index error: {error}"),
            Self::Cancelled => format!("CCH index error: {self}"),
        }
    }
}

/// The hierarchy. Cheap to share behind an `Arc`: the six structural arrays never change after
/// `build`, and a new metric is a new hierarchy sharing them ([`Cch::customized`]).
///
/// The internal layout — the two id spaces, what an arc id means, and what each of the four
/// arrays holds — is documented in [`customize`], which scoped recustomization (#3217) and
/// any later change to how the weight arrays are stored code against.
pub struct Cch {
    /// Phase 1's output: the elimination order and the chordal supergraph. Metric
    /// independent, so every version of this index shares one.
    topology: Arc<Topology>,
    /// Phase 2's output: the four per-arc arrays, indexed by arc id.
    metric: Metric,
}

impl Cch {
    /// Cheapest weight and node sequence. Mapping node pairs to edge ids is the caller's job.
    ///
    /// `None` when `dst` is unreachable from `src`, or when either is not a node of this
    /// hierarchy. `src == dst` is the single-node path of weight `0.0`.
    ///
    /// Also `None` when every path's weight overflows `f64` (a chain of weights near
    /// `f64::MAX`): the sum is `+inf`, which reads as "no path", as in C.
    ///
    /// Also `None` when unpacking a shortcut on the route finds an arc missing: the
    /// hierarchy no longer matches the graph it was built from, which is a bug, not an
    /// unreachable pair.
    ///
    /// # Panics
    /// On that missing arc when `debug_assertions` are on.
    #[must_use]
    pub fn query(
        &self,
        src: u64,
        dst: u64,
    ) -> Option<(f64, Vec<u64>)> {
        query::query(self, src, dst)
    }
}

/// What [`Cch::recustomize_seeded`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recustomized {
    /// Only the affected cone was recomputed; `arcs` is how many arcs that was.
    Scoped { arcs: usize },
    /// More than `n / 8` arcs changed, so a full customization is due, and it needs the
    /// complete [`WeightCsr`] this path never builds. **Nothing was written.**
    NeedsFull,
    /// A changed pair has no arc in the hierarchy (a new adjacency) or names a node the
    /// hierarchy does not have. Recustomizing cannot represent it: the structure has to be
    /// rebuilt. **Nothing was written.**
    NeedsRebuild,
}

/// What a CCH index holds: nothing yet while its build runs, the built hierarchy, or why its
/// build failed.
#[derive(Clone)]
pub(crate) enum Hierarchy {
    /// The build has not been adopted yet: the entry is `Building`.
    Pending,
    /// The hierarchy the build produced, shared by every version that holds this entry.
    Built(Arc<Cch>),
    /// The build failed for this reason (`GraphTooLarge`-like resource limits, the ordering
    /// backend, a bug: never the graph's data, which `CREATE` checked and every write since
    /// keeps valid). The entry is `Ready` in the builder's sense, so no job runs for it; it
    /// stays failed until it is dropped and created again. Only the definition is saved, so an
    /// RDB load builds it again.
    Failed(Arc<str>),
}

/// One CCH index as held by `Graph`: the kind of an [`IndexEntry`] in
/// `FalkorDbIndex::cch_indexes`, which adds its build state. `new_version` clones the `Arc`.
#[derive(Clone)]
pub(crate) struct CchIndex {
    pub def: CchIndexDef,
    pub(crate) hierarchy: Hierarchy,
    /// Builds started again in a row because writes committed during them could not be taken
    /// ([`Absorbed::Restart`]). An adoption resets it; at [`MAX_REDOS`] the build is given up
    /// as failed, so writes faster than a build cannot keep it building for as long as they
    /// last.
    redos: u8,
}

/// How many builds in a row writers may overtake with a change the hierarchy cannot take: the
/// index is given up as failed at the last of them, after `MAX_REDOS - 1` restarts.
const MAX_REDOS: u8 = 5;

/// The reason a build given up after [`MAX_REDOS`] restarts is failed with.
const OVERTAKEN: &str = "structural writes kept overtaking its build";

impl CchIndex {
    /// An index whose build has not been adopted yet, for a `Building` entry.
    #[must_use]
    pub(crate) const fn pending(def: CchIndexDef) -> Self {
        Self {
            def,
            hierarchy: Hierarchy::Pending,
            redos: 0,
        }
    }

    /// An index over a hierarchy built already, for a `Ready` entry.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn built(
        def: CchIndexDef,
        cch: Arc<Cch>,
    ) -> Self {
        Self {
            def,
            hierarchy: Hierarchy::Built(cch),
            redos: 0,
        }
    }

    /// The hierarchy, if the build produced one.
    #[must_use]
    pub(crate) const fn cch(&self) -> Option<&Arc<Cch>> {
        match &self.hierarchy {
            Hierarchy::Built(cch) => Some(cch),
            Hierarchy::Pending | Hierarchy::Failed(_) => None,
        }
    }

    /// Start counting restarts again, for another generation.
    pub(crate) fn restart_count_reset(&mut self) {
        self.redos = 0;
    }

    /// Why the build failed, if it did.
    #[must_use]
    pub(crate) fn failure(&self) -> Option<&str> {
        match &self.hierarchy {
            Hierarchy::Failed(reason) => Some(reason),
            Hierarchy::Pending | Hierarchy::Built(_) => None,
        }
    }
}

/// CCH as a kind of versioned index, built in the background by the index builder
/// ([`online_build`](super::online_build)) like a column. An entry is found by its whole
/// definition.
///
/// - `Cursor = ()`: a hierarchy is not built in resumable pieces; one indexer batch builds it
///   whole, so there is no place to resume from.
/// - `Batch = Result<Arc<Cch>, CchError>`: what that batch builds off the lock, the hierarchy
///   to adopt, or why it could not (the trait's read has no other error channel). Either way
///   the batch ends the build: a failure is adopted as [`Hierarchy::Failed`].
/// - `Changes = RoaringTreemap`: the ids of the edges a unit of work wrote in the index's
///   types, which a `Building` entry keeps for its build's commit
///   ([`maint`]'s flush hands them over; a built entry is updated by the flush itself).
/// - `Scratch = ()`: nothing is reused from one index to the next.
///
/// **Writes during a build.** The build reads the version the builder took. Writers that
/// commit meanwhile leave the ids of the edges they wrote in the entry's written set
/// ([`Self::writer_record`]), and the build's commit catches the hierarchy up on them
/// ([`Self::indexer_catch_up`]); a writer whose fork the adoption overtook does the same for
/// its own writes as it escalates ([`Self::writer_rebase`]). Either takes them the way a write
/// to a built index is taken (`maint::absorb`): scoped, or a full customization, under the
/// host's lock; one the hierarchy cannot take starts the build again
/// ([`Absorbed::Restart`]).
impl VersionedIndex for CchIndex {
    type Cursor = ();
    type Changes = RoaringTreemap;
    type Batch = Result<Arc<Cch>, CchError>;
    type Name = CchIndexDef;
    type Scratch = ();

    fn entry<'a>(
        index: &'a FalkorDbIndex,
        def: &CchIndexDef,
    ) -> Option<&'a IndexEntry<Self>> {
        index.cch_indexes.get(def)
    }

    fn entry_mut<'a>(
        index: &'a mut FalkorDbIndex,
        def: &CchIndexDef,
    ) -> Option<&'a mut IndexEntry<Self>> {
        index.cch_indexes.get_mut(def)
    }

    /// A `Building` entry keeps the ids of the edges written, for its build's commit: nothing is
    /// below its cursor, so nothing is applied now. A built entry is not updated here: its
    /// update reads the graph (the seeds of the recomputed arcs), which this hook is not given,
    /// so [`maint`]'s flush does it and calls this for `Building` entries only.
    fn writer_record(
        &mut self,
        state: &mut BuildState<()>,
        changes: RoaringTreemap,
        _place: impl Fn(u64) -> Option<()>,
        _scratch: &mut (),
    ) {
        if let BuildState::Building { written, .. } = state {
            written.by_this_version |= changes;
        }
    }

    /// This is the published copy, the hierarchy adopted after the writer's fork was made.
    /// Take the edges the fork wrote (`written`): their pairs in `published` (an edge the fork
    /// deleted) and in `fork` (one it created or reweighted), seeded from `fork`. A change the
    /// hierarchy cannot take starts another build, under a new generation
    /// ([`Absorbed::Restart`]). So does an error taking them (a resource, a bug): it is not the
    /// writer's fault, and the hierarchy is healthy, so the builder builds it again rather than
    /// the index being failed, as a flush that cannot update an index hands it to the builder
    /// ([`maint::flush`]). A copy still `Building`, or failed, takes nothing.
    fn writer_rebase(
        &mut self,
        state: &BuildState<()>,
        def: &CchIndexDef,
        written: &RoaringTreemap,
        published: &Graph,
        fork: &Graph,
    ) -> Absorbed<()> {
        if written.is_empty() || matches!(state, BuildState::Building { .. }) {
            return Absorbed::Yes;
        }
        let absorbed = self
            .absorb(def, written, published, fork)
            .unwrap_or_else(|_| {
                self.hierarchy = Hierarchy::Pending;
                Absorbed::Restart(())
            });
        if absorbed == Absorbed::Restart(()) {
            // Another generation: its own count of restarts.
            self.redos = 0;
        }
        absorbed
    }

    /// This is the batch's copy, the hierarchy just built from `batch_version`. Take the edges
    /// writers wrote since (`written`): their pairs in `batch_version` (an edge deleted since)
    /// and in `published` (one created or reweighted since), seeded from `published`. A change
    /// the hierarchy cannot take starts the build again under the same generation, so the same
    /// job reads a newer version ([`Absorbed::Restart`]); the [`MAX_REDOS`]-th build in a row
    /// overtaken that way gives the index up as failed instead. An error taking the edges (a
    /// resource, a bug) is this build's result, held as failed with its reason, as a build that
    /// fails is.
    fn indexer_catch_up(
        &mut self,
        _state: &BuildState<()>,
        def: &CchIndexDef,
        written: &RoaringTreemap,
        batch_version: &Graph,
        published: &Graph,
    ) -> Absorbed<()> {
        if written.is_empty() {
            self.redos = 0;
            return Absorbed::Yes;
        }
        let absorbed = match self.absorb(def, written, batch_version, published) {
            Ok(absorbed) => absorbed,
            Err(cause) => {
                self.hierarchy = Hierarchy::Failed(Arc::from(cause.to_string()));
                return Absorbed::Yes;
            }
        };
        if absorbed == Absorbed::Restart(()) && self.redos >= MAX_REDOS {
            self.hierarchy = Hierarchy::Failed(Arc::from(OVERTAKEN));
            return Absorbed::Yes;
        }
        absorbed
    }

    /// Never called: the CCH arm of [`Graph::indexer_fit`] never chooses
    /// [`CatchUp::Insert`](super::versioned::CatchUp::Insert): a batch cannot be inserted into a
    /// hierarchy.
    fn indexer_insert(
        &mut self,
        _def: &CchIndexDef,
        _batch: &Self::Batch,
        _written: &RoaringTreemap,
        _from: (),
        _next: Option<()>,
        _published: &Graph,
    ) {
        unreachable!("a CCH batch is never inserted")
    }

    /// Never called, as [`Self::indexer_insert`]: only the insert weighs a batch's length.
    fn batch_len(_batch: &Self::Batch) -> usize {
        unreachable!("a CCH batch is never inserted")
    }

    /// The whole build, off the lock, on `version`, the builder's own copy of a published
    /// version: the input, Phase 1 and Phase 2. One batch, so the next is always `None`, and
    /// applying it ends the build, failed or not ([`Self::indexer_add`]). The budget is
    /// meaningless for a hierarchy and ignored.
    ///
    /// Between the input, the ordering and the customization it gives up if the builder is
    /// stopping ([`online_build::stopping`](super::online_build::stopping)): the ordering
    /// cannot be interrupted, and a build outlives the stop's wait. A batch given up is never
    /// published, so its error is never adopted.
    fn indexer_read(
        &self,
        version: &Graph,
        def: &CchIndexDef,
        _from: (),
        _budget: usize,
    ) -> (Self::Batch, Option<()>) {
        (build_off_the_lock(version, def).map(Arc::new), None)
    }

    /// Adopt what the batch built: the hierarchy, or, for an error, its reason as
    /// [`Hierarchy::Failed`].
    fn indexer_add(
        &mut self,
        batch: &mut Self::Batch,
    ) {
        self.hierarchy = match batch {
            Ok(cch) => Hierarchy::Built(Arc::clone(cch)),
            Err(cause) => Hierarchy::Failed(Arc::from(cause.to_string())),
        };
    }

    /// The hierarchy ([`Cch::memory_usage`]) if built, the reason if the build failed, and the
    /// definition's type list. A `Topology` shared with another version is counted here too, as
    /// every index counts a shared page. A build in flight belongs to no version and is not
    /// counted, as a column's batch is not.
    fn memory_usage(&self) -> usize {
        self.cch().map_or(0, |cch| cch.memory_usage())
            + self.failure().map_or(0, str::len)
            + size_of_val(self.def.rel_types())
    }
}

impl CchIndex {
    /// Take the edges `written` between `before` and `after` into this copy's hierarchy
    /// (`maint::absorb`). A hierarchy that cannot take them is dropped, and the build starts
    /// again: [`Absorbed::Restart`], counted in [`Self::redos`]. A failed copy takes nothing.
    ///
    /// # Errors
    /// What `maint::absorb` reports (a resource, a bug), with this copy unchanged: the caller
    /// answers it, a writer's rebase by starting the build again, a build's commit by holding
    /// the index failed.
    fn absorb(
        &mut self,
        def: &CchIndexDef,
        written: &RoaringTreemap,
        before: &Graph,
        after: &Graph,
    ) -> Result<Absorbed<()>, CchError> {
        let Hierarchy::Built(cch) = &self.hierarchy else {
            return Ok(Absorbed::Yes);
        };
        Ok(match maint::absorb(cch, def, written, before, after)? {
            Some(updated) => {
                self.hierarchy = Hierarchy::Built(Arc::new(updated));
                self.redos = 0;
                Absorbed::Yes
            }
            None => {
                self.hierarchy = Hierarchy::Pending;
                self.redos = self.redos.saturating_add(1);
                Absorbed::Restart(())
            }
        })
    }
}

/// [`CchIndex::indexer_read`]'s build: input, ordering, customization, with a check for the
/// builder's stop before each of the three ([`Cch::build_checked`] makes the last one, between
/// the ordering and the customization).
fn build_off_the_lock(
    version: &Graph,
    def: &CchIndexDef,
) -> Result<Cch, CchError> {
    let stopping = || {
        if online_build::stopping() {
            Err(CchError::Cancelled)
        } else {
            Ok(())
        }
    };
    stopping()?;
    let input = CchInput::build(version, def)?;
    stopping()?;
    Cch::build_checked(&input, stopping)
}

#[cfg(test)]
mod tests {
    use super::{Cch, CchIndex, CchIndexDef, Csr, TypeId};
    use crate::graph::{graph::Graph, graphblas::test_init};
    use crate::index::falkordb::cch::testkit::{empty, to_input};
    use crate::index::falkordb::versioned::{BuildState, IndexEntry, VersionedIndex};
    use std::sync::Arc;

    #[test]
    fn new_sorts_rel_types() {
        let def = CchIndexDef::new(vec![TypeId(7), TypeId(2), TypeId(5)], 3);
        assert_eq!(def.rel_types(), &[TypeId(2), TypeId(5), TypeId(7)]);
        assert_eq!(def.weight_attr, 3);
    }

    #[test]
    fn new_dedups_rel_types() {
        let def = CchIndexDef::new(vec![TypeId(4), TypeId(1), TypeId(4), TypeId(1)], 0);
        assert_eq!(def.rel_types(), &[TypeId(1), TypeId(4)]);
        // `memory_usage` counts the length, so no capacity is left over.
        assert_eq!(def.rel_types.capacity(), def.rel_types.len());
    }

    #[test]
    fn equal_sets_are_equal() {
        let a = CchIndexDef::new(vec![TypeId(3), TypeId(1), TypeId(3)], 9);
        let b = CchIndexDef::new(vec![TypeId(1), TypeId(3)], 9);
        assert_eq!(a, b);
    }

    /// `Csr::row` is the slice `row_offsets[node]..row_offsets[node + 1]` of `neighbors`,
    /// empty rows included, and refuses a row that does not exist rather than reading the
    /// next one.
    #[test]
    fn csr_row_slices_one_row() {
        // 0: [1, 2]   1: []   2: [0]
        let csr = Csr {
            node_count: 3,
            row_offsets: vec![0, 2, 2, 3],
            neighbors: vec![1, 2, 0],
        };
        assert_eq!(csr.row(0), &[1, 2]);
        assert!(csr.row(1).is_empty());
        assert_eq!(csr.row(2), &[0]);
        assert!(std::panic::catch_unwind(|| csr.row(3).len()).is_err());
    }

    /// Each structural defect fails `is_well_formed`, and each defect of a simple
    /// undirected adjacency fails `is_symmetric_simple`, from the one clean path 0 - 1 - 2.
    #[test]
    fn csr_checks_catch_each_defect() {
        let csr = |node_count: usize, row_offsets: &[u64], neighbors: &[u64]| Csr {
            node_count,
            row_offsets: row_offsets.into(),
            neighbors: neighbors.into(),
        };
        let clean = csr(3, &[0, 1, 3, 4], &[1, 0, 2, 1]);
        assert!(clean.is_well_formed() && clean.is_symmetric_simple());
        for bad in [
            csr(3, &[0, 1, 3], &[1, 0, 2]),       // n offsets, not n + 1
            csr(3, &[1, 1, 3, 4], &[1, 0, 2, 1]), // does not start at 0
            csr(3, &[0, 1, 3, 3], &[1, 0, 2, 1]), // does not end at neighbors.len()
            csr(3, &[0, 2, 1, 4], &[1, 0, 2, 1]), // decreasing
            csr(3, &[0, 1, 3, 4], &[1, 0, 3, 1]), // id past n
        ] {
            assert!(!bad.is_well_formed() && !bad.is_symmetric_simple());
        }
        // Refused, not an overflow on `n + 1`.
        let huge = csr(usize::MAX, &[0], &[]);
        assert!(!huge.is_well_formed() && !huge.is_symmetric_simple());
        for unsimple in [
            csr(3, &[0, 1, 3, 4], &[1, 2, 0, 1]),    // row 1 unsorted
            csr(3, &[0, 1, 4, 5], &[1, 0, 2, 2, 1]), // row 1 repeats 2
            csr(3, &[0, 2, 4, 5], &[0, 1, 0, 2, 1]), // self-loop on 0
            csr(3, &[0, 1, 3, 3], &[1, 0, 2]),       // 1 -> 2 not mirrored
        ] {
            assert!(unsimple.is_well_formed() && !unsimple.is_symmetric_simple());
        }
    }

    /// `new_version` must carry the indexes forward by sharing the `Arc`, not by copying the
    /// hierarchy. Drop `cch_indexes` from `Graph::new_version` and this fails on the length;
    /// deep-copy the hierarchy instead of cloning the `Arc` and it fails on `ptr_eq`.
    ///
    /// Inserts into the field directly, so it checks `new_version` alone, independent of
    /// `create_cch_index` and the graph input.
    #[test]
    fn new_version_shares_the_hierarchy() {
        test_init::ensure_init();
        let mut g = Graph::new(16, 16, 1, 0, "cch_new_version");
        assert!(
            g.falkordb_index.cch_indexes.is_empty(),
            "a fresh graph holds no CCH index"
        );

        let def = CchIndexDef::new(vec![TypeId(1)], 0);
        let cch = Arc::new(Cch::build(&to_input(&empty(0))).expect("empty hierarchy"));
        g.falkordb_index.cch_indexes.insert(
            def.clone(),
            IndexEntry::new(
                CchIndex::built(def.clone(), Arc::clone(&cch)),
                BuildState::Ready,
                // Any generation: nothing here compares it.
                1,
            ),
        );

        let next = g.new_version();
        assert_eq!(next.falkordb_index.cch_indexes.len(), 1);
        let held = CchIndex::entry(&next.falkordb_index, &def).expect("found by its definition");
        assert_eq!(held.state, BuildState::Ready);
        assert!(
            Arc::ptr_eq(held.index.cch().expect("built"), &cch),
            "new_version must share the hierarchy, not copy it"
        );
    }
}
