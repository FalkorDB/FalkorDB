//! Customizable Contraction Hierarchies (CCH) path index — contract skeleton.
//!
//! [`Cch`]'s methods are stubs: signatures the implementation changes build against. The
//! engine does not call any of it yet: `Graph` only holds an empty list of indexes, and
//! the query pipeline never reaches the `unimplemented!()` bodies below. (The public
//! `Graph` methods in `graph.rs` are stubs too, or delegate to `FalkorDbIndex`'s CCH methods,
//! which are, so a caller that invokes them directly reaches `unimplemented!()`.)
//!
//! Layout (one file per area, so changes to different areas do not touch each other's files):
//! - `order.rs`     Phase 1: nested-dissection ordering and triangulation (#3208)
//! - `nd.rs`        the nested-dissection call behind Phase 1's ordering
//! - `customize.rs` Phase 2: customization, `WeightArr` (#3209)
//! - `partial.rs`   scoped recustomization (#3217)
//! - `query.rs`     bidirectional upward search and path unpacking (#3210)
//! - `input.rs`     `Graph` -> [`CchInput`] (#3211)
//!
//! The types themselves live here in `mod.rs` because every part needs them; the other
//! files are where the `impl` blocks land.

mod customize;
mod input;
mod nd;
mod order;
mod query;
#[cfg(test)]
pub(crate) mod testkit;

use std::sync::Arc;

use roaring::RoaringTreemap;

use super::falkordb_index::FalkorDbIndex;
use super::versioned::{BuildState, IndexEntry, VersionedIndex};
use crate::graph::graph::{Graph, TypeId};
use crate::graph::graphblas::tensor::GrB_INDEX_MAX;
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
pub struct Csr {
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
        &self.neighbors[id_index(self.row_offsets[node])..id_index(self.row_offsets[node + 1])]
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
    pub csr: Csr,
    pub w: Vec<f64>,
}

/// What `Graph` hands the hierarchy builder (built by #3211, consumed by #3208 and #3209).
pub struct CchInput {
    pub n: usize,
    /// Symmetric, no self-loops ([`Csr::is_symmetric_simple`]): `CchInput::build` takes it
    /// as the pattern union of the weights and their transpose after dropping the diagonal.
    /// Phase 1 only.
    pub skeleton: Csr,
    /// Directed. One entry per node pair: the cheapest weight over all parallel edges and all
    /// indexed relationship types. A missing or non-numeric attribute counts as 1.0.
    /// An absent entry means "no edge". Weights must be `>= 0`; `+inf` is legal and means a
    /// closed edge: it exists, but no path can use it.
    pub weights: WeightCsr,
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
    pub fn new(
        mut rel_types: Vec<TypeId>,
        weight_attr: u16,
    ) -> Self {
        rel_types.sort_unstable();
        rel_types.dedup();
        Self {
            rel_types,
            weight_attr,
        }
    }

    /// Sorted, duplicate-free — normalised by [`Self::new`], which is the only constructor.
    #[cfg(test)] // only the tests read it until the input build (#3211) does
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
    /// A bug, not bad input. The message says which.
    #[error("internal CCH error: {0}")]
    Internal(&'static str),
    /// A bug, not bad input: no elimination order could be produced. The message is the
    /// ordering's own error, the backend's included, as text: the backend is a dependency
    /// of this module, not of its callers.
    #[error("internal CCH error: {0}")]
    Ordering(String),
}

impl From<order::OrderError> for CchError {
    fn from(error: order::OrderError) -> Self {
        Self::Ordering(error.to_string())
    }
}

/// The hierarchy. Cheap to share behind an `Arc`: the six structural arrays never change after
/// `build`; only the four weight arrays are rewritten by `customize`.
pub struct Cch {
    // filled in by the ordering (#3208) and customization (#3209) changes
}

impl Cch {
    /// Phase 1 + Phase 2.
    ///
    /// # Errors
    /// [`CchError::InvalidWeight`] for a negative or NaN weight.
    ///
    /// # Panics
    /// Always — unimplemented until #3208 and #3209. Nothing calls it yet.
    pub fn build(_input: &CchInput) -> Result<Self, CchError> {
        unimplemented!("hierarchy build, #3208 and #3209")
    }

    /// Phase 2 only, same structure, new weights.
    ///
    /// # Errors
    /// [`CchError::InvalidWeight`] for a negative or NaN weight.
    ///
    /// # Panics
    /// Always — unimplemented until #3209. Nothing calls it yet.
    pub fn customize(
        &mut self,
        _weights: &WeightCsr,
    ) -> Result<(), CchError> {
        unimplemented!("customization, #3209")
    }

    /// Recompute the metric after the edges in `changed` changed weight, touching only the
    /// arcs those changes can reach.
    ///
    /// `changed` lists node-space `(u, v)` pairs whose cheapest weight may differ from the
    /// last customization — raised, lowered, removed, or added where the hierarchy already
    /// has an arc. Direction does not matter (an arc carries both), duplicates are fine,
    /// and self-loops are ignored. `weights` is the **complete** new metric, the same
    /// [`WeightCsr`] a full [`Cch::customize`] would take:
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
    /// [`CchError::InvalidWeight`] if a changed pair's new weight is negative or NaN.
    /// Nothing is written in that case, nor on [`Recustomized::NeedsRebuild`].
    ///
    /// # Panics
    /// Always — unimplemented until #3217. Nothing calls it yet.
    pub fn recustomize_scoped(
        &mut self,
        _changed: &[(u64, u64)],
        _weights: &WeightCsr,
    ) -> Result<Recustomized, CchError> {
        unimplemented!("scoped recustomization, #3217")
    }

    /// Cheapest weight and node sequence. Mapping node pairs to edge ids is the caller's job.
    ///
    /// # Panics
    /// Always — unimplemented until #3210. Nothing calls it yet.
    #[must_use]
    pub fn query(
        &self,
        _src: u64,
        _dst: u64,
    ) -> Option<(f64, Vec<u64>)> {
        unimplemented!("query, #3210")
    }

    /// # Panics
    /// Always — unimplemented until #3209. Nothing calls it yet.
    #[must_use]
    pub fn node_count(&self) -> usize {
        unimplemented!("customization, #3209")
    }

    /// Only the tests call it.
    ///
    /// # Panics
    /// Always — unimplemented until #3209. Nothing calls it yet.
    #[cfg(test)]
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        unimplemented!("customization, #3209")
    }
}

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

/// One CCH index as held by `Graph`: the kind of an [`IndexEntry`] in
/// `FalkorDbIndex::cch_indexes`, which adds its build state. `new_version` clones the `Arc`.
#[derive(Clone)]
pub struct CchIndex {
    pub def: CchIndexDef,
    pub cch: Arc<Cch>,
}

/// CCH as a kind of versioned index. An entry is found by its whole definition, and until the
/// background build every entry is built synchronously and held [`BuildState::Ready`].
///
/// - `Cursor = ()`: a hierarchy is not built in resumable pieces; one indexer turn builds it
///   whole, so there is no place to resume from.
/// - `Batch = Arc<Cch>`: what that turn builds off the lock, the hierarchy to adopt.
/// - `Changes = CchDirty`: what a unit of work's writes touched, as its graph write methods
///   stage it; an index keeps what [`CchDirty`] says touches its definition.
/// - `Scratch = ()`: nothing is reused from one index to the next.
impl VersionedIndex for CchIndex {
    type Cursor = ();
    type Changes = CchDirty;
    type Batch = Arc<Cch>;
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

    /// Never called: `IndexId` has no CCH variant until the background build, so no writer
    /// dispatches here.
    fn writer_record(
        &mut self,
        _state: &mut BuildState<()>,
        _changes: CchDirty,
        _place: impl Fn(u64) -> Option<()>,
        _scratch: &mut (),
    ) {
        unreachable!("CCH is not dispatched through IndexId until the background build")
    }

    /// Never called: a writer is rebased only onto batches on a `Building` column, and every
    /// CCH entry is held `Ready`.
    fn writer_rebase(
        &mut self,
        _state: &BuildState<()>,
        _def: &CchIndexDef,
        _written: &RoaringTreemap,
        _published: &Graph,
        _fork: &Graph,
    ) {
        unreachable!("CCH is not dispatched through IndexId until the background build")
    }

    /// Never called: `IndexId` has no CCH variant until the background build, so no indexer
    /// batch commits on one.
    fn indexer_catch_up(
        &mut self,
        _state: &BuildState<()>,
        _def: &CchIndexDef,
        _written: &RoaringTreemap,
        _batch_version: &Graph,
        _published: &Graph,
    ) {
        unreachable!("CCH is not dispatched through IndexId until the background build")
    }

    /// Never called: `IndexId` has no CCH variant until the background build, so no indexer
    /// batch commits on one.
    fn indexer_insert(
        &mut self,
        _def: &CchIndexDef,
        _batch: &Arc<Cch>,
        _written: &RoaringTreemap,
        _from: (),
        _next: Option<()>,
        _published: &Graph,
    ) {
        unreachable!("CCH is not dispatched through IndexId until the background build")
    }

    /// Never called: `IndexId` has no CCH variant until the background build, so no indexer
    /// batch commits on one.
    fn batch_len(_batch: &Arc<Cch>) -> usize {
        unreachable!("CCH is not dispatched through IndexId until the background build")
    }

    /// Never called: `IndexId` has no CCH variant until the background build, so no indexer
    /// batch is read for one.
    fn indexer_read(
        &self,
        _version: &Graph,
        _def: &CchIndexDef,
        _from: (),
        _budget: usize,
    ) -> (Arc<Cch>, Option<()>) {
        unreachable!("CCH is not dispatched through IndexId until the background build")
    }

    /// Never called: `IndexId` has no CCH variant until the background build, so no indexer
    /// batch is added to one.
    fn indexer_add(
        &mut self,
        _batch: &mut Arc<Cch>,
    ) {
        unreachable!("CCH is not dispatched through IndexId until the background build")
    }

    /// Never called: [`FalkorDbIndex::memory_usage`] counts the node and edge columns only, since
    /// the hierarchy holds no arrays until #3208 and #3209.
    fn memory_usage(&self) -> usize {
        unreachable!("GRAPH.MEMORY USAGE counts no CCH index until the hierarchy holds arrays")
    }
}

/// What a unit of work's writes touched that the graph's CCH indexes might care about. The
/// graph's write methods stage it in the unit's
/// [`IndexDocs::cch`](crate::index::index_docs::IndexDocs::cch), next to the native index's
/// staged tuples, and the owner of the unit flushes it into rebuilt indexes once the unit's
/// writes are in (#3214).
///
/// Empty for every graph with no CCH index, so the maintenance hook costs one
/// `is_empty()` check on the existing write path.
#[derive(Clone, Default)]
pub struct CchDirty {
    // filled in by the maintenance change (#3214): touched rel types and weight attributes, created/deleted nodes
}

#[cfg(test)]
mod tests {
    use super::{Cch, CchIndex, CchIndexDef, Csr, TypeId};
    use crate::graph::{graph::Graph, graphblas::test_init};
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
    /// Reaches past the (still unimplemented) DDL by inserting into the field directly, which is
    /// also why this test and `CchIndex::entry` are the only things in the crate that touch
    /// `cch_indexes` at all.
    #[test]
    fn new_version_shares_the_hierarchy() {
        test_init::ensure_init();
        let mut g = Graph::new(16, 16, 1, 0, "cch_new_version");
        assert!(
            g.falkordb_index.cch_indexes.is_empty(),
            "a fresh graph holds no CCH index"
        );

        let def = CchIndexDef::new(vec![TypeId(1)], 0);
        let cch = Arc::new(Cch {});
        g.falkordb_index.cch_indexes.insert(
            def.clone(),
            IndexEntry::new(
                CchIndex {
                    def: def.clone(),
                    cch: Arc::clone(&cch),
                },
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
            Arc::ptr_eq(&held.index.cch, &cch),
            "new_version must share the hierarchy, not copy it"
        );
    }
}
