//! Customizable Contraction Hierarchies (CCH) path index — contract skeleton.
//!
//! Everything here is a stub: types and signatures the implementation changes build against.
//! The engine does not call any of it yet: `Graph` only holds an empty list of indexes, and
//! the query pipeline never reaches the `unimplemented!()` bodies below. (The public
//! `Graph` methods in `graph.rs` delegate to `index.rs`, whose bodies are stubs too, so a
//! caller that invokes them directly reaches `unimplemented!()`.)
//!
//! Layout (one file per area, so changes to different areas do not touch each other's files):
//! - `order.rs`     Phase 1: nested-dissection ordering and triangulation (#3208)
//! - `nd.rs`        the nested-dissection call behind Phase 1's ordering
//! - `customize.rs` Phase 2: customization, `WeightArr` (#3209)
//! - `partial.rs`   scoped recustomization (#3217)
//! - `query.rs`     bidirectional upward search and path unpacking (#3210)
//! - `input.rs`     `Graph` -> [`CchInput`] (#3211)
//! - `index.rs`     the logic behind `Graph`'s CCH methods, over [`CchIndex`] (#3212)
//!
//! The types themselves live here in `mod.rs` because every part needs them; the other
//! files are where the `impl` blocks land.

pub mod customize;
pub mod index;
pub mod input;
mod nd;
pub mod order;
pub mod query;
#[cfg(test)]
pub(crate) mod testkit;

use std::sync::Arc;

use super::falkordb_index::FalkorDbIndex;
use super::versioned::{BuildState, IndexEntry, VersionedIndex};
use crate::graph::graph::{Graph, TypeId};
use crate::graph::graphblas::tensor::GrB_INDEX_MAX;
use customize::Metric;
use order::Topology;
use thiserror::Error;

/// The largest node count (`CchInput::n`, the graph's node capacity) a hierarchy is built
/// for: the engine's own graph limit, `GrB_INDEX_MAX` (`2^60 - 1`). The id space refuses an
/// id at or past it (`IdSpace::create`) and `node_cap`, the dimension of every node matrix,
/// never grows past it, so [`Cch::build`] refuses more with [`CchError::GraphTooLarge`] only
/// as a backstop. Node ids, ranks and mids are `u64` throughout, so `u64::MAX` is free to be
/// the "none" sentinel of the rank and mid arrays.
///
/// A limit, not a promise: the hierarchy holds several dense arrays of `n + 1` entries, so in
/// practice memory stops a graph well before this.
#[allow(clippy::cast_possible_truncation)]
pub const MAX_NODES: usize = GrB_INDEX_MAX as usize;

/// Whether a graph of `n` nodes is within [`MAX_NODES`]. [`Cch::build`] calls it before
/// touching any array.
///
/// # Errors
/// [`CchError::GraphTooLarge`] past the limit.
pub fn check_size(n: usize) -> Result<(), CchError> {
    if n > MAX_NODES {
        return Err(CchError::GraphTooLarge);
    }
    Ok(())
}

/// Compressed sparse row over `n` nodes, ids `u64`. Its arrays are sized once, when it is
/// built, and never grow, so they are boxed slices rather than `Vec`s.
///
/// Every `Csr` here holds, by construction:
/// * `p` has `n + 1` offsets, starting at 0, never decreasing, ending at `i.len()`;
/// * every id in `i` is below `n`;
/// * every row is sorted ascending and duplicate-free.
///
/// A `Csr` exported from a GraphBLAS matrix gets all three from the export (`n + 1` row
/// pointers, column indices below the dimension, sorted and unique within a row); the ones
/// this module builds itself keep them.
pub struct Csr {
    pub n: usize,
    /// `n + 1` offsets into `i`.
    pub p: Box<[u64]>,
    /// Column ids, sorted and duplicate-free within each row.
    pub i: Box<[u64]>,
}

impl Csr {
    /// The column ids of row `u`: `i[p[u]..p[u + 1]]`, sorted and duplicate-free.
    ///
    /// # Panics
    /// If `u >= n`, or an offset is out of range of `i`.
    #[must_use]
    pub fn row(
        &self,
        u: usize,
    ) -> &[u64] {
        &self.i[self.p[u] as usize..self.p[u + 1] as usize]
    }

    /// Whether `p` is shaped as the type requires (`n + 1` offsets from 0 to `i.len()`,
    /// never decreasing) and every id is below `n`. `O(n + nnz)`, no allocation: the check
    /// that keeps a C library reading this `Csr` within its arrays.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        let n = self.n;
        // `len - 1`, not `n + 1`: the fields are public, and `n` may be anything.
        self.p.len().checked_sub(1) == Some(n)
            && self.p[0] == 0
            && self.p[n] as usize == self.i.len()
            && self.p.windows(2).all(|w| w[0] <= w[1])
            && self.i.iter().all(|&v| (v as usize) < n)
    }

    /// Whether this is the adjacency of a simple undirected graph: well formed, rows
    /// sorted and duplicate-free, no self-loop, and every entry mirrored (`v` in row `u`
    /// iff `u` in row `v`). What [`CchInput::skeleton`] holds by construction; checked
    /// again before the ordering backend reads it.
    #[must_use]
    pub fn is_symmetric_simple(&self) -> bool {
        self.is_well_formed()
            && (0..self.n).all(|u| {
                let r = self.row(u);
                r.windows(2).all(|w| w[0] < w[1])
                    && r.iter().all(|&v| {
                        v as usize != u && self.row(v as usize).binary_search(&(u as u64)).is_ok()
                    })
            })
    }
}

/// Directed weights: same shape as a [`Csr`], with one weight per entry.
pub struct WeightCsr {
    pub csr: Csr,
    pub w: Box<[f64]>,
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
    pub weight_attr: u16,
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
    pub fn rel_types(&self) -> &[TypeId] {
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
    /// More than [`MAX_NODES`] nodes (see [`check_size`]).
    #[error("graph is too large for a CCH index")]
    GraphTooLarge,
    /// A weight was negative (`-inf` included) or NaN: weights must be `>= 0`. `+inf` is
    /// legal (a closed edge).
    #[error("a weight is negative or NaN")]
    InvalidWeight,
    /// A bug, not bad input: the ordering backend failed on a well-formed graph or
    /// returned something other than a permutation, or `customize` was handed weights for
    /// a different graph than the hierarchy was built for. The message says which.
    #[error("internal CCH error: {0}")]
    Internal(&'static str),
}

/// The hierarchy. Cheap to share behind an `Arc`: the six structural arrays never change after
/// `build`; only the four weight arrays are rewritten by `customize`.
///
/// The internal layout — the two id spaces, what an arc id means, and what each of the four
/// arrays holds — is documented in [`customize`], which scoped recustomization (#3217) and
/// any later change to how the weight arrays are stored code against.
pub struct Cch {
    /// Phase 1's output: the elimination order and the chordal supergraph. Metric
    /// independent, so every version of this index shares one.
    topo: Arc<Topology>,
    /// Phase 2's output: the four per-arc arrays, indexed by arc id.
    metric: Metric,
}

impl Cch {
    /// Phase 1 + Phase 2.
    ///
    /// # Errors
    /// [`CchError::GraphTooLarge`] above [`MAX_NODES`] nodes (see [`check_size`]),
    /// [`CchError::InvalidWeight`] for a negative or NaN weight, and
    /// [`CchError::Internal`] if the ordering backend fails.
    pub fn build(input: &CchInput) -> Result<Self, CchError> {
        Self::build_impl(input)
    }

    /// Phase 2 only, same structure, new weights.
    ///
    /// # Errors
    /// [`CchError::InvalidWeight`] for a negative or NaN weight, and
    /// [`CchError::Internal`] if `weights` is not shaped for this hierarchy — a different
    /// node count, or an edge between two nodes it has no arc for. Both need a rebuild.
    /// Nothing is written on error.
    pub fn customize(
        &mut self,
        weights: &WeightCsr,
    ) -> Result<(), CchError> {
        self.customize_impl(weights)
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
    /// `None` when `dst` is unreachable from `src`, or when either is not a node of this
    /// hierarchy. `src == dst` is the single-node path of weight `0.0`.
    ///
    /// Also `None`, with a warning logged once per process, when unpacking a shortcut on the
    /// route finds an arc missing: the hierarchy no longer matches the graph it was built from,
    /// which is a bug, not an unreachable pair.
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

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_count_impl()
    }

    /// Bytes held by this hierarchy: the struct itself, the shared [`Topology`], and the
    /// four weight arrays. Two versions sharing one
    /// `Topology` each report its bytes.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.memory_usage_impl()
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
        index.cch_indexes.iter().find(|idx| &idx.index.def == def)
    }

    fn entry_mut<'a>(
        index: &'a mut FalkorDbIndex,
        def: &CchIndexDef,
    ) -> Option<&'a mut IndexEntry<Self>> {
        index
            .cch_indexes
            .iter_mut()
            .find(|idx| &idx.index.def == def)
    }

    /// Never called: `IndexId` has no CCH variant until the background build, so no writer
    /// dispatches here.
    fn writer_record(
        &mut self,
        _state: BuildState<()>,
        _changes: CchDirty,
        _place: impl Fn(u64) -> Option<()>,
        _scratch: &mut (),
    ) {
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
    use super::{Cch, CchError, CchIndex, CchIndexDef, Csr, MAX_NODES, TypeId, check_size};
    use crate::graph::{graph::Graph, graphblas::test_init};
    use crate::index::falkordb::cch::testkit::{empty, to_input};
    use crate::index::falkordb::versioned::{BuildState, IndexEntry, VersionedIndex};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;

    fn hash_of(def: &CchIndexDef) -> u64 {
        let mut h = DefaultHasher::new();
        def.hash(&mut h);
        h.finish()
    }

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
    fn equal_sets_are_equal_and_hash_equal() {
        let a = CchIndexDef::new(vec![TypeId(3), TypeId(1), TypeId(3)], 9);
        let b = CchIndexDef::new(vec![TypeId(1), TypeId(3)], 9);
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn different_weight_attr_is_a_different_index() {
        let a = CchIndexDef::new(vec![TypeId(1)], 0);
        let b = CchIndexDef::new(vec![TypeId(1)], 1);
        assert_ne!(a, b);
    }

    #[test]
    fn different_rel_sets_are_different_indexes() {
        let a = CchIndexDef::new(vec![TypeId(1), TypeId(2)], 0);
        let b = CchIndexDef::new(vec![TypeId(1)], 0);
        assert_ne!(a, b);
    }

    /// The ceiling is the engine's own, tested at the boundary rather than with a graph
    /// that large.
    #[test]
    fn check_size_holds_the_line_at_the_graph_limit() {
        assert_eq!(check_size(0), Ok(()));
        assert_eq!(check_size(MAX_NODES), Ok(()));
        assert_eq!(check_size(MAX_NODES + 1), Err(CchError::GraphTooLarge));
    }

    /// `Csr::row` is the slice `p[u]..p[u + 1]` of `i`, empty rows included, and refuses a
    /// row that does not exist rather than reading the next one.
    #[test]
    fn csr_row_slices_one_row() {
        // 0: [1, 2]   1: []   2: [0]
        let csr = Csr {
            n: 3,
            p: Box::new([0, 2, 2, 3]),
            i: Box::new([1, 2, 0]),
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
        let csr = |n: usize, p: &[u64], i: &[u64]| Csr {
            n,
            p: p.into(),
            i: i.into(),
        };
        let clean = csr(3, &[0, 1, 3, 4], &[1, 0, 2, 1]);
        assert!(clean.is_well_formed() && clean.is_symmetric_simple());
        for bad in [
            csr(3, &[0, 1, 3], &[1, 0, 2]),       // n offsets, not n + 1
            csr(3, &[1, 1, 3, 4], &[1, 0, 2, 1]), // does not start at 0
            csr(3, &[0, 1, 3, 3], &[1, 0, 2, 1]), // does not end at i.len()
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
    /// Reaches past the (still unimplemented) DDL by pushing onto the field directly, which is
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
        let cch = Arc::new(Cch::build(&to_input(&empty(0))).expect("empty hierarchy"));
        g.falkordb_index.cch_indexes.push(IndexEntry::new(
            CchIndex {
                def: def.clone(),
                cch: Arc::clone(&cch),
            },
            BuildState::Ready,
        ));

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
