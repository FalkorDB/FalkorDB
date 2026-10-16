//! Customizable Contraction Hierarchies (CCH) path index.
//!
//! A graph version holds its indexes as `Arc<Cch>`; the `Graph` methods in `index.rs`
//! create, drop, list and rebuild them.
//!
//! Layout (one file per area, so changes to different areas do not touch each other's files):
//! - `order.rs`     Phase 1: nested-dissection ordering and triangulation (#3208)
//! - `customize.rs` Phase 2: customization, `WeightArr` (#3209, #3217)
//! - `query.rs`     bidirectional upward search and path unpacking (#3210)
//! - `input.rs`     `Graph` -> [`CchInput`] (#3211)
//! - `index.rs`     the `Graph` API over [`CchIndex`] (#3212)
//!
//! The types themselves live here in `mod.rs` because every part needs them; the other
//! files are where the `impl` blocks land.

pub mod customize;
pub mod index;
pub mod input;
pub mod order;
pub mod partial;
pub mod query;
#[cfg(test)]
pub(crate) mod testkit;

use std::sync::Arc;

use crate::graph::graph::TypeId;
use customize::Metric;
use order::Topology;
use thiserror::Error;

/// The largest node count (`CchInput::n`, the graph's node capacity) a hierarchy is built
/// for; above it [`Cch::build`] refuses with [`CchError::GraphTooLarge`]. It is the ordering
/// backend's own index width (`i32::MAX`, `2^31 - 1`), so [`check_size`] has one node limit
/// rather than a lower one with the backend's as a backstop; the `u32` id width sits above
/// it.
///
/// A limit, not a promise: the CSR offset arrays are dense, `n + 1` offsets each (`4 * n`
/// bytes for the input's `u32` offsets, `8 * n` for the hierarchy's `u64` ones, ~16 GiB
/// per array at the ceiling), and the hierarchy holds several arrays of that length, so in
/// practice memory is what stops a graph well before this.
pub const MAX_NODES: usize = i32::MAX as usize;

/// The largest number of entries (`skeleton.i.len()`) the skeleton may hold; above it
/// [`Cch::build`] refuses with [`CchError::GraphTooLarge`]. This is the ordering backend's
/// index width again (`i32::MAX`, `2^31 - 1` = 2,147,483,647): `feral-metis` addresses its
/// adjacency array with `i32`, so the backend cannot take more.
///
/// **This is the limit a large graph reaches first**, well before [`MAX_NODES`]: a graph
/// of 200 million nodes with an average degree of 10 (2 billion entries) is already near
/// it, while 2 billion nodes would need an average degree of about 1.
///
/// What is counted, exactly:
/// * **Skeleton entries, not graph edges.** The skeleton is symmetric, so it stores each
///   connected node pair `{u, v}` twice, once per direction. At most
///   `i32::MAX / 2` = 1,073,741,823 pairs fit; that is the real "edge" limit.
/// * **Pairs, not relationships.** Parallel edges, edges in both directions and edges of
///   several indexed types between the same two nodes collapse into one pair, and
///   self-loops are not counted at all (see [`CchInput`]). A graph with more relationships
///   than this is accepted if it has fewer connected pairs.
/// * **Not a CSR limit.** It does fit the input's `u32` offsets ([`CchInput`] stores
///   `Csr<u32>`), but the limit comes from the ordering backend alone; a backend with
///   64-bit indices would lift it, and the input would then need `u64` offsets.
///
/// So, to the question "are graphs with this many edges rejected?": yes, with
/// [`CchError::GraphTooLarge`], through [`check_size`]. As with [`MAX_NODES`] it is a
/// limit, not a promise: a skeleton near it is already about 8.6 GB of `u32` column ids
/// before the weights and the hierarchy, so memory stops a graph earlier in practice.
pub const MAX_SKELETON_ENTRIES: u64 = i32::MAX as u64;

/// Whether a graph of `n` nodes and `nnz` skeleton entries is within what a hierarchy is
/// built for: at most [`MAX_NODES`] nodes, and at most [`MAX_SKELETON_ENTRIES`] skeleton
/// entries (twice the number of connected node pairs; the ordering backend's index
/// width). Standalone so the ceilings are testable without allocating a graph that size;
/// [`Cch::build`] calls it before touching any array.
///
/// # Errors
/// [`CchError::GraphTooLarge`] past either limit.
pub fn check_size(
    n: usize,
    nnz: u64,
) -> Result<(), CchError> {
    if n > MAX_NODES || nnz > MAX_SKELETON_ENTRIES {
        return Err(CchError::GraphTooLarge);
    }
    Ok(())
}

/// Compressed sparse row, generic over its two integer widths: offsets `P` (see
/// [`CsrOffset`]) and column ids `I` (see [`CsrIndex`]).
///
/// The defaults are the widths that always fit: `u64` offsets and `u32` node ids (a graph
/// above [`MAX_NODES`] nodes is refused with [`CchError::GraphTooLarge`]). [`CchInput`]
/// narrows the offsets to `u32`, which holds because a skeleton above
/// [`MAX_SKELETON_ENTRIES`] (below `2^31`) entries is refused too. The chordal supergraph
/// built on top keeps `u64` offsets: fill-in can take it past `2^32` arcs on a skeleton
/// far below that ceiling.
pub struct Csr<P = u64, I = u32> {
    pub n: usize,
    /// `n + 1` offsets into `i`.
    pub p: Vec<P>,
    /// Column ids, sorted and duplicate-free within each row.
    pub i: Vec<I>,
}

/// An unsigned integer type a [`Csr`] stores its offsets (`p`) in: `u32` when the entry
/// count is known to stay below `2^32`, `u64` otherwise.
pub trait CsrOffset: Copy + Ord + Into<u64> + 'static {
    /// The offset as a slice index into `i`. Lossless: an offset never exceeds the entry
    /// count, which is a `usize`.
    fn to_usize(self) -> usize;
}

/// An unsigned integer type a [`Csr`] stores its column ids (`i`) in. Every CCH array uses
/// `u32`, the node id width; `u64` is there so the type is not tied to it.
pub trait CsrIndex: Copy + Ord + Into<u64> + 'static {
    /// The column id as a slice index (a row of a [`Csr`] with `n` rows). Lossless: a
    /// column id is below `n`, which is a `usize`.
    fn to_usize(self) -> usize;
}

impl CsrOffset for u32 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl CsrOffset for u64 {
    #[allow(clippy::cast_possible_truncation)]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl CsrIndex for u32 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl CsrIndex for u64 {
    #[allow(clippy::cast_possible_truncation)]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl<P: CsrOffset, I: CsrIndex> Csr<P, I> {
    /// The column ids of row `u`: `i[p[u]..p[u + 1]]`, sorted and duplicate-free.
    ///
    /// `u` is a `usize` because it indexes `p`, like every other slice index here; callers
    /// holding a `u32` node id write `row(u as usize)`.
    ///
    /// # Panics
    /// If `u >= n`, or an offset is out of range of `i`. Nothing else is checked: `p` must
    /// hold `n + 1` non-decreasing offsets ending at `i.len()`, as every `Csr` built by this
    /// module does. A malformed `p` that stays within bounds yields the wrong slice instead
    /// of panicking.
    #[must_use]
    pub fn row(
        &self,
        u: usize,
    ) -> &[I] {
        &self.i[self.p[u].to_usize()..self.p[u + 1].to_usize()]
    }
}

/// Directed weights: same shape as a [`Csr`], with one weight per entry.
pub struct WeightCsr {
    pub csr: Csr<u32>,
    pub w: Vec<f64>,
}

/// What `Graph` hands the hierarchy builder (built by #3211, consumed by #3208 and #3209).
pub struct CchInput {
    pub n: usize,
    /// Symmetric, no self-loops. Phase 1 only. At most [`MAX_SKELETON_ENTRIES`] entries
    /// (`skeleton.i.len()`, counting each connected pair twice); more is
    /// [`CchError::GraphTooLarge`].
    pub skeleton: Csr<u32>,
    /// Directed. One entry per node pair: the cheapest weight over all parallel edges and all
    /// indexed relationship types. A missing or non-numeric attribute counts as 1.0.
    /// An absent entry means "no edge". Weights must be finite and >= 0.
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
    /// More than [`MAX_NODES`] nodes (2^31 - 1), or more than [`MAX_SKELETON_ENTRIES`]
    /// skeleton entries (2^31 - 1, that is, up to 1,073,741,823 connected node pairs: the
    /// skeleton stores each pair in both directions). The entry limit is the one a large
    /// graph meets first; it is the ordering backend's `i32` index width. See
    /// [`check_size`].
    #[error("graph is too large for a CCH index")]
    GraphTooLarge,
    /// A weight was negative, infinite or NaN: weights must be finite and `>= 0`.
    #[error("a weight is negative, infinite or NaN")]
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
    /// [`CchError::GraphTooLarge`] above [`MAX_NODES`] nodes or [`MAX_SKELETON_ENTRIES`]
    /// skeleton entries (see [`check_size`]), [`CchError::InvalidWeight`] for a negative,
    /// infinite or NaN weight, and [`CchError::Internal`] if the ordering backend fails.
    pub fn build(input: &CchInput) -> Result<Self, CchError> {
        Self::build_impl(input)
    }

    /// Phase 2 only, same structure, new weights.
    ///
    /// # Errors
    /// [`CchError::InvalidWeight`] for a negative, infinite or NaN weight, and
    /// [`CchError::Internal`] if `weights` is not shaped for this hierarchy — a different
    /// node count, or an edge between two nodes it has no arc for. Both need a rebuild.
    /// Nothing is written on error.
    pub fn customize(
        &mut self,
        weights: &WeightCsr,
    ) -> Result<(), CchError> {
        self.customize_impl(weights)
    }

    /// Cheapest weight and node sequence. Mapping node pairs to edge ids is the caller's job.
    ///
    /// `None` when `dst` is unreachable from `src`, or when either is not a node of this
    /// hierarchy. `src == dst` is the single-node path of weight `0.0`.
    #[must_use]
    pub fn query(
        &self,
        src: u32,
        dst: u32,
    ) -> Option<(f64, Vec<u32>)> {
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

/// One CCH index as held by `Graph`. `new_version` clones the `Arc`.
#[derive(Clone)]
pub struct CchIndex {
    pub def: CchIndexDef,
    pub cch: Arc<Cch>,
}

mod maint;
pub use maint::CchDirty;

#[cfg(test)]
mod tests {
    use super::{
        Cch, CchError, CchIndex, CchIndexDef, Csr, MAX_NODES, MAX_SKELETON_ENTRIES, TypeId,
        check_size,
    };
    use crate::graph::{graph::Graph, graphblas::test_init};
    use crate::index::cch::testkit::{empty, to_input};
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

    /// The ceiling is a pure function of the two sizes, so it is tested here at the
    /// boundary values rather than with a graph that large.
    #[test]
    fn check_size_holds_the_line_at_both_ceilings() {
        assert_eq!(check_size(0, 0), Ok(()));
        assert_eq!(check_size(MAX_NODES, MAX_SKELETON_ENTRIES), Ok(()));
        assert_eq!(check_size(MAX_NODES + 1, 0), Err(CchError::GraphTooLarge));
        assert_eq!(
            check_size(10, MAX_SKELETON_ENTRIES + 1),
            Err(CchError::GraphTooLarge)
        );
        assert_eq!(
            check_size(u32::MAX as usize + 1, 0),
            Err(CchError::GraphTooLarge)
        );
    }

    /// `Csr::row` is the slice `p[u]..p[u + 1]` of `i`, empty rows included, and refuses a
    /// row that does not exist rather than reading the next one.
    #[test]
    fn csr_row_slices_one_row() {
        // 0: [1, 2]   1: []   2: [0]
        let csr: Csr = Csr {
            n: 3,
            p: vec![0, 2, 2, 3],
            i: vec![1, 2, 0],
        };
        assert_eq!(csr.row(0), &[1, 2]);
        assert!(csr.row(1).is_empty());
        assert_eq!(csr.row(2), &[0]);
        assert!(std::panic::catch_unwind(|| csr.row(3).len()).is_err());
    }

    /// `new_version` must carry the indexes forward by sharing the `Arc`, not by copying the
    /// hierarchy. Drop `cch_indexes` from `Graph::new_version` and this fails on the length;
    /// deep-copy the hierarchy instead of cloning the `Arc` and it fails on `ptr_eq`.
    ///
    /// Pushes onto the field directly, so it checks `new_version` alone, independent of
    /// `create_cch_index` and the graph input.
    #[test]
    fn new_version_shares_the_hierarchy() {
        test_init::ensure_init();
        let mut g = Graph::new(16, 16, 1, 0, "cch_new_version");
        assert!(g.cch_indexes.is_empty(), "a fresh graph holds no CCH index");

        let def = CchIndexDef::new(vec![TypeId(1)], 0);
        let cch = Arc::new(Cch::build(&to_input(&empty(0))).expect("empty hierarchy"));
        g.cch_indexes.push(CchIndex {
            def: def.clone(),
            cch: Arc::clone(&cch),
        });

        let next = g.new_version();
        assert_eq!(next.cch_indexes.len(), 1);
        assert_eq!(next.cch_indexes[0].def, def);
        assert!(
            Arc::ptr_eq(&next.cch_indexes[0].cch, &cch),
            "new_version must share the hierarchy, not copy it"
        );
    }
}
