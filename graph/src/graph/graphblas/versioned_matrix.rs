//! Copy-on-write sparse matrix with MVCC delta tracking.
//!
//! This module provides [`VersionedMatrix`], which wraps a base [`Matrix`] with
//! two delta matrices to track pending additions and deletions. This is the
//! building block for snapshot isolation: readers see the committed base state
//! while writers accumulate changes in separate delta matrices.
//!
//! ## Internal Structure
//!
//! ```text
//!   VersionedMatrix
//!     |
//!     |-- m   Cow<Matrix>   Base matrix (committed / shared with readers)
//!     |-- dp  Cow<Matrix>   Delta-plus  (pending additions)
//!     |-- dm  Cow<Matrix>   Delta-minus (pending deletions)
//!
//!   Effective state = (m UNION dp) MINUS dm
//! ```
//!
//! Each inner matrix is wrapped in [`Cow`] (copy-on-write). When a new version
//! is created via [`Dup`], the `Cow` clones share the underlying `Arc<Matrix>`
//! until a mutation triggers a deep copy.
//!
//! ## Read Path
//!
//! ```text
//!   get(i, j):
//!     m has (i,j)?
//!       yes --> dm has (i,j)?  -->  yes: None (deleted)
//!                                   no:  Some(true)
//!       no  --> dp has (i,j)?  -->  yes: Some(true)
//!                                   no:  None
//! ```
//!
//! ## Write Path
//!
//! ```text
//!   set(i, j):
//!     m has (i,j)?
//!       yes --> remove (i,j) from dm   (un-delete)
//!       no  --> add (i,j) to dp        (new addition)
//!
//!   remove(i, j):
//!     m has (i,j)?
//!       yes --> add (i,j) to dm        (mark deleted)
//!       no  --> remove (i,j) from dp   (undo pending add)
//! ```
//!
//! UINT64 matrices additionally support in-place *value* updates of a
//! committed entry: `set` masks the old entry in `dm` and writes the new
//! value to `dp`, so `dp` never shadows a live `m` entry (see the
//! `VersionedMatrix<u64>` impl).
//!
//! ## Flush
//!
//! When delta matrices exceed 10,000 entries, [`flush`](VersionedMatrix::flush)
//! merges them into the base matrix (`dp` via element-wise add, `dm` via
//! masked removal) and clears the deltas.
//!
//! ## Iterator
//!
//! [`Iter`] chains the base matrix iterator (skipping entries present in `dm`)
//! with the delta-plus iterator, producing the effective state without
//! materializing a merged matrix.

use super::{
    GxB_Print_Level,
    matrix::{self, Dup, Matrix},
    serialization::{Decode, Encode, Reader, Writer},
};
use crate::graph::{
    cow::Cow,
    graphblas::matrix::{BoolExtract, IterExtract, Uint64Extract},
};

/// A matrix with MVCC delta tracking for snapshot isolation.
///
/// Wraps a base matrix with separate matrices for tracking additions
/// and deletions, enabling concurrent reads during writes.
///
/// The type parameter `T` tags the element type of the *valued* layers (base
/// `m` and delta-plus `dp`): `bool` for pure structure/presence, `u64` for
/// valued matrices such as inline edge ids. The delta-minus `dm` is always a
/// `bool` deletion mask. `T` defaults to `bool`.
pub struct VersionedMatrix<T> {
    /// Base committed matrix
    m: Cow<Matrix<T>>,
    /// Delta-plus: edges added in current transaction
    dp: Cow<Matrix<T>>,
    /// Delta-minus: edges removed in current transaction (always a bool mask)
    dm: Cow<Matrix<bool>>,
}

// Manual `Clone` so it holds for every `V` without a `V: Clone` bound (see the
// note on `Matrix`'s manual `Clone`).
impl<T> Clone for VersionedMatrix<T> {
    fn clone(&self) -> Self {
        Self {
            m: self.m.clone(),
            dp: self.dp.clone(),
            dm: self.dm.clone(),
        }
    }
}

unsafe impl<T> Send for VersionedMatrix<T> {}
unsafe impl<T> Sync for VersionedMatrix<T> {}

impl<T> VersionedMatrix<T> {
    /// Base committed matrix. Element values are typed by `T`; structural
    /// consumers (`ANY_PAIR` mxm, masks) may use it regardless of `T`.
    #[must_use]
    pub fn m(&self) -> &Matrix<T> {
        &self.m
    }

    /// Delta-plus matrix (pending additions), element values typed by `T`.
    #[must_use]
    pub fn dp(&self) -> &Matrix<T> {
        &self.dp
    }

    /// Delta-minus deletion mask (always `bool`).
    #[must_use]
    pub fn dm(&self) -> &Matrix<bool> {
        &self.dm
    }

    #[must_use]
    pub fn nrows(&self) -> u64 {
        self.m.nrows()
    }

    #[must_use]
    pub fn ncols(&self) -> u64 {
        self.m.ncols()
    }

    pub fn resize(
        &mut self,
        nrows: u64,
        ncols: u64,
    ) {
        self.wait();
        self.m.resize(nrows, ncols);
        self.dp.resize(nrows, ncols);
        self.dm.resize(nrows, ncols);
    }

    pub fn wait(&self) {
        debug_assert!(!self.m.pending());
        self.dp.wait();
        self.dm.wait();
    }

    /// Wait on all three internal matrices (m, dp, dm).
    /// Used for fork safety — ensures no GrB internal locks are held.
    pub fn wait_all(&self) {
        self.m.wait();
        self.dp.wait();
        self.dm.wait();
    }

    /// Returns true if every internal matrix has no pending GraphBLAS
    /// operations — i.e. wait_all was effective.
    #[must_use]
    pub fn is_synced(&self) -> bool {
        self.m.is_synced() && self.dp.is_synced() && self.dm.is_synced()
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.m.memory_usage() + self.dp.memory_usage() + self.dm.memory_usage()
    }

    /// Materialize the effective structure as a `bool` matrix: `(m - dm) ∪ dp`,
    /// values discarded. Works for both bool and valued (uint64) bases — only
    /// structure is preserved, which is all the structure-only consumers
    /// (traversal, relationship-matrix building) need.
    #[must_use]
    pub fn extract(&self) -> Matrix<bool> {
        self.wait();
        let mut m = Matrix::<bool>::new(self.m.nrows(), self.m.ncols());
        m.element_wise_add(None, None, Some(&*self.m), None);
        if self.dm.nvals() > 0 {
            m.remove_all(&self.dm);
        }
        if self.dp.nvals() > 0 {
            m.element_wise_add(None, None, Some(&*self.dp), None);
        }
        m
    }

    /// Effective entry count: `|m| + |dp| − |dm|`.
    ///
    /// Relies on the invariants maintained by [`Self::set`] /
    /// [`Self::remove`]: `dm ⊆ m` and `dp ∩ (m ∖ dm) = ∅` (bool matrices
    /// keep the stricter `dp ∩ m = ∅`; `u64` matrices mask any in-place
    /// update in `dm`).
    #[must_use]
    pub fn nvals(&self) -> u64 {
        self.wait();
        self.m.nvals() + self.dp.nvals() - self.dm.nvals()
    }

    pub fn print(
        &self,
        level: GxB_Print_Level,
    ) {
        self.m.print(level);
        self.dp.print(level);
        self.dm.print(level);
    }

    /// Bulk-remove all entries matching a mask matrix.
    ///
    /// Equivalent to calling `remove(i, j)` for every entry `(i, j)` in `mask`,
    /// but executes in two GraphBLAS bulk operations instead of N individual calls:
    /// - Entries in base `m` matching `mask` are marked deleted in `dm`
    /// - Entries in delta-plus `dp` matching `mask` are removed from `dp`
    pub fn remove_mask(
        &mut self,
        mask: &Matrix<bool>,
    ) {
        // dm |= (m & mask): mark deleted every committed entry that `mask`
        // selects. The set added to `dm` is the intersection `m ∩ mask`, which
        // is symmetric — so `m`'s values are irrelevant and it can flow through
        // the (structure-only, `PAIR`-semiring) generic `b` slot while the bool
        // `mask` acts as the GraphBLAS write mask.
        self.dm
            .element_wise_add(Some(mask), None, Some(&*self.m), None);
        // dp &= ~mask: remove entries from dp that exist in mask
        self.dp.remove_all(mask);
    }
}

impl VersionedMatrix<bool> {
    #[must_use]
    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> Iter {
        self.wait();
        Iter::<BoolExtract>::new(self, min_row, max_row)
    }

    pub fn remove(
        &mut self,
        i: u64,
        j: u64,
    ) {
        if self.m.get(i, j).is_some() {
            debug_assert!(self.dp.get(i, j).is_none());
            self.dm.set(i, j, true);
        } else {
            self.dp.remove(i, j);
        }
    }

    #[must_use]
    pub fn get(
        &self,
        i: u64,
        j: u64,
    ) -> Option<bool> {
        self.wait();
        self.m.get(i, j).map_or_else(
            || self.dp.get(i, j),
            |value| {
                if self.dm.get(i, j).is_some() {
                    None
                } else {
                    Some(value)
                }
            },
        )
    }

    pub fn set(
        &mut self,
        i: u64,
        j: u64,
        value: bool,
    ) {
        debug_assert!(!self.m.pending());
        if self.m.get(i, j).is_some() {
            debug_assert!(self.dp.get(i, j).is_none());
            self.dm.remove(i, j);
        } else {
            debug_assert!(self.dm.get(i, j).is_none());
            self.dp.set(i, j, value);
        }
    }

    #[must_use]
    pub fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        Self {
            m: Cow::new(Matrix::<bool>::new(nrows, ncols)),
            dp: Cow::new(Matrix::<bool>::new(nrows, ncols)),
            dm: Cow::new(Matrix::<bool>::new(nrows, ncols)),
        }
    }

    /// Wrap an owned `Matrix` as a `VersionedMatrix` with empty delta-plus /
    /// delta-minus.  Used when callers materialize a merged matrix and then
    /// want to expose it through the versioned-matrix iter API without the
    /// dup overhead of re-building inside the versioned wrapper.
    #[must_use]
    pub fn from_matrix(m: Matrix<bool>) -> Self {
        let nrows = m.nrows();
        let ncols = m.ncols();
        Self {
            m: Cow::new(m),
            dp: Cow::new(Matrix::<bool>::new(nrows, ncols)),
            dm: Cow::new(Matrix::<bool>::new(nrows, ncols)),
        }
    }

    pub fn flush(&mut self) {
        self.wait();
        if self.dp.nvals() >= 10000 {
            self.m.element_wise_add(None, None, Some(&self.dp), None);
            self.dp.clear();
        }
        if self.dm.nvals() >= 10000 {
            self.m.remove_all(&self.dm);
            self.dm.clear();
        }
    }

    /// Set multiple entries, checking dm emptiness once upfront.
    ///
    /// If dm is empty, uses the fast path (1 FFI call per entry).
    /// Otherwise falls back to the full `set` path (2+ FFI calls per entry).
    pub fn set_all(
        &mut self,
        entries: impl Iterator<Item = (u64, u64)>,
    ) {
        if self.dm.nvals() == 0 {
            for (i, j) in entries {
                self.dp.set(i, j, true);
            }
        } else {
            for (i, j) in entries {
                self.set(i, j, true);
            }
        }
    }
}

impl<T> Dup<Self> for VersionedMatrix<T> {
    fn dup(&self) -> Self {
        Self {
            m: self.m.new_version(),
            dp: self.dp.new_version(),
            dm: self.dm.new_version(),
        }
    }
}

/// UINT64-valued overlay support.
///
/// A UINT64 `VersionedMatrix` carries a `u64` *value* at each `(i, j)` (e.g. an
/// edge id) rather than a plain boolean presence bit. The base `m` and
/// delta-plus `dp` are UINT64-typed; the delta-minus `dm` stays BOOL (it is a
/// pure deletion mask).
///
/// Unlike the bool model, a committed entry's *value* can change in place
/// (e.g. an edge id changes on multi-edge promotion, or delete-then-re-add
/// within one transaction). [`Self::set`] handles this by masking the old
/// committed entry in `dm` and writing the new value to `dp`, preserving the
/// no-shadow invariant `dp ∩ (m ∖ dm) = ∅`: the effective content is always
/// `(m ∖ dm) ∪ dp` with the union disjoint, so iterators and `nvals()` need
/// no overlap handling. Note this means `dp ∩ dm` may be non-empty (an
/// in-place update has the pair in both), unlike the bool model.
impl VersionedMatrix<u64> {
    /// Construct a UINT64-valued versioned matrix: `m`/`dp` UINT64, `dm` BOOL.
    #[must_use]
    pub fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        Self {
            m: Cow::new(Matrix::<u64>::new(nrows, ncols)),
            dp: Cow::new(Matrix::<u64>::new(nrows, ncols)),
            dm: Cow::new(Matrix::<bool>::new(nrows, ncols)),
        }
    }

    /// Stream effective UINT64 `(row, col, value)` triples over rows in
    /// `[min_row, max_row]`: `(m ∖ dm) ∪ dp`.
    #[must_use]
    pub fn iter(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> Iter<Uint64Extract> {
        self.wait();
        Iter::<Uint64Extract>::new(self, min_row, max_row)
    }

    /// Structure-only `(row, col)` iterator over the effective matrix,
    /// ignoring the stored edge-id values. Same semantics as the bool
    /// [`VersionedMatrix::iter`].
    #[must_use]
    pub fn structural_iter(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> Iter {
        self.wait();
        Iter::<BoolExtract>::new(self, min_row, max_row)
    }

    /// Write `value` at `(i, j)`: the new value lands in `dp`, and if the
    /// committed base holds an entry at `(i, j)` it is masked in `dm` so the
    /// old value never shadows the new one (`dp ∩ (m ∖ dm) = ∅`).
    pub fn set(
        &mut self,
        i: u64,
        j: u64,
        value: u64,
    ) {
        debug_assert!(!self.m.pending());
        self.dp.set(i, j, value);
        if self.m.contains(i, j) {
            self.dm.set(i, j, true);
        }
    }

    /// Bulk UINT64 set. Unlike per-element [`VersionedMatrix::set`] callers,
    /// this checks base emptiness once up front and never calls `get`/`wait`
    /// per entry, so it stays O(n) for a batch of `n` writes (critical for
    /// bulk edge creation).
    pub fn set_all(
        &mut self,
        entries: impl Iterator<Item = (u64, u64, u64)>,
    ) {
        debug_assert!(!self.m.pending());
        if self.m.nvals() == 0 {
            for (i, j, v) in entries {
                self.dp.set(i, j, v);
            }
        } else {
            for (i, j, v) in entries {
                self.dp.set(i, j, v);
                if self.m.contains(i, j) {
                    self.dm.set(i, j, true);
                }
            }
        }
    }

    /// Effective UINT64 value at `(i, j)`: `dp` wins, then `m` unless masked by
    /// `dm`. Returns `None` if absent or deleted.
    #[must_use]
    pub fn get(
        &self,
        i: u64,
        j: u64,
    ) -> Option<u64> {
        self.wait();
        if let Some(v) = self.dp.get(i, j) {
            return Some(v);
        }
        if self.dm.nvals() != 0 && self.dm.get(i, j).is_some() {
            return None;
        }
        self.m.get(i, j)
    }

    /// Remove `(i, j)` (value-agnostic): drop any pending add and mask the
    /// committed entry as deleted.
    pub fn remove(
        &mut self,
        i: u64,
        j: u64,
    ) {
        if self.dp.get(i, j).is_some() {
            self.dp.remove(i, j);
        }
        if self.m.get(i, j).is_some() {
            self.dm.set(i, j, true);
        }
    }
}

impl VersionedMatrix<bool> {
    /// Transposes the matrix.
    ///
    /// # Returns
    /// A new matrix that is the transpose of the original.
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self {
            m: Cow::new(self.m.transpose()),
            dp: Cow::new(self.dp.transpose()),
            dm: Cow::new(self.dm.transpose()),
        }
    }
}

impl<V> Encode<19> for VersionedMatrix<V> {
    fn encode(
        &self,
        w: &mut dyn Writer,
    ) {
        self.m.encode(w);
        self.dp.encode(w);
        self.dm.encode(w);
    }
}

impl<V> Decode<19> for VersionedMatrix<V> {
    fn decode(r: &mut dyn Reader) -> Result<Self, String> {
        let m = Matrix::<V>::decode(r)?;
        let dp = Matrix::<V>::decode(r)?;
        let dm = Matrix::<bool>::decode(r)?;
        Ok(Self {
            m: Cow::new(m),
            dp: Cow::new(dp),
            dm: Cow::new(dm),
        })
    }
}

pub struct Iter<E: IterExtract = BoolExtract> {
    mit: matrix::Iter<E>,
    /// Delta-plus iterator. Lazily left `None` when `dp` is empty (the common
    /// read-only hot path on a freshly loaded graph) so we skip allocating and
    /// freeing a `GxB_Iterator` that would never yield anything. `dp` is a
    /// stable read snapshot for the life of this iterator, so once `None` it
    /// stays `None` across `seek` calls.
    dpit: Option<matrix::Iter<E>>,
    dm: Cow<Matrix<bool>>,
    /// True when the deletion mask is empty, so the `m` phase can stream `mit`
    /// without per-edge `dm` lookups. Hot path for read-only queries on a
    /// freshly loaded graph.
    dm_empty: bool,
}

unsafe impl<E: IterExtract> Send for Iter<E> {}
unsafe impl<E: IterExtract> Sync for Iter<E> {}

impl<E: IterExtract> Iter<E> {
    /// Streams the effective content `(m ∖ dm) ∪ dp` — a disjoint union
    /// thanks to the no-shadow invariant, so the `m` phase only needs the
    /// `dm` mask. Valid for a `VersionedMatrix` of any element type when
    /// `E = BoolExtract` (only the sparsity pattern is read).
    fn new<V>(
        vm: &VersionedMatrix<V>,
        min_row: u64,
        max_row: u64,
    ) -> Self {
        Self {
            mit: matrix::Iter::new(&vm.m, min_row, max_row),
            dpit: if vm.dp.nvals() == 0 {
                None
            } else {
                Some(matrix::Iter::new(&vm.dp, min_row, max_row))
            },
            dm: vm.dm.clone(),
            dm_empty: vm.dm.nvals() == 0,
        }
    }

    /// Re-seek both inner GraphBLAS iterators to a new row range without
    /// re-allocating them. Hot-loop callers (e.g. `CondTraverseOp` and
    /// `ExpandInto` looking up edges by `(src, dst)`) use this to amortize
    /// the per-pair iterator allocation.
    pub fn seek(
        &mut self,
        min_row: u64,
        max_row: u64,
    ) {
        self.mit.seek(min_row, max_row);
        if let Some(dpit) = &mut self.dpit {
            dpit.seek(min_row, max_row);
        }
    }
}

impl Iterator for Iter<BoolExtract> {
    type Item = (u64, u64);

    /// Advances the iterator and returns the next element in the matrix.
    ///
    /// # Returns
    /// - `Some((u64, u64))`: The next element in the matrix.
    /// - `None`: The iterator is depleted.
    fn next(&mut self) -> Option<Self::Item> {
        for (i, j) in &mut self.mit {
            if !self.dm_empty && self.dm.get(i, j).is_some() {
                continue; // deleted
            }
            return Some((i, j));
        }
        self.dpit.as_mut().and_then(Iterator::next)
    }
}

impl Iterator for Iter<Uint64Extract> {
    type Item = (u64, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        for (i, j, v) in &mut self.mit {
            if !self.dm_empty && self.dm.contains(i, j) {
                continue; // deleted (or overridden in-place; dp has the new value)
            }
            return Some((i, j, v));
        }
        self.dpit.as_mut().and_then(Iterator::next)
    }
}
