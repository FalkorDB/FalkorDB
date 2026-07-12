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
use crate::graph::cow::Cow;

mod sealed {
    pub trait Sealed {}
    impl Sealed for bool {}
    impl Sealed for u64 {}
}

/// Element types storable in a [`VersionedMatrix`]: `bool` (pure structure)
/// and `u64` (valued entries such as inline edge ids). Sealed — the delta
/// invariants below are only proven for these two types.
///
/// The distinguishing capability is the **dp overlay**: `u64` writes may
/// leave an entry in *both* `m` and `dp` (an in-place value update of a
/// committed cell — `dp` wins on read), whereas `bool` `set`/`remove` always
/// keep `dp ∩ m = ∅`. Structural consumers ([`Iter`], `nvals`) query this via
/// [`Element::dp_overlay`] instead of guessing from the type.
pub trait Element: sealed::Sealed + Sized {
    /// The delta-plus handle when this element type permits `dp ∩ m ≠ ∅`,
    /// `None` when disjointness is guaranteed. Only `u64` has overlay
    /// semantics, so the overlay handle is concretely `Matrix<u64>`; callers
    /// use it to skip base entries shadowed by `dp`.
    fn dp_overlay(dp: &Cow<Matrix<Self>>) -> Option<Cow<Matrix<u64>>>;
}

impl Element for bool {
    fn dp_overlay(_dp: &Cow<Matrix<Self>>) -> Option<Cow<Matrix<u64>>> {
        None
    }
}

impl Element for u64 {
    fn dp_overlay(dp: &Cow<Matrix<Self>>) -> Option<Cow<Matrix<u64>>> {
        Some(dp.clone())
    }
}

/// A matrix with MVCC delta tracking for snapshot isolation.
///
/// Wraps a base matrix with separate matrices for tracking additions
/// and deletions, enabling concurrent reads during writes.
///
/// The type parameter `V` tags the element type of the *valued* layers (base
/// `m` and delta-plus `dp`): `bool` for pure structure/presence, `u64` for
/// valued matrices such as inline edge ids. The delta-minus `dm` is always a
/// `bool` deletion mask. `V` defaults to `bool`.
pub struct VersionedMatrix<V> {
    /// Base committed matrix
    m: Cow<Matrix<V>>,
    /// Delta-plus: edges added in current transaction
    dp: Cow<Matrix<V>>,
    /// Delta-minus: edges removed in current transaction (always a bool mask)
    dm: Cow<Matrix<bool>>,
}

// Manual `Clone` so it holds for every `V` without a `V: Clone` bound (see the
// note on `Matrix`'s manual `Clone`).
impl<V> Clone for VersionedMatrix<V> {
    fn clone(&self) -> Self {
        Self {
            m: self.m.clone(),
            dp: self.dp.clone(),
            dm: self.dm.clone(),
        }
    }
}

unsafe impl<V> Send for VersionedMatrix<V> {}
unsafe impl<V> Sync for VersionedMatrix<V> {}

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

    pub fn nrows(&self) -> u64 {
        self.m.nrows()
    }

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
}

impl VersionedMatrix<bool> {
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

    /// Effective entry count: `|m| + |dp| − |dm|`.
    ///
    /// Relies on the bool invariants maintained by [`Self::set`] /
    /// [`Self::remove`]: `dp ∩ m = ∅`, `dm ⊆ m`, `dp ∩ dm = ∅`. (`u64`
    /// matrices relax the first invariant — see the overlay-aware `nvals`
    /// in the UINT64 impl below.)
    pub fn nvals(&self) -> u64 {
        self.wait();
        self.m.nvals() + self.dp.nvals() - self.dm.nvals()
    }
}

impl<V> Dup<Self> for VersionedMatrix<V> {
    fn dup(&self) -> Self {
        Self {
            m: self.m.new_version(),
            dp: self.dp.new_version(),
            dm: self.dm.new_version(),
        }
    }
}

impl<V> VersionedMatrix<V> {
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

    #[must_use]
    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> Iter
    where
        V: Element,
    {
        self.wait();
        Iter::new(self, min_row, max_row)
    }

    /// Materialize the effective structure as a `bool` matrix: `(m - dm) ∪ dp`,
    /// values discarded. Works for both bool and valued (uint64) bases — only
    /// structure is preserved, which is all the structure-only consumers
    /// (traversal, relationship-matrix building) need.
    #[must_use]
    pub fn to_matrix(&self) -> Matrix<bool> {
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

    #[must_use]
    pub fn extract_m_dp(&self) -> (Matrix<bool>, Matrix<bool>) {
        if self.dm.nvals() == 0 {
            // Fast path: no deletions, return dups of m and dp directly
            (self.m.dup(), self.dp.dup())
        } else {
            let mut m = Matrix::<bool>::new(self.m.nrows(), self.m.ncols());
            let mut dp = Matrix::<bool>::new(self.dp.nrows(), self.dp.ncols());
            m.select(&self.dm, &self.m);
            dp.select(&self.dm, &self.dp);
            (m, dp)
        }
    }

    /// Bulk-extract all effective entries as (row, col) arrays.
    ///
    /// Returns `(rows, cols)` from `(m - dm) ∪ dp`, avoiding iterator overhead
    /// on matrices with huge dimensions (e.g., GrB_INDEX_MAX).
    #[must_use]
    pub fn extract_all_tuples(&self) -> (Vec<u64>, Vec<u64>) {
        self.wait();
        if self.dm.nvals() == 0 {
            // Fast path: no deletions, just combine m and dp tuples
            let (mut rows_m, mut cols_m) = self.m.extract_tuples_bool();
            let (rows_dp, cols_dp) = self.dp.extract_tuples_bool();
            rows_m.extend_from_slice(&rows_dp);
            cols_m.extend_from_slice(&cols_dp);
            (rows_m, cols_m)
        } else {
            // Slow path: materialize effective matrix then extract
            let effective = self.to_matrix();
            effective.extract_tuples_bool()
        }
    }

    /// Bulk-extract tuples from base `m` and delta-plus `dp` separately.
    ///
    /// Returns `((m_rows, m_cols), (dp_rows, dp_cols))`.
    /// Only valid when `dm` is empty (asserted in debug builds).
    #[must_use]
    pub fn extract_m_dp_tuples(&self) -> ((Vec<u64>, Vec<u64>), (Vec<u64>, Vec<u64>)) {
        self.wait();
        debug_assert_eq!(self.dm.nvals(), 0, "extract_m_dp_tuples requires empty dm");
        let m_tuples = self.m.extract_tuples_bool();
        let dp_tuples = self.dp.extract_tuples_bool();
        (m_tuples, dp_tuples)
    }
}

impl VersionedMatrix<u64> {
    /// Iterate UINT64 entries with full overlay semantics (`dp` wins over `m`,
    /// minus `dm`) — equivalent to `uint64_iter_range(0, u64::MAX)`.
    ///
    /// Used during RDB decode to read C-produced relation matrices where
    /// single-edge entries store the edge ID as a UINT64 value.
    pub fn uint64_iter(&self) -> impl Iterator<Item = (u64, u64, u64)> + '_ {
        self.uint64_iter_range(0, u64::MAX)
    }

    /// Effective entry count with dp-overlay semantics:
    /// `|m| + |dp| − |dm| − |m ∩ dp|`.
    ///
    /// Unlike the bool variant, `dp` may *shadow* committed `m` entries (an
    /// in-place value update or delete-then-re-add of a committed cell), so the
    /// structural overlap must be subtracted or shadowed cells count twice.
    /// The overlap is computed only when `dp` is non-empty; committed
    /// snapshots with folded deltas pay nothing extra.
    pub fn nvals(&self) -> u64 {
        self.wait();
        let dp_nvals = self.dp.nvals();
        let mut nvals = self.m.nvals() + dp_nvals - self.dm.nvals();
        if dp_nvals > 0 {
            let mut overlap = Matrix::<bool>::new(self.m.nrows(), self.m.ncols());
            overlap.intersect_structure(&self.m, &self.dp);
            nvals -= overlap.nvals();
        }
        nvals
    }
}

impl VersionedMatrix<bool> {
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

/// UINT64-valued overlay support.
///
/// A UINT64 `VersionedMatrix` carries a `u64` *value* at each `(i, j)` (e.g. an
/// edge id) rather than a plain boolean presence bit. The base `m` and
/// delta-plus `dp` are UINT64-typed; the delta-minus `dm` stays BOOL (it is a
/// pure deletion mask).
///
/// Read precedence is **dp wins over m, minus dm**: `dp` is the newest overlay,
/// so a value written in the current transaction (insert *or* update of a
/// committed entry, including delete-then-re-add) is visible immediately while
/// readers on the committed snapshot still see `m`. This is a superset of the
/// bool existence model (which requires `dp` disjoint from `m`); the extra
/// generality is needed because an edge id can change in place (single→multi).
///
/// Because `dp` can shadow `m`, every aggregate over the effective content
/// must be overlap-aware: `nvals()` subtracts `|m ∩ dp|`, and the structural
/// [`Iter`] / valued [`UintIter`] skip `m` entries present in `dp`.
impl VersionedMatrix<u64> {
    /// Construct a UINT64-valued versioned matrix: `m`/`dp` UINT64, `dm` BOOL.
    #[must_use]
    pub fn new_uint64(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        Self {
            m: Cow::new(Matrix::<u64>::new(nrows, ncols)),
            dp: Cow::new(Matrix::<u64>::new(nrows, ncols)),
            dm: Cow::new(Matrix::<bool>::new(nrows, ncols)),
        }
    }

    /// Write `value` at `(i, j)` with dp-overlay semantics: the new value lands
    /// in `dp` (newest), and any pending deletion of `(i, j)` is cleared.
    pub fn set_uint64(
        &mut self,
        i: u64,
        j: u64,
        value: u64,
    ) {
        debug_assert!(!self.m.pending());
        self.dp.set_uint64(i, j, value);
        if self.dm.nvals() != 0 {
            self.dm.remove(i, j);
        }
    }

    /// Bulk UINT64 set with dp-overlay semantics. Unlike per-element
    /// [`VersionedMatrix::set_uint64`] callers, this checks `dm` emptiness once
    /// up front and never calls `get`/`wait` per entry, so it stays O(n) for a
    /// batch of `n` writes (critical for bulk edge creation).
    pub fn set_all_uint64(
        &mut self,
        entries: impl Iterator<Item = (u64, u64, u64)>,
    ) {
        debug_assert!(!self.m.pending());
        if self.dm.nvals() == 0 {
            for (i, j, v) in entries {
                self.dp.set_uint64(i, j, v);
            }
        } else {
            for (i, j, v) in entries {
                self.dp.set_uint64(i, j, v);
                self.dm.remove(i, j);
            }
        }
    }

    /// Effective UINT64 value at `(i, j)`: `dp` wins, then `m` unless masked by
    /// `dm`. Returns `None` if absent or deleted.
    #[must_use]
    pub fn get_uint64(
        &self,
        i: u64,
        j: u64,
    ) -> Option<u64> {
        self.wait();
        if let Some(v) = self.dp.get_uint64(i, j) {
            return Some(v);
        }
        if self.dm.nvals() != 0 && self.dm.get(i, j).is_some() {
            return None;
        }
        self.m.get_uint64(i, j)
    }

    /// Remove `(i, j)` (value-agnostic): drop any pending add and mask the
    /// committed entry as deleted.
    pub fn remove_uint64(
        &mut self,
        i: u64,
        j: u64,
    ) {
        if self.dp.get_uint64(i, j).is_some() {
            self.dp.remove(i, j);
        }
        if self.m.get_uint64(i, j).is_some() {
            self.dm.set(i, j, true);
        }
    }

    /// Collect effective `(row, col, value)` triples (full range) into a `Vec`.
    /// Convenience wrapper over [`VersionedMatrix::uint64_iter_range`].
    #[must_use]
    pub fn collect_uint64(&self) -> Vec<(u64, u64, u64)> {
        self.uint64_iter_range(0, u64::MAX).collect()
    }

    /// Flush UINT64 deltas into the base: apply deletions, then fold `dp` in
    /// with dp winning on overlap. Clears both deltas.
    pub fn flush_uint64(&mut self) {
        self.wait();
        if self.dm.nvals() >= 10000 {
            self.m.remove_all(&self.dm);
            self.dm.clear();
        }
        if self.dp.nvals() >= 10000 {
            self.m.merge_overwrite_uint64(&self.dp);
            self.dp.clear();
        }
    }

    /// Test-only: unconditionally fold UINT64 deltas into the base (simulate a
    /// commit so subsequent writes exercise the dp-overlay-over-committed path).
    #[cfg(test)]
    pub fn force_commit_uint64(&mut self) {
        self.wait();
        if self.dm.nvals() != 0 {
            self.m.remove_all(&self.dm);
            self.dm.clear();
        }
        if self.dp.nvals() != 0 {
            self.m.merge_overwrite_uint64(&self.dp);
            self.dp.clear();
        }
        self.wait();
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

pub struct Iter {
    mit: matrix::Iter,
    /// Delta-plus iterator. Lazily left `None` when `dp` is empty (the common
    /// read-only hot path on a freshly loaded graph) so we skip allocating and
    /// freeing a `GxB_Iterator` that would never yield anything. `dp` is a
    /// stable read snapshot for the life of this iterator, so once `None` it
    /// stays `None` across `seek` calls.
    dpit: Option<matrix::Iter>,
    dm: Cow<Matrix<bool>>,
    /// The dp overlay handle ([`Element::dp_overlay`]) when `m` entries may be
    /// shadowed by `dp` (`u64` matrices with a non-empty `dp`): such entries
    /// must be skipped in the `m` phase or they'd be emitted twice. `None` for
    /// bool matrices (whose `set`/`remove` keep `dp ∩ m = ∅`) and whenever
    /// `dp` is empty.
    shadow_dp: Option<Cow<Matrix<u64>>>,
    /// True when the deletion mask is empty, so the `m` phase can stream `mit`
    /// without per-edge `dm` lookups. Hot path for read-only queries on a
    /// freshly loaded graph.
    dm_empty: bool,
    dp_empty: bool,
}

unsafe impl Send for Iter {}
unsafe impl Sync for Iter {}

impl Iter {
    /// Creates a new iterator for traversing all elements in a matrix.
    ///
    /// # Parameters
    /// - `m`: The matrix to iterate over.
    /// - `min_row`: The minimum row index to start iterating from.
    /// - `max_row`: The maximum row index to stop iterating at.
    #[must_use]
    pub fn new<V: Element>(
        m: &VersionedMatrix<V>,
        min_row: u64,
        max_row: u64,
    ) -> Self {
        let dm_empty = m.dm.nvals() == 0;
        let dp_empty = m.dp.nvals() == 0;
        Self {
            mit: m.m.iter(min_row, max_row),
            dpit: if dp_empty {
                None
            } else {
                Some(m.dp.iter(min_row, max_row))
            },
            dm: m.dm.clone(),
            shadow_dp: if dp_empty { None } else { V::dp_overlay(&m.dp) },
            dm_empty,
            dp_empty,
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

impl Iterator for Iter {
    type Item = (u64, u64);

    /// Advances the iterator and returns the next element in the matrix.
    ///
    /// # Returns
    /// - `Some((u64, u64))`: The next element in the matrix.
    /// - `None`: The iterator is depleted.
    fn next(&mut self) -> Option<Self::Item> {
        if self.dm_empty && self.shadow_dp.is_none() {
            if let Some(item) = self.mit.next() {
                return Some(item);
            }
            if self.dp_empty {
                return None;
            }
            return self.dpit.as_mut().and_then(Iterator::next);
        }
        for (i, j) in &mut self.mit {
            if !self.dm_empty && self.dm.get(i, j).is_some() {
                continue; // deleted
            }
            if let Some(dp) = &self.shadow_dp
                && dp.contains(i, j)
            {
                continue; // shadowed by the dp overlay; emitted in the dp phase
            }
            return Some((i, j));
        }
        self.dpit.as_mut().and_then(Iterator::next)
    }
}

/// Streaming UINT64 row-range iterator with dp-overlay semantics.
///
/// Yields effective `(row, col, value)` triples over rows in `[min_row,
/// max_row]`: every `m` entry not masked by `dm` and not shadowed by `dp`,
/// followed by every `dp` entry. Mirrors the bool [`Iter`] but carries the
/// `u64` value. Supports [`UintIter::seek`] for amortized per-row scans.
pub struct UintIter {
    mit: matrix::Iter<matrix::Uint64Extract>,
    dpit: matrix::Iter<matrix::Uint64Extract>,
    dm: Cow<Matrix<bool>>,
    dp: Cow<Matrix<u64>>,
    dm_empty: bool,
    dp_empty: bool,
}

unsafe impl Send for UintIter {}
unsafe impl Sync for UintIter {}

impl UintIter {
    fn new(
        vm: &VersionedMatrix<u64>,
        min_row: u64,
        max_row: u64,
    ) -> Self {
        let dm_empty = vm.dm.nvals() == 0;
        let dp_empty = vm.dp.nvals() == 0;
        Self {
            mit: vm.m.uint64_iter_range(min_row, max_row),
            dpit: vm.dp.uint64_iter_range(min_row, max_row),
            dm: vm.dm.clone(),
            dp: vm.dp.clone(),
            dm_empty,
            dp_empty,
        }
    }

    /// Re-seek both inner iterators to a new row range without re-allocating.
    pub fn seek(
        &mut self,
        min_row: u64,
        max_row: u64,
    ) {
        self.mit.seek(min_row, max_row);
        self.dpit.seek(min_row, max_row);
    }
}

impl Iterator for UintIter {
    type Item = (u64, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.dm_empty && self.dp_empty {
            return self.mit.next();
        }
        for (i, j, v) in &mut self.mit {
            if !self.dm_empty && self.dm.contains(i, j) {
                continue; // deleted
            }
            if !self.dp_empty && self.dp.contains(i, j) {
                continue; // overridden by dp; emitted in the dp phase
            }
            return Some((i, j, v));
        }
        self.dpit.next()
    }
}

impl VersionedMatrix<u64> {
    /// Stream effective UINT64 `(row, col, value)` triples over rows in
    /// `[min_row, max_row]` with dp-overlay semantics.
    #[must_use]
    pub fn uint64_iter_range(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> UintIter {
        self.wait();
        UintIter::new(self, min_row, max_row)
    }
}

#[cfg(test)]
mod uint64_overlay_tests {
    use super::*;
    use crate::graph::graphblas::test_init::ensure_init;

    fn sorted(mut v: Vec<(u64, u64, u64)>) -> Vec<(u64, u64, u64)> {
        v.sort_unstable();
        v
    }

    #[test]
    fn insert_get_iter() {
        ensure_init();
        let mut vm = VersionedMatrix::new_uint64(100, 100);
        vm.set_uint64(1, 2, 42);
        vm.set_uint64(3, 4, 7);
        assert_eq!(vm.get_uint64(1, 2), Some(42));
        assert_eq!(vm.get_uint64(3, 4), Some(7));
        assert_eq!(vm.get_uint64(5, 6), None);
        assert_eq!(sorted(vm.collect_uint64()), vec![(1, 2, 42), (3, 4, 7)]);
    }

    #[test]
    fn update_in_place_dp_wins() {
        ensure_init();
        let mut vm = VersionedMatrix::new_uint64(100, 100);
        vm.set_uint64(1, 2, 42);
        // Simulate commit: fold dp into base by forcing a flush.
        vm.force_commit_uint64();
        vm.set_uint64(1, 2, 999); // update committed entry → dp overlay
        assert_eq!(vm.get_uint64(1, 2), Some(999));
        assert_eq!(sorted(vm.collect_uint64()), vec![(1, 2, 999)]);
    }

    #[test]
    fn delete_then_readd_new_value() {
        ensure_init();
        let mut vm = VersionedMatrix::new_uint64(100, 100);
        vm.set_uint64(1, 2, 42);
        vm.force_commit_uint64(); // commit
        vm.remove_uint64(1, 2);
        assert_eq!(vm.get_uint64(1, 2), None);
        vm.set_uint64(1, 2, 77); // re-add with a different id
        assert_eq!(vm.get_uint64(1, 2), Some(77));
        assert_eq!(sorted(vm.collect_uint64()), vec![(1, 2, 77)]);
    }

    #[test]
    fn remove_pending_add() {
        ensure_init();
        let mut vm = VersionedMatrix::new_uint64(100, 100);
        vm.set_uint64(1, 2, 42);
        vm.remove_uint64(1, 2); // remove an uncommitted add
        assert_eq!(vm.get_uint64(1, 2), None);
        assert!(vm.collect_uint64().is_empty());
    }

    #[test]
    fn nvals_with_overlay() {
        ensure_init();
        let mut vm = VersionedMatrix::new_uint64(100, 100);
        vm.set_uint64(1, 2, 42);
        assert_eq!(vm.nvals(), 1);
        vm.force_commit_uint64();
        assert_eq!(vm.nvals(), 1);
        // In-place update of a committed cell: dp shadows m.
        vm.set_uint64(1, 2, 999);
        assert_eq!(vm.nvals(), 1);
        // A second, non-overlapping pending entry.
        vm.set_uint64(3, 4, 7);
        assert_eq!(vm.nvals(), 2);

        // Delete a committed cell, then re-add it.
        let mut vm2 = VersionedMatrix::new_uint64(100, 100);
        vm2.set_uint64(1, 2, 42);
        vm2.force_commit_uint64();
        vm2.remove_uint64(1, 2);
        assert_eq!(vm2.nvals(), 0);
        vm2.set_uint64(1, 2, 77);
        assert_eq!(vm2.nvals(), 1);
    }

    #[test]
    fn iter_no_duplicates_with_overlay() {
        ensure_init();
        let mut vm = VersionedMatrix::new_uint64(100, 100);
        vm.set_uint64(1, 2, 42);
        vm.set_uint64(5, 6, 11);
        vm.force_commit_uint64();
        vm.set_uint64(1, 2, 999); // dp shadows m
        vm.set_uint64(3, 4, 7); // dp-only entry
        vm.remove_uint64(5, 6); // committed delete → dm

        // Structural iterator must emit each effective cell exactly once.
        let mut items: Vec<(u64, u64)> = vm.iter(0, u64::MAX).collect();
        items.sort_unstable();
        assert_eq!(items, vec![(1, 2), (3, 4)]);

        // Valued full-range iterator must be shadow- and dm-aware too.
        let vals = sorted(vm.uint64_iter().collect());
        assert_eq!(vals, vec![(1, 2, 999), (3, 4, 7)]);
    }
}
