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
//! UINT64-valued layers (e.g. `Tensor`'s forward adjacency, which owns its
//! own `m`/`dp`/`dm` triple) additionally support in-place *value* updates of
//! a committed entry: the new value is written to `dp`, *shadowing* the live
//! `m` entry (no `dm` mask — `dm` marks pure deletions only, so `dp ∩ dm = ∅`
//! and `dm ⊆ m`). [`Iter`]'s sorted merge yields the live `dp` value for
//! shadowed pairs and skips the stale `m` entry.
//!
//! ## Flush
//!
//! When delta matrices exceed 10,000 entries, [`flush`](VersionedMatrix::flush)
//! merges them into the base matrix (`dp` via element-wise add, `dm` via
//! masked removal) and clears the deltas.
//!
//! ## Iterator
//!
//! [`Iter`] performs a three-way sorted merge over the `m`, `dp`, and `dm`
//! iterators (GraphBLAS row iterators yield ascending `(row, col)` order):
//! `m` entries matched by the `dm` lookahead are dropped, `dp` entries are
//! interleaved in order (winning over a shadowed `m` entry at the same
//! position), producing the effective state — sorted by `(row, col)` —
//! without materializing a merged matrix or issuing per-entry point lookups.

use super::{
    GxB_Print_Level,
    matrix::{self, Dup, Matrix},
    serialization::{Decode, Encode, Reader, Writer},
};
use crate::graph::{
    cow::Cow,
    graphblas::matrix::{BoolExtract, IterExtract},
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
        m.set_pattern(None, &*self.m, None);
        if self.dm.nvals() > 0 {
            m.remove_all(&self.dm);
        }
        if self.dp.nvals() > 0 {
            m.set_pattern(None, &*self.dp, None);
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
        // dm<mask> = mask ∩ m: mark deleted every committed entry that `mask`
        // selects. eWiseMult's `PAIR` semiring never reads `m`'s values — an
        // eWiseAdd copy would typecast a u64 value of 0 to `false`, which
        // valued masks then skip.
        self.dm
            .element_wise_multiply(Some(mask), Some(mask), Some(&*self.m), None);
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
    /// Lookahead on `mit` (buffered while waiting for the merge to pick it).
    m_next: Option<E::Item>,
    /// Delta-plus iterator. Lazily left `None` when `dp` is empty (the common
    /// read-only hot path on a freshly loaded graph) so we skip allocating and
    /// freeing a `GxB_Iterator` that would never yield anything. `dp` is a
    /// stable read snapshot for the life of this iterator, so once `None` it
    /// stays `None` across `seek` calls.
    dpit: Option<matrix::Iter<E>>,
    /// Lookahead on `dpit`.
    dp_next: Option<E::Item>,
    /// Delta-minus iterator, `None` when `dm` is empty (same rationale as
    /// `dpit`).
    dmit: Option<matrix::Iter<BoolExtract>>,
    /// Lookahead on `dmit`.
    dm_next: Option<(u64, u64)>,
}

unsafe impl<E: IterExtract> Send for Iter<E> {}
unsafe impl<E: IterExtract> Sync for Iter<E> {}

impl<E: IterExtract> Iter<E> {
    /// Streams the effective content `(m ∖ dm) ∪ dp` as a sorted merge of the
    /// three layer iterators. Valid for a `VersionedMatrix` of any element
    /// type when `E = BoolExtract` (only the sparsity pattern is read).
    fn new<V>(
        vm: &VersionedMatrix<V>,
        min_row: u64,
        max_row: u64,
    ) -> Self {
        Self::from_layers(&vm.m, &vm.dp, &vm.dm, min_row, max_row)
    }

    /// Build the effective-content iterator directly from the three delta
    /// layers, for owners that manage `m`/`dp`/`dm` themselves (e.g.
    /// `Tensor`'s forward adjacency). Callers must have waited `dp`/`dm`.
    /// When `dp` shadows a committed `m` pair (an in-place update with no
    /// `dm` mask), the merge yields the live `dp` value and skips the `m`
    /// entry.
    pub(crate) fn from_layers<V>(
        m: &Matrix<V>,
        dp: &Matrix<V>,
        dm: &Cow<Matrix<bool>>,
        min_row: u64,
        max_row: u64,
    ) -> Self {
        let mut dmit = if dm.nvals() == 0 {
            None
        } else {
            Some(matrix::Iter::<BoolExtract>::new(&**dm, min_row, max_row))
        };
        let dm_next = dmit.as_mut().and_then(Iterator::next);
        Self {
            mit: matrix::Iter::new(m, min_row, max_row),
            m_next: None,
            dpit: if dp.nvals() == 0 {
                None
            } else {
                Some(matrix::Iter::new(dp, min_row, max_row))
            },
            dp_next: None,
            dmit,
            dm_next,
        }
    }

    /// Re-seek the inner GraphBLAS iterators to a new row range without
    /// re-allocating them. Hot-loop callers (e.g. `CondTraverseOp` and
    /// `ExpandInto` looking up edges by `(src, dst)`) use this to amortize
    /// the per-pair iterator allocation.
    pub fn seek(
        &mut self,
        min_row: u64,
        max_row: u64,
    ) {
        self.mit.seek(min_row, max_row);
        self.m_next = None;
        if let Some(dpit) = &mut self.dpit {
            dpit.seek(min_row, max_row);
        }
        self.dp_next = None;
        if let Some(dmit) = &mut self.dmit {
            dmit.seek(min_row, max_row);
            self.dm_next = dmit.next();
        }
    }
}

/// Three-way sorted merge over the layer iterators. GraphBLAS row iterators
/// yield entries in ascending `(row, col)` order, so the output is sorted by
/// `(row, col)` too. `m` entries masked by `dm` are dropped; when `dp` holds
/// the same pair as `m` (shadow), the live `dp` item wins.
impl<E: IterExtract> Iterator for Iter<E> {
    type Item = E::Item;

    fn next(&mut self) -> Option<E::Item> {
        if self.m_next.is_none() {
            self.m_next = self.mit.next();
        }
        // Skip m entries deleted by dm (both streams ascending, dm ⊆ m).
        while let Some(m) = &self.m_next {
            let mp = E::pos(m);
            while let Some(dm) = self.dm_next {
                if dm < mp {
                    self.dm_next = self.dmit.as_mut().and_then(Iterator::next);
                } else {
                    break;
                }
            }
            if self.dm_next == Some(mp) {
                self.dm_next = self.dmit.as_mut().and_then(Iterator::next);
                self.m_next = self.mit.next();
            } else {
                break;
            }
        }
        if self.dp_next.is_none() {
            self.dp_next = self.dpit.as_mut().and_then(Iterator::next);
        }
        match (&self.m_next, &self.dp_next) {
            (Some(m), Some(dp)) => {
                let mp = E::pos(m);
                let dpp = E::pos(dp);
                if dpp <= mp {
                    if dpp == mp {
                        self.m_next = None; // shadowed; dp yields the live value
                    }
                    self.dp_next.take()
                } else {
                    self.m_next.take()
                }
            }
            (Some(_), None) => self.m_next.take(),
            (None, _) => self.dp_next.take(),
        }
    }
}
