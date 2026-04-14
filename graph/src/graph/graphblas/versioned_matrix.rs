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
    matrix::{self, Dup, Get, MaskedElementWiseAdd, Matrix, New, Remove, Set, Size, Transpose},
    serialization::{Decode, Encode, Reader, Writer},
};
use crate::graph::cow::Cow;

/// A matrix with MVCC delta tracking for snapshot isolation.
///
/// Wraps a base matrix with separate matrices for tracking additions
/// and deletions, enabling concurrent reads during writes.
pub struct VersionedMatrix {
    /// Base committed matrix
    m: Cow<Matrix>,
    /// Delta-plus: edges added in current transaction
    dp: Cow<Matrix>,
    /// Delta-minus: edges removed in current transaction
    dm: Cow<Matrix>,
}

unsafe impl Send for VersionedMatrix {}
unsafe impl Sync for VersionedMatrix {}

impl Size for VersionedMatrix {
    fn nrows(&self) -> u64 {
        self.m.nrows()
    }

    fn ncols(&self) -> u64 {
        self.m.ncols()
    }

    fn resize(
        &mut self,
        nrows: u64,
        ncols: u64,
    ) {
        self.wait();
        self.m.resize(nrows, ncols);
        self.dp.resize(nrows, ncols);
        self.dm.resize(nrows, ncols);
    }

    fn nvals(&self) -> u64 {
        self.wait();
        self.m.nvals() + self.dp.nvals() - self.dm.nvals()
    }
}

impl New for VersionedMatrix {
    fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        Self {
            m: Cow::new(Matrix::new(nrows, ncols)),
            dp: Cow::new(Matrix::new(nrows, ncols)),
            dm: Cow::new(Matrix::new(nrows, ncols)),
        }
    }
}

impl Dup<Self> for VersionedMatrix {
    fn dup(&self) -> Self {
        Self {
            m: self.m.new_version(),
            dp: self.dp.new_version(),
            dm: self.dm.new_version(),
        }
    }
}

impl VersionedMatrix {
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

    pub fn wait(&self) {
        debug_assert!(!self.m.pending());
        self.dp.wait();
        self.dm.wait();
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
    ) -> Iter {
        self.wait();
        Iter::new(self, min_row, max_row)
    }

    #[must_use]
    pub fn to_matrix(&self) -> Matrix {
        // TODO: remove
        self.wait();
        let mut m = self.m.dup();
        m.remove_all(&self.dm);
        m.element_wise_add(None, None, Some(&self.dp), None);
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

    #[must_use]
    pub fn extract_m_dp(&self) -> (Matrix, Matrix) {
        let mut m = Matrix::new(self.m.nrows(), self.m.ncols());
        let mut dp = Matrix::new(self.dp.nrows(), self.dp.ncols());

        m.select(&self.dm, &self.m);
        dp.select(&self.dm, &self.dp);

        (m, dp)
    }

    /// Returns true if the base matrix has UINT64 element type.
    ///
    /// C-produced relation matrices store edge IDs as UINT64, while
    /// Rust-produced ones use BOOL.
    #[must_use]
    pub fn is_uint64(&self) -> bool {
        self.m.is_uint64()
    }

    /// Iterate UINT64 entries from the base M and delta-plus DP matrices.
    ///
    /// Used during RDB decode to read C-produced relation matrices where
    /// single-edge entries store the edge ID as a UINT64 value.
    /// Returns an empty iterator for Rust-produced BOOL matrices.
    pub fn uint64_iter(&self) -> impl Iterator<Item = (u64, u64, u64)> + '_ {
        self.m.uint64_iter().chain(self.dp.uint64_iter())
    }
}

impl Remove for VersionedMatrix {
    fn remove(
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
}

impl Get for VersionedMatrix {
    fn get(
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
}

impl Set for VersionedMatrix {
    fn set(
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

impl Transpose for VersionedMatrix
where
    Self: New,
{
    /// Transposes the matrix.
    ///
    /// # Returns
    /// A new matrix that is the transpose of the original.
    fn transpose(&self) -> Self {
        Self {
            m: Cow::new(self.m.transpose()),
            dp: Cow::new(self.dp.transpose()),
            dm: Cow::new(self.dm.transpose()),
        }
    }
}

impl Encode<19> for VersionedMatrix {
    fn encode(
        &self,
        w: &mut dyn Writer,
    ) {
        self.m.encode(w);
        self.dp.encode(w);
        self.dm.encode(w);
    }
}

impl Decode<19> for VersionedMatrix {
    fn decode(r: &mut dyn Reader) -> Result<Self, String> {
        let m = Matrix::decode(r)?;
        let dp = Matrix::decode(r)?;
        let dm = Matrix::decode(r)?;
        Ok(Self {
            m: Cow::new(m),
            dp: Cow::new(dp),
            dm: Cow::new(dm),
        })
    }
}

pub struct Iter {
    mit: matrix::Iter,
    dpit: matrix::Iter,
    dm: Cow<Matrix>,
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
    pub fn new(
        m: &VersionedMatrix,
        min_row: u64,
        max_row: u64,
    ) -> Self {
        Self {
            mit: m.m.iter(min_row, max_row),
            dpit: m.dp.iter(min_row, max_row),
            dm: m.dm.clone(),
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
        for (i, j) in &mut self.mit {
            if self.dm.get(i, j).is_none() {
                return Some((i, j));
            }
        }
        self.dpit.next()
    }
}
