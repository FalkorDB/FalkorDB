//! [`Matrix<T>`]: an owned `GrB_Matrix` with a checked element type, and
//! [`transpose`], the first operation, which exercises the mask and
//! descriptor plumbing every other operation will share.
//!
//! Operations are free functions in the C API's argument order: `(C, Mask,
//! accum, op, A, B, desc)`. Inputs are [`Operand`]s: `&Matrix<T>`, or
//! [`Output`](crate::Output) for the output itself.

use std::{
    fmt,
    marker::PhantomData,
    mem::{MaybeUninit, forget},
    ptr::null_mut,
};

use crate::{
    Result,
    error::{check, expect_success},
    handle::{self, run},
    init::{self, WaitMode},
    mask::{Desc, Mask},
    operand::Operand,
    ops::{BinaryOp, accum_raw},
    sealed::Token,
    sys::{self, GrB_Index, GrB_Info, GrB_Matrix},
    types::GrbType,
};

/// `GrB_Matrix_nrows` on a raw handle.
pub(crate) fn raw_nrows(m: GrB_Matrix) -> GrB_Index {
    let mut n = 0;
    // SAFETY: a live matrix and a valid out-pointer.
    let info = unsafe { sys::GrB_Matrix_nrows(&raw mut n, m) };
    expect_success(info, "GrB_Matrix_nrows");
    n
}

/// `GrB_Matrix_ncols` on a raw handle.
pub(crate) fn raw_ncols(m: GrB_Matrix) -> GrB_Index {
    let mut n = 0;
    // SAFETY: a live matrix and a valid out-pointer.
    let info = unsafe { sys::GrB_Matrix_ncols(&raw mut n, m) };
    expect_success(info, "GrB_Matrix_ncols");
    n
}

/// An owned `GrB_Matrix` whose entries are `T`.
///
/// `T` is checked, not advisory: the handle was created with `T`'s GraphBLAS
/// type (or adopted by an `unsafe fn` that promised it), so a typed operator
/// applied to it meets the domains it was declared with.
///
/// Methods that take `&self` may still finish the matrix's pending work
/// inside GraphBLAS (`nvals`, `extract_element`, every operation that reads
/// it). That is a write to the handle's internals; it is sound on one thread
/// and is why this type is not `Sync`.
pub struct Matrix<T> {
    raw: GrB_Matrix,
    _elem: PhantomData<T>,
}

// SAFETY: a GraphBLAS handle is not tied to the thread that created it; the
// wrapper holds nothing else. Deliberately not `Sync` (see the type docs).
unsafe impl<T: Send> Send for Matrix<T> {}

impl<T: GrbType> Matrix<T> {
    /// An empty `nrows x ncols` matrix.
    ///
    /// # Errors
    ///
    /// Dimensions over `GrB_INDEX_MAX + 1`, out of memory, not initialised.
    pub fn new(
        nrows: GrB_Index,
        ncols: GrB_Index,
    ) -> Result<Self> {
        init::ensure_initialized()?;
        let mut raw = null_mut();
        // SAFETY: a valid out-pointer and a live type handle.
        check(unsafe { sys::GrB_Matrix_new(&raw mut raw, T::grb_type(), nrows, ncols) })?;
        // SAFETY: just created with `T`'s type; owned here from now on.
        Ok(unsafe { Self::from_raw(raw) })
    }

    /// Take ownership of a raw handle.
    ///
    /// # Safety
    ///
    /// `raw` is a live, non-null matrix whose GraphBLAS type is `T`'s, and
    /// nothing else uses or frees it afterwards: the result owns it alone.
    pub unsafe fn from_raw(raw: GrB_Matrix) -> Self {
        debug_assert!(!raw.is_null());
        Self {
            raw,
            _elem: PhantomData,
        }
    }

    /// Give up ownership; the caller must free the handle with GraphBLAS.
    #[must_use]
    pub const fn into_raw(self) -> GrB_Matrix {
        let raw = self.raw;
        forget(self);
        raw
    }

    /// The raw handle, still owned here. For the LAGraph wrapper and for the
    /// engine's domain wrapper while it migrates; not for new code.
    #[must_use]
    pub const fn as_raw(&self) -> GrB_Matrix {
        self.raw
    }

    #[must_use]
    pub fn nrows(&self) -> GrB_Index {
        raw_nrows(self.raw)
    }

    #[must_use]
    pub fn ncols(&self) -> GrB_Index {
        raw_ncols(self.raw)
    }

    /// Number of stored entries. Finishes pending work first.
    #[must_use]
    pub fn nvals(&self) -> GrB_Index {
        let mut n = 0;
        // SAFETY: a live matrix and a valid out-pointer.
        let info = unsafe { sys::GrB_Matrix_nvals(&raw mut n, self.raw) };
        expect_success(info, "GrB_Matrix_nvals");
        n
    }

    /// A copy, pending work included (GraphBLAS copies it unfinished).
    ///
    /// # Errors
    ///
    /// Out of memory.
    pub fn dup(&self) -> Result<Self> {
        let mut raw = null_mut();
        // SAFETY: a valid out-pointer and a live matrix.
        handle::check_read(unsafe { sys::GrB_Matrix_dup(&raw mut raw, self.raw) })?;
        // SAFETY: a fresh matrix of `T`'s type, owned here from now on.
        Ok(unsafe { Self::from_raw(raw) })
    }

    /// Remove every entry.
    ///
    /// # Errors
    ///
    /// Never in practice; kept as a `Result` because the C call has one.
    pub fn clear(&mut self) -> Result<()> {
        // SAFETY: a live matrix.
        self.check(unsafe { sys::GrB_Matrix_clear(self.raw) })
    }

    /// Change the dimensions, dropping entries that fall outside.
    ///
    /// # Errors
    ///
    /// Dimensions over `GrB_INDEX_MAX + 1`, out of memory.
    pub fn resize(
        &mut self,
        nrows: GrB_Index,
        ncols: GrB_Index,
    ) -> Result<()> {
        // SAFETY: a live matrix.
        self.check(unsafe { sys::GrB_Matrix_resize(self.raw, nrows, ncols) })
    }

    /// Finish pending work.
    ///
    /// # Errors
    ///
    /// Out of memory while assembling.
    pub fn wait(
        &mut self,
        mode: WaitMode,
    ) -> Result<()> {
        handle::wait(self.raw, mode)
    }

    /// Whether the matrix has pending work (`GxB_WILL_WAIT` query).
    #[must_use]
    pub fn is_pending(&self) -> bool {
        handle::is_pending(self.raw)
    }

    /// `C(i, j) = x`.
    ///
    /// # Errors
    ///
    /// Index out of range, out of memory.
    pub fn set_element(
        &mut self,
        i: GrB_Index,
        j: GrB_Index,
        x: T,
    ) -> Result<()> {
        // SAFETY: a live matrix of `T`'s type.
        self.check(unsafe { T::matrix_set_element(self.raw, x, i, j) })
    }

    /// Delete the entry at `(i, j)` if there is one.
    ///
    /// # Errors
    ///
    /// Index out of range.
    pub fn remove_element(
        &mut self,
        i: GrB_Index,
        j: GrB_Index,
    ) -> Result<()> {
        // SAFETY: a live matrix.
        self.check(unsafe { sys::GrB_Matrix_removeElement(self.raw, i, j) })
    }

    /// The entry at `(i, j)`, or `None` when there is none. Finishes pending
    /// work first.
    ///
    /// # Errors
    ///
    /// Index out of range.
    pub fn extract_element(
        &self,
        i: GrB_Index,
        j: GrB_Index,
    ) -> Result<Option<T>> {
        let mut x = MaybeUninit::<T>::uninit();
        // SAFETY: a live matrix of `T`'s type and room for one `T`.
        let info = unsafe { T::matrix_extract_element(x.as_mut_ptr(), self.raw, i, j) };
        match info {
            // SAFETY: GraphBLAS wrote the value on success.
            GrB_Info::GrB_SUCCESS => Ok(Some(unsafe { x.assume_init() })),
            GrB_Info::GrB_NO_VALUE => Ok(None),
            other => handle::check_read(other).map(|()| None),
        }
    }

    /// Whether an entry is stored at `(i, j)`; never reads the value.
    /// Pending work is seen: GraphBLAS finishes it when the entry is not
    /// found at first.
    ///
    /// # Errors
    ///
    /// Index out of range.
    pub fn contains(
        &self,
        i: GrB_Index,
        j: GrB_Index,
    ) -> Result<bool> {
        // SAFETY: a live matrix.
        match unsafe { sys::GxB_Matrix_isStoredElement(self.raw, i, j) } {
            GrB_Info::GrB_SUCCESS => Ok(true),
            GrB_Info::GrB_NO_VALUE => Ok(false),
            other => handle::check_read(other).map(|()| false),
        }
    }

    /// Bulk-load `(rows[k], cols[k]) = values[k]` into an empty matrix.
    /// Duplicates are combined with `dup`; with `dup = None` a duplicate is
    /// an error.
    ///
    /// # Panics
    ///
    /// If the three slices differ in length.
    ///
    /// # Errors
    ///
    /// The matrix is not empty, an index is out of range, a duplicate with
    /// no `dup`, out of memory.
    pub fn build(
        &mut self,
        rows: &[GrB_Index],
        cols: &[GrB_Index],
        values: &[T],
        dup: Option<&BinaryOp<T, T, T>>,
    ) -> Result<()> {
        assert_eq!(rows.len(), cols.len(), "rows and cols differ in length");
        assert_eq!(rows.len(), values.len(), "rows and values differ in length");
        // SAFETY: a live matrix of `T`'s type; three slices of the stated
        // length that outlive the call.
        self.check(unsafe {
            T::matrix_build(
                self.raw,
                rows.as_ptr(),
                cols.as_ptr(),
                values.as_ptr(),
                rows.len() as GrB_Index,
                accum_raw(dup),
            )
        })
    }

    /// Every stored entry as `(rows, cols, values)`, in storage order.
    /// Finishes pending work first.
    ///
    /// This copies the whole matrix to the host; reach for it when the data
    /// leaves GraphBLAS for good, not as a way to loop.
    ///
    /// # Errors
    ///
    /// Out of memory.
    pub fn extract_tuples(&self) -> Result<(Vec<GrB_Index>, Vec<GrB_Index>, Vec<T>)> {
        let capacity = handle::entry_count(self.nvals());
        let mut rows = Vec::with_capacity(capacity);
        let mut cols = Vec::with_capacity(capacity);
        let mut vals = Vec::<T>::with_capacity(capacity);
        let mut n = capacity as GrB_Index;
        // SAFETY: a live matrix of `T`'s type and three buffers with room
        // for `n` entries; GraphBLAS writes at most `n` and stores how many.
        let info = unsafe {
            T::matrix_extract_tuples(
                rows.as_mut_ptr(),
                cols.as_mut_ptr(),
                vals.as_mut_ptr(),
                &raw mut n,
                self.raw,
            )
        };
        handle::check_read(info)?;
        let n = handle::entry_count(n);
        debug_assert!(n <= capacity);
        // SAFETY: GraphBLAS initialised the first `n` elements of each.
        unsafe {
            rows.set_len(n);
            cols.set_len(n);
            vals.set_len(n);
        }
        Ok((rows, cols, vals))
    }

    /// Bytes GraphBLAS holds for this matrix (`GxB_Matrix_memoryUsage`).
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        handle::memory_usage(self.raw)
    }

    /// Whether the matrix is iso (one shared value).
    #[must_use]
    pub fn is_iso(&self) -> bool {
        handle::is_iso(self.raw)
    }

    /// `Ok` on success, else the error with this matrix's message.
    pub(crate) fn check(
        &self,
        info: GrB_Info,
    ) -> Result<()> {
        handle::check(info, self.raw)
    }
}

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        // SAFETY: `raw` is owned by this value and freed exactly once.
        let _ = unsafe { sys::GrB_Matrix_free(&raw mut self.raw) };
    }
}

impl<T: GrbType> fmt::Debug for Matrix<T> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "Matrix<{}>({}x{} @ {:p})",
            T::NAME,
            self.nrows(),
            self.ncols(),
            self.raw
        )
    }
}

// ───────────────────────────── operations ─────────────────────────────

/// `C<Mask> = accum(C, Aᵀ)`.
///
/// With `desc.transpose_first` the input is transposed first, so the call is a
/// plain masked copy, `C<Mask> = A`: the GraphBLAS idiom for "apply a mask to
/// a matrix".
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn transpose<'a, T, A>(
    c: &mut Matrix<T>,
    mask: Mask<'a, GrB_Matrix>,
    accum: Option<&BinaryOp<T, T, T>>,
    a: A,
    desc: Desc,
) -> Result<()>
where
    T: GrbType,
    A: Operand<'a, T, GrB_Matrix, Elem = T>,
{
    run(c.raw, &mask, desc, |c, m, d| {
        // SAFETY: live handles of one element type; aliasing allowed.
        unsafe { sys::GrB_transpose(c, m, accum_raw(accum), a.raw(c, Token), d) }
    })
}
