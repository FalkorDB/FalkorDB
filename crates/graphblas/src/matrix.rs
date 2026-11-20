//! [`Matrix<T>`] and every operation whose output is a matrix, as free
//! functions in the C API's argument order: `(C, Mask, accum, op, A, B,
//! desc)`.
//!
//! The type parameters say what GraphBLAS would otherwise typecast silently:
//!
//! * `X: Accepts<A::Elem>`: the operator's input domain is the operand's
//!   element type, or [`Pattern`](crate::Pattern).
//! * `Z: Produces<C>`: the operator's output domain is the output's element
//!   type.
//! * `accum` is `BinaryOp<C, C, C>` on the output type.
//! * Operations that copy entries through unchanged ([`ewise_add`],
//!   [`transpose`], [`select`], [`assign`], [`extract`]) require the input
//!   and output element types to be the same.
//!
//! An operator passed as `op.typecast()` lifts the first two, and for
//! [`ewise_add`] the last: GraphBLAS then casts among built-in types at
//! that operator's edges, as [`Typecast`](crate::Typecast) describes.
//! [`select`] is the exception and takes its predicate as declared.
//!
//! Inputs are [`Operand`]s: `&Matrix<T>`, or [`Output`](crate::Output) for the output
//! itself. `C += A*B` runs in place in SuiteSparse (dot4 / saxpy5) when `C`
//! is full, the accumulator is `semiring.add_op()` and `C` is **not** aliased
//! with the mask or an input; `C<M> += T` lands in place through
//! `GB_subassign` under the same condition. `apply` with `Output`, no mask
//! and no accumulator rewrites the values in place. Every other aliased form
//! is computed into a temporary and transplanted by GraphBLAS.

use std::{
    fmt,
    marker::PhantomData,
    mem::{MaybeUninit, forget},
    ptr::null_mut,
};

use crate::{
    Result, Scalar,
    error::{check, expect_success},
    frozen::{Freezable, Frozen},
    handle::{self, run},
    index::Indices,
    init::{self, WaitMode},
    mask::{Desc, Mask, MaskMode, descriptor},
    operand::Operand,
    ops::{BinaryOp, IndexUnaryOp, Monoid, Semiring, UnaryOp, accum_raw},
    sealed::{Sealed, Token},
    settings::{IntegerWidth, Orientation, Sparsity},
    sys::{self, GrB_Index, GrB_Info, GrB_Matrix, GrB_Vector, GxB_Option_Field},
    types::{Accepts, BuiltinType, GrbType, PassesThrough, Produces},
    vector::Vector,
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
/// and is why this type is not `Sync`. To read one matrix from several
/// threads, [`freeze`](Self::freeze) it.
///
/// A `&self` method must do nothing but read once the matrix is
/// materialized: [`Frozen`] hands `&Matrix<T>` to several threads on that
/// promise (see [`Freezable`]).
pub struct Matrix<T> {
    raw: GrB_Matrix,
    _elem: PhantomData<T>,
}

// SAFETY: a GraphBLAS handle is not tied to the thread that created it; the
// wrapper holds nothing else. Deliberately not `Sync` (see the type docs).
unsafe impl<T: Send> Send for Matrix<T> {}

impl<T> Sealed for Matrix<T> {}

// SAFETY: with nothing pending, the `&self` methods are `GrB_Matrix_nrows`,
// `ncols`, `nvals`, `dup`, `extractElement`, `isStoredElement`,
// `extractTuples`, `GrB_get`, `memoryUsage`, `iso` and `GrB_Matrix_error`,
// and the input and mask slots of the operations. In SuiteSparse 10.5 none
// of them names the matrix as its error logger (`GB_WHERE_1`, or no `Werk`
// at all), each waits only when `GB_will_wait`, and the hyper-hash an input
// may need was built by the materializing wait (`GrB_Matrix_wait` calls
// `GB_hyper_hash_build`). Two things qualify "only reads". An input whose
// count of non-empty vectors is unknown gets it cached by `dup` and `mxm`
// (`GB_nvec_nonempty_update`): one idempotent 64-bit store, atomic where
// the platform needs it. And "the hash was built" holds for the global
// hyper-hash threshold in force at the wait, which this crate never
// changes; a setter for it would have to be reconciled with this impl.
unsafe impl<T: GrbType> Freezable for Matrix<T> {
    unsafe fn materialize(&self) -> Result<()> {
        handle::wait(self.raw, WaitMode::Materialize)
    }
}

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

    /// Take ownership of a raw handle if its element type is `T`'s;
    /// otherwise hand it back untouched. For a handle another library made,
    /// where the type is that library's promise and not this crate's.
    ///
    /// # Safety
    ///
    /// `raw` is a live, non-null matrix and nothing else uses or frees it
    /// afterwards.
    ///
    /// # Errors
    ///
    /// `Err(raw)` when the element type is not `T`'s.
    pub unsafe fn try_from_raw(raw: GrB_Matrix) -> std::result::Result<Self, GrB_Matrix> {
        if handle::has_type(raw, T::grb_type()) {
            // SAFETY: live and owned by the caller's promise; of `T`'s type
            // by the check.
            Ok(unsafe { Self::from_raw(raw) })
        } else {
            Err(raw)
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
    ///
    /// # Panics
    ///
    /// When finishing the pending work runs out of memory, the one way it
    /// can fail; the matrix is unusable afterwards.
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

    /// Give up write access for a handle that can be cloned and read from
    /// several threads. Does no work: pending work is finished by the first
    /// [`Frozen::read`]. Settings that convert the storage
    /// ([`set_sparsity`](Self::set_sparsity),
    /// [`set_orientation`](Self::set_orientation)) have to be made before.
    #[must_use]
    pub fn freeze(self) -> Frozen<Self> {
        Frozen::new(self)
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

    /// Bulk-load the pattern `(rows[k], cols[k])` with every entry equal to
    /// `value`, as an iso matrix (one shared value, no values array).
    /// Duplicates are allowed.
    ///
    /// # Panics
    ///
    /// If the two slices differ in length.
    ///
    /// # Errors
    ///
    /// The matrix is not empty, an index is out of range, out of memory.
    pub fn build_iso(
        &mut self,
        rows: &[GrB_Index],
        cols: &[GrB_Index],
        value: T,
    ) -> Result<()> {
        assert_eq!(rows.len(), cols.len(), "rows and cols differ in length");
        let scalar = Scalar::from_value(value)?;
        // SAFETY: a live matrix of `T`'s type, a live scalar of the same
        // type, two slices of the stated length that outlive the call.
        self.check(unsafe {
            sys::GxB_Matrix_build_Scalar(
                self.raw,
                rows.as_ptr(),
                cols.as_ptr(),
                scalar.as_raw(),
                rows.len() as GrB_Index,
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

    /// The current storage format (`GxB_SPARSITY_STATUS`).
    #[must_use]
    pub fn sparsity(&self) -> Sparsity {
        handle::sparsity(self.raw)
    }

    /// Pin or free the storage format (`GxB_SPARSITY_CONTROL`).
    ///
    /// # Errors
    ///
    /// Never in practice.
    pub fn set_sparsity(
        &mut self,
        control: Sparsity,
    ) -> Result<()> {
        handle::set_sparsity(self.raw, control)
    }

    /// `GxB_FORMAT`.
    #[must_use]
    pub fn orientation(&self) -> Orientation {
        Orientation::from_raw(self.get_i32(GxB_Option_Field::GxB_FORMAT))
    }

    /// Set `GxB_FORMAT`; converts the storage if it changes.
    ///
    /// # Errors
    ///
    /// Out of memory.
    pub fn set_orientation(
        &mut self,
        orientation: Orientation,
    ) -> Result<()> {
        self.set_i32(GxB_Option_Field::GxB_FORMAT, orientation.raw())
    }

    /// The width GraphBLAS actually uses for row indices
    /// (`GxB_ROWINDEX_INTEGER_BITS`).
    #[must_use]
    pub fn row_index_width(&self) -> IntegerWidth {
        IntegerWidth::from_bits(self.get_i32(GxB_Option_Field::GxB_ROWINDEX_INTEGER_BITS))
    }

    /// The width GraphBLAS actually uses for column indices.
    #[must_use]
    pub fn col_index_width(&self) -> IntegerWidth {
        IntegerWidth::from_bits(self.get_i32(GxB_Option_Field::GxB_COLINDEX_INTEGER_BITS))
    }

    /// The width GraphBLAS actually uses for offsets.
    #[must_use]
    pub fn offset_width(&self) -> IntegerWidth {
        IntegerWidth::from_bits(self.get_i32(GxB_Option_Field::GxB_OFFSET_INTEGER_BITS))
    }

    /// Hint the index widths (`GxB_*_INTEGER_HINT`). A hint: GraphBLAS keeps
    /// 64 bits where 32 cannot hold the dimension, so read the getters back
    /// before relying on it.
    ///
    /// # Errors
    ///
    /// Never in practice.
    pub fn set_index_width_hint(
        &mut self,
        rows: IntegerWidth,
        cols: IntegerWidth,
        offsets: IntegerWidth,
    ) -> Result<()> {
        self.set_i32(GxB_Option_Field::GxB_ROWINDEX_INTEGER_HINT, rows.raw())?;
        self.set_i32(GxB_Option_Field::GxB_COLINDEX_INTEGER_HINT, cols.raw())?;
        self.set_i32(GxB_Option_Field::GxB_OFFSET_INTEGER_HINT, offsets.raw())
    }

    /// `Ok` on success, else the error with this matrix's message.
    pub(crate) fn check(
        &self,
        info: GrB_Info,
    ) -> Result<()> {
        handle::check(info, self.raw)
    }

    /// `GrB_Matrix_get_INT32` for a field that cannot fail on a live handle.
    fn get_i32(
        &self,
        field: GxB_Option_Field,
    ) -> i32 {
        handle::get_i32(self.raw, field)
    }

    /// `GrB_Matrix_set_INT32`.
    #[allow(
        clippy::needless_pass_by_ref_mut,
        reason = "it writes the matrix, through the handle"
    )]
    fn set_i32(
        &mut self,
        field: GxB_Option_Field,
        value: i32,
    ) -> Result<()> {
        handle::set_i32(self.raw, field, value)
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

/// `C<Mask> = accum(C, A * B)` over `semiring`.
///
/// `A`, `B` and `C` are independent element types. Each must equal the
/// semiring's domain for its slot (`X` for `A`, `Y` for `B`, `C` for the
/// output), except that a `Pattern` domain takes any type. GraphBLAS would
/// cast a mismatched built-in instead; this layer refuses (the casting rule
/// in the crate docs), so a semiring over `f64` does not run on an `i64`
/// matrix as it stands. Pass `&semiring.typecast()` to let GraphBLAS cast
/// in this call, or store a converted copy with [`cast`] first. The same
/// holds for every operation below that takes an operator.
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn mxm<'a, A, B, C, X, Y, Z>(
    c: &mut Matrix<C>,
    mask: Mask<'a, GrB_Matrix>,
    accum: Option<&BinaryOp<C, C, C>>,
    semiring: &Semiring<X, Y, Z>,
    a: A,
    b: B,
    desc: Desc,
) -> Result<()>
where
    C: GrbType,
    A: Operand<'a, C, GrB_Matrix>,
    B: Operand<'a, C, GrB_Matrix>,
    X: Accepts<A::Elem>,
    Y: Accepts<B::Elem>,
    Z: Produces<C>,
{
    run(c.raw, &mask, desc, |c, m, d| {
        // SAFETY: live handles whose types match the semiring's domains, or
        // cast to them by GraphBLAS's rule; GraphBLAS allows any aliasing
        // among them.
        unsafe {
            sys::GrB_mxm(
                c,
                m,
                accum_raw(accum),
                semiring.as_raw(),
                a.raw(c, Token),
                b.raw(c, Token),
                d,
            )
        }
    })
}

/// `C<Mask> = accum(C, A ⊕ B)`: set union. `op` combines entries present in
/// both; entries present in one side are copied through.
///
/// That copy is why the inputs and the output share one element type
/// (`X: PassesThrough`).
///
/// With `&op.typecast()` the inputs may be of other built-in types, for an
/// operator that reads and writes one type ([`PassesThrough`] says why).
/// The entries only one side has are then cast too, to the operator's type
/// and from there to the output's, without the operator being called: a
/// `u64` id of `0` unioned into a `bool` matrix lands as `false`. For a
/// pattern union of a *valued* matrix into a `bool` one use [`apply`] with
/// `ONE_BOOL` and `accum = Some(&ANY_BOOL)` instead.
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn ewise_add<'a, A, B, C, X, Y, Z>(
    c: &mut Matrix<C>,
    mask: Mask<'a, GrB_Matrix>,
    accum: Option<&BinaryOp<C, C, C>>,
    op: &BinaryOp<X, Y, Z>,
    a: A,
    b: B,
    desc: Desc,
) -> Result<()>
where
    C: GrbType,
    A: Operand<'a, C, GrB_Matrix>,
    B: Operand<'a, C, GrB_Matrix>,
    X: Accepts<A::Elem> + PassesThrough<A::Elem, Z>,
    Y: Accepts<B::Elem> + PassesThrough<B::Elem, Z>,
    Z: Produces<C>,
{
    run(c.raw, &mask, desc, |c, m, d| {
        // SAFETY: live handles of one element type, or cast to the
        // operator's by GraphBLAS's rule; aliasing allowed.
        unsafe {
            sys::GrB_Matrix_eWiseAdd_BinaryOp(
                c,
                m,
                accum_raw(accum),
                op.as_raw(),
                a.raw(c, Token),
                b.raw(c, Token),
                d,
            )
        }
    })
}

/// `C<Mask> = accum(C, A ⊗ B)`: set intersection.
///
/// Only positions present in both inputs get an entry, always `op`'s
/// result, so the output type is `op`'s output type. For a monoid or
/// semiring operator pass [`Monoid::op`] or [`Semiring::multiply`].
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn ewise_mult<'a, A, B, C, X, Y, Z>(
    c: &mut Matrix<C>,
    mask: Mask<'a, GrB_Matrix>,
    accum: Option<&BinaryOp<C, C, C>>,
    op: &BinaryOp<X, Y, Z>,
    a: A,
    b: B,
    desc: Desc,
) -> Result<()>
where
    C: GrbType,
    A: Operand<'a, C, GrB_Matrix>,
    B: Operand<'a, C, GrB_Matrix>,
    X: Accepts<A::Elem>,
    Y: Accepts<B::Elem>,
    Z: Produces<C>,
{
    run(c.raw, &mask, desc, |c, m, d| {
        // SAFETY: live handles whose types match the operator's domains, or
        // cast to them by GraphBLAS's rule; aliasing allowed.
        unsafe {
            sys::GrB_Matrix_eWiseMult_BinaryOp(
                c,
                m,
                accum_raw(accum),
                op.as_raw(),
                a.raw(c, Token),
                b.raw(c, Token),
                d,
            )
        }
    })
}

/// `C<Mask> = accum(C, op(A))`.
///
/// With `a = Output`, no mask and no accumulator, GraphBLAS rewrites the
/// values in place.
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn apply<'a, A, C, X, Z>(
    c: &mut Matrix<C>,
    mask: Mask<'a, GrB_Matrix>,
    accum: Option<&BinaryOp<C, C, C>>,
    op: &UnaryOp<X, Z>,
    a: A,
    desc: Desc,
) -> Result<()>
where
    C: GrbType,
    A: Operand<'a, C, GrB_Matrix>,
    X: Accepts<A::Elem>,
    Z: Produces<C>,
{
    run(c.raw, &mask, desc, |c, m, d| {
        // SAFETY: live handles whose types match the operator's domains, or
        // cast to them by GraphBLAS's rule; aliasing allowed.
        unsafe { sys::GrB_Matrix_apply(c, m, accum_raw(accum), op.as_raw(), a.raw(c, Token), d) }
    })
}

/// `C<Mask> = accum(C, op(A, i, j, y))`: an index-unary operator applied to
/// every entry with its position and the scalar `y`.
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn apply_index<'a, A, C, X, Y, Z>(
    c: &mut Matrix<C>,
    mask: Mask<'a, GrB_Matrix>,
    accum: Option<&BinaryOp<C, C, C>>,
    op: &IndexUnaryOp<X, Y, Z>,
    a: A,
    y: Y,
    desc: Desc,
) -> Result<()>
where
    C: GrbType,
    A: Operand<'a, C, GrB_Matrix>,
    X: Accepts<A::Elem>,
    Y: GrbType,
    Z: Produces<C>,
{
    run(c.raw, &mask, desc, |c, m, d| {
        // SAFETY: live handles whose types match the operator's domains, or
        // cast to them by GraphBLAS's rule, `y` passed through the entry
        // point of its own type; aliasing allowed.
        unsafe { Y::matrix_apply_index(c, m, accum_raw(accum), op.as_raw(), a.raw(c, Token), y, d) }
    })
}

/// `C<Mask> = accum(C, A[op(A, i, j, y)])`: keep the entries of `A` the
/// predicate accepts, values unchanged.
///
/// The predicate may be `op.typecast()`: a `VALUEGT_FP64` on a `u64`
/// matrix compares each value, cast to `f64`, with the scalar.
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn select<'a, T, A, X, Y, Z>(
    c: &mut Matrix<T>,
    mask: Mask<'a, GrB_Matrix>,
    accum: Option<&BinaryOp<T, T, T>>,
    op: &IndexUnaryOp<X, Y, Z>,
    a: A,
    y: Y,
    desc: Desc,
) -> Result<()>
where
    T: GrbType,
    A: Operand<'a, T, GrB_Matrix, Elem = T>,
    X: Accepts<T>,
    Y: GrbType,
    Z: Produces<bool>,
{
    run(c.raw, &mask, desc, |c, m, d| {
        // SAFETY: as in `apply_index`.
        unsafe { Y::matrix_select(c, m, accum_raw(accum), op.as_raw(), a.raw(c, Token), y, d) }
    })
}

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

/// `C<Mask>(rows, cols) = accum(C(rows, cols), x)`: one value over a
/// sub-grid of positions.
///
/// # Errors
///
/// Index out of range, out of memory.
pub fn assign_scalar<T: GrbType>(
    c: &mut Matrix<T>,
    mask: Mask<'_, GrB_Matrix>,
    accum: Option<&BinaryOp<T, T, T>>,
    x: T,
    rows: Indices<'_>,
    cols: Indices<'_>,
    desc: Desc,
) -> Result<()> {
    let (ri, rn) = rows.raw(|| c.nrows());
    let (ci, cn) = cols.raw(|| c.ncols());
    run(c.raw, &mask, desc, |c, m, d| {
        // SAFETY: a live matrix of `T`'s type; index lists that outlive the
        // call.
        unsafe { T::matrix_assign_scalar(c, m, accum_raw(accum), x, ri, rn, ci, cn, d) }
    })
}

/// `C<Mask>(rows, cols) = accum(C(rows, cols), A)`: write a matrix into a
/// sub-grid of positions (`GrB_Matrix_assign`). The mask is over all of `C`.
///
/// # Errors
///
/// Dimension mismatch, index out of range, out of memory.
pub fn assign<'a, T, A>(
    c: &mut Matrix<T>,
    mask: Mask<'a, GrB_Matrix>,
    accum: Option<&BinaryOp<T, T, T>>,
    a: A,
    rows: Indices<'_>,
    cols: Indices<'_>,
    desc: Desc,
) -> Result<()>
where
    T: GrbType,
    A: Operand<'a, T, GrB_Matrix, Elem = T>,
{
    let (ri, rn) = rows.raw(|| c.nrows());
    let (ci, cn) = cols.raw(|| c.ncols());
    run(c.raw, &mask, desc, |c, m, d| {
        // SAFETY: live handles of one element type; index lists that
        // outlive the call; aliasing allowed.
        unsafe {
            sys::GrB_Matrix_assign(c, m, accum_raw(accum), a.raw(c, Token), ri, rn, ci, cn, d)
        }
    })
}

/// `C<Mask> = accum(C, A(rows, cols))`: a sub-matrix (`GrB_Matrix_extract`).
///
/// # Errors
///
/// Dimension mismatch, index out of range, out of memory.
pub fn extract<'a, T, A>(
    c: &mut Matrix<T>,
    mask: Mask<'a, GrB_Matrix>,
    accum: Option<&BinaryOp<T, T, T>>,
    a: A,
    rows: Indices<'_>,
    cols: Indices<'_>,
    desc: Desc,
) -> Result<()>
where
    T: GrbType,
    A: Operand<'a, T, GrB_Matrix, Elem = T>,
{
    // `All` spans the input's dimensions, after the transpose if any.
    let a_raw = a.raw(c.raw, Token);
    let (a_rows, a_cols) = if desc.transpose_first {
        (raw_ncols(a_raw), raw_nrows(a_raw))
    } else {
        (raw_nrows(a_raw), raw_ncols(a_raw))
    };
    let (ri, rn) = rows.raw(|| a_rows);
    let (ci, cn) = cols.raw(|| a_cols);
    run(c.raw, &mask, desc, |c, m, d| {
        // SAFETY: live handles of one element type; index lists that
        // outlive the call; aliasing allowed.
        unsafe { sys::GrB_Matrix_extract(c, m, accum_raw(accum), a_raw, ri, rn, ci, cn, d) }
    })
}

/// `c = accum(c, ⊕ A)`: fold every entry of `A` with `monoid` into the
/// scalar `c`. With no accumulator `c` is overwritten; an empty `A` gives
/// the monoid's identity.
///
/// # Errors
///
/// Out of memory.
pub fn reduce<A, C, M>(
    c: &mut C,
    accum: Option<&BinaryOp<C, C, C>>,
    monoid: &Monoid<M>,
    a: &Matrix<A>,
    desc: Desc,
) -> Result<()>
where
    C: GrbType,
    M: Accepts<A> + Produces<C>,
{
    let d = descriptor(MaskMode::NONE, desc);
    // SAFETY: `c` is a valid `C`, written through the entry point of its
    // own type; live handles whose types are the monoid's, or cast to it by
    // GraphBLAS's rule.
    check(unsafe { C::matrix_reduce(c, accum_raw(accum), monoid.as_raw(), a.raw, d) })
}

/// `w<mask> = accum(w, ⊕ A(i, :))`: fold each row of `A` into `w(i)`; with
/// `desc.transpose_first`, each column (`GrB_Matrix_reduce_Monoid`).
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn reduce_rows<A, C, M>(
    w: &mut Vector<C>,
    mask: Mask<'_, GrB_Vector>,
    accum: Option<&BinaryOp<C, C, C>>,
    monoid: &Monoid<M>,
    a: &Matrix<A>,
    desc: Desc,
) -> Result<()>
where
    C: GrbType,
    M: Accepts<A> + Produces<C>,
{
    let m = mask.raw(w.as_raw());
    let d = descriptor(mask.mode(), desc);
    // SAFETY: live handles whose types are the monoid's, or cast to it by
    // GraphBLAS's rule.
    w.check(unsafe {
        sys::GrB_Matrix_reduce_Monoid(w.as_raw(), m, accum_raw(accum), monoid.as_raw(), a.raw, d)
    })
}

/// `C<Mask> = accum(C, (C's type) A)`: copy `A` into `C`, converting the
/// element type with GraphBLAS's own rules.
///
/// The one place in this crate where a converted copy is stored;
/// `op.typecast()` converts inside an operation without one.
///
/// Implemented as `GrB_assign` over `GrB_ALL`, LAGraph's idiom for a
/// typecast copy, which SuiteSparse routes to its whole-matrix `C = A`
/// method. Built-in types only: GraphBLAS cannot cast user-defined ones.
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn cast<A, C>(
    c: &mut Matrix<C>,
    mask: Mask<'_, GrB_Matrix>,
    accum: Option<&BinaryOp<C, C, C>>,
    a: &Matrix<A>,
    desc: Desc,
) -> Result<()>
where
    A: BuiltinType,
    C: BuiltinType,
{
    let (ri, rn) = Indices::All.raw(|| c.nrows());
    let (ci, cn) = Indices::All.raw(|| c.ncols());
    run(c.raw, &mask, desc, |c, m, d| {
        // SAFETY: live handles of built-in types, which GraphBLAS casts
        // between; `GrB_ALL` with the output's dimensions.
        unsafe { sys::GrB_Matrix_assign(c, m, accum_raw(accum), a.raw, ri, rn, ci, cn, d) }
    })
}
