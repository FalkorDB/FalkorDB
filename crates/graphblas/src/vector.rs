//! [`Vector<T>`] and every operation whose output is a vector, mirroring
//! [`matrix`](crate::matrix) one for one, plus `mxv` and `vxm`.

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
    matrix::Matrix,
    operand::Operand,
    ops::{BinaryOp, IndexUnaryOp, Monoid, Semiring, UnaryOp, accum_raw},
    sealed::{Sealed, Token},
    settings::{IntegerWidth, Sparsity},
    sys::{self, GrB_Index, GrB_Info, GrB_Vector, GxB_Option_Field},
    types::{Accepts, BuiltinType, GrbType},
};

/// `GrB_Vector_size` on a raw handle.
fn raw_size(v: GrB_Vector) -> GrB_Index {
    let mut n = 0;
    // SAFETY: a live vector and a valid out-pointer.
    let info = unsafe { sys::GrB_Vector_size(&raw mut n, v) };
    expect_success(info, "GrB_Vector_size");
    n
}

/// An owned `GrB_Vector` whose entries are `T`. See [`Matrix`] for the
/// contract on `T` and on `&self` methods.
pub struct Vector<T> {
    raw: GrB_Vector,
    _elem: PhantomData<T>,
}

// SAFETY: a GraphBLAS handle is not tied to the thread that created it; the
// wrapper holds nothing else. Deliberately not `Sync`.
unsafe impl<T: Send> Send for Vector<T> {}

impl<T> Sealed for Vector<T> {}

// SAFETY: as for `Matrix<T>`: with nothing pending, every `&self` method and
// every input or mask slot only reads the vector.
unsafe impl<T: GrbType> Freezable for Vector<T> {
    unsafe fn materialize(&self) -> Result<()> {
        handle::wait(self.raw, WaitMode::Materialize)
    }
}

impl<T: GrbType> Vector<T> {
    /// An empty vector of length `size`.
    ///
    /// # Errors
    ///
    /// Size over `GrB_INDEX_MAX + 1`, out of memory, not initialised.
    pub fn new(size: GrB_Index) -> Result<Self> {
        init::ensure_initialized()?;
        let mut raw = null_mut();
        // SAFETY: a valid out-pointer and a live type handle.
        check(unsafe { sys::GrB_Vector_new(&raw mut raw, T::grb_type(), size) })?;
        // SAFETY: just created with `T`'s type; owned here from now on.
        Ok(unsafe { Self::from_raw(raw) })
    }

    /// Take ownership of a raw handle.
    ///
    /// # Safety
    ///
    /// `raw` is a live, non-null vector whose GraphBLAS type is `T`'s, and
    /// nothing else uses or frees it afterwards: the result owns it alone.
    pub unsafe fn from_raw(raw: GrB_Vector) -> Self {
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
    /// `raw` is a live, non-null vector and nothing else uses or frees it
    /// afterwards.
    ///
    /// # Errors
    ///
    /// `Err(raw)` when the element type is not `T`'s.
    pub unsafe fn try_from_raw(raw: GrB_Vector) -> std::result::Result<Self, GrB_Vector> {
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
    pub const fn into_raw(self) -> GrB_Vector {
        let raw = self.raw;
        forget(self);
        raw
    }

    /// The raw handle, still owned here. For the LAGraph wrapper and for the
    /// engine's domain wrapper while it migrates; not for new code.
    #[must_use]
    pub const fn as_raw(&self) -> GrB_Vector {
        self.raw
    }

    #[must_use]
    pub fn size(&self) -> GrB_Index {
        raw_size(self.raw)
    }

    /// Number of stored entries. Finishes pending work first.
    ///
    /// # Panics
    ///
    /// When finishing the pending work runs out of memory, the one way it
    /// can fail; the vector is unusable afterwards.
    #[must_use]
    pub fn nvals(&self) -> GrB_Index {
        let mut n = 0;
        // SAFETY: a live vector and a valid out-pointer.
        let info = unsafe { sys::GrB_Vector_nvals(&raw mut n, self.raw) };
        expect_success(info, "GrB_Vector_nvals");
        n
    }

    /// A copy, pending work included.
    ///
    /// # Errors
    ///
    /// Out of memory.
    pub fn dup(&self) -> Result<Self> {
        let mut raw = null_mut();
        // SAFETY: a valid out-pointer and a live vector.
        handle::check_read(unsafe { sys::GrB_Vector_dup(&raw mut raw, self.raw) })?;
        // SAFETY: a fresh vector of `T`'s type, owned here from now on.
        Ok(unsafe { Self::from_raw(raw) })
    }

    /// Remove every entry.
    ///
    /// # Errors
    ///
    /// Never in practice.
    pub fn clear(&mut self) -> Result<()> {
        // SAFETY: a live vector.
        self.check(unsafe { sys::GrB_Vector_clear(self.raw) })
    }

    /// Change the length, dropping entries that fall outside.
    ///
    /// # Errors
    ///
    /// Size over `GrB_INDEX_MAX + 1`, out of memory.
    pub fn resize(
        &mut self,
        size: GrB_Index,
    ) -> Result<()> {
        // SAFETY: a live vector.
        self.check(unsafe { sys::GrB_Vector_resize(self.raw, size) })
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
    /// several threads; see [`Matrix::freeze`](crate::Matrix::freeze).
    #[must_use]
    pub fn freeze(self) -> Frozen<Self> {
        Frozen::new(self)
    }

    /// Whether the vector has pending work.
    #[must_use]
    pub fn is_pending(&self) -> bool {
        handle::is_pending(self.raw)
    }

    /// `w(i) = x`.
    ///
    /// # Errors
    ///
    /// Index out of range, out of memory.
    pub fn set_element(
        &mut self,
        i: GrB_Index,
        x: T,
    ) -> Result<()> {
        // SAFETY: a live vector of `T`'s type.
        self.check(unsafe { T::vector_set_element(self.raw, x, i) })
    }

    /// Delete the entry at `i` if there is one.
    ///
    /// # Errors
    ///
    /// Index out of range.
    pub fn remove_element(
        &mut self,
        i: GrB_Index,
    ) -> Result<()> {
        // SAFETY: a live vector.
        self.check(unsafe { sys::GrB_Vector_removeElement(self.raw, i) })
    }

    /// The entry at `i`, or `None` when there is none. Finishes pending work
    /// first.
    ///
    /// # Errors
    ///
    /// Index out of range.
    pub fn extract_element(
        &self,
        i: GrB_Index,
    ) -> Result<Option<T>> {
        let mut x = MaybeUninit::<T>::uninit();
        // SAFETY: a live vector of `T`'s type and room for one `T`.
        let info = unsafe { T::vector_extract_element(x.as_mut_ptr(), self.raw, i) };
        match info {
            // SAFETY: GraphBLAS wrote the value on success.
            GrB_Info::GrB_SUCCESS => Ok(Some(unsafe { x.assume_init() })),
            GrB_Info::GrB_NO_VALUE => Ok(None),
            other => handle::check_read(other).map(|()| None),
        }
    }

    /// Whether an entry is stored at `i`; never reads the value. Pending
    /// work may not be visible.
    ///
    /// # Errors
    ///
    /// Index out of range.
    pub fn contains(
        &self,
        i: GrB_Index,
    ) -> Result<bool> {
        // SAFETY: a live vector.
        match unsafe { sys::GxB_Vector_isStoredElement(self.raw, i) } {
            GrB_Info::GrB_SUCCESS => Ok(true),
            GrB_Info::GrB_NO_VALUE => Ok(false),
            other => handle::check_read(other).map(|()| false),
        }
    }

    /// Bulk-load `w(indices[k]) = values[k]` into an empty vector.
    /// Duplicates are combined with `dup`; with `dup = None` a duplicate is
    /// an error.
    ///
    /// # Panics
    ///
    /// If the two slices differ in length.
    ///
    /// # Errors
    ///
    /// The vector is not empty, an index is out of range, a duplicate with
    /// no `dup`, out of memory.
    pub fn build(
        &mut self,
        indices: &[GrB_Index],
        values: &[T],
        dup: Option<&BinaryOp<T, T, T>>,
    ) -> Result<()> {
        assert_eq!(
            indices.len(),
            values.len(),
            "indices and values differ in length"
        );
        // SAFETY: a live vector of `T`'s type; two slices of the stated
        // length that outlive the call.
        self.check(unsafe {
            T::vector_build(
                self.raw,
                indices.as_ptr(),
                values.as_ptr(),
                indices.len() as GrB_Index,
                accum_raw(dup),
            )
        })
    }

    /// Bulk-load the pattern `indices` with every entry equal to `value`, as
    /// an iso vector. Duplicates are allowed.
    ///
    /// # Errors
    ///
    /// The vector is not empty, an index is out of range, out of memory.
    pub fn build_iso(
        &mut self,
        indices: &[GrB_Index],
        value: T,
    ) -> Result<()> {
        let scalar = Scalar::from_value(value)?;
        // SAFETY: a live vector of `T`'s type, a live scalar of the same
        // type, a slice of the stated length that outlives the call.
        self.check(unsafe {
            sys::GxB_Vector_build_Scalar(
                self.raw,
                indices.as_ptr(),
                scalar.as_raw(),
                indices.len() as GrB_Index,
            )
        })
    }

    /// Every stored entry as `(indices, values)`, in storage order, which
    /// is index order only when the vector is not jumbled. Finishes pending
    /// work first. See [`Matrix::extract_tuples`] on when to use it.
    ///
    /// # Errors
    ///
    /// Out of memory.
    pub fn extract_tuples(&self) -> Result<(Vec<GrB_Index>, Vec<T>)> {
        let capacity = handle::entry_count(self.nvals());
        let mut idx = Vec::with_capacity(capacity);
        let mut vals = Vec::<T>::with_capacity(capacity);
        let mut n = capacity as GrB_Index;
        // SAFETY: a live vector of `T`'s type and two buffers with room for
        // `n` entries; GraphBLAS writes at most `n` and stores how many.
        let info = unsafe {
            T::vector_extract_tuples(idx.as_mut_ptr(), vals.as_mut_ptr(), &raw mut n, self.raw)
        };
        handle::check_read(info)?;
        let n = handle::entry_count(n);
        debug_assert!(n <= capacity);
        // SAFETY: GraphBLAS initialised the first `n` elements of each.
        unsafe {
            idx.set_len(n);
            vals.set_len(n);
        }
        Ok((idx, vals))
    }

    /// Bytes GraphBLAS holds for this vector.
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        handle::memory_usage(self.raw)
    }

    /// Whether the vector is iso (one shared value).
    #[must_use]
    pub fn is_iso(&self) -> bool {
        handle::is_iso(self.raw)
    }

    /// The current storage format.
    #[must_use]
    pub fn sparsity(&self) -> Sparsity {
        handle::sparsity(self.raw)
    }

    /// Pin or free the storage format.
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

    /// The width GraphBLAS actually uses for indices.
    #[must_use]
    pub fn index_width(&self) -> IntegerWidth {
        IntegerWidth::from_bits(self.get_i32(GxB_Option_Field::GxB_ROWINDEX_INTEGER_BITS))
    }

    /// `Ok` on success, else the error with this vector's message.
    pub(crate) fn check(
        &self,
        info: GrB_Info,
    ) -> Result<()> {
        handle::check(info, self.raw)
    }

    /// `GrB_Vector_get_INT32` for a field that cannot fail on a live handle.
    fn get_i32(
        &self,
        field: GxB_Option_Field,
    ) -> i32 {
        handle::get_i32(self.raw, field)
    }
}

impl<T> Drop for Vector<T> {
    fn drop(&mut self) {
        // SAFETY: `raw` is owned by this value and freed exactly once.
        let _ = unsafe { sys::GrB_Vector_free(&raw mut self.raw) };
    }
}

impl<T: GrbType> fmt::Debug for Vector<T> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(f, "Vector<{}>({} @ {:p})", T::NAME, self.size(), self.raw)
    }
}

// ───────────────────────────── operations ─────────────────────────────

/// `w<mask> = accum(w, A * u)` over `semiring`; types as in
/// [`matrix::mxm`](crate::matrix::mxm).
///
/// `A` is the first input: [`Desc::T0`] (`transpose_first`) uses `Aᵀ`, and
/// `transpose_second` is ignored.
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn mxv<'a, A, U, C, X, Y>(
    w: &mut Vector<C>,
    mask: Mask<'a, GrB_Vector>,
    accum: Option<&BinaryOp<C, C, C>>,
    semiring: &Semiring<X, Y, C>,
    a: &Matrix<A>,
    u: U,
    desc: Desc,
) -> Result<()>
where
    A: GrbType,
    C: GrbType,
    U: Operand<'a, C, GrB_Vector>,
    X: Accepts<A>,
    Y: Accepts<U::Elem>,
{
    run(w.raw, &mask, desc, |w, m, d| {
        // SAFETY: live handles whose types match the semiring's domains;
        // aliasing allowed.
        unsafe {
            sys::GrB_mxv(
                w,
                m,
                accum_raw(accum),
                semiring.as_raw(),
                a.as_raw(),
                u.raw(w, Token),
                d,
            )
        }
    })
}

/// `w<mask> = accum(w, u * A)` over `semiring`; types as in
/// [`matrix::mxm`](crate::matrix::mxm).
///
/// `A` is the second input: [`Desc::T1`] (`transpose_second`) uses `Aᵀ`,
/// and `transpose_first` is ignored.
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn vxm<'a, U, A, C, X, Y>(
    w: &mut Vector<C>,
    mask: Mask<'a, GrB_Vector>,
    accum: Option<&BinaryOp<C, C, C>>,
    semiring: &Semiring<X, Y, C>,
    u: U,
    a: &Matrix<A>,
    desc: Desc,
) -> Result<()>
where
    A: GrbType,
    C: GrbType,
    U: Operand<'a, C, GrB_Vector>,
    X: Accepts<U::Elem>,
    Y: Accepts<A>,
{
    run(w.raw, &mask, desc, |w, m, d| {
        // SAFETY: as in `mxv`.
        unsafe {
            sys::GrB_vxm(
                w,
                m,
                accum_raw(accum),
                semiring.as_raw(),
                u.raw(w, Token),
                a.as_raw(),
                d,
            )
        }
    })
}

/// `w<mask> = accum(w, u ⊕ v)`: set union; see
/// [`matrix::ewise_add`](crate::matrix::ewise_add).
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn ewise_add<'a, T, X, Y, U, V>(
    w: &mut Vector<T>,
    mask: Mask<'a, GrB_Vector>,
    accum: Option<&BinaryOp<T, T, T>>,
    op: &BinaryOp<X, Y, T>,
    u: U,
    v: V,
    desc: Desc,
) -> Result<()>
where
    T: GrbType,
    U: Operand<'a, T, GrB_Vector, Elem = T>,
    V: Operand<'a, T, GrB_Vector, Elem = T>,
    X: Accepts<T>,
    Y: Accepts<T>,
{
    run(w.raw, &mask, desc, |w, m, d| {
        // SAFETY: live handles of one element type; aliasing allowed.
        unsafe {
            sys::GrB_Vector_eWiseAdd_BinaryOp(
                w,
                m,
                accum_raw(accum),
                op.as_raw(),
                u.raw(w, Token),
                v.raw(w, Token),
                d,
            )
        }
    })
}

/// `w<mask> = accum(w, u ⊗ v)`: set intersection; see
/// [`matrix::ewise_mult`](crate::matrix::ewise_mult).
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn ewise_mult<'a, U, V, C, X, Y>(
    w: &mut Vector<C>,
    mask: Mask<'a, GrB_Vector>,
    accum: Option<&BinaryOp<C, C, C>>,
    op: &BinaryOp<X, Y, C>,
    u: U,
    v: V,
    desc: Desc,
) -> Result<()>
where
    C: GrbType,
    U: Operand<'a, C, GrB_Vector>,
    V: Operand<'a, C, GrB_Vector>,
    X: Accepts<U::Elem>,
    Y: Accepts<V::Elem>,
{
    run(w.raw, &mask, desc, |w, m, d| {
        // SAFETY: live handles whose types match the operator's domains;
        // aliasing allowed.
        unsafe {
            sys::GrB_Vector_eWiseMult_BinaryOp(
                w,
                m,
                accum_raw(accum),
                op.as_raw(),
                u.raw(w, Token),
                v.raw(w, Token),
                d,
            )
        }
    })
}

/// `w<mask> = accum(w, op(u))`. With `u = Output`, no mask and no
/// accumulator, GraphBLAS rewrites the values in place.
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn apply<'a, U, C, X>(
    w: &mut Vector<C>,
    mask: Mask<'a, GrB_Vector>,
    accum: Option<&BinaryOp<C, C, C>>,
    op: &UnaryOp<X, C>,
    u: U,
    desc: Desc,
) -> Result<()>
where
    C: GrbType,
    U: Operand<'a, C, GrB_Vector>,
    X: Accepts<U::Elem>,
{
    run(w.raw, &mask, desc, |w, m, d| {
        // SAFETY: live handles whose types match the operator's domains;
        // aliasing allowed.
        unsafe { sys::GrB_Vector_apply(w, m, accum_raw(accum), op.as_raw(), u.raw(w, Token), d) }
    })
}

/// `w<mask> = accum(w, op(u, i, 0, y))`.
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn apply_index<'a, U, C, X, Y>(
    w: &mut Vector<C>,
    mask: Mask<'a, GrB_Vector>,
    accum: Option<&BinaryOp<C, C, C>>,
    op: &IndexUnaryOp<X, Y, C>,
    u: U,
    y: Y,
    desc: Desc,
) -> Result<()>
where
    C: GrbType,
    U: Operand<'a, C, GrB_Vector>,
    X: Accepts<U::Elem>,
    Y: GrbType,
{
    run(w.raw, &mask, desc, |w, m, d| {
        // SAFETY: live handles whose types match the operator's domains, `y`
        // passed through the entry point of its own type; aliasing allowed.
        unsafe { Y::vector_apply_index(w, m, accum_raw(accum), op.as_raw(), u.raw(w, Token), y, d) }
    })
}

/// `w<mask> = accum(w, u[op(u, i, 0, y)])`: keep the entries the predicate
/// accepts, values unchanged.
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn select<'a, T, U, X, Y>(
    w: &mut Vector<T>,
    mask: Mask<'a, GrB_Vector>,
    accum: Option<&BinaryOp<T, T, T>>,
    op: &IndexUnaryOp<X, Y, bool>,
    u: U,
    y: Y,
    desc: Desc,
) -> Result<()>
where
    T: GrbType,
    U: Operand<'a, T, GrB_Vector, Elem = T>,
    X: Accepts<T>,
    Y: GrbType,
{
    run(w.raw, &mask, desc, |w, m, d| {
        // SAFETY: as in `apply_index`.
        unsafe { Y::vector_select(w, m, accum_raw(accum), op.as_raw(), u.raw(w, Token), y, d) }
    })
}

/// `w<mask>(indices) = accum(w(indices), x)`.
///
/// # Errors
///
/// Index out of range, out of memory.
pub fn assign_scalar<T: GrbType>(
    w: &mut Vector<T>,
    mask: Mask<'_, GrB_Vector>,
    accum: Option<&BinaryOp<T, T, T>>,
    x: T,
    indices: Indices<'_>,
    desc: Desc,
) -> Result<()> {
    let (ix, n) = indices.raw(|| w.size());
    run(w.raw, &mask, desc, |w, m, d| {
        // SAFETY: a live vector of `T`'s type; an index list that outlives
        // the call.
        unsafe { T::vector_assign_scalar(w, m, accum_raw(accum), x, ix, n, d) }
    })
}

/// `w<mask>(indices) = accum(w(indices), u)` (`GrB_Vector_assign`).
///
/// # Errors
///
/// Dimension mismatch, index out of range, out of memory.
pub fn assign<'a, T, U>(
    w: &mut Vector<T>,
    mask: Mask<'a, GrB_Vector>,
    accum: Option<&BinaryOp<T, T, T>>,
    u: U,
    indices: Indices<'_>,
    desc: Desc,
) -> Result<()>
where
    T: GrbType,
    U: Operand<'a, T, GrB_Vector, Elem = T>,
{
    let (ix, n) = indices.raw(|| w.size());
    run(w.raw, &mask, desc, |w, m, d| {
        // SAFETY: live handles of one element type; an index list that
        // outlives the call; aliasing allowed.
        unsafe { sys::GrB_Vector_assign(w, m, accum_raw(accum), u.raw(w, Token), ix, n, d) }
    })
}

/// `w<mask> = accum(w, u(indices))`: a sub-vector (`GrB_Vector_extract`).
///
/// # Errors
///
/// Dimension mismatch, index out of range, out of memory.
pub fn extract<'a, T, U>(
    w: &mut Vector<T>,
    mask: Mask<'a, GrB_Vector>,
    accum: Option<&BinaryOp<T, T, T>>,
    u: U,
    indices: Indices<'_>,
    desc: Desc,
) -> Result<()>
where
    T: GrbType,
    U: Operand<'a, T, GrB_Vector, Elem = T>,
{
    // `All` spans the input's size.
    let u_raw = u.raw(w.raw, Token);
    let (ix, n) = indices.raw(|| raw_size(u_raw));
    run(w.raw, &mask, desc, |w, m, d| {
        // SAFETY: live handles of one element type; an index list that
        // outlives the call; aliasing allowed.
        unsafe { sys::GrB_Vector_extract(w, m, accum_raw(accum), u_raw, ix, n, d) }
    })
}

/// `c = accum(c, ⊕ u)`: fold every entry of `u` with `monoid` into the
/// scalar `c`; see [`matrix::reduce`](crate::matrix::reduce).
///
/// # Errors
///
/// Out of memory.
pub fn reduce<T: GrbType>(
    c: &mut T,
    accum: Option<&BinaryOp<T, T, T>>,
    monoid: &Monoid<T>,
    u: &Vector<T>,
    desc: Desc,
) -> Result<()> {
    let d = descriptor(MaskMode::NONE, desc);
    // SAFETY: `c` is a valid `T`; live handles whose types are `T`.
    check(unsafe { T::vector_reduce(c, accum_raw(accum), monoid.as_raw(), u.raw, d) })
}

/// `w<mask> = accum(w, (w's type) u)`: copy `u` into `w`, converting the
/// element type; see [`matrix::cast`](crate::matrix::cast).
///
/// # Errors
///
/// Dimension mismatch, out of memory.
pub fn cast<U, C>(
    w: &mut Vector<C>,
    mask: Mask<'_, GrB_Vector>,
    accum: Option<&BinaryOp<C, C, C>>,
    u: &Vector<U>,
    desc: Desc,
) -> Result<()>
where
    U: BuiltinType,
    C: BuiltinType,
{
    let (ix, n) = Indices::All.raw(|| w.size());
    run(w.raw, &mask, desc, |w, m, d| {
        // SAFETY: live handles of built-in types, which GraphBLAS casts
        // between; `GrB_ALL` with the output's size.
        unsafe { sys::GrB_Vector_assign(w, m, accum_raw(accum), u.raw, ix, n, d) }
    })
}
