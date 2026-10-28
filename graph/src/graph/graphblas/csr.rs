//! Row-major CSR export of a GraphBLAS matrix: the result and error types of
//! [`Matrix::take_csr`], and the handover of GraphBLAS's own container arrays
//! to Rust.
//!
//! The policy (which handles may be exported, and the lock that keeps
//! everyone else off the handle meanwhile) stays with the wrapper in
//! `matrix.rs`; this module checks the storage it unloads and hands it over.

#![allow(clippy::doc_markdown)]

use std::{
    mem::MaybeUninit,
    os::raw::{c_int, c_void},
    ptr::null_mut,
};

use thiserror::Error;

#[cfg(doc)]
use super::matrix::Matrix;
use super::matrix::grb;
use super::{
    GrB_Desc_Value, GrB_FP64, GrB_INT32, GrB_INT64, GrB_Index, GrB_Info, GrB_Matrix,
    GrB_Orientation, GrB_Type, GrB_UINT32, GrB_UINT64, GrB_Vector, GrB_Vector_extractTuples_FP64,
    GrB_Vector_extractTuples_UINT32, GrB_Vector_extractTuples_UINT64, GxB_Container,
    GxB_Container_free, GxB_Container_new, GxB_SPARSE, GxB_Vector_load, GxB_Vector_unload,
    GxB_load_Matrix_from_Container, GxB_unload_Matrix_into_Container,
};

/// Row-major CSR export of a GraphBLAS matrix: what [`Matrix::take_csr`] hands
/// back.
///
/// Row `r` owns the entries `row_ptr[r] .. row_ptr[r + 1]`: their column
/// indices are that slice of `col_idx`, ascending and duplicate-free, and —
/// unless `V` is `()`, the pattern alone — their values are the same slice of
/// `vals`.
/// `row_ptr` has `nrows + 1` entries and ends at the entry count. The widths
/// are GraphBLAS's own, `P` for the offsets and `I` for the column indices
/// (`u32` or `u64` each), which the caller asks the matrix for before
/// exporting.
pub struct CsrExport<P = u64, I = u32, V = ()> {
    pub nrows: u64,
    pub ncols: u64,
    /// `nrows + 1` offsets into `col_idx` (and `vals`, when non-empty).
    pub row_ptr: Vec<P>,
    /// One column index per entry, ascending within each row.
    pub col_idx: Vec<I>,
    /// One value per entry; empty for the pattern export (`V = ()`).
    pub vals: Vec<V>,
}

/// Why [`Matrix::take_csr`] refused a matrix. The export never converts the
/// storage or casts an array: the caller sets the matrix up
/// ([`Matrix::prepare_csr`]) and anything else is refused, the matrix left as
/// it was. The handle and the value type are checked before the unload; the
/// storage is checked on what GraphBLAS unloads, which is loaded back on a
/// refusal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum CsrExportError {
    /// Another `Matrix` shares this handle, and would see the matrix empty
    /// while it is unloaded.
    #[error("another handle shares this matrix")]
    SharedHandle,
    /// Values were asked for (`V` is `f64`, i.e. `FP64`), and the matrix holds
    /// another type.
    #[error("the matrix does not hold FP64 values")]
    NotFp64,
    /// The matrix is not stored as sorted row-major sparse, i.e. CSR: it is
    /// hypersparse, bitmap or full, stored by column, or jumbled.
    #[error("the matrix is not stored as sorted row-major sparse (CSR)")]
    NotCsr,
    /// Values were asked for and the matrix stores one value for every entry.
    #[error("the values are stored iso")]
    IsoValues,
    /// An array (named) is not of the type the export was asked for, or is
    /// shorter than the matrix needs. For the offsets or the column indices
    /// that is a width: a 32-bit hint is ignored, rightly, on a matrix with
    /// `2^32` entries or more (offsets) or more than `2^31` columns (column
    /// indices).
    #[error("{0} are not of the requested type")]
    Array(&'static str),
}

impl CsrExportError {
    /// A fixed description, for error types that carry a `&'static str`.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::SharedHandle => "CSR export: another handle shares the matrix",
            Self::NotFp64 => "CSR export: the matrix does not hold FP64 values",
            Self::NotCsr => "CSR export: the matrix is not stored as sorted row-major sparse",
            Self::IsoValues => "CSR export: the values are stored iso",
            Self::Array(_) => "CSR export: an array is not of the requested type",
        }
    }
}

/// A Rust type a GraphBLAS array is read as bit for bit, with no conversion.
/// Public only because it bounds the CSR export's type parameters.
///
/// # Safety
/// The export trusts an implementor's answers to size, read and free
/// GraphBLAS's arrays as `Vec<Self>`, so an implementor promises:
/// - [`Self::holds`] is `true` only for a GraphBLAS type whose values have
///   exactly `size_of::<Self>()` bytes and alignment at most
///   `align_of::<Self>()`, and every bit pattern of which is a valid `Self`;
/// - [`Self::extract`] writes at most `*n` valid values of `Self` to `out`
///   and leaves in `*n` how many it wrote.
pub unsafe trait GrbElem: Copy {
    /// Whether an array GraphBLAS reports as `ty` holds exactly this type's bits.
    fn holds(ty: GrB_Type) -> bool;

    /// `GrB_Vector_extractTuples_<T>` with no indices: up to `*n` values of
    /// `v`, written (and cast, if `v` is another type) into `out`.
    ///
    /// # Safety
    /// `out` has room for `*n` values, and `v` is a live vector.
    #[doc(hidden)]
    unsafe fn extract(
        v: GrB_Vector,
        out: *mut Self,
        n: *mut GrB_Index,
    ) -> GrB_Info;
}

unsafe impl GrbElem for u64 {
    /// GraphBLAS hands its 64-bit index arrays out as `INT64`. An index is at
    /// most `GrB_INDEX_MAX` (`2^60 - 1`), so it has the same bits as a `u64`.
    fn holds(ty: GrB_Type) -> bool {
        unsafe { ty == GrB_UINT64 || ty == GrB_INT64 }
    }

    unsafe fn extract(
        v: GrB_Vector,
        out: *mut Self,
        n: *mut GrB_Index,
    ) -> GrB_Info {
        unsafe { GrB_Vector_extractTuples_UINT64(null_mut(), out, n, v) }
    }
}

unsafe impl GrbElem for u32 {
    /// GraphBLAS hands its 32-bit index arrays out as `INT32`. It keeps an index
    /// array 32-bit only while the dimension is at most `2^31`, so every value
    /// is below `2^31` and has the same bits as a `u32`.
    fn holds(ty: GrB_Type) -> bool {
        unsafe { ty == GrB_UINT32 || ty == GrB_INT32 }
    }

    unsafe fn extract(
        v: GrB_Vector,
        out: *mut Self,
        n: *mut GrB_Index,
    ) -> GrB_Info {
        unsafe { GrB_Vector_extractTuples_UINT32(null_mut(), out, n, v) }
    }
}

unsafe impl GrbElem for f64 {
    fn holds(ty: GrB_Type) -> bool {
        unsafe { ty == GrB_FP64 }
    }

    unsafe fn extract(
        v: GrB_Vector,
        out: *mut Self,
        n: *mut GrB_Index,
    ) -> GrB_Info {
        unsafe { GrB_Vector_extractTuples_FP64(null_mut(), out, n, v) }
    }
}

/// No GraphBLAS array is read as `()`: it is the value type of the pattern
/// export, which leaves the `x` array alone.
unsafe impl GrbElem for () {
    fn holds(_: GrB_Type) -> bool {
        false
    }

    unsafe fn extract(
        _: GrB_Vector,
        _: *mut Self,
        n: *mut GrB_Index,
    ) -> GrB_Info {
        unsafe { *n = 0 };
        GrB_Info::GrB_SUCCESS
    }
}

/// The value type of a CSR export: `()` for the pattern alone, `f64` for
/// `FP64` values. The export reads the matrix's `x` array only when
/// [`Self::HAS_VALUES`], and then requires the matrix to be of this type
/// ([`GrbElem::holds`]) and not iso.
///
/// # Safety
/// [`Self::HAS_VALUES`] is `true` only for a type whose [`GrbElem::holds`]
/// names a GraphBLAS value type: the export then adopts the `x` array as a
/// `Vec<Self>`.
pub unsafe trait CsrValue: GrbElem {
    /// Whether the export carries values.
    const HAS_VALUES: bool;
}

unsafe impl CsrValue for () {
    const HAS_VALUES: bool = false;
}

unsafe impl CsrValue for f64 {
    const HAS_VALUES: bool = true;
}

/// One array of a `GxB_Container` (`p`, `i` or `x`), unloaded from its
/// vector: until it is adopted or loaded back, nobody else owns it.
///
/// SAFETY: only [`Self::unload`] constructs an `Unloaded`, so its caller's
/// promise (a live vector of a container nobody else can reach) covers every
/// later method; that is why `unload` is `unsafe` and `reload`, `fits` and
/// `into_vec` are not.
struct Unloaded {
    v: GrB_Vector,
    x: *mut c_void,
    ty: GrB_Type,
    n: u64,
    /// Bytes GraphBLAS allocated for `x`, at least `n` entries.
    size: u64,
    handling: c_int,
}

impl Unloaded {
    /// Take the array out of `v`, which is left empty.
    ///
    /// # Safety
    /// `v` is a live vector of a container nobody else can reach.
    unsafe fn unload(v: GrB_Vector) -> Self {
        let mut a = Self {
            v,
            x: null_mut(),
            ty: null_mut(),
            n: 0,
            size: 0,
            handling: 0,
        };
        unsafe {
            grb!(GxB_Vector_unload(
                v,
                &raw mut a.x,
                &raw mut a.ty,
                &raw mut a.n,
                &raw mut a.size,
                &raw mut a.handling,
                null_mut(),
            ));
        }
        a
    }

    /// Load the array back into its vector, which owns it again.
    fn reload(mut self) {
        // SAFETY: `x` is what `unload` took out of `v`, with its own type,
        // length, size and handling.
        unsafe {
            grb!(GxB_Vector_load(
                self.v,
                &raw mut self.x,
                self.ty,
                self.n,
                self.size,
                self.handling,
                null_mut()
            ));
        }
    }

    /// The soundness check [`Self::into_vec`] relies on: the array is of
    /// type `T` (read bit for bit, never cast) and holds at least `len`
    /// entries. The export refuses an array that does not fit.
    fn fits<T: GrbElem>(
        &self,
        len: usize,
    ) -> bool {
        len == 0 || (T::holds(self.ty) && self.n as usize >= len)
    }

    /// The first `len` entries as a `Vec<T>`, which [`Self::fits`] allows.
    ///
    /// An array GraphBLAS owns outright (the default handling: not read-only,
    /// no arena) in a whole number of aligned `T`s becomes the `Vec`, in O(1):
    /// GraphBLAS no longer refers to it, and the global allocator frees it,
    /// which [`init`](super::matrix::init)'s contract allows. Any other array
    /// goes back to its vector, for GraphBLAS to free with the container, and
    /// GraphBLAS copies its entries into the `Vec` (`extractTuples`).
    fn into_vec<T: GrbElem>(
        self,
        len: usize,
    ) -> Vec<T> {
        let size = self.size as usize;
        if !self.x.is_null()
            && self.handling == GrB_Desc_Value::GrB_DEFAULT as c_int
            && size.is_multiple_of(size_of::<T>())
            && self.x.cast::<T>().is_aligned()
        {
            // SAFETY: `x` holds at least `len` initialized `T`s (`fits`) in
            // a block of exactly `size` bytes, which GraphBLAS's malloc
            // returned and has just given up (the default handling: it is not
            // a borrowed, read-only array). The global allocator frees that
            // block, under `Vec`'s layout of `size` bytes, as GraphBLAS would
            // (the contract of `init`).
            return unsafe { Vec::from_raw_parts(self.x.cast::<T>(), len, size / size_of::<T>()) };
        }
        let (v, n) = (self.v, self.n);
        self.reload();
        if len == 0 {
            return Vec::new();
        }
        let mut out = Vec::<T>::with_capacity(n as usize);
        let mut got = n;
        // SAFETY: `out` has room for `n` values, the length of `v`, which holds
        // the array again; `fits` checked that `n >= len`.
        unsafe {
            grb!(T::extract(v, out.as_mut_ptr(), &raw mut got));
            out.set_len(got as usize);
        }
        out.truncate(len);
        out
    }
}

/// Take `m` out as CSR: unload it into a `GxB_Container` and hand its `p`, `i`
/// and (unless `V` is `()`) `x` arrays over as the export's `Vec`s, adopting
/// them where GraphBLAS owns them outright ([`Unloaded::into_vec`]).
///
/// The storage is checked here, on the unloaded container: sparse,
/// row-major and not jumbled ([`CsrExportError::NotCsr`]), not iso when `V`
/// carries values ([`CsrExportError::IsoValues`]), and each array of its
/// type and long enough ([`Unloaded::fits`], [`CsrExportError::Array`]).
///
/// On success `m` is left as the unload leaves it: a valid `0 x 0` matrix
/// with no entries and its settings. On a refusal it is loaded back exactly
/// as it was.
///
/// # Safety
/// `m` is a live matrix nobody else can reach until this returns, already
/// materialized and, unless `V` is `()`, of value type `V`: the wrapper's
/// `take_csr` does that first.
pub(super) unsafe fn take_out_csr<P, I, V>(
    m: GrB_Matrix
) -> Result<CsrExport<P, I, V>, CsrExportError>
where
    P: GrbElem + Into<u64>,
    I: GrbElem + Into<u64>,
    V: CsrValue,
{
    unsafe {
        let mut container: MaybeUninit<GxB_Container> = MaybeUninit::uninit();
        grb!(GxB_Container_new(container.as_mut_ptr()));
        let mut container = container.assume_init();
        grb!(GxB_unload_Matrix_into_Container(m, container, null_mut()));
        // Refused: hand everything back, each unloaded array to its vector
        // and the container to `m`, which then holds exactly what it held on
        // entry.
        let refuse = |mut container: GxB_Container, unloaded: Vec<Unloaded>, error| {
            unloaded.into_iter().for_each(Unloaded::reload);
            grb!(GxB_load_Matrix_from_Container(m, container, null_mut()));
            grb!(GxB_Container_free(&raw mut container));
            Err(error)
        };

        let nrows = (*container).nrows;
        let ncols = (*container).ncols;
        let nvals = (*container).nvals as usize;
        if (*container).format != GxB_SPARSE as i32
            || (*container).orientation != GrB_Orientation::GrB_ROWMAJOR as i32
            || (*container).jumbled
        {
            return refuse(container, Vec::new(), CsrExportError::NotCsr);
        }
        // The pattern export leaves `x`, iso or not, with GraphBLAS.
        if V::HAS_VALUES && (*container).iso {
            return refuse(container, Vec::new(), CsrExportError::IsoValues);
        }

        let p = Unloaded::unload((*container).p);
        let i = Unloaded::unload((*container).i);
        let x = V::HAS_VALUES.then(|| Unloaded::unload((*container).x));
        let what = if !p.fits::<P>(nrows as usize + 1) {
            Some("the offsets")
        } else if !i.fits::<I>(nvals) {
            Some("the column indices")
        } else if x.as_ref().is_some_and(|x| !x.fits::<V>(nvals)) {
            Some("the values")
        } else {
            None
        };
        if let Some(what) = what {
            return refuse(
                container,
                [Some(p), Some(i), x].into_iter().flatten().collect(),
                CsrExportError::Array(what),
            );
        }

        // Checked: the arrays leave the container for good.
        let row_ptr = p.into_vec::<P>(nrows as usize + 1);
        let col_idx = i.into_vec::<I>(nvals);
        let vals = x.map_or_else(Vec::new, |x| x.into_vec::<V>(nvals));
        // What the container still holds (an iso pattern value, an array
        // copied rather than adopted) is GraphBLAS's to free.
        grb!(GxB_Container_free(&raw mut container));
        Ok(CsrExport {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            vals,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{CsrExport, CsrExportError, Unloaded};
    use crate::graph::graphblas::matrix::{Matrix, grb};
    use crate::graph::graphblas::{
        GrB_Matrix_get_INT32, GrB_Matrix_set_INT32, GrB_Orientation, GxB_HYPERSPARSE,
        GxB_Option_Field, test_init,
    };

    /// What a caller does before exporting: store `m` as plain CSR with the
    /// export's widths, `u64` offsets and `u32` column indices. The export
    /// checks all of it and sets none of it. Iso values are left alone, for
    /// the iso test; a caller exporting values asks for `V = f64`.
    fn csr_ready<T>(m: &mut Matrix<T>) {
        grb!(m.prepare_csr::<u64, u32, ()>());
    }

    /// Break one storage setting behind `prepare_csr`'s back, as a matrix
    /// stored some other way would be.
    fn set_field<T>(
        m: &Matrix<T>,
        field: GxB_Option_Field,
        value: i32,
    ) {
        grb!(unsafe { GrB_Matrix_set_INT32(m.inner(), value, field as _) });
    }

    fn get_field<T>(
        m: &Matrix<T>,
        field: GxB_Option_Field,
    ) -> i32 {
        let mut value = 0;
        grb!(unsafe { GrB_Matrix_get_INT32(m.inner(), &raw mut value, field as _) });
        value
    }

    /// Rebuild the `(row, col)` set from a CSR export, so a test can compare
    /// against the triples it put in without reimplementing CSR indexing.
    fn entries<V>(csr: &CsrExport<u64, u32, V>) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        for r in 0..csr.nrows as usize {
            for k in csr.row_ptr[r]..csr.row_ptr[r + 1] {
                out.push((r as u64, u64::from(csr.col_idx[k as usize])));
            }
        }
        out
    }

    /// What a successful export leaves: a valid, empty `0 x 0` matrix.
    fn taken<T>(m: &Matrix<T>) {
        assert_eq!((m.nrows(), m.ncols(), m.nvals()), (0, 0, 0));
    }

    #[test]
    fn pattern_round_trips_and_rows_are_sorted() {
        test_init::ensure_init();
        let mut m = Matrix::<bool>::new(5, 5);
        // Deliberately out of row-major order, and row 1 and row 4 left empty,
        // so the export has to sort within rows and emit repeated offsets for
        // the empty ones rather than skipping them.
        for (i, j) in [(3, 1), (0, 4), (0, 0), (2, 2), (3, 0), (0, 2)] {
            m.set(i, j, true);
        }
        m.wait();
        csr_ready(&mut m);

        let csr = m
            .take_csr::<u64, u32, ()>()
            .expect("a CSR-ready matrix exports");
        assert_eq!(csr.nrows, 5);
        assert_eq!(csr.ncols, 5);
        assert_eq!(csr.row_ptr.len(), 6);
        assert_eq!(csr.row_ptr, vec![0, 3, 3, 4, 6, 6]);
        assert_eq!(
            entries(&csr),
            vec![(0, 0), (0, 2), (0, 4), (2, 2), (3, 0), (3, 1)]
        );
        assert!(csr.vals.is_empty(), "pattern export carries no values");
        // The entries moved out: the matrix keeps none of them.
        taken(&m);
    }

    /// The offset width is the caller's choice: with the 32-bit hint the same
    /// entries come out as `u32` offsets, read bit for bit, and asking that
    /// matrix for `u64` offsets is refused rather than widened.
    #[test]
    fn exports_32_bit_offsets_when_asked() {
        test_init::ensure_init();
        let mut m = Matrix::<f64>::new(3, 3);
        for (i, j, w) in [(2, 0, 7.25), (0, 2, 2.5), (0, 1, 1.5)] {
            m.set(i, j, w);
        }
        m.wait();
        grb!(m.prepare_csr::<u32, u32, f64>());

        assert_eq!(
            m.take_csr::<u64, u32, f64>().err(),
            Some(CsrExportError::Array("the offsets"))
        );
        let csr = m
            .take_csr::<u32, u32, f64>()
            .expect("a CSR-ready matrix exports");
        assert_eq!(csr.row_ptr, vec![0u32, 2, 2, 3]);
        assert_eq!(csr.col_idx, vec![1, 2, 0]);
        assert_eq!(csr.vals, vec![1.5, 2.5, 7.25]);
    }

    #[test]
    fn empty_matrix_exports_an_all_zero_row_ptr() {
        test_init::ensure_init();
        let mut m = Matrix::<bool>::new(4, 7);
        csr_ready(&mut m);
        let csr = m
            .take_csr::<u64, u32, ()>()
            .expect("a CSR-ready matrix exports");
        assert_eq!(csr.nrows, 4);
        assert_eq!(csr.ncols, 7);
        assert_eq!(csr.row_ptr, vec![0, 0, 0, 0, 0]);
        assert!(csr.col_idx.is_empty());
        taken(&m);

        let mut m = Matrix::<f64>::new(2, 3);
        csr_ready(&mut m);
        let csr = m
            .take_csr::<u64, u32, f64>()
            .expect("an empty FP64 matrix exports");
        assert_eq!(csr.row_ptr, vec![0, 0, 0]);
        assert!(csr.col_idx.is_empty() && csr.vals.is_empty());
        taken(&m);
    }

    /// The matrix an export leaves is a real one: resized, filled and
    /// exported again, it behaves like a new matrix, and the first export's
    /// arrays are unaffected by any of it.
    #[test]
    fn a_taken_matrix_can_be_refilled_and_exported_again() {
        test_init::ensure_init();
        let mut m = Matrix::<f64>::new(3, 3);
        m.set(0, 1, 1.5);
        m.set(2, 2, 2.5);
        m.wait();
        csr_ready(&mut m);
        let first = m.take_csr::<u64, u32, f64>().expect("exports");
        taken(&m);

        m.resize(2, 4);
        m.set(1, 3, 9.0);
        m.set(1, 0, 8.0);
        m.wait();
        assert_eq!(m.get(1, 3), Some(9.0));
        csr_ready(&mut m);
        let second = m.take_csr::<u64, u32, f64>().expect("exports again");
        assert_eq!((second.nrows, second.ncols), (2, 4));
        assert_eq!(second.row_ptr, vec![0, 0, 2]);
        assert_eq!(second.col_idx, vec![0, 3]);
        assert_eq!(second.vals, vec![8.0, 9.0]);

        assert_eq!(first.row_ptr, vec![0, 1, 1, 2]);
        assert_eq!(first.col_idx, vec![1, 2]);
        assert_eq!(first.vals, vec![1.5, 2.5]);
    }

    /// The exported `Vec`s are ordinary ones: growing them reallocates
    /// through the global allocator, which a GraphBLAS-allocated array must
    /// survive as well as a Rust one.
    #[test]
    fn exported_vecs_grow_and_drop_like_any_vec() {
        test_init::ensure_init();
        let mut m = Matrix::<f64>::new(64, 64);
        for k in 0..64 {
            m.set(k, (k * 7) % 64, k as f64);
        }
        m.wait();
        csr_ready(&mut m);
        let mut csr = m.take_csr::<u64, u32, f64>().expect("exports");
        let vals = csr.vals.clone();
        for k in 0..10_000 {
            csr.vals.push(f64::from(k));
            csr.col_idx.push(k);
        }
        assert_eq!(&csr.vals[..64], &vals[..]);
        csr.vals.shrink_to_fit();
        csr.row_ptr.truncate(1);
        csr.row_ptr.shrink_to_fit();
        assert_eq!(csr.row_ptr, vec![0]);
    }

    #[test]
    fn exported_values_line_up_with_their_columns() {
        test_init::ensure_init();
        let mut m = Matrix::<f64>::new(3, 3);
        for (i, j, w) in [(0, 2, 2.5), (0, 1, 1.5), (2, 0, 7.25)] {
            m.set(i, j, w);
        }
        m.wait();
        csr_ready(&mut m);

        let csr = m
            .take_csr::<u64, u32, f64>()
            .expect("a CSR-ready matrix exports");
        assert_eq!(csr.row_ptr, vec![0, 2, 2, 3]);
        assert_eq!(csr.col_idx, vec![1, 2, 0]);
        assert_eq!(csr.vals, vec![1.5, 2.5, 7.25]);
    }

    /// An all-equal FP64 matrix is stored iso (one value for every entry). The
    /// export refuses to expand it; the caller expands it with `prepare_csr`
    /// for `f64` values.
    #[test]
    fn iso_values_are_refused_until_the_caller_expands_them() {
        test_init::ensure_init();
        let iso = || {
            let mut m = Matrix::<f64>::new(3, 3);
            for (i, j) in [(0, 0), (1, 2), (2, 1)] {
                m.set(i, j, 4.0);
            }
            m.wait();
            csr_ready(&mut m);
            m
        };
        let mut m = iso();
        assert_eq!(
            m.take_csr::<u64, u32, f64>().err(),
            Some(CsrExportError::IsoValues)
        );
        assert_eq!(m.nvals(), 3, "refused, the matrix keeps its entries");
        grb!(m.prepare_csr::<u64, u32, f64>());
        let csr = m
            .take_csr::<u64, u32, f64>()
            .expect("expanded, the matrix exports");
        assert_eq!(csr.vals, vec![4.0, 4.0, 4.0]);
        assert_eq!(csr.col_idx, vec![0, 2, 1]);

        // The pattern does not care, and its iso value stays with GraphBLAS.
        let mut m = iso();
        assert_eq!(entries(&m.take_csr::<u64, u32, ()>().unwrap()).len(), 3);
    }

    /// Each precondition, broken on its own, is refused, and the matrix is
    /// left as it was (loaded back, if it was unloaded): it keeps its entries
    /// and the setting the export refused. The export never repairs a setting.
    #[test]
    fn each_broken_precondition_is_refused_and_left_alone() {
        test_init::ensure_init();
        let fresh = || {
            let mut m = Matrix::<f64>::new(4, 4);
            m.set(0, 3, 1.0);
            m.set(2, 1, 2.0);
            m.wait();
            csr_ready(&mut m);
            m
        };
        let intact = |m: &Matrix<f64>| {
            assert_eq!(m.nvals(), 2, "a refused matrix keeps its entries");
            assert_eq!((m.nrows(), m.ncols()), (4, 4));
            assert_eq!(m.get(0, 3), Some(1.0));
            assert_eq!(m.get(2, 1), Some(2.0));
        };

        let mut m = fresh();
        set_field(
            &m,
            GxB_Option_Field::GxB_SPARSITY_CONTROL,
            GxB_HYPERSPARSE as i32,
        );
        assert_eq!(
            m.take_csr::<u64, u32, f64>().err(),
            Some(CsrExportError::NotCsr)
        );
        assert_eq!(m.sparsity_status(), "hypersparse");
        intact(&m);

        let col_major = GrB_Orientation::GrB_COLMAJOR as i32;
        let mut m = fresh();
        set_field(
            &m,
            GxB_Option_Field::GrB_STORAGE_ORIENTATION_HINT,
            col_major,
        );
        assert_eq!(
            m.take_csr::<u64, u32, f64>().err(),
            Some(CsrExportError::NotCsr)
        );
        assert_eq!(
            get_field(&m, GxB_Option_Field::GrB_STORAGE_ORIENTATION_HINT),
            col_major
        );
        intact(&m);

        let mut m = fresh();
        set_field(&m, GxB_Option_Field::GxB_OFFSET_INTEGER_HINT, 32);
        assert_eq!(
            m.take_csr::<u64, u32, f64>().err(),
            Some(CsrExportError::Array("the offsets"))
        );
        assert_eq!(get_field(&m, GxB_Option_Field::GxB_OFFSET_INTEGER_BITS), 32);
        intact(&m);

        let mut m = fresh();
        set_field(&m, GxB_Option_Field::GxB_COLINDEX_INTEGER_HINT, 64);
        assert_eq!(
            m.take_csr::<u64, u32, f64>().err(),
            Some(CsrExportError::Array("the column indices"))
        );
        assert_eq!(
            get_field(&m, GxB_Option_Field::GxB_COLINDEX_INTEGER_BITS),
            64
        );
        intact(&m);

        // A clone shares the handle and would see the matrix empty mid-export.
        let mut m = fresh();
        let clone = m.clone();
        assert_eq!(
            m.take_csr::<u64, u32, f64>().err(),
            Some(CsrExportError::SharedHandle)
        );
        drop(clone);
        intact(&m);
        assert!(
            m.take_csr::<u64, u32, f64>().is_ok(),
            "alone again, it exports"
        );
    }

    /// A clone dropped on another thread releases the handle: once that
    /// thread is joined the export sees it unshared and goes ahead. And a
    /// clone alive on another thread is refused like a local one.
    #[test]
    fn a_clone_on_another_thread_is_refused_until_it_is_dropped() {
        test_init::ensure_init();
        let mut m = Matrix::<f64>::new(2, 2);
        m.set(1, 0, 3.0);
        m.set(1, 1, 4.0);
        m.wait();
        csr_ready(&mut m);

        let clone = m.clone();
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let reader = std::thread::spawn(move || {
            assert_eq!(clone.get(1, 0), Some(3.0));
            rx.recv().expect("told to finish");
            drop(clone);
        });
        assert_eq!(
            m.take_csr::<u64, u32, f64>().err(),
            Some(CsrExportError::SharedHandle)
        );
        tx.send(()).expect("reader alive");
        reader.join().expect("reader finished");
        let csr = m.take_csr::<u64, u32, f64>().expect("alone, it exports");
        assert_eq!(csr.vals, vec![3.0, 4.0]);
    }

    /// A matrix wider than `2^31` columns cannot have 32-bit column indices:
    /// GraphBLAS ignores the hint, and the export says so instead of widening.
    #[test]
    fn a_column_hint_that_cannot_hold_is_refused() {
        test_init::ensure_init();
        let mut m = Matrix::<bool>::new(2, (1 << 31) + 1);
        m.set(1, 1 << 31, true);
        m.wait();
        csr_ready(&mut m);
        assert_eq!(
            m.take_csr::<u64, u32, ()>().err(),
            Some(CsrExportError::Array("the column indices"))
        );
        assert!(m.contains(1, 1 << 31));
    }

    /// Asking a `u64` matrix for FP64 values is refused, before the matrix is
    /// unloaded, rather than cast.
    #[test]
    fn a_wrong_type_value_export_is_refused_without_emptying_the_matrix() {
        test_init::ensure_init();
        let mut m = Matrix::<u64>::new(3, 3);
        m.set(0, 1, 7);
        m.set(2, 2, 9);
        m.wait();
        csr_ready(&mut m);
        assert_eq!(
            m.take_csr::<u64, u32, f64>().err(),
            Some(CsrExportError::NotFp64)
        );
        assert_eq!(m.nvals(), 2, "the matrix must survive the refused export");
        assert_eq!(m.get(2, 2), Some(9));
    }

    /// An array GraphBLAS does not own outright (here a read-only one, which
    /// Rust lent it) is not adopted: GraphBLAS copies it into the `Vec`, and it
    /// goes back to its vector, still the lender's.
    #[test]
    fn a_borrowed_array_is_copied_not_adopted() {
        use crate::graph::graphblas::{
            GrB_UINT64, GrB_Vector, GrB_Vector_free, GrB_Vector_new, GrB_Vector_nvals,
            GxB_Option_Field, GxB_Vector_load,
        };
        use std::{os::raw::c_void, ptr::null_mut};

        test_init::ensure_init();
        let lent: Vec<u64> = (10..20).collect();
        unsafe {
            let mut v: GrB_Vector = null_mut();
            grb!(GrB_Vector_new(&raw mut v, GrB_UINT64, 0));
            let mut x = lent.as_ptr().cast_mut().cast::<c_void>();
            grb!(GxB_Vector_load(
                v,
                &raw mut x,
                GrB_UINT64,
                lent.len() as u64,
                (lent.len() * size_of::<u64>()) as u64,
                GxB_Option_Field::GxB_IS_READONLY as _,
                null_mut(),
            ));
            let a = Unloaded::unload(v);
            assert!(a.fits::<u64>(7) && !a.fits::<u32>(7) && !a.fits::<u64>(11));
            let out = a.into_vec::<u64>(7);
            assert_eq!(out, (10..17).collect::<Vec<u64>>());
            assert_ne!(out.as_ptr(), lent.as_ptr(), "copied, not adopted");
            let mut nvals = 0;
            grb!(GrB_Vector_nvals(&raw mut nvals, v));
            assert_eq!(nvals, 10, "the vector holds its array again");
            grb!(GrB_Vector_free(&raw mut v));
        }
        assert_eq!(lent, (10..20).collect::<Vec<u64>>(), "the lender's array");
    }
}
