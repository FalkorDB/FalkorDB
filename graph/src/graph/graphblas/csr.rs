//! Row-major CSR export of a GraphBLAS matrix: the result and error types of
//! [`Matrix::to_csr_pattern`] and [`Matrix::to_csr_f64`], and the copy out of
//! GraphBLAS's own container arrays.
//!
//! The policy (which matrices may be exported, and the lock that keeps
//! everyone else off the handle meanwhile) stays with the wrapper in
//! `matrix.rs`; this module only reads a matrix the wrapper has already
//! checked.

#![allow(clippy::doc_markdown)]

use std::{
    mem::MaybeUninit,
    os::raw::{c_int, c_void},
    ptr::null_mut,
    slice,
};

use thiserror::Error;

#[cfg(doc)]
use super::matrix::Matrix;
use super::matrix::grb;
use super::settings::{IntegerWidth, Orientation, Sparsity};
use super::{
    GrB_FP64, GrB_INT32, GrB_Matrix, GrB_Type, GrB_UINT32, GrB_UINT64, GrB_Vector, GxB_Container,
    GxB_Container_free, GxB_Container_new, GxB_SPARSE, GxB_Vector_load, GxB_Vector_unload,
    GxB_load_Matrix_from_Container, GxB_unload_Matrix_into_Container,
};

/// Row-major CSR export of a GraphBLAS matrix: what
/// [`Matrix::to_csr_pattern`] and [`Matrix::to_csr_f64`] hand back.
///
/// Row `r` owns the entries `row_ptr[r] .. row_ptr[r + 1]`: their column
/// indices are that slice of `col_idx`, ascending and duplicate-free, and —
/// when values were asked for — their values are the same slice of `vals`.
/// `row_ptr` has `nrows + 1` entries and ends at the entry count. The widths
/// are GraphBLAS's own, `P` for the offsets and `I` for the column indices
/// (`u32` or `u64` each), which the caller asks the matrix for before
/// exporting.
pub struct CsrExport<P = u64, I = u32> {
    pub nrows: u64,
    pub ncols: u64,
    /// `nrows + 1` offsets into `col_idx` (and `vals`, when non-empty).
    pub row_ptr: Vec<P>,
    /// One column index per entry, ascending within each row.
    pub col_idx: Vec<I>,
    /// One value per entry, or empty when only the pattern was exported.
    pub vals: Vec<f64>,
}

/// Why [`Matrix::to_csr_pattern`] or [`Matrix::to_csr_f64`] refused a
/// matrix. The export never converts the storage or casts an array: the
/// caller sets the matrix up (see [`Matrix::to_csr_f64`]) and anything else is
/// refused. Every check except [`Self::Unexpected`] runs before the matrix is
/// unloaded, and even that one reloads it first, so a refused matrix is left
/// as it was.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum CsrExportError {
    /// Another `Matrix` shares this handle, and would see the matrix empty
    /// while it is unloaded.
    #[error("another handle shares this matrix")]
    SharedHandle,
    /// Values were asked for as `FP64`, and the matrix holds another type.
    #[error("the matrix does not hold FP64 values")]
    NotFp64,
    /// Stored in another format than sparse.
    #[error("the matrix is stored {0:?}, not sparse")]
    NotSparse(Sparsity),
    /// Stored by column.
    #[error("the matrix is stored column-major")]
    NotRowMajor,
    /// The offsets are not the width the export was asked for. A 32-bit hint
    /// is ignored, rightly, on a matrix with `2^32` entries or more.
    #[error("the offsets are {0:?}, not the requested width")]
    OffsetWidth(IntegerWidth),
    /// The column indices are not the width the export was asked for. A
    /// 32-bit hint is ignored, rightly, on a matrix with more than `2^31`
    /// columns.
    #[error("the column indices are {0:?}, not the requested width")]
    ColIndexWidth(IntegerWidth),
    /// Values were asked for and the matrix stores one value for every entry;
    /// `set_iso(false)` expands it.
    #[error("the values are stored iso")]
    IsoValues,
    /// GraphBLAS unloaded something the checks above rule out (named here):
    /// a GraphBLAS invariant broke, not a caller mistake.
    #[error("GraphBLAS unloaded {0} in a form the checks ruled out")]
    Unexpected(&'static str),
}

impl CsrExportError {
    /// A fixed description, for error types that carry a `&'static str`.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::SharedHandle => "CSR export: another handle shares the matrix",
            Self::NotFp64 => "CSR export: the matrix does not hold FP64 values",
            Self::NotSparse(_) => "CSR export: the matrix is not stored sparse",
            Self::NotRowMajor => "CSR export: the matrix is stored column-major",
            Self::OffsetWidth(_) => "CSR export: the offsets are not the requested width",
            Self::ColIndexWidth(_) => "CSR export: the column indices are not the requested width",
            Self::IsoValues => "CSR export: the values are stored iso",
            Self::Unexpected(_) => "CSR export: GraphBLAS unloaded an unexpected form",
        }
    }
}

/// A Rust type a GraphBLAS array is read as bit for bit, with no conversion.
/// Public only because it bounds the CSR export's type parameters.
pub trait GrbElem: Copy {
    /// Whether an array GraphBLAS reports as `ty` holds exactly this type's bits.
    fn holds(ty: GrB_Type) -> bool;
}

impl GrbElem for u64 {
    fn holds(ty: GrB_Type) -> bool {
        unsafe { ty == GrB_UINT64 }
    }
}

impl GrbElem for u32 {
    /// GraphBLAS hands its 32-bit index arrays out as `INT32`. It keeps an index
    /// array 32-bit only while the dimension is at most `2^31`, so every value
    /// is below `2^31` and has the same bits as a `u32`.
    fn holds(ty: GrB_Type) -> bool {
        unsafe { ty == GrB_UINT32 || ty == GrB_INT32 }
    }
}

impl GrbElem for f64 {
    fn holds(ty: GrB_Type) -> bool {
        unsafe { ty == GrB_FP64 }
    }
}

/// Copy the first `len` entries of a container vector (`p`, `i` or `x`) out as
/// `T`, refusing rather than converting when the stored type is not `T`. The
/// array is handed straight back to the vector either way, leaving the
/// container exactly as it was found. `what` names the array in the error.
unsafe fn container_array<T: GrbElem>(
    v: GrB_Vector,
    len: usize,
    what: &'static str,
) -> Result<Vec<T>, CsrExportError> {
    if len == 0 {
        return Ok(Vec::new());
    }
    unsafe {
        let mut x: *mut c_void = null_mut();
        let mut ty: GrB_Type = null_mut();
        let mut n: u64 = 0;
        let mut x_size: u64 = 0;
        let mut handling: c_int = 0;
        grb!(GxB_Vector_unload(
            v,
            &raw mut x,
            &raw mut ty,
            &raw mut n,
            &raw mut x_size,
            &raw mut handling,
            null_mut(),
        ));

        let out = if T::holds(ty) && n as usize >= len {
            Ok(slice::from_raw_parts(x.cast::<T>(), len).to_vec())
        } else {
            Err(CsrExportError::Unexpected(what))
        };

        grb!(GxB_Vector_load(
            v,
            &raw mut x,
            ty,
            n,
            x_size,
            handling,
            null_mut()
        ));
        out
    }
}

/// Copy `m` out as CSR: unload it into a `GxB_Container`, copy its `p`, `i`
/// and (`with_values`) `x` arrays, and load it straight back, refused or not.
///
/// # Safety
/// `m` is a live matrix nobody else can reach until this returns, already
/// materialized and checked to be sparse, row-major, with offsets as wide as
/// `P`, column indices as wide as `I` and, `with_values`, non-iso `FP64`
/// values: the wrapper's `export_csr` does all of that first.
pub(super) unsafe fn copy_out_csr<P, I>(
    m: GrB_Matrix,
    with_values: bool,
) -> Result<CsrExport<P, I>, CsrExportError>
where
    P: GrbElem + Into<u64>,
    I: GrbElem + Into<u64>,
{
    unsafe {
        let mut container: MaybeUninit<GxB_Container> = MaybeUninit::uninit();
        grb!(GxB_Container_new(container.as_mut_ptr()));
        let container = container.assume_init();
        grb!(GxB_unload_Matrix_into_Container(m, container, null_mut()));

        let nrows = (*container).nrows;
        let ncols = (*container).ncols;
        let nvals = (*container).nvals as usize;
        let arrays = if (*container).format != GxB_SPARSE as i32
            || (*container).orientation != Orientation::RowMajor as i32
            || (*container).jumbled
        {
            Err(CsrExportError::Unexpected(
                "a container that is not sorted row-major CSR",
            ))
        } else {
            container_array::<P>((*container).p, nrows as usize + 1, "the offsets").and_then(
                |row_ptr| {
                    let col_idx =
                        container_array::<I>((*container).i, nvals, "the column indices")?;
                    let vals = if with_values {
                        container_array::<f64>((*container).x, nvals, "the values")?
                    } else {
                        Vec::new()
                    };
                    Ok((row_ptr, col_idx, vals))
                },
            )
        };

        // Hand everything back, refused or not: the container still owns
        // the same arrays (`container_array` reloads what it unloads), so
        // this restores `m` to exactly what it held on entry.
        grb!(GxB_load_Matrix_from_Container(m, container, null_mut()));
        let mut c = container;
        grb!(GxB_Container_free(&raw mut c));

        let (row_ptr, col_idx, vals) = arrays?;
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
    use super::{CsrExport, CsrExportError};
    use crate::graph::graphblas::matrix::Matrix;
    use crate::graph::graphblas::settings::{IntegerWidth, Orientation, Sparsity};
    use crate::graph::graphblas::test_init;

    /// What a caller does before exporting: store `m` as plain CSR with the
    /// export's widths. The export checks all of it and sets none of it.
    fn csr_ready<T>(m: &mut Matrix<T>) {
        m.set_sparsity(Sparsity::Sparse);
        m.set_orientation(Orientation::RowMajor);
        m.set_offset_width(IntegerWidth::Bits64);
        m.set_col_index_width(IntegerWidth::Bits32);
    }

    /// Rebuild the `(row, col)` set from a CSR export, so a test can compare
    /// against the triples it put in without reimplementing CSR indexing.
    fn entries(csr: &CsrExport) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        for r in 0..csr.nrows as usize {
            for k in csr.row_ptr[r]..csr.row_ptr[r + 1] {
                out.push((r as u64, u64::from(csr.col_idx[k as usize])));
            }
        }
        out
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
            .to_csr_pattern::<u64, u32>()
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

        // The matrix itself must be untouched: the export unloads it into a
        // container and loads it straight back. Drop the reload in `export_csr`
        // and this is what fails.
        assert_eq!(m.nvals(), 6);
        assert!(m.contains(0, 4) && m.contains(3, 1) && !m.contains(1, 1));
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
        csr_ready(&mut m);
        m.set_offset_width(IntegerWidth::Bits32);

        let csr = m
            .to_csr_f64::<u32, u32>()
            .expect("a CSR-ready matrix exports");
        assert_eq!(csr.row_ptr, vec![0u32, 2, 2, 3]);
        assert_eq!(csr.col_idx, vec![1, 2, 0]);
        assert_eq!(csr.vals, vec![1.5, 2.5, 7.25]);
        assert_eq!(
            m.to_csr_f64::<u64, u32>().err(),
            Some(CsrExportError::OffsetWidth(IntegerWidth::Bits32))
        );
    }

    #[test]
    fn empty_matrix_exports_an_all_zero_row_ptr() {
        test_init::ensure_init();
        let mut m = Matrix::<bool>::new(4, 7);
        csr_ready(&mut m);
        let csr = m
            .to_csr_pattern::<u64, u32>()
            .expect("a CSR-ready matrix exports");
        assert_eq!(csr.nrows, 4);
        assert_eq!(csr.ncols, 7);
        assert_eq!(csr.row_ptr, vec![0, 0, 0, 0, 0]);
        assert!(csr.col_idx.is_empty());
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
            .to_csr_f64::<u64, u32>()
            .expect("a CSR-ready matrix exports");
        assert_eq!(csr.row_ptr, vec![0, 2, 2, 3]);
        assert_eq!(csr.col_idx, vec![1, 2, 0]);
        assert_eq!(csr.vals, vec![1.5, 2.5, 7.25]);
    }

    /// An all-equal FP64 matrix is stored iso (one value for every entry). The
    /// export refuses to expand it; the caller expands it with `set_iso(false)`.
    #[test]
    fn iso_values_are_refused_until_the_caller_expands_them() {
        test_init::ensure_init();
        let mut m = Matrix::<f64>::new(3, 3);
        for (i, j) in [(0, 0), (1, 2), (2, 1)] {
            m.set(i, j, 4.0);
        }
        m.wait();
        csr_ready(&mut m);
        assert_eq!(
            m.to_csr_f64::<u64, u32>().err(),
            Some(CsrExportError::IsoValues)
        );
        // The pattern does not care.
        assert_eq!(entries(&m.to_csr_pattern::<u64, u32>().unwrap()).len(), 3);

        m.set_iso(false);
        let csr = m
            .to_csr_f64::<u64, u32>()
            .expect("expanded, the matrix exports");
        assert_eq!(csr.vals, vec![4.0, 4.0, 4.0]);
        assert_eq!(csr.col_idx, vec![0, 2, 1]);
    }

    /// Each precondition, broken on its own, is refused with its own error and
    /// before anything is unloaded: the matrix keeps its entries and the
    /// setting the export refused. The export never repairs a setting.
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
            assert_eq!(m.get(0, 3), Some(1.0));
            assert_eq!(m.get(2, 1), Some(2.0));
        };

        let mut m = fresh();
        m.set_sparsity(Sparsity::Hypersparse);
        assert_eq!(
            m.to_csr_f64::<u64, u32>().err(),
            Some(CsrExportError::NotSparse(Sparsity::Hypersparse))
        );
        assert_eq!(m.sparsity(), Sparsity::Hypersparse);
        intact(&m);

        let mut m = fresh();
        m.set_orientation(Orientation::ColMajor);
        assert_eq!(
            m.to_csr_f64::<u64, u32>().err(),
            Some(CsrExportError::NotRowMajor)
        );
        assert_eq!(m.orientation(), Orientation::ColMajor);
        intact(&m);

        let mut m = fresh();
        m.set_offset_width(IntegerWidth::Bits32);
        assert_eq!(
            m.to_csr_f64::<u64, u32>().err(),
            Some(CsrExportError::OffsetWidth(IntegerWidth::Bits32))
        );
        assert_eq!(m.offset_width(), IntegerWidth::Bits32);
        intact(&m);

        let mut m = fresh();
        m.set_col_index_width(IntegerWidth::Bits64);
        assert_eq!(
            m.to_csr_f64::<u64, u32>().err(),
            Some(CsrExportError::ColIndexWidth(IntegerWidth::Bits64))
        );
        assert_eq!(m.col_index_width(), IntegerWidth::Bits64);
        intact(&m);

        // A clone shares the handle and would see the matrix empty mid-export.
        let mut m = fresh();
        let clone = m.clone();
        assert_eq!(
            m.to_csr_f64::<u64, u32>().err(),
            Some(CsrExportError::SharedHandle)
        );
        drop(clone);
        intact(&m);
        assert!(
            m.to_csr_f64::<u64, u32>().is_ok(),
            "alone again, it exports"
        );
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
            m.to_csr_pattern::<u64, u32>().err(),
            Some(CsrExportError::ColIndexWidth(IntegerWidth::Bits64))
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
            m.to_csr_f64::<u64, u32>().err(),
            Some(CsrExportError::NotFp64)
        );
        assert_eq!(m.nvals(), 2, "the matrix must survive the refused export");
        assert_eq!(m.get(2, 2), Some(9));
    }
}
