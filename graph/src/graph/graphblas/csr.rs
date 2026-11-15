//! Row-major CSR export of a GraphBLAS matrix, copied into Rust `Vec`s: the
//! result type of [`Matrix::into_csr`] and [`Matrix::into_csr_pattern`], and
//! the copy itself.
//!
//! GraphBLAS's own `GrB_Matrix_export` writes the arrays Rust allocated. It
//! finishes the matrix's pending work first, so every row comes out sorted; it
//! converts hypersparse, bitmap and full storage in a scratch copy of its own;
//! it widens 32-bit indices to 64; and it expands iso values. A matrix stored
//! by column is first copied by row (`Matrix::copy_csr`), since the export's
//! own transpose can leave rows unsorted. So a matrix needs no setting up
//! before it is exported, and nothing GraphBLAS allocated is ever freed by Rust.

use std::os::raw::c_int;

use super::length;
#[cfg(doc)]
use super::matrix::Matrix;
use super::matrix::grb;
use super::{GrB_Format, GrB_Index, GrB_Info, GrB_Matrix, GrB_Matrix_exportSize};

/// Row-major CSR export of a GraphBLAS matrix.
///
/// Row `r` owns the entries `row_offsets[r] .. row_offsets[r + 1]`: their column
/// indices are that slice of `column_indices`, ascending and duplicate-free, and —
/// unless `V` is `()`, the pattern alone — their values are the same slice of
/// `values`. `row_offsets` has one entry per row plus one, and ends at the entry
/// count.
pub struct CsrExport<V> {
    /// One offset into `column_indices` (and `values`, when non-empty) per
    /// row, plus one.
    pub row_offsets: Vec<u64>,
    /// One column index per entry, ascending within each row.
    pub column_indices: Vec<u64>,
    /// One value per entry; empty for the pattern export (`V = ()`).
    pub values: Vec<V>,
}

impl<V> CsrExport<V> {
    /// The matrix's row count.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.row_offsets.len() - 1
    }
}

/// A typed `GrB_Matrix_export_*`: its `Ax` must be of the matrix's own type.
pub(super) type ExportFn<V> = unsafe extern "C" fn(
    *mut GrB_Index,
    *mut GrB_Index,
    *mut V,
    *mut GrB_Index,
    *mut GrB_Index,
    *mut GrB_Index,
    c_int,
    GrB_Matrix,
) -> GrB_Info;

/// Copy `matrix` out as row-major CSR through `export`.
///
/// Rows come out sorted when `matrix` is stored by row: see `Matrix::copy_csr`.
/// An `export` of another type than the matrix's is refused by GraphBLAS
/// (`GrB_DOMAIN_MISMATCH`), which panics here.
///
/// # Safety
/// `matrix` is live and nobody else uses it until this returns: the export
/// finishes its pending work, a mutation.
pub(super) unsafe fn copy_out_csr<V: Copy + Default>(
    matrix: GrB_Matrix,
    export: ExportFn<V>,
) -> CsrExport<V> {
    let format = GrB_Format::GrB_CSR_FORMAT as c_int;
    let (mut offset_count, mut index_count, mut value_count) = (0, 0, 0);
    // SAFETY: `matrix` is live (the caller's contract); the counts are locals.
    grb!(unsafe {
        GrB_Matrix_exportSize(
            &raw mut offset_count,
            &raw mut index_count,
            &raw mut value_count,
            format,
            matrix,
        )
    });
    let mut row_offsets = vec![0; length(offset_count)];
    let mut column_indices = vec![0; length(index_count)];
    let mut values = vec![V::default(); length(value_count)];
    // SAFETY: each array holds the count `GrB_Matrix_exportSize` asked for,
    // which the export checks against before writing; `export` matches the
    // matrix's type (the caller's contract).
    grb!(unsafe {
        export(
            row_offsets.as_mut_ptr(),
            column_indices.as_mut_ptr(),
            values.as_mut_ptr(),
            &raw mut offset_count,
            &raw mut index_count,
            &raw mut value_count,
            format,
            matrix,
        )
    });
    row_offsets.truncate(length(offset_count));
    column_indices.truncate(length(index_count));
    values.truncate(length(value_count));
    CsrExport {
        row_offsets,
        column_indices,
        values,
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use std::ptr::null_mut;

    use super::CsrExport;
    use crate::graph::graphblas::matrix::{Matrix, grb};
    use crate::graph::graphblas::{
        GrB_Matrix_get_INT32, GrB_Matrix_set_INT32, GrB_Orientation, GxB_BITMAP, GxB_Context,
        GxB_Context_Field, GxB_Context_disengage, GxB_Context_engage, GxB_Context_free,
        GxB_Context_get_INT32, GxB_Context_new, GxB_Context_set_INT32, GxB_FULL, GxB_HYPERSPARSE,
        GxB_Option_Field, test_init,
    };

    /// Run `body` with this thread's GraphBLAS calls on `threads` threads,
    /// however many cores the machine has: a `GxB_Context` engaged on this
    /// thread only, so other tests are not affected.
    fn on_threads<R>(
        threads: i32,
        body: impl FnOnce() -> R,
    ) -> R {
        let nthreads = GxB_Context_Field::GxB_CONTEXT_NTHREADS as _;
        let mut context: GxB_Context = null_mut();
        let mut set = 0;
        grb!(unsafe { GxB_Context_new(&raw mut context) });
        grb!(unsafe { GxB_Context_set_INT32(context, nthreads, threads) });
        grb!(unsafe { GxB_Context_get_INT32(context, nthreads, &raw mut set) });
        assert_eq!(set, threads, "the context took the thread count");
        grb!(unsafe { GxB_Context_engage(context) });
        let result = body();
        grb!(unsafe { GxB_Context_disengage(context) });
        grb!(unsafe { GxB_Context_free(&raw mut context) });
        result
    }

    /// Change how GraphBLAS stores `m`, behind the export's back.
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

    fn arrays(csr: CsrExport<f64>) -> (Vec<u64>, Vec<u64>, Vec<f64>) {
        (csr.row_offsets, csr.column_indices, csr.values)
    }

    #[test]
    fn pattern_round_trips_and_rows_are_sorted() {
        test_init::ensure_init();
        let mut m = Matrix::<bool>::new(5, 5);
        // Deliberately out of row-major order and left pending, with row 1 and
        // row 4 empty, so the export has to finish the writes, sort within
        // rows, and emit repeated offsets for the empty ones.
        for (i, j) in [(3, 1), (0, 4), (0, 0), (2, 2), (3, 0), (0, 2)] {
            m.set(i, j, true);
        }
        let csr = m.into_csr_pattern();
        assert_eq!(csr.row_count(), 5);
        assert_eq!(csr.row_offsets, vec![0, 3, 3, 4, 6, 6]);
        assert_eq!(csr.column_indices, vec![0, 2, 4, 2, 0, 1]);
        assert!(
            csr.values.is_empty(),
            "the pattern export carries no values"
        );
    }

    /// Values follow their entries; an entry removed before the export is gone
    /// and every other one is exported as it was.
    #[test]
    fn values_follow_their_entries() {
        test_init::ensure_init();
        let mut m = Matrix::<f64>::new(4, 4);
        for (i, j, w) in [(3, 0, 4.0), (0, 3, 2.0), (1, 2, 3.0), (0, 1, 1.0)] {
            m.set(i, j, w);
        }
        m.wait();
        m.remove(0, 1);
        assert_eq!(
            arrays(m.into_csr()),
            (vec![0, 1, 2, 2, 3], vec![3, 2, 0], vec![2.0, 3.0, 4.0])
        );
    }

    /// An all-equal matrix is stored iso, one value for every entry; the export
    /// hands every entry its own copy.
    #[test]
    fn iso_values_are_expanded() {
        test_init::ensure_init();
        let mut m = Matrix::<f64>::new(3, 3);
        for (i, j) in [(0, 0), (1, 2), (2, 1)] {
            m.set(i, j, 4.0);
        }
        m.wait();
        assert_eq!(get_field(&m, GxB_Option_Field::GxB_ISO), 1, "stored iso");
        assert_eq!(
            arrays(m.into_csr()),
            (vec![0, 1, 2, 3], vec![0, 2, 1], vec![4.0; 3])
        );
    }

    /// However GraphBLAS stores the matrix (hypersparse, bitmap, full, by
    /// column, or with 64-bit offsets or column indices rather than its default
    /// 32), the export is the same 64-bit CSR. Each storage is set after the
    /// last write and checked before the export, so the export really meets it.
    #[test]
    fn any_storage_and_width_exports_the_same() {
        use GxB_Option_Field::{
            GrB_STORAGE_ORIENTATION_HINT, GxB_COLINDEX_INTEGER_BITS, GxB_COLINDEX_INTEGER_HINT,
            GxB_OFFSET_INTEGER_BITS, GxB_OFFSET_INTEGER_HINT, GxB_SPARSITY_CONTROL,
            GxB_SPARSITY_STATUS,
        };
        test_init::ensure_init();
        let col_major = GrB_Orientation::GrB_COLMAJOR as i32;
        // (setting, value, the field that reports the storage, what it reports)
        let storages = [
            (
                GxB_SPARSITY_CONTROL,
                GxB_HYPERSPARSE as i32,
                GxB_SPARSITY_STATUS,
                GxB_HYPERSPARSE as i32,
            ),
            (
                GxB_SPARSITY_CONTROL,
                GxB_BITMAP as i32,
                GxB_SPARSITY_STATUS,
                GxB_BITMAP as i32,
            ),
            (
                GxB_SPARSITY_CONTROL,
                GxB_FULL as i32,
                GxB_SPARSITY_STATUS,
                GxB_FULL as i32,
            ),
            (
                GrB_STORAGE_ORIENTATION_HINT,
                col_major,
                GrB_STORAGE_ORIENTATION_HINT,
                col_major,
            ),
            (GxB_OFFSET_INTEGER_HINT, 64, GxB_OFFSET_INTEGER_BITS, 64),
            (GxB_COLINDEX_INTEGER_HINT, 64, GxB_COLINDEX_INTEGER_BITS, 64),
        ];
        // Full, so that every sparsity, full included, can hold it.
        let filled = || {
            let mut m = Matrix::<f64>::new(2, 2);
            for (i, j, w) in [(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)] {
                m.set(i, j, w);
            }
            m.wait();
            m
        };
        let expected = (vec![0, 2, 4], vec![0, 1, 0, 1], vec![1.0, 2.0, 3.0, 4.0]);
        let default = filled();
        assert_eq!(get_field(&default, GxB_OFFSET_INTEGER_BITS), 32);
        assert_eq!(get_field(&default, GxB_COLINDEX_INTEGER_BITS), 32);
        assert_eq!(arrays(default.into_csr()), expected);
        for (setting, value, status, stored) in storages {
            let m = filled();
            set_field(&m, setting, value);
            assert_eq!(get_field(&m, status), stored, "{setting:?} took");
            assert_eq!(arrays(m.into_csr()), expected, "{setting:?}");
        }
    }

    /// A matrix stored by column comes out with every row sorted. Random
    /// entries, enough of them, and 8 GraphBLAS threads whatever the machine,
    /// so that the transpose runs on several threads, which can leave rows
    /// unsorted: exporting such a matrix directly gave 78,583 out-of-order rows
    /// at this size, and none at 2 threads.
    #[test]
    fn a_matrix_stored_by_column_exports_sorted_rows() {
        test_init::ensure_init();
        on_threads(
            8,
            a_matrix_stored_by_column_exports_sorted_rows_on_this_thread,
        );
    }

    fn a_matrix_stored_by_column_exports_sorted_rows_on_this_thread() {
        const N: u64 = 200_000;
        let filled = |by_column: bool| {
            let mut m = Matrix::<f64>::new(N, N);
            if by_column {
                let col_major = GrB_Orientation::GrB_COLMAJOR as i32;
                set_field(
                    &m,
                    GxB_Option_Field::GrB_STORAGE_ORIENTATION_HINT,
                    col_major,
                );
            }
            let mut rng = SmallRng::seed_from_u64(1);
            for k in 0..2 * N {
                m.set(rng.random_range(0..N), rng.random_range(0..N), k as f64);
            }
            m.wait();
            m
        };
        let by_column = filled(true);
        assert_eq!(
            get_field(&by_column, GxB_Option_Field::GrB_STORAGE_ORIENTATION_HINT),
            GrB_Orientation::GrB_COLMAJOR as i32,
            "stored by column"
        );
        let exported = by_column.into_csr();
        let unsorted = exported
            .row_offsets
            .windows(2)
            .filter(|row| {
                let row = &exported.column_indices[row[0] as usize..row[1] as usize];
                row.windows(2).any(|pair| pair[0] >= pair[1])
            })
            .count();
        assert_eq!(unsorted, 0, "rows out of order");
        assert!(
            arrays(exported) == arrays(filled(false).into_csr()),
            "the same CSR as the matrix stored by row"
        );
    }

    /// Past `2^32` columns a column index needs all 64 bits.
    #[test]
    fn column_indices_past_two_to_the_32_keep_every_bit() {
        test_init::ensure_init();
        let wide = (1u64 << 32) + 5;
        let mut m = Matrix::<f64>::new(2, wide + 1);
        m.set(1, wide, 1.5);
        m.set(0, 3, 0.5);
        assert_eq!(
            arrays(m.into_csr()),
            (vec![0, 1, 2], vec![3, wide], vec![0.5, 1.5])
        );
    }

    /// An empty matrix still has one offset per row, plus one; a `0 x 0` one
    /// has just the one.
    #[test]
    fn an_empty_matrix_exports_its_rows() {
        test_init::ensure_init();
        assert_eq!(
            arrays(Matrix::<f64>::new(3, 3).into_csr()),
            (vec![0; 4], vec![], vec![])
        );
        let csr = Matrix::<bool>::new(0, 0).into_csr_pattern();
        assert_eq!((csr.row_count(), csr.row_offsets), (0, vec![0]));
    }

    /// The export reads the matrix and frees only its own handle: a clone that
    /// shares the GraphBLAS matrix keeps every entry.
    #[test]
    fn a_clone_keeps_its_entries() {
        test_init::ensure_init();
        let mut m = Matrix::<f64>::new(3, 3);
        m.set(0, 2, 2.5);
        m.set(2, 1, 7.0);
        let clone = m.clone();
        assert_eq!(arrays(m.into_csr()).2, vec![2.5, 7.0]);
        assert_eq!(clone.nvals(), 2);
        assert_eq!(clone.get(0, 2), Some(2.5));
        assert_eq!(clone.get(2, 1), Some(7.0));
    }
}
