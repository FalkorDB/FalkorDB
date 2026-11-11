//! Row-major CSR export of a GraphBLAS matrix: the result and error types of
//! [`Matrix::take_csr`], and the handover of GraphBLAS's own container arrays
//! to Rust.
//!
//! The policy (which handles may be exported, and the lock that keeps
//! everyone else off the handle meanwhile) stays with the wrapper in
//! `matrix.rs`; this module checks the storage it unloads and hands it over.

#![allow(clippy::doc_markdown)]

use std::{
    fmt,
    os::raw::{c_int, c_void},
    ptr::null_mut,
};

use thiserror::Error;

use super::length;
#[cfg(doc)]
use super::matrix::Matrix;
use super::matrix::grb;
use super::{
    GrB_Desc_Value, GrB_FP64, GrB_INT32, GrB_INT64, GrB_Index, GrB_Info, GrB_Matrix,
    GrB_Orientation, GrB_Type, GrB_UINT32, GrB_UINT64, GrB_Vector, GxB_Container,
    GxB_Container_free, GxB_Container_new, GxB_SPARSE, GxB_Vector_load, GxB_Vector_unload,
    GxB_load_Matrix_from_Container, GxB_unload_Matrix_into_Container,
};

/// Row-major CSR export of a GraphBLAS matrix: what [`Matrix::take_csr`] hands
/// back.
///
/// Row `r` owns the entries `row_offsets[r] .. row_offsets[r + 1]`: their column
/// indices are that slice of `column_indices`, ascending and duplicate-free, and —
/// unless `V` is `()`, the pattern alone — their values are the same slice of
/// `values`.
/// `row_offsets` has one entry per row plus one, and ends at the entry count.
/// The widths are GraphBLAS's own, `P` for the offsets and `I` for the column
/// indices (`u32` or `u64` each), which the caller asks the matrix for before
/// exporting.
pub struct CsrExport<P, I, V> {
    /// One offset into `column_indices` (and `values`, when non-empty) per
    /// row, plus one.
    pub row_offsets: Vec<P>,
    /// One column index per entry, ascending within each row.
    pub column_indices: Vec<I>,
    /// One value per entry; empty for the pattern export (`V = ()`).
    pub values: Vec<V>,
}

impl<P, I, V> CsrExport<P, I, V> {
    /// The matrix's row count.
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.row_offsets.len() - 1
    }
}

/// One array of a CSR export, as [`CsrExportError::Array`] names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsrArray {
    RowOffsets,
    ColumnIndices,
    Values,
}

impl fmt::Display for CsrArray {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.write_str(match self {
            Self::RowOffsets => "the row offsets",
            Self::ColumnIndices => "the column indices",
            Self::Values => "the values",
        })
    }
}

/// Why [`Matrix::take_csr`] (or [`Matrix::prepare_csr`]) refused a matrix. The export never converts the
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
    /// Values were asked for, and the matrix holds another type than `V`.
    #[error("the matrix does not hold the requested value type")]
    WrongValueType,
    /// The matrix is not stored as sorted row-major sparse, i.e. CSR: it is
    /// hypersparse, bitmap or full, or stored by column. (Never jumbled: the
    /// export materializes first, which sorts every row, and asserts it.)
    #[error("the matrix is not stored as sorted row-major sparse (CSR)")]
    NotCsr,
    /// Values were asked for and the matrix stores one value for every entry.
    #[error("the values are stored iso")]
    IsoValues,
    /// An array is not of the type the export was asked for, or is shorter
    /// than the matrix needs. For the offsets or the column indices that is a
    /// width: a 32-bit hint is ignored, rightly, on a matrix with `2^32`
    /// entries or more (offsets) or more than `2^31` columns (column indices).
    #[error("{0} are not of the requested type")]
    Array(CsrArray),
    /// GraphBLAS does not own the array outright (it is read-only, or was lent
    /// to GraphBLAS), so it cannot be handed over.
    #[error("{0} are not GraphBLAS's own to hand over")]
    NotOwned(CsrArray),
    /// [`Matrix::prepare_csr`] only: GraphBLAS refused one of the storage
    /// settings, with this `GrB_Info` (a conversion allocates in proportion to
    /// the matrix, so it can run out of memory).
    #[error("GraphBLAS refused a storage setting: {0:?}")]
    Setting(GrB_Info),
}

/// Only this module implements [`GrbElem`], and so [`CsrIndex`] and
/// [`CsrValue`]: the export relies on every implementation's answers.
mod sealed {
    pub trait Sealed {}
    impl Sealed for u64 {}
    impl Sealed for u32 {}
    impl Sealed for f64 {}
    impl Sealed for () {}
}

/// A Rust type a GraphBLAS array is read as bit for bit, with no conversion:
/// `u32`, `u64`, `f64`, and `()` for no array. Public only because it bounds
/// the CSR export's type parameters, and sealed: the export adopts GraphBLAS's
/// arrays as `Vec<Self>` on the strength of [`Self::holds`], which is `true`
/// only for a GraphBLAS type whose values have exactly `size_of::<Self>()`
/// bytes and every bit pattern of which is a valid `Self`.
pub trait GrbElem: Copy + sealed::Sealed {
    /// Whether an array GraphBLAS reports as `element_type` holds exactly this
    /// type's bits.
    fn holds(element_type: GrB_Type) -> bool;
}

impl GrbElem for u64 {
    /// GraphBLAS hands its 64-bit index arrays out as `INT64`. An index is at
    /// most `GrB_INDEX_MAX` (`2^60 - 1`), so it has the same bits as a `u64`.
    fn holds(element_type: GrB_Type) -> bool {
        // SAFETY: reads of the type globals `GrB_init` set.
        unsafe { element_type == GrB_UINT64 || element_type == GrB_INT64 }
    }
}

impl GrbElem for u32 {
    /// GraphBLAS hands its 32-bit index arrays out as `INT32`. It keeps an index
    /// array 32-bit only while the dimension is at most `2^31`, so every value
    /// is below `2^31` and has the same bits as a `u32`.
    fn holds(element_type: GrB_Type) -> bool {
        // SAFETY: reads of the type globals `GrB_init` set.
        unsafe { element_type == GrB_UINT32 || element_type == GrB_INT32 }
    }
}

impl GrbElem for f64 {
    fn holds(element_type: GrB_Type) -> bool {
        // SAFETY: a read of the type global `GrB_init` set.
        unsafe { element_type == GrB_FP64 }
    }
}

/// No GraphBLAS array is read as `()`: it is the value type of the pattern
/// export, which leaves the `x` array alone.
impl GrbElem for () {
    fn holds(_: GrB_Type) -> bool {
        false
    }
}

/// The integer type of a CSR export's offsets or column indices: `u32` or
/// `u64`, the two widths GraphBLAS stores them in.
pub trait CsrIndex: GrbElem {
    /// The width, as GraphBLAS's `*_INTEGER_HINT` fields take it.
    const BITS: i32;
}

impl CsrIndex for u32 {
    const BITS: i32 = 32;
}

impl CsrIndex for u64 {
    const BITS: i32 = 64;
}

/// The value type of a CSR export: `()` for the pattern alone, `f64` for
/// `FP64` values. The export reads the matrix's `x` array only when
/// [`Self::HAS_VALUES`], and then requires the matrix to be of this type
/// ([`GrbElem::holds`]) and not iso.
///
/// [`Self::HAS_VALUES`] is `true` only for a type whose [`GrbElem::holds`]
/// names a GraphBLAS value type: the export then adopts the `x` array as a
/// `Vec<Self>`.
pub trait CsrValue: GrbElem {
    /// Whether the export carries values.
    const HAS_VALUES: bool;
}

impl CsrValue for () {
    const HAS_VALUES: bool = false;
}

impl CsrValue for f64 {
    const HAS_VALUES: bool = true;
}

/// A `GxB_Container` this module owns and frees on drop, with whatever it still
/// holds then.
struct Container(GxB_Container);

impl Container {
    fn new() -> Self {
        let mut container: GxB_Container = null_mut();
        // SAFETY: `GxB_Container_new` writes a new container, which `Self`
        // then frees.
        grb!(unsafe { GxB_Container_new(&raw mut container) });
        Self(container)
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        // SAFETY: `self.0` is live and ours alone.
        let info = unsafe { GxB_Container_free(&raw mut self.0) };
        debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
    }
}

/// One array of a `GxB_Container` (`p`, `i` or `x`), unloaded from its
/// vector: until it is adopted or loaded back, nobody else owns it.
///
/// Only [`Self::unload`] constructs an `Unloaded`, so its caller's promise (a
/// live vector of a container nobody else can reach) covers every later
/// method; that is why `unload` is `unsafe` and `reload`, `fits` and
/// `into_vec` are not.
struct Unloaded {
    vector: GrB_Vector,
    array: *mut c_void,
    element_type: GrB_Type,
    length: u64,
    /// Bytes GraphBLAS allocated for `array`, at least `length` entries.
    byte_size: u64,
    handling: c_int,
}

impl Unloaded {
    /// Take the array out of `vector`, which is left empty.
    ///
    /// # Safety
    /// `vector` is a live vector of a container nobody else can reach.
    unsafe fn unload(vector: GrB_Vector) -> Self {
        let mut unloaded = Self {
            vector,
            array: null_mut(),
            element_type: null_mut(),
            length: 0,
            byte_size: 0,
            handling: 0,
        };
        // SAFETY: `vector` is live and ours (the caller's contract); the
        // unload writes its array and that array's description into the
        // fields.
        unsafe {
            grb!(GxB_Vector_unload(
                vector,
                &raw mut unloaded.array,
                &raw mut unloaded.element_type,
                &raw mut unloaded.length,
                &raw mut unloaded.byte_size,
                &raw mut unloaded.handling,
                null_mut(),
            ));
        }
        unloaded
    }

    /// Load the array back into its vector, which owns it again.
    fn reload(mut self) {
        // SAFETY: `array` is what `unload` took out of `vector`, with its own
        // type, length, size and handling.
        unsafe {
            grb!(GxB_Vector_load(
                self.vector,
                &raw mut self.array,
                self.element_type,
                self.length,
                self.byte_size,
                self.handling,
                null_mut()
            ));
        }
    }

    /// Whether the first `len` entries can become a `Vec<T>`, and if not,
    /// why: the array must be of type `T` (read bit for bit, never cast), hold
    /// at least `len` entries, and be GraphBLAS's own (the default handling:
    /// not read-only, no arena) in a whole number of aligned `T`s.
    fn check<T: GrbElem>(
        &self,
        len: usize,
        array: CsrArray,
    ) -> Result<(), CsrExportError> {
        if len == 0 {
            return Ok(());
        }
        if !T::holds(self.element_type) || length(self.length) < len {
            return Err(CsrExportError::Array(array));
        }
        let byte_size = length(self.byte_size);
        let owned = !self.array.is_null()
            && self.handling == GrB_Desc_Value::GrB_DEFAULT as c_int
            && byte_size.is_multiple_of(size_of::<T>())
            && len <= byte_size / size_of::<T>()
            && self.array.cast::<T>().is_aligned();
        if owned {
            Ok(())
        } else {
            Err(CsrExportError::NotOwned(array))
        }
    }

    /// The first `len` entries as a `Vec<T>`, in O(1): the array itself, which
    /// GraphBLAS no longer refers to and the global allocator frees, as
    /// [`init`](super::matrix::init)'s contract allows. [`Self::check`] has
    /// passed. An empty array goes back to its vector, for GraphBLAS to free.
    fn into_vec<T: GrbElem>(
        self,
        len: usize,
    ) -> Vec<T> {
        if len == 0 {
            self.reload();
            return Vec::new();
        }
        let capacity = length(self.byte_size) / size_of::<T>();
        // SAFETY: `check` passed: `array` holds at least `len` initialized
        // `T`s in a block of exactly `capacity` aligned `T`s, which GraphBLAS's
        // allocator returned and has just given up. The global allocator frees
        // that block as GraphBLAS would (the contract of `init`).
        unsafe { Vec::from_raw_parts(self.array.cast::<T>(), len, capacity) }
    }
}

/// Take `matrix` out as CSR: unload it into a `GxB_Container` and hand its `p`,
/// `i` and (unless `V` is `()`) `x` arrays over as the export's `Vec`s
/// ([`Unloaded::into_vec`]).
///
/// The storage is checked here, on the unloaded container: sparse and
/// row-major ([`CsrExportError::NotCsr`]), not iso when `V` carries values
/// ([`CsrExportError::IsoValues`]), and each array ([`Unloaded::check`]).
///
/// On success `matrix` is left as the unload leaves it: a valid `0 x 0`
/// matrix with no entries and its settings. On a refusal it is loaded back
/// exactly as it was.
///
/// # Safety
/// `matrix` is a live matrix nobody else can reach until this returns, already
/// materialized and, unless `V` is `()`, of value type `V`: the wrapper's
/// `take_csr` does that first.
pub(super) unsafe fn take_out_csr<P: CsrIndex, I: CsrIndex, V: CsrValue>(
    matrix: GrB_Matrix
) -> Result<CsrExport<P, I, V>, CsrExportError> {
    // Whatever the container still holds when it drops (an iso pattern value,
    // an empty array) is GraphBLAS's to free.
    let container = Container::new();
    // SAFETY: `matrix` is live and nobody else can reach it (the caller's
    // contract); the container is new and ours.
    grb!(unsafe { GxB_unload_Matrix_into_Container(matrix, container.0, null_mut()) });
    // SAFETY: a live container, ours alone; a copy of its fields, read before
    // anything changes them.
    let fields = unsafe { *container.0 };
    // Refused: hand everything back, each unloaded array to its vector and
    // the container to `matrix`, which then holds exactly what it held on
    // entry.
    let refuse = |unloaded: Vec<Unloaded>, error| {
        unloaded.into_iter().for_each(Unloaded::reload);
        // SAFETY: the container is the one `matrix` was unloaded into, whole
        // again; `matrix` is still nobody else's.
        grb!(unsafe { GxB_load_Matrix_from_Container(matrix, container.0, null_mut()) });
        Err(error)
    };

    // Materialized (the caller's contract), so every row is sorted: a
    // jumbled container is a bug, not a matrix to refuse. The matrix gets its
    // entries back before the panic.
    if fields.jumbled {
        let _ = refuse(Vec::new(), CsrExportError::NotCsr);
        panic!("a materialized matrix is never jumbled");
    }
    if fields.format != GxB_SPARSE as i32
        || fields.orientation != GrB_Orientation::GrB_ROWMAJOR as i32
    {
        return refuse(Vec::new(), CsrExportError::NotCsr);
    }
    // The pattern export leaves `x`, iso or not, with GraphBLAS.
    if V::HAS_VALUES && fields.iso {
        return refuse(Vec::new(), CsrExportError::IsoValues);
    }

    let offset_count = length(fields.nrows) + 1;
    let entry_count = length(fields.nvals);
    // SAFETY: `p`, `i` and `x` are live vectors of our container, which
    // nobody else can reach.
    let (row_offsets, column_indices, values) = unsafe {
        (
            Unloaded::unload(fields.p),
            Unloaded::unload(fields.i),
            V::HAS_VALUES.then(|| Unloaded::unload(fields.x)),
        )
    };
    let checked = row_offsets
        .check::<P>(offset_count, CsrArray::RowOffsets)
        .and_then(|()| column_indices.check::<I>(entry_count, CsrArray::ColumnIndices))
        .and_then(|()| {
            values.as_ref().map_or(Ok(()), |values| {
                values.check::<V>(entry_count, CsrArray::Values)
            })
        });
    if let Err(error) = checked {
        let unloaded = [Some(row_offsets), Some(column_indices), values];
        return refuse(unloaded.into_iter().flatten().collect(), error);
    }

    // Checked: the arrays leave the container for good.
    Ok(CsrExport {
        row_offsets: row_offsets.into_vec(offset_count),
        column_indices: column_indices.into_vec(entry_count),
        values: values.map_or_else(Vec::new, |values| values.into_vec(entry_count)),
    })
}

#[cfg(test)]
mod tests {
    use super::{CsrArray, CsrExport, CsrExportError, CsrIndex, Unloaded};
    use crate::graph::graphblas::matrix::{Matrix, grb};
    use crate::graph::graphblas::{
        GrB_Matrix_get_INT32, GrB_Matrix_set_INT32, GrB_Orientation, GxB_BITMAP, GxB_FULL,
        GxB_HYPERSPARSE, GxB_Option_Field, test_init,
    };

    /// What a caller does before exporting: store `m` as plain CSR with the
    /// export's widths, `u64` offsets and `u32` column indices. The export
    /// checks all of it and sets none of it. Iso values are left alone, for
    /// the iso test; a caller exporting values asks for `V = f64`.
    fn csr_ready<T>(m: &mut Matrix<T>) {
        m.prepare_csr::<u64, u32, ()>()
            .expect("an unshared matrix is set up");
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
        (0..csr.row_count())
            .flat_map(|row| {
                let entries = csr.row_offsets[row] as usize..csr.row_offsets[row + 1] as usize;
                csr.column_indices[entries]
                    .iter()
                    .map(move |&column| (row as u64, u64::from(column)))
            })
            .collect()
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
        assert_eq!(csr.row_count(), 5);
        assert_eq!(csr.row_offsets.len(), 6);
        assert_eq!(csr.row_offsets, vec![0, 3, 3, 4, 6, 6]);
        assert_eq!(
            entries(&csr),
            vec![(0, 0), (0, 2), (0, 4), (2, 2), (3, 0), (3, 1)]
        );
        assert!(csr.values.is_empty(), "pattern export carries no values");
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
        m.prepare_csr::<u32, u32, f64>()
            .expect("an unshared matrix is set up");

        assert_eq!(
            m.take_csr::<u64, u32, f64>().err(),
            Some(CsrExportError::Array(CsrArray::RowOffsets))
        );
        let csr = m
            .take_csr::<u32, u32, f64>()
            .expect("a CSR-ready matrix exports");
        assert_eq!(csr.row_offsets, vec![0u32, 2, 2, 3]);
        assert_eq!(csr.column_indices, vec![1, 2, 0]);
        assert_eq!(csr.values, vec![1.5, 2.5, 7.25]);
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
        m.prepare_csr::<u64, u32, f64>()
            .expect("an unshared matrix is set up");
        let csr = m
            .take_csr::<u64, u32, f64>()
            .expect("expanded, the matrix exports");
        assert_eq!(csr.values, vec![4.0, 4.0, 4.0]);
        assert_eq!(csr.column_indices, vec![0, 2, 1]);

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
            Some(CsrExportError::Array(CsrArray::RowOffsets))
        );
        assert_eq!(get_field(&m, GxB_Option_Field::GxB_OFFSET_INTEGER_BITS), 32);
        intact(&m);

        let mut m = fresh();
        set_field(&m, GxB_Option_Field::GxB_COLINDEX_INTEGER_HINT, 64);
        assert_eq!(
            m.take_csr::<u64, u32, f64>().err(),
            Some(CsrExportError::Array(CsrArray::ColumnIndices))
        );
        assert_eq!(
            get_field(&m, GxB_Option_Field::GxB_COLINDEX_INTEGER_BITS),
            64
        );
        intact(&m);

        // A clone shares the handle and would see the matrix empty mid-export,
        // or its storage convert under it mid-read.
        let mut m = fresh();
        let clone = m.clone();
        assert_eq!(
            m.prepare_csr::<u64, u32, f64>().err(),
            Some(CsrExportError::SharedHandle)
        );
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

    /// An entry removed just before the matrix is set up is gone from the
    /// export, and every other entry is exported as it was, whichever way the
    /// setup changes the column index width.
    #[test]
    fn a_removal_before_prepare_csr_leaves_the_other_entries_intact() {
        fn removed_then_exported<I: CsrIndex + Into<u64>>(
            stored_bits: i32
        ) -> (Vec<u64>, Vec<u64>, Vec<f64>) {
            let mut m = Matrix::<f64>::new(4, 4);
            set_field(&m, GxB_Option_Field::GxB_COLINDEX_INTEGER_HINT, stored_bits);
            for (i, j, w) in [(0, 1, 1.0), (0, 3, 2.0), (1, 2, 3.0), (3, 0, 4.0)] {
                m.set(i, j, w);
            }
            m.wait();
            m.remove(0, 1);
            m.prepare_csr::<u64, I, f64>()
                .expect("an unshared matrix is set up");
            let csr = m.take_csr::<u64, I, f64>().expect("exports");
            let column_indices = csr.column_indices.into_iter().map(Into::into).collect();
            (csr.row_offsets, column_indices, csr.values)
        }

        test_init::ensure_init();
        let expected = (vec![0, 1, 2, 2, 3], vec![3, 2, 0], vec![2.0, 3.0, 4.0]);
        assert_eq!(removed_then_exported::<u32>(64), expected);
        assert_eq!(removed_then_exported::<u64>(32), expected);
    }

    /// Bitmap and full storage are refused like hypersparse, and the matrix
    /// keeps its entries.
    #[test]
    fn bitmap_and_full_storage_are_refused() {
        test_init::ensure_init();
        for sparsity in [GxB_BITMAP, GxB_FULL] {
            let mut m = Matrix::<f64>::new(2, 2);
            for (i, j, w) in [(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)] {
                m.set(i, j, w);
            }
            csr_ready(&mut m);
            set_field(&m, GxB_Option_Field::GxB_SPARSITY_CONTROL, sparsity as i32);
            assert_eq!(
                m.take_csr::<u64, u32, f64>().err(),
                Some(CsrExportError::NotCsr),
                "sparsity {sparsity}"
            );
            assert_eq!(m.nvals(), 4);
        }
    }

    /// Past `2^31` columns GraphBLAS keeps 64-bit column indices whatever the
    /// hint, and the 32-bit export is refused rather than narrowed.
    #[test]
    fn more_than_two_to_the_31_columns_refuse_32_bit_column_indices() {
        test_init::ensure_init();
        let wide = (1 << 31) + 1;
        let mut m = Matrix::<f64>::new(3, wide);
        m.set(0, wide - 1, 1.5);
        m.set(2, 7, 2.5);
        m.prepare_csr::<u64, u32, f64>()
            .expect("an unshared matrix is set up");
        assert_eq!(
            m.take_csr::<u64, u32, f64>().err(),
            Some(CsrExportError::Array(CsrArray::ColumnIndices))
        );
        let csr = m.take_csr::<u64, u64, f64>().expect("64-bit exports");
        assert_eq!(csr.column_indices, vec![wide - 1, 7]);
    }

    /// A write after `prepare_csr` can undo it (GraphBLAS v10.5 picks the
    /// integer widths again when it finishes the write), and the export then
    /// refuses rather than hand over arrays of another width. Either way,
    /// preparing again exports every write, rows sorted.
    #[test]
    fn a_write_after_prepare_csr_needs_another_prepare() {
        test_init::ensure_init();
        let mut m = Matrix::<f64>::new(3, 3);
        m.set(1, 2, 1.0);
        m.wait();
        m.prepare_csr::<u64, u32, f64>()
            .expect("an unshared matrix is set up");
        m.set(1, 0, 2.0);
        m.set(0, 1, 3.0);
        m.remove(1, 2);
        let csr = match m.take_csr::<u64, u32, f64>() {
            Ok(csr) => csr,
            Err(refused) => {
                assert_eq!(refused, CsrExportError::Array(CsrArray::RowOffsets));
                m.prepare_csr::<u64, u32, f64>()
                    .expect("an unshared matrix is set up");
                m.take_csr::<u64, u32, f64>().expect("exports")
            }
        };
        assert_eq!(csr.row_offsets, vec![0, 1, 2, 2]);
        assert_eq!(csr.column_indices, vec![1, 0]);
        assert_eq!(csr.values, vec![3.0, 2.0]);
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
            Some(CsrExportError::WrongValueType)
        );
        assert_eq!(m.nvals(), 2, "the matrix must survive the refused export");
        assert_eq!(m.get(2, 2), Some(9));
    }

    /// An array GraphBLAS does not own outright (here a read-only one, which
    /// Rust lent it) is refused, not adopted, and goes back to its vector,
    /// still the lender's. The type and length are checked first.
    #[test]
    fn a_borrowed_array_is_refused_not_adopted() {
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
            let values = CsrArray::Values;
            assert_eq!(
                a.check::<u32>(7, values),
                Err(CsrExportError::Array(values))
            );
            assert_eq!(
                a.check::<u64>(11, values),
                Err(CsrExportError::Array(values))
            );
            assert_eq!(
                a.check::<u64>(7, values),
                Err(CsrExportError::NotOwned(values))
            );
            assert_eq!(a.check::<u64>(0, values), Ok(()));
            a.reload();
            let mut nvals = 0;
            grb!(GrB_Vector_nvals(&raw mut nvals, v));
            assert_eq!(nvals, 10, "the vector holds its array again");
            grb!(GrB_Vector_free(&raw mut v));
        }
        assert_eq!(lent, (10..20).collect::<Vec<u64>>(), "the lender's array");
    }
}
