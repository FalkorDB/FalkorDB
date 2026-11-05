//! Safe Rust wrapper around GraphBLAS boolean sparse matrices.
//!
//! This module provides [`Matrix`], which wraps `GrB_Matrix` from the
//! SuiteSparse:GraphBLAS C library. Matrices represent graph adjacency:
//! each relationship type in the graph gets its own sparse boolean matrix.
//!
//! ## Sparse Storage
//!
//! Only non-zero entries consume memory. For a graph with N nodes but
//! sparse connectivity, this is far more efficient than a dense N x N array.
//!
//! ```text
//!   GrB_Matrix (boolean, sparse)
//!
//!   Logical view:               Stored entries only:
//!     0 1 2 3 4                   (0,1) = true
//!   +----------+                  (1,2) = true
//! 0 | . T . . . |                 (3,0) = true
//! 1 | . . T . . |                 (3,4) = true
//! 2 | . . . . . |
//! 3 | T . . . T |               nvals = 4
//! 4 | . . . . . |               nrows = 5, ncols = 5
//!   +----------+
//! ```
//!
//! ## Thread Safety
//!
//! [`Matrix`] shares an owned `GrB_Matrix` handle through an `Arc` for
//! reference-counted cleanup. A `Mutex` guards operations that require serialization (e.g., `wait`).
//! `Clone` is shallow (Arc clone); use [`Dup`] for a deep copy.
//!
//! ## Initialization
//!
//! Call [`init`] once before creating any matrices. It initializes GraphBLAS
//! and LAGraph with optional custom allocators for Redis memory integration.
//!
//! ## Key Operations
//!
//! | Operation      | Method                         | Description                        |
//! |----------------|--------------------------------|------------------------------------|
//! | Create edge    | `Set::set(i, j, true)`         | Set entry at (row, col)            |
//! | Remove edge    | `Remove::remove(i, j)`         | Delete entry at (row, col)         |
//! | Check edge     | `Get::get(i, j)`               | Returns `Some(true)` or `None`     |
//! | Multi-hop      | `MxM::lmxm(b)` / `rmxm(b)`    | Matrix multiply (path traversal)   |
//! | Union          | `MaskedElementWiseAdd`          | OR of two matrices                 |
//! | Intersection   | `MaskedElementWiseMultiply`     | AND of two matrices                |
//! | Transpose      | `Transpose::transpose()`       | Reverse all edge directions        |
//!
//! ## Row Iterator
//!
//! [`Iter`] traverses non-zero entries row by row within a `[min_row, max_row]`
//! range, yielding `(row, col)` pairs. It attaches a `GxB_Iterator` to a
//! snapshot (Arc-cloned) of the underlying `GrB_Matrix`.

#![allow(clippy::doc_markdown)]

use std::{
    borrow::Cow,
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    os::raw::c_void,
    ptr::null_mut,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use parking_lot::Mutex;

use crate::graph::graphblas::{
    lagraph_bindings::{LAGraph_Finalize, LAGraph_Init},
    serialization::{Decode, DecodeLimits, Encode, Reader, Writer},
};

/// Size of the `GxB_Container_struct` in bytes.
const CONTAINER_STRUCT_SIZE: usize = std::mem::size_of::<super::GxB_Container_struct>();

use super::vector::{Component, Vector};
use super::{
    GrB_BOOL, GrB_BinaryOp, GrB_DESC_C, GrB_DESC_CT0, GrB_DESC_CT0T1, GrB_DESC_CT1, GrB_DESC_R,
    GrB_DESC_RC, GrB_DESC_RCT0, GrB_DESC_RCT0T1, GrB_DESC_RCT1, GrB_DESC_RS, GrB_DESC_RSC,
    GrB_DESC_RSCT0, GrB_DESC_RSCT0T1, GrB_DESC_RSCT1, GrB_DESC_RST0, GrB_DESC_RST0T1,
    GrB_DESC_RST1, GrB_DESC_RT0, GrB_DESC_RT0T1, GrB_DESC_RT1, GrB_DESC_S, GrB_DESC_SC,
    GrB_DESC_SCT0, GrB_DESC_SCT0T1, GrB_DESC_SCT1, GrB_DESC_ST0, GrB_DESC_ST0T1, GrB_DESC_ST1,
    GrB_DESC_T0, GrB_DESC_T0T1, GrB_DESC_T1, GrB_Descriptor, GrB_GLOBAL, GrB_Global_set_INT32,
    GrB_Info, GrB_Matrix, GrB_Matrix_apply, GrB_Matrix_assign_BOOL, GrB_Matrix_build_BOOL,
    GrB_Matrix_build_UINT64, GrB_Matrix_clear, GrB_Matrix_dup, GrB_Matrix_eWiseAdd_BinaryOp,
    GrB_Matrix_eWiseMult_Semiring, GrB_Matrix_extractElement_BOOL,
    GrB_Matrix_extractElement_UINT64, GrB_Matrix_free, GrB_Matrix_get_INT32, GrB_Matrix_ncols,
    GrB_Matrix_new, GrB_Matrix_nrows, GrB_Matrix_nvals, GrB_Matrix_removeElement,
    GrB_Matrix_resize, GrB_Matrix_select_UINT64, GrB_Matrix_set_INT32, GrB_Matrix_setElement_BOOL,
    GrB_Matrix_setElement_UINT64, GrB_Matrix_wait, GrB_Mode, GrB_Orientation, GrB_SECOND_UINT64,
    GrB_Scalar, GrB_Scalar_free, GrB_Scalar_new, GrB_Scalar_setElement_BOOL, GrB_Type, GrB_UINT64,
    GrB_VALUEEQ_UINT64, GrB_WaitMode, GrB_finalize, GrB_mxm, GrB_mxv, GrB_transpose, GxB_ANY_BOOL,
    GxB_ANY_PAIR_BOOL, GxB_ANY_UINT64, GxB_Container_free, GxB_Container_new,
    GxB_Global_Option_set_INT32, GxB_HYPERSPARSE, GxB_Iterator, GxB_Iterator_free,
    GxB_Iterator_get_UINT64, GxB_Iterator_new, GxB_JIT_Control, GxB_Matrix_build_Scalar,
    GxB_Matrix_fprint, GxB_Matrix_isStoredElement, GxB_Matrix_memoryUsage, GxB_Matrix_type,
    GxB_NTHREADS, GxB_ONE_BOOL, GxB_Option_Field, GxB_PLUS_PAIR_UINT64, GxB_Print_Level,
    GxB_SPARSE, GxB_Type_size, GxB_init, GxB_load_Matrix_from_Container, GxB_rowIterator_attach,
    GxB_rowIterator_getColIndex, GxB_rowIterator_getRowIndex, GxB_rowIterator_kount,
    GxB_rowIterator_nextCol, GxB_rowIterator_nextRow, GxB_rowIterator_seekRow,
    GxB_unload_Matrix_into_Container,
};
use super::{
    GrB_FP64, GrB_Matrix_build_FP64, GrB_Matrix_extractTuples_FP64, GrB_Matrix_select_INT64,
    GrB_Matrix_setElement_FP64, GrB_OFFDIAG,
};
use super::{
    GrB_GT_UINT64, GrB_Matrix_eWiseMult_BinaryOp, GrB_Matrix_select_BOOL, GrB_VALUEEQ_BOOL,
};

/// Initializes the GraphBLAS library in non-blocking mode.
///
/// Custom allocators can be provided to integrate with Redis memory management.
/// This ensures GraphBLAS memory counts toward Redis limits.
///
/// # Errors
///
/// Returns `Err` with a descriptive message if `GxB_init` or `LAGraph_Init`
/// fail. The caller (Redis module-load path) should propagate this as
/// `Status::Err` so Redis refuses to load the module rather than aborting
/// the whole server process.
#[allow(clippy::similar_names)]
pub fn init(
    user_malloc_function: Option<unsafe extern "C" fn(arg1: usize) -> *mut c_void>,
    user_calloc_function: Option<unsafe extern "C" fn(arg1: usize, arg2: usize) -> *mut c_void>,
    user_realloc_function: Option<
        unsafe extern "C" fn(arg1: *mut c_void, arg2: usize) -> *mut c_void,
    >,
    user_free_function: Option<unsafe extern "C" fn(arg1: *mut c_void)>,
) -> Result<(), String> {
    unsafe {
        let info = GxB_init(
            GrB_Mode::GrB_NONBLOCKING as _,
            user_malloc_function,
            user_calloc_function,
            user_realloc_function,
            user_free_function,
        );
        if info != GrB_Info::GrB_SUCCESS {
            return Err(format!("GraphBLAS GxB_init failed: {info:?}"));
        }

        // Pick GraphBLAS JIT control level:
        //
        //   * Default — GxB_JIT_RUN: mirror the FalkorDB C module
        //     (src/module.c:106). PreJIT kernels statically linked into
        //     libgraphblas.a (vendored from build/graphblas/PreJIT/ by
        //     the native-deps GraphBLAS recipe) are used for hot ops; RUN additionally permits
        //     dlopen of any kernel already present in the on-disk cache,
        //     without any runtime compilation. In the shipped runtime image
        //     the cache is empty and no compiler is installed, so any op
        //     not covered by PreJIT silently falls back to generic kernels
        //     (no panic, no dlopen attempts that would deadlock fork()).
        //     Local arm64 A/B vs GxB_JIT_OFF (which main shipped) shows
        //     +6% to +87% across the benchmark suite; aligning with
        //     the C module's choice keeps the runtime semantics
        //     interchangeable.
        //
        //   * `--features prejit_harvest` — GxB_JIT_ON: full JIT including
        //     compile-on-demand. Selected at build time, never at runtime —
        //     prevents an env-var typo from accidentally enabling JIT in
        //     a shipped binary. Used exclusively by `native-deps prejit` to
        //     populate ~/.SuiteSparse/GrBx.y.z/c/ with the .c kernel
        //     sources we then check in as the next generation of vendored
        //     PreJIT (see native-deps/src/prejit.rs).
        #[cfg(feature = "prejit_harvest")]
        let (jit_level, jit_name) = (GxB_JIT_Control::GxB_JIT_ON, "JIT_ON (harvest)");
        #[cfg(not(feature = "prejit_harvest"))]
        let (jit_level, jit_name) = (GxB_JIT_Control::GxB_JIT_RUN, "JIT_RUN");
        let info = GrB_Global_set_INT32(
            GrB_GLOBAL,
            jit_level as i32,
            GxB_Option_Field::GxB_JIT_C_CONTROL as _,
        );
        if info != GrB_Info::GrB_SUCCESS {
            return Err(format!("GraphBLAS {jit_name} failed: {info:?}"));
        }

        // Initialize LAGraph after GraphBLAS
        // `c_char` (not `i8`) because char signedness is platform-dependent:
        // signed on amd64, unsigned on arm64 Linux. LAGraph FFI takes *mut c_char.
        let mut msg: [std::os::raw::c_char; 256] = [0; 256];
        let rc = LAGraph_Init(msg.as_mut_ptr());
        if rc != 0 {
            return Err(format!(
                "LAGraph_Init failed (rc={rc}): {}",
                std::ffi::CStr::from_ptr(msg.as_ptr()).to_string_lossy(),
            ));
        }
    }
    Ok(())
}

/// Set the number of threads GraphBLAS and OpenMP may use internally.
///
/// Call with `n = 1` in a fork child process to prevent GraphBLAS/OpenMP from
/// touching thread pool handles that are invalid after `fork()`.
pub fn set_nthreads(n: i32) {
    unsafe {
        // Tell OpenMP directly — after fork, its thread team is invalid.
        omp_set_num_threads(n);
        // Also tell GraphBLAS, which gates its own parallel-for loops.
        GxB_Global_Option_set_INT32(GxB_NTHREADS as i32, n);
    }
}

unsafe extern "C" {
    fn omp_set_num_threads(num_threads: i32);
}

/// Enable or disable GraphBLAS diagnostic output (burble mode).
pub fn burble(burble: bool) {
    unsafe {
        GrB_Global_set_INT32(
            GrB_GLOBAL,
            i32::from(burble),
            GxB_Option_Field::GxB_BURBLE as _,
        );
    }
}

/// Finalizes LAGraph and GraphBLAS, releasing all resources.
pub fn shutdown() {
    unsafe {
        let mut msg: [std::os::raw::c_char; 256] = [0; 256];
        LAGraph_Finalize(msg.as_mut_ptr());
    }
}

pub enum Descriptor {
    T0,
    T1,
    T0T1,
    C,
    CT0,
    CT1,
    CT0T1,
    S,
    ST0,
    ST1,
    ST0T1,
    SC,
    SCT0,
    SCT1,
    SCT0T1,
    R,
    RT0,
    RT1,
    RT0T1,
    RC,
    RCT0,
    RCT1,
    RCT0T1,
    RS,
    RST0,
    RST1,
    RST0T1,
    RSC,
    RSCT0,
    RSCT1,
    RSCT0T1,
}

/// The `⊕` [`Matrix::element_wise_add`] folds intersecting entries with,
/// selected by the matrix's element type.
///
/// The operator is a property of what the values *mean*, which is fixed by the
/// element type, so there is nothing for a caller to choose: a `bool` matrix
/// carries only a sparsity pattern, where either of the two `true`s will do; a
/// `u64` matrix carries edge ids, where `b` is the delta layer whose value
/// shadows the base's on a pair present in both.
pub trait EWiseAdd {
    fn add_op() -> GrB_BinaryOp;
}

impl EWiseAdd for bool {
    fn add_op() -> GrB_BinaryOp {
        // The additive monoid of `GxB_ANY_PAIR_BOOL`, which is what the
        // eWiseAdd-by-semiring form this replaced resolved to.
        unsafe { GxB_ANY_BOOL }
    }
}

impl EWiseAdd for u64 {
    fn add_op() -> GrB_BinaryOp {
        unsafe { GrB_SECOND_UINT64 }
    }
}

/// Element types a [`Matrix`] can be constructed for.
///
/// `Matrix::<T>::new` is written per element type because it names a GraphBLAS
/// type handle, which leaves generic code with no way to make an empty layer of
/// its own type. This bridges that, in the same shape as [`EWiseAdd`] above.
pub trait MatrixType: Sized {
    fn new_matrix(
        nrows: u64,
        ncols: u64,
    ) -> Matrix<Self>;
}

impl MatrixType for bool {
    fn new_matrix(
        nrows: u64,
        ncols: u64,
    ) -> Matrix<bool> {
        Matrix::<bool>::new(nrows, ncols)
    }
}

impl MatrixType for u64 {
    fn new_matrix(
        nrows: u64,
        ncols: u64,
    ) -> Matrix<u64> {
        Matrix::<u64>::new(nrows, ncols)
    }
}

impl From<Descriptor> for GrB_Descriptor {
    fn from(descriptor: Descriptor) -> Self {
        unsafe {
            match descriptor {
                Descriptor::T0 => GrB_DESC_T0,
                Descriptor::T1 => GrB_DESC_T1,
                Descriptor::T0T1 => GrB_DESC_T0T1,
                Descriptor::C => GrB_DESC_C,
                Descriptor::CT0 => GrB_DESC_CT0,
                Descriptor::CT1 => GrB_DESC_CT1,
                Descriptor::CT0T1 => GrB_DESC_CT0T1,
                Descriptor::S => GrB_DESC_S,
                Descriptor::ST0 => GrB_DESC_ST0,
                Descriptor::ST1 => GrB_DESC_ST1,
                Descriptor::ST0T1 => GrB_DESC_ST0T1,
                Descriptor::SC => GrB_DESC_SC,
                Descriptor::SCT0 => GrB_DESC_SCT0,
                Descriptor::SCT1 => GrB_DESC_SCT1,
                Descriptor::SCT0T1 => GrB_DESC_SCT0T1,
                Descriptor::R => GrB_DESC_R,
                Descriptor::RT0 => GrB_DESC_RT0,
                Descriptor::RT1 => GrB_DESC_RT1,
                Descriptor::RT0T1 => GrB_DESC_RT0T1,
                Descriptor::RC => GrB_DESC_RC,
                Descriptor::RCT0 => GrB_DESC_RCT0,
                Descriptor::RCT1 => GrB_DESC_RCT1,
                Descriptor::RCT0T1 => GrB_DESC_RCT0T1,
                Descriptor::RS => GrB_DESC_RS,
                Descriptor::RST0 => GrB_DESC_RST0,
                Descriptor::RST1 => GrB_DESC_RST1,
                Descriptor::RST0T1 => GrB_DESC_RST0T1,
                Descriptor::RSC => GrB_DESC_RSC,
                Descriptor::RSCT0 => GrB_DESC_RSCT0,
                Descriptor::RSCT1 => GrB_DESC_RSCT1,
                Descriptor::RSCT0T1 => GrB_DESC_RSCT0T1,
            }
        }
    }
}

/// Sole owner of a `GrB_Matrix` handle, freeing it on drop.
///
/// Clones of a [`Matrix`] (and its [`Iter`]s) share one of these through an
/// `Arc`, so the free runs exactly once, when the `Arc`'s own atomic count
/// reaches zero. Freeing from `Matrix::drop` behind `Arc::get_mut` instead
/// raced: two clones dropped concurrently could each see the other still
/// alive, skip the free, and then both decrement — leaking the matrix.
///
/// No `Deref`: code reads the raw handle through [`Matrix::inner`] (or
/// `Iter::matrix`), so call sites never write `**x.m`.
struct Handle(GrB_Matrix);

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            let info = GrB_Matrix_free(&raw mut self.0);
            // debug_assert in Drop: panicking while unwinding aborts the
            // process. A GrB_*_free failure is logically a leak, not
            // state corruption — surface in debug, swallow in release.
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

/// A wrapper around a GraphBLAS matrix.
///
/// The type parameter `T` is a compile-time tag for the element type the matrix
/// carries (`bool` for pure structure / presence, `u64` for valued matrices such
/// as inline edge ids). It is a zero-sized [`PhantomData`] marker — every
/// `Matrix<T>` has identical layout regardless of `T` — so it only documents and
/// type-checks intent; it does not change the runtime representation.
pub struct Matrix<T> {
    /// The underlying GraphBLAS matrix.
    m: Arc<Handle>,
    lock: Arc<Mutex<()>>,
    /// Set to `true` by every mutating op; `wait()` short-circuits when `false`
    /// to skip the lock + GrB_Matrix_wait FFI under read-heavy contention.
    has_pending: Arc<AtomicBool>,
    phantom: PhantomData<T>,
}

// Manual `Clone` (not derived) so it holds for every `T` without a `T: Clone`
// bound — the only `T`-dependent field is the ZST `PhantomData<T>`. This keeps
// generic `Matrix<T>` / `VersionedMatrix<V>` code free of spurious bounds.
impl<T> Clone for Matrix<T> {
    fn clone(&self) -> Self {
        Self {
            m: self.m.clone(),
            lock: self.lock.clone(),
            has_pending: self.has_pending.clone(),
            phantom: PhantomData,
        }
    }
}

unsafe impl<T> Send for Matrix<T> {}
unsafe impl<T> Sync for Matrix<T> {}

/// Pin a matrix to (hyper)sparse storage, like the C implementation's delta
/// matrices. Without this GraphBLAS auto-converts dense-ish matrices (e.g.
/// node-labels: nodes × few-labels) to bitmap, and every later fold/wait
/// pays a whole-bitmap memset + assign.
unsafe fn pin_sparse(m: GrB_Matrix) {
    let info = unsafe {
        GrB_Matrix_set_INT32(
            m,
            (GxB_SPARSE | GxB_HYPERSPARSE) as i32,
            GxB_Option_Field::GxB_SPARSITY_CONTROL as _,
        )
    };
    debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
    // SuiteSparse stores n-by-1 matrices by column by default; row iterators
    // (GxB_rowIterator_attach) fail with GrB_NOT_IMPLEMENTED on such a
    // matrix, and a later resize keeps the orientation. Pin row-major so a
    // matrix created while a dimension happens to be 1 stays iterable.
    let info = unsafe {
        GrB_Matrix_set_INT32(
            m,
            GrB_Orientation::GrB_ROWMAJOR as i32,
            GxB_Option_Field::GrB_STORAGE_ORIENTATION_HINT as _,
        )
    };
    debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
}

/// Frees a `GxB_Container` and whatever vectors it still holds, on every exit
/// from [`Matrix::decode`] — the early error returns included.
struct ContainerGuard(super::GxB_Container);

impl Drop for ContainerGuard {
    fn drop(&mut self) {
        let info = unsafe { GxB_Container_free(&raw mut self.0) };
        debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
    }
}

/// Most a matrix dimension can be: GraphBLAS's `GrB_INDEX_MAX`.
const MAX_DIM: u64 = 1 << 60;

/// The scalar fields of a matrix header as a payload states them.
///
/// Only these are taken from the payload. The rest of the container (arenas,
/// the reserved "future" words, every pointer) stays as `GxB_Container_new`
/// made it: a payload has no business setting them, and `header_arena` in
/// particular picks the allocator GraphBLAS later frees with.
struct ContainerHeader {
    nrows: u64,
    ncols: u64,
    nvals: u64,
    format: i32,
    orientation: i32,
    iso: bool,
}

impl ContainerHeader {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        // Offsets are those of `GxB_Container_struct`; the size assertion in
        // the bindings pins its layout and `container_header_offsets` checks
        // these against it.
        fn at<const N: usize>(
            bytes: &[u8],
            off: usize,
        ) -> [u8; N] {
            bytes[off..off + N].try_into().expect("slice has N bytes")
        }
        let h = Self {
            nrows: u64::from_le_bytes(at(bytes, OFF_NROWS)),
            ncols: u64::from_le_bytes(at(bytes, OFF_NROWS + 8)),
            nvals: u64::from_le_bytes(at(bytes, OFF_NROWS + 32)),
            format: i32::from_le_bytes(at(bytes, OFF_FORMAT)),
            orientation: i32::from_le_bytes(at(bytes, OFF_FORMAT + 4)),
            iso: bytes[OFF_ISO] != 0,
        };
        // Pinned sparse or hypersparse on both ends (`pin_sparse`); bitmap and
        // full would allocate nrows * ncols.
        if h.format != GxB_SPARSE as i32 && h.format != GxB_HYPERSPARSE as i32 {
            return Err(format!("matrix format {} is not sparse", h.format));
        }
        if h.orientation != GrB_Orientation::GrB_ROWMAJOR as i32
            && h.orientation != GrB_Orientation::GrB_COLMAJOR as i32
        {
            return Err(format!("matrix orientation {} is unknown", h.orientation));
        }
        let max_dim = DecodeLimits::current().map_or(MAX_DIM, |l| l.max_dim().min(MAX_DIM));
        if h.nrows > max_dim || h.ncols > max_dim {
            return Err(format!(
                "matrix dimensions {}x{} exceed the {max_dim} the graph allows",
                h.nrows, h.ncols
            ));
        }
        Ok(h)
    }
}

const OFF_NROWS: usize = std::mem::offset_of!(super::GxB_Container_struct, nrows);
const OFF_FORMAT: usize = std::mem::offset_of!(super::GxB_Container_struct, format);
const OFF_ISO: usize = std::mem::offset_of!(super::GxB_Container_struct, iso);

/// Bytes per entry of an index array (`p`, `h` or `i`): GraphBLAS keeps them
/// as 32- or 64-bit integers, whichever the matrix's size calls for.
fn index_width(c: &Component) -> Result<usize, String> {
    match c.type_name() {
        b"GrB_UINT32" | b"GrB_INT32" => Ok(4),
        b"GrB_UINT64" | b"GrB_INT64" => Ok(8),
        other => Err(format!(
            "matrix index array has type {}",
            String::from_utf8_lossy(other)
        )),
    }
}

/// Entry `k` of an index array of `w`-byte entries.
#[inline(always)]
fn index_at(
    bytes: &[u8],
    w: usize,
    k: usize,
) -> u64 {
    if w == 4 {
        u64::from(u32::from_le_bytes(
            bytes[4 * k..4 * k + 4].try_into().expect("4 bytes"),
        ))
    } else {
        u64::from_le_bytes(bytes[8 * k..8 * k + 8].try_into().expect("8 bytes"))
    }
}

/// Check that the arrays of a sparse or hypersparse matrix are a matrix: the
/// pointer array starts at zero, never decreases and ends at `nvals`; the
/// hyperlist, if there is one, is strictly increasing and inside the
/// dimension; every row of indices is strictly increasing and inside the other
/// dimension. Returns the number of non-empty vectors, which the pass over the
/// pointers counts for free.
///
/// `GxB_load_Matrix_from_Container` checks none of that, only that the arrays
/// are long enough, and everything after it — iterators above all — trusts it:
/// a pointer or index out of range reads past the arrays and crashes the server.
/// GraphBLAS has a checker for this (`GxB_Matrix_fprint` at `GxB_SILENT`), but it
/// runs at 10 to 35 ns an entry, which more than doubled the cost of loading a
/// graph of a few million edges; a loop over the raw arrays costs next to nothing.
fn check_structure(
    header: &ContainerHeader,
    h: &Component,
    p: &Component,
    i: &Component,
) -> Result<i64, String> {
    let bad = |what: &str| Err(format!("corrupt matrix structure: {what}"));
    let (vlen, vdim) = if header.orientation == GrB_Orientation::GrB_COLMAJOR as i32 {
        (header.nrows, header.ncols)
    } else {
        (header.ncols, header.nrows)
    };
    let (pw, iw) = (index_width(p)?, index_width(i)?);
    // The arrays may be allocated past the entries in use, so the counts are
    // the components' own, held to fit in what was sent.
    let (plen1, nnz) = (p.n_entries as usize, i.n_entries);
    if p.n_entries
        .checked_mul(pw as u64)
        .is_none_or(|n| n > p.arr_data.len() as u64)
        || i.n_entries
            .checked_mul(iw as u64)
            .is_none_or(|n| n > i.arr_data.len() as u64)
        || plen1 == 0
    {
        return bad("pointer or index array is shorter than it says");
    }
    let (pa, ia) = (&p.arr_data[..], &i.arr_data[..]);
    let nvec = plen1 - 1;

    if header.format == GxB_HYPERSPARSE as i32 {
        let hw = index_width(h)?;
        let ha = &h.arr_data[..];
        if h.n_entries != nvec as u64
            || nvec as u64 > vdim
            || h.n_entries
                .checked_mul(hw as u64)
                .is_none_or(|n| n > ha.len() as u64)
        {
            return bad("hyperlist length does not match the pointer array");
        }
        let mut last = None;
        for k in 0..nvec {
            let j = index_at(ha, hw, k);
            if j >= vdim || last.is_some_and(|l| l >= j) {
                return bad("hyperlist is not strictly increasing within the dimension");
            }
            last = Some(j);
        }
    } else if nvec as u64 != vdim {
        return bad("pointer array does not cover the dimension");
    }

    if index_at(pa, pw, 0) != 0 || index_at(pa, pw, nvec) != header.nvals || header.nvals > nnz {
        return bad("pointers do not span exactly the entries");
    }
    let outcome = match (pw, iw) {
        (4, 4) => check_rows::<u32, u32>(pa, ia, nvec, vlen, header.nvals),
        (4, _) => check_rows::<u32, u64>(pa, ia, nvec, vlen, header.nvals),
        (_, 4) => check_rows::<u64, u32>(pa, ia, nvec, vlen, header.nvals),
        _ => check_rows::<u64, u64>(pa, ia, nvec, vlen, header.nvals),
    };
    outcome.or_else(bad)
}

/// An unsigned integer an index array is made of.
trait Index: Copy + Ord + Into<u64> {
    const WIDTH: usize;
    fn from_le(bytes: &[u8]) -> Self;
}

impl Index for u32 {
    const WIDTH: usize = 4;
    fn from_le(bytes: &[u8]) -> Self {
        Self::from_le_bytes(bytes.try_into().expect("4 bytes"))
    }
}

impl Index for u64 {
    const WIDTH: usize = 8;
    fn from_le(bytes: &[u8]) -> Self {
        Self::from_le_bytes(bytes.try_into().expect("8 bytes"))
    }
}

/// The first `n` entries of an index array, in place when the bytes are aligned
/// for `T` (they nearly always are), copied when not.
fn typed<T: Index>(
    bytes: &[u8],
    n: usize,
) -> Cow<'_, [T]> {
    let bytes = &bytes[..n * T::WIDTH];
    if cfg!(target_endian = "little") {
        // SAFETY: every bit pattern is a valid `u32` or `u64`, and `align_to`
        // hands back only the part of the bytes that is aligned and whole.
        let (head, body, tail) = unsafe { bytes.align_to::<T>() };
        if head.is_empty() && tail.is_empty() {
            return Cow::Borrowed(body);
        }
    }
    Cow::Owned(bytes.chunks_exact(T::WIDTH).map(T::from_le).collect())
}

/// The pointers run from 0 to `nvals` without decreasing, and each row of
/// indices they delimit is strictly increasing and below `vlen`. Returns how
/// many of those rows have entries.
fn check_rows<P: Index, I: Index>(
    pa: &[u8],
    ia: &[u8],
    nvec: usize,
    vlen: u64,
    nvals: u64,
) -> Result<i64, &'static str> {
    let ptrs = typed::<P>(pa, nvec + 1);
    // `nvals` is at most the entries held (checked by the caller), and no
    // pointer is allowed past it, so every slice below stays inside `idx`.
    let idx = typed::<I>(ia, usize::try_from(nvals).map_err(|_| "entry count")?);
    let mut start: u64 = ptrs[0].into();
    let mut nonempty = 0i64;
    for &end in &ptrs[1..] {
        let end: u64 = end.into();
        if end < start || end > nvals {
            return Err("pointers decrease or pass the entry count");
        }
        let row = &idx[start as usize..end as usize];
        // Strictly increasing, so only the last can be out of range.
        if row.last().is_some_and(|&last| last.into() >= vlen)
            || row.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err("indices are out of range or not strictly increasing");
        }
        nonempty += i64::from(end > start);
        start = end;
    }
    Ok(nonempty)
}

impl<T> Decode<19> for Matrix<T> {
    fn decode(r: &mut dyn Reader) -> Result<Self, String> {
        let container_bytes = r.read_buffer()?;

        // Validate container size before reading it
        if container_bytes.len() < CONTAINER_STRUCT_SIZE {
            return Err(format!(
                "container buffer too small: {} bytes < {} bytes required",
                container_bytes.len(),
                CONTAINER_STRUCT_SIZE
            ));
        }
        let header = ContainerHeader::parse(&container_bytes)?;

        unsafe {
            let mut container: MaybeUninit<super::GxB_Container> = MaybeUninit::uninit();
            let info = GxB_Container_new(container.as_mut_ptr());
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GxB_Container_new failed: {info:?}"
            );
            let guard = ContainerGuard(container.assume_init());
            let container = guard.0;

            (*container).nrows = header.nrows;
            (*container).ncols = header.ncols;
            // The payload's own counts are not trusted, and kernels size
            // workspace by them; `check_structure` counts the real one below.
            (*container).nrows_nonempty = -1;
            (*container).ncols_nonempty = -1;
            (*container).nvals = header.nvals;
            (*container).format = header.format;
            (*container).orientation = header.orientation;
            (*container).iso = header.iso;
            // A saved matrix is waited, so never jumbled; the structure check
            // below rejects one that is not sorted.
            (*container).jumbled = false;

            // Read the 5 component vectors, x, h, p, i, b, and check the
            // structure they spell out before GraphBLAS gets any of them. Each
            // is stored in the container the moment it is loaded, so the guard
            // frees it on a later error.
            let x = Component::read(r)?;
            let h = Component::read(r)?;
            let p = Component::read(r)?;
            let i = Component::read(r)?;
            let b = Component::read(r)?;
            let nonempty = check_structure(&header, &h, &p, &i)?;
            // Leaving it unknown (-1) makes GraphBLAS count it again on first
            // use: a pass over every vector, per matrix.
            if header.orientation == GrB_Orientation::GrB_COLMAJOR as i32 {
                (*container).ncols_nonempty = nonempty;
            } else {
                (*container).nrows_nonempty = nonempty;
            }
            (*container).x = ManuallyDrop::new(x.load()?).ptr();
            (*container).h = ManuallyDrop::new(h.load()?).ptr();
            (*container).p = ManuallyDrop::new(p.load()?).ptr();
            (*container).i = ManuallyDrop::new(i.load()?).ptr();
            (*container).b = ManuallyDrop::new(b.load()?).ptr();

            // Create matrix and load from container
            let mut m: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
            let info = GrB_Matrix_new(m.as_mut_ptr(), GrB_BOOL, 0, 0);
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GrB_Matrix_new failed: {info:?}"
            );
            let mut m = m.assume_init();
            pin_sparse(m);

            // `GxB_load_Matrix_from_Container` checks that the arrays are
            // long enough for what the header says; `check_structure` has
            // already checked what is in them. A failure is bad input, not a
            // broken invariant.
            let info = GxB_load_Matrix_from_Container(m, container, null_mut());
            if info != GrB_Info::GrB_SUCCESS {
                GrB_Matrix_free(&raw mut m);
                return Err(format!("GxB_load_Matrix_from_Container failed: {info:?}"));
            }

            // The container import takes the payload's orientation over the
            // row-major hint `pin_sparse` set, and row iterators (every reader of
            // a decoded matrix) refuse a column-major one. Convert it; a row-major
            // payload, which is every one this engine writes, pays nothing.
            if header.orientation == GrB_Orientation::GrB_COLMAJOR as i32 {
                let info = GrB_Matrix_set_INT32(
                    m,
                    GrB_Orientation::GrB_ROWMAJOR as i32,
                    GxB_Option_Field::GrB_STORAGE_ORIENTATION_HINT as _,
                );
                if info != GrB_Info::GrB_SUCCESS {
                    GrB_Matrix_free(&raw mut m);
                    return Err(format!("matrix could not be stored by row: {info:?}"));
                }
            }

            // The element type is whatever the payload's value vector said.
            // Readers of a `Matrix<T>` take `size_of::<T>()` bytes per entry
            // straight from the array, so a narrower type would be read past
            // its end. A wider one is fine, and is what a bool matrix comes
            // back as when it was saved as UINT64 (a tensor's delta-minus).
            let mut ty: MaybeUninit<GrB_Type> = MaybeUninit::uninit();
            let mut size = 0usize;
            if GxB_Matrix_type(ty.as_mut_ptr(), m) != GrB_Info::GrB_SUCCESS
                || GxB_Type_size(&raw mut size, ty.assume_init()) != GrB_Info::GrB_SUCCESS
                || size < std::mem::size_of::<T>()
            {
                GrB_Matrix_free(&raw mut m);
                return Err(format!(
                    "matrix element size {size} is narrower than the {} expected",
                    std::mem::size_of::<T>()
                ));
            }

            // The hyper-hash (Y) was nullified above and is not serialized, so
            // a hypersparse matrix comes back with GxB_WILL_WAIT set. Rebuild
            // it now so `pending()` reflects real pending work (no-op for
            // non-hypersparse matrices).
            let info = GrB_Matrix_wait(m, GrB_WaitMode::GrB_MATERIALIZE as _);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);

            Ok(Self {
                m: Arc::new(Handle(m)),
                lock: Arc::new(Mutex::new(())),
                has_pending: Arc::new(AtomicBool::new(false)),
                phantom: PhantomData,
            })
        }
    }
}

impl<T> Encode<19> for Matrix<T> {
    fn encode(
        &self,
        w: &mut dyn Writer,
    ) {
        unsafe {
            let mut container: MaybeUninit<super::GxB_Container> = MaybeUninit::uninit();
            let info = GxB_Container_new(container.as_mut_ptr());
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GxB_Container_new failed: {info:?}"
            );
            let container = container.assume_init();

            let info = GxB_unload_Matrix_into_Container(self.inner(), container, null_mut());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);

            // Write container struct bytes. Its pointer fields are heap
            // addresses in this process: meaningless to a reader (decode
            // rebuilds every component), and they would disclose the memory
            // layout to anyone holding a DUMP / GRAPH.COPY payload or an RDB.
            // Write them as null.
            let mut header = std::ptr::read(container);
            header.p = null_mut();
            header.h = null_mut();
            header.b = null_mut();
            header.i = null_mut();
            header.x = null_mut();
            header.vector_future = [null_mut(); 11];
            header.Y = null_mut();
            header.matrix_future = [null_mut(); 15];
            header.void_future = [null_mut(); 16];
            let container_bytes =
                std::slice::from_raw_parts((&raw const header).cast::<u8>(), CONTAINER_STRUCT_SIZE);
            w.write_buffer(container_bytes);

            // Write 5 vectors: x, h, p, i, b
            ManuallyDrop::new(Vector::<bool>::from((*container).x)).encode(w);
            ManuallyDrop::new(Vector::<bool>::from((*container).h)).encode(w);
            ManuallyDrop::new(Vector::<bool>::from((*container).p)).encode(w);
            ManuallyDrop::new(Vector::<bool>::from((*container).i)).encode(w);
            ManuallyDrop::new(Vector::<bool>::from((*container).b)).encode(w);

            let info = GxB_load_Matrix_from_Container(self.inner(), container, null_mut());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);

            let mut c = container;
            let info = GxB_Container_free(&raw mut c);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

impl<T> Matrix<T> {
    /// Pin this matrix to hypersparse storage and return it (builder-style,
    /// for delta matrices at construction). Deltas inherit the base's
    /// dimensions but the fold policy keeps their nvals small; in plain
    /// sparse format every `GB_wait` / zombie-select on them pays `O(nrows)`
    /// row-pointer work (measured 18x on 100-node creates after a bulk write
    /// inflated capacity to 1m rows). Hypersparse makes those ops `O(nvec)`.
    /// The C implementation pins its delta matrices hypersparse for the same
    /// reason.
    #[must_use]
    pub(super) fn into_hyper(self) -> Self {
        unsafe {
            let info = GrB_Matrix_set_INT32(
                self.inner(),
                GxB_HYPERSPARSE as i32,
                GxB_Option_Field::GxB_SPARSITY_CONTROL as _,
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            // Deltas also opt out of the hyper hash: GrB_Matrix_wait
            // (MATERIALIZE) rebuilds A->Y from scratch on every commit, an
            // O(nvec) sort that grows with the accumulating delta — measured
            // as the dominant cost of small repeated creates. Without A->Y,
            // lookups binary-search A->h, which is fine at delta sizes.
            let info = GrB_Matrix_set_INT32(self.inner(), 0, GxB_Option_Field::GxB_HYPER_HASH as _);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self
    }

    /// Number of vectors (rows) the storage holds — for a hypersparse matrix,
    /// the number of non-empty rows, read from the data structure rather than
    /// counted.
    ///
    /// Two caveats, both measured rather than assumed:
    ///
    /// * `GxB_rowIterator_kount` is documented only as an *upper* bound: "if A
    ///   is hypersparse, kount is the # of vectors held in the data structure,
    ///   some of which may be empty". SuiteSparse in fact prunes emptied
    ///   vectors on `wait`, so for an assembled hypersparse matrix the two
    ///   coincide — verified directly, including after removals that empty a
    ///   row, which is the case that matters. For a sparse, bitmap or full
    ///   matrix kount is `nrows`, which is why this returns `None` unless the
    ///   storage is hypersparse: at `me`'s dimensions `nrows` is meaningless as
    ///   a row count and a caller must not silently compare against it.
    /// * attaching an iterator materializes pending work, so `self` must
    ///   already be waited. Callers reach this through
    ///   [`Tensor::multi_pairs_in_me`], which waits first.
    #[must_use]
    pub fn hyper_vector_count(&self) -> Option<u64> {
        debug_assert!(
            !self.has_pending.load(Ordering::Relaxed),
            "hyper_vector_count on a pending matrix: the attach below would \
             materialize it, which is unsound on a shared layer"
        );
        let mut sparsity: i32 = 0;
        let info = unsafe {
            GrB_Matrix_get_INT32(
                self.inner(),
                &raw mut sparsity,
                GxB_Option_Field::GxB_SPARSITY_STATUS as _,
            )
        };
        debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        if sparsity != GxB_HYPERSPARSE as i32 {
            return None;
        }
        unsafe {
            let mut it: GxB_Iterator = null_mut();
            let info = GxB_Iterator_new(&raw mut it);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            let info = GxB_rowIterator_attach(it, self.inner(), null_mut());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            let kount = GxB_rowIterator_kount(it);
            let info = GxB_Iterator_free(&raw mut it);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            Some(kount)
        }
    }

    /// Transposes the matrix.
    ///
    /// # Returns
    /// A new matrix of the same GraphBLAS type that is the transpose of the
    /// original.
    #[must_use]
    pub fn transpose(&self) -> Self {
        unsafe {
            let mut type_: MaybeUninit<GrB_Type> = MaybeUninit::uninit();
            let info = GxB_Matrix_type(type_.as_mut_ptr(), self.inner());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            let mut m: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
            let info = GrB_Matrix_new(
                m.as_mut_ptr(),
                type_.assume_init(),
                self.ncols(),
                self.nrows(),
            );
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GrB_Matrix_new failed: {info:?}"
            );
            let m = m.assume_init();
            pin_sparse(m);
            let transpose = Self {
                m: Arc::new(Handle(m)),
                lock: Arc::new(Mutex::new(())),
                has_pending: Arc::new(AtomicBool::new(true)),
                phantom: PhantomData,
            };
            let info = GrB_transpose(
                transpose.inner(),
                null_mut(),
                null_mut(),
                self.inner(),
                null_mut(),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            transpose
        }
    }

    /// A copy of this matrix at larger dimensions, every entry at its original
    /// coordinate.
    ///
    /// Implemented as `dup` then `GrB_Matrix_resize`. It previously built the
    /// enlarged matrix with `GxB_Matrix_concat` and empty padding tiles, on the
    /// reasoning that resizing meant a copy at the old dims followed by a
    /// mutation that assembled pending work internally, whereas concat writes
    /// once straight into a matrix already at the target dims.
    ///
    /// Measured, that reasoning was wrong. Standalone, `dup` + `resize` costs
    /// 0.26-0.56x the concat path on an assembled matrix -- 0.059 ms against
    /// 0.214 at 10k rows, 1.90 ms against 3.41 at 1m -- and 0.71-0.87x on a
    /// pending one. (Measured against 10.3.1, the version linked at the time.)
    /// It produces an identical result: `eWiseAdd(MINUS)` over the two outputs
    /// finds no differing entry at 1k or 50k rows.
    ///
    /// GraphBLAS 10.5.0, now the linked version, lets `GB_resize` skip its wait
    /// entirely when the storage is sparse or hypersparse and neither dimension
    /// shrinks. That does not help here and is not why this is a `resize`: both
    /// callers `wait` the grown matrix immediately, because publishing an
    /// unmaterialized layer to an MVCC snapshot is unsound, so the skipped wait
    /// is paid one line later. Building 10.5.0 into the engine and measuring the
    /// suite moved it 0.9998x.
    ///
    /// Shrinking is not expressible this way -- it would drop entries -- so both
    /// dimensions must be `>=` the current ones.
    ///
    /// Unlike the concat formulation, `self` need not be waited, and is never
    /// mutated: `GB_dup` copies the source's pending tuples and zombies into the
    /// copy without finishing them ("Pending work in A is copied into C; it is
    /// not finished", `Source/dup/GB_dup.c`). That matters because `self` is a
    /// layer that may still be shared with a published snapshot, where the
    /// internal wait a GraphBLAS call makes on a pending input would be a
    /// mutation. `resize` then flags the copy pending, so the caller's `wait`
    /// assembles the whole thing once.
    #[must_use]
    pub fn grown(
        &self,
        nrows: u64,
        ncols: u64,
    ) -> Self {
        let (r0, c0) = (self.nrows(), self.ncols());
        assert!(
            nrows >= r0 && ncols >= c0,
            "grown must not shrink: {r0}x{c0} -> {nrows}x{ncols}"
        );
        let mut out = self.dup();
        if nrows != r0 || ncols != c0 {
            out.resize(nrows, ncols);
        }
        out
    }

    /// Returns the raw GrB_Matrix handle for FFI calls (e.g. LAGraph).
    /// The caller must NOT free the returned handle.
    #[must_use]
    pub fn inner(&self) -> GrB_Matrix {
        self.m.0
    }

    /// Whether an entry is stored at `(i, j)`, for **any** element type — a
    /// pure sparsity-pattern probe (`GxB_Matrix_isStoredElement`) that never
    /// reads or typecasts the element value.
    ///
    /// Pending updates may not be visible; call [`Self::wait`] first when the
    /// matrix may have queued mutations.
    #[must_use]
    pub fn contains(
        &self,
        i: u64,
        j: u64,
    ) -> bool {
        unsafe { GxB_Matrix_isStoredElement(self.inner(), i, j) == GrB_Info::GrB_SUCCESS }
    }

    /// Number of positions stored in both `self` and `b` (structural
    /// intersection size). `ANY_PAIR` only inspects the sparsity pattern, so
    /// the element types of the operands are never read.
    #[must_use]
    pub fn intersection_nvals<TB>(
        &self,
        b: &Matrix<TB>,
    ) -> u64 {
        let t = Matrix::<bool>::new(self.nrows(), self.ncols());
        unsafe {
            let info = GrB_Matrix_eWiseMult_Semiring(
                t.inner(),
                null_mut(),
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                self.inner(),
                b.inner(),
                null_mut(),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        t.nvals()
    }

    /// How many entries the rows `rows` selects hold: `Σ_{i ∈ rows} |A(i, :)|`.
    ///
    /// One `w = Aᵀ·x` over `PLUS_PAIR`, then a `PLUS` reduction of `w`. On a
    /// by-row matrix GraphBLAS computes the transposed product by pushing each
    /// selected row into `w`, so the cost follows `rows` and their lengths,
    /// not the size of the matrix — where `A·x` would visit every row. `PAIR`
    /// reads only the two patterns, so the element type of `self` does not
    /// matter. `rows` must be `nrows` long. Call [`Self::wait`] first if
    /// `self` may have queued mutations.
    #[must_use]
    pub fn count_in_rows(
        &self,
        rows: &Vector<bool>,
    ) -> u64 {
        let w = Vector::<u64>::new(self.ncols());
        unsafe {
            let info = GrB_mxv(
                w.ptr(),
                null_mut(),
                null_mut(),
                GxB_PLUS_PAIR_UINT64,
                self.inner(),
                rows.ptr(),
                GrB_DESC_T0,
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        w.sum()
    }

    #[must_use]
    pub fn pending(&self) -> bool {
        unsafe {
            let mut pending = MaybeUninit::uninit();
            let info = GrB_Matrix_get_INT32(
                self.inner(),
                pending.as_mut_ptr(),
                GxB_Option_Field::GxB_WILL_WAIT as _,
            );
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GrB_Matrix_get_INT32 failed: {info:?}"
            );
            pending.assume_init() == 1
        }
    }

    pub fn wait(&self) {
        if !self.has_pending.load(Ordering::Acquire) {
            return;
        }
        let lock = self.lock.lock();
        if !self.has_pending.load(Ordering::Relaxed) {
            drop(lock);
            return;
        }
        unsafe {
            let info = GrB_Matrix_wait(self.inner(), GrB_WaitMode::GrB_MATERIALIZE as _);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(false, Ordering::Release);
        drop(lock);
    }

    /// Returns true if this matrix has no pending GraphBLAS operations,
    /// i.e. is in a state where the fork child can safely serialize it
    /// without first calling `wait()`. Cheap atomic load.
    #[must_use]
    pub fn is_synced(&self) -> bool {
        !self.has_pending.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn memory_usage(&self) -> usize {
        unsafe {
            let mut usage = 0usize;
            let info = GxB_Matrix_memoryUsage(&raw mut usage, self.inner());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            usage
        }
    }

    /// The index widths GraphBLAS chose for this matrix, as
    /// `(row bits, column bits)`. Read-only in GraphBLAS and derived from the
    /// declared dimensions, so this is how a shape choice is checked rather
    /// than assumed.
    #[cfg(test)]
    pub(super) fn integer_bits_for_test(&self) -> (i32, i32) {
        unsafe {
            let (mut r, mut c) = (0i32, 0i32);
            // Checked: a failed getter leaves the initialised zero behind, and
            // reporting "0-bit indices" would read as a result rather than an
            // error.
            let ri = GrB_Matrix_get_INT32(
                self.inner(),
                &raw mut r,
                GxB_Option_Field::GxB_ROWINDEX_INTEGER_BITS as i32,
            );
            assert_eq!(ri, GrB_Info::GrB_SUCCESS, "row index bits: {ri:?}");
            let ci = GrB_Matrix_get_INT32(
                self.inner(),
                &raw mut c,
                GxB_Option_Field::GxB_COLINDEX_INTEGER_BITS as i32,
            );
            assert_eq!(ci, GrB_Info::GrB_SUCCESS, "column index bits: {ci:?}");
            (r, c)
        }
    }

    /// Ask GraphBLAS for 32-bit row and column index arrays. Only a *hint*: it
    /// is honoured when the matrix's declared dimensions fit, and ignored when
    /// they do not, so a caller cannot make indices too narrow for the data.
    #[cfg(test)]
    pub(super) fn hint_32bit_indices_for_test(&mut self) -> (i32, i32) {
        unsafe {
            let r = GrB_Matrix_set_INT32(
                self.inner(),
                32,
                GxB_Option_Field::GxB_ROWINDEX_INTEGER_HINT as i32,
            );
            let c = GrB_Matrix_set_INT32(
                self.inner(),
                32,
                GxB_Option_Field::GxB_COLINDEX_INTEGER_HINT as i32,
            );
            (r as i32, c as i32)
        }
    }

    /// Ask GraphBLAS globally for 32-bit indices on matrices created after this
    /// point. Returns the two status codes.
    #[cfg(test)]
    pub(super) fn global_hint_32bit_for_test() -> (i32, i32) {
        unsafe {
            let r = GrB_Global_set_INT32(
                GrB_GLOBAL,
                32,
                GxB_Option_Field::GxB_ROWINDEX_INTEGER_HINT as i32,
            );
            let c = GrB_Global_set_INT32(
                GrB_GLOBAL,
                32,
                GxB_Option_Field::GxB_COLINDEX_INTEGER_HINT as i32,
            );
            (r as i32, c as i32)
        }
    }

    pub fn clear(&mut self) {
        unsafe {
            let info = GrB_Matrix_clear(self.inner());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(false, Ordering::Relaxed);
    }

    pub fn remove_all<U>(
        &mut self,
        b: &Matrix<U>,
    ) {
        unsafe {
            let info = GrB_transpose(
                self.inner(),
                b.inner(),
                null_mut(),
                self.inner(),
                GrB_DESC_RCT0,
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    /// Remove every entry on the main diagonal (`GrB_select` with
    /// `GrB_OFFDIAG`, whose thunk 0 names the main diagonal), in place.
    /// So far only the CCH test kit uses it, to drop a weight matrix's self-loops.
    #[cfg(test)]
    pub(crate) fn drop_diagonal(&mut self) {
        let info = unsafe {
            GrB_Matrix_select_INT64(
                self.inner(),
                null_mut(),
                null_mut(),
                GrB_OFFDIAG,
                self.inner(),
                0,
                null_mut(),
            )
        };
        assert_eq!(
            info,
            GrB_Info::GrB_SUCCESS,
            "GrB_Matrix_select_INT64: {info:?}"
        );
        self.has_pending.store(true, Ordering::Relaxed);
    }

    pub fn select(
        &mut self,
        mask: &Matrix<bool>,
        a: &Self,
    ) {
        unsafe {
            let info = GrB_transpose(
                self.inner(),
                mask.inner(),
                null_mut(),
                a.inner(),
                GrB_DESC_RCT0,
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    /// `self<mask> = a ⊕ b`, with `a` and `b` defaulting to `self`.
    ///
    /// The `⊕` comes from the output's element type via [`EWiseAdd`]: pattern
    /// union for `bool`, value-preserving `SECOND` for `u64`, so a delta layer
    /// merged into its base keeps the delta's (live) value on a shadowed pair.
    pub fn element_wise_add<TB>(
        &mut self,
        mask: Option<&Matrix<bool>>,
        a: Option<&Self>,
        b: Option<&Matrix<TB>>,
        descriptor: Option<Descriptor>,
    ) where
        T: EWiseAdd,
    {
        unsafe {
            let info = GrB_Matrix_eWiseAdd_BinaryOp(
                self.inner(),
                mask.map_or(null_mut(), |m| m.inner()),
                null_mut(),
                T::add_op(),
                a.map_or(self.inner(), |a| a.inner()),
                b.map_or(self.inner(), |b| b.inner()),
                descriptor.map_or(null_mut(), std::convert::Into::into),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    pub fn element_wise_multiply<TB>(
        &mut self,
        mask: Option<&Matrix<bool>>,
        a: Option<&Matrix<bool>>,
        b: Option<&Matrix<TB>>,
        descriptor: Option<Descriptor>,
    ) {
        unsafe {
            let info = GrB_Matrix_eWiseMult_Semiring(
                self.inner(),
                mask.map_or(null_mut(), |m| m.inner()),
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                a.map_or(self.inner(), |a| a.inner()),
                b.map_or(self.inner(), |b| b.inner()),
                descriptor.map_or(null_mut(), std::convert::Into::into),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    /// `self<mask> ∪= pattern(a)`: union `a`'s sparsity pattern into `self`
    /// as all-`true` entries (`GrB_Matrix_apply` with `GxB_ONE_BOOL`, which
    /// never reads `a`'s values).
    ///
    /// Use this instead of a single-sided `element_wise_add` when `a` is a
    /// valued matrix: eWiseAdd *typecasts* single-side entries, so a `u64`
    /// edge id of 0 would land as `false` — invisible to valued masks and a
    /// source of corruption.
    pub fn set_pattern<TB>(
        &mut self,
        mask: Option<&Matrix<bool>>,
        a: &Matrix<TB>,
        descriptor: Option<Descriptor>,
    ) {
        unsafe {
            let info = GrB_Matrix_apply(
                self.inner(),
                mask.map_or(null_mut(), |m| m.inner()),
                GxB_ANY_BOOL,
                GxB_ONE_BOOL,
                a.inner(),
                descriptor.map_or(null_mut(), std::convert::Into::into),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    /// `self = self * b` over the `ANY_PAIR_BOOL` semiring. Accepts a matrix
    /// of any element type: `PAIR` only inspects the sparsity pattern, so the
    /// values of `b` (e.g. inline edge ids in a UINT64 relationship matrix)
    /// are never read.
    pub fn lmxm<TB>(
        &mut self,
        b: &Matrix<TB>,
    ) {
        unsafe {
            let info = GrB_mxm(
                self.inner(),
                null_mut(),
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                self.inner(),
                b.inner(),
                null_mut(),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    /// `self = b * self` over the `ANY_PAIR_BOOL` semiring. Accepts a matrix
    /// of any element type (see [`Self::lmxm`]).
    pub fn rmxm<TB>(
        &mut self,
        b: &Matrix<TB>,
    ) {
        unsafe {
            let info = GrB_mxm(
                self.inner(),
                null_mut(),
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                b.inner(),
                self.inner(),
                null_mut(),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    #[must_use]
    pub fn nrows(&self) -> u64 {
        unsafe {
            let mut nrows = 0u64;
            let info = GrB_Matrix_nrows(&raw mut nrows, self.inner());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            nrows
        }
    }

    #[must_use]
    pub fn ncols(&self) -> u64 {
        unsafe {
            let mut ncols = 0u64;
            let info = GrB_Matrix_ncols(&raw mut ncols, self.inner());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            ncols
        }
    }

    pub fn resize(
        &mut self,
        nrows: u64,
        ncols: u64,
    ) {
        unsafe {
            let info = GrB_Matrix_resize(self.inner(), nrows, ncols);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    /// Which storage GraphBLAS currently holds this matrix in. Diagnostic: the
    /// choice decides whether `GrB_Matrix_resize` can take its no-wait fast
    /// path, which needs sparse or hypersparse.
    #[must_use]
    pub fn sparsity_status(&self) -> &'static str {
        let mut sparsity: i32 = 0;
        let info = unsafe {
            GrB_Matrix_get_INT32(
                self.inner(),
                &raw mut sparsity,
                GxB_Option_Field::GxB_SPARSITY_STATUS as _,
            )
        };
        debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        match sparsity as u32 {
            GxB_HYPERSPARSE => "hypersparse",
            GxB_SPARSE => "sparse",
            GxB_BITMAP => "bitmap",
            GxB_FULL => "full",
            _ => "unknown",
        }
    }

    #[must_use]
    pub fn nvals(&self) -> u64 {
        unsafe {
            let mut nvals = 0u64;
            let info = GrB_Matrix_nvals(&raw mut nvals, self.inner());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            nvals
        }
    }

    pub fn remove(
        &mut self,
        i: u64,
        j: u64,
    ) {
        unsafe {
            let info = GrB_Matrix_removeElement(self.inner(), i, j);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    pub fn print(
        &self,
        level: GxB_Print_Level,
    ) {
        unsafe {
            let info = GxB_Matrix_fprint(self.inner(), null_mut(), level as _, null_mut());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
    }
}

pub trait Dup<T> {
    fn dup(&self) -> T;
}

impl<T> Dup<Self> for Matrix<T> {
    fn dup(&self) -> Self {
        // Serialize against concurrent `wait()` on a shared-handle clone
        // (e.g. BGSAVE main thread calling `wait_all` on the committed
        // snapshot while a writer thread duplicates the same Matrix to
        // build a new MVCC version). `Cow<Matrix>::new_version` performs
        // a shallow clone, so both Cow instances share `lock` and
        // `has_pending` — taking the lock here closes the race with
        // `wait()`, which takes the same lock before mutating GrB state.
        let pending = self.has_pending.load(Ordering::Acquire);
        let _guard = if pending {
            Some(self.lock.lock())
        } else {
            None
        };
        let dup_pending = if pending {
            self.has_pending.load(Ordering::Relaxed)
        } else {
            false
        };
        Self {
            m: Arc::new(Handle(unsafe {
                let mut m: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
                let info = GrB_Matrix_dup(m.as_mut_ptr(), self.inner());
                assert_eq!(
                    info,
                    GrB_Info::GrB_SUCCESS,
                    "GrB_Matrix_dup failed: {info:?}"
                );
                let m = m.assume_init();
                // GrB_Matrix_dup copies sparsity_control but resets
                // no_hyper_hash (GB_new), so delta matrices would regain the
                // per-commit O(nvec) hyper-hash rebuild after their first COW
                // dup — carry the opt-out over explicitly.
                let mut hyper_hash: i32 = 1;
                let info = GrB_Matrix_get_INT32(
                    self.inner(),
                    &raw mut hyper_hash,
                    GxB_Option_Field::GxB_HYPER_HASH as _,
                );
                debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
                if hyper_hash == 0 {
                    let info = GrB_Matrix_set_INT32(m, 0, GxB_Option_Field::GxB_HYPER_HASH as _);
                    debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
                }
                m
            })),
            lock: Arc::new(Mutex::new(())),
            has_pending: Arc::new(AtomicBool::new(dup_pending)),
            phantom: PhantomData,
        }
    }
}

impl Matrix<u64> {
    /// How many stored entries equal `value`: `GrB_select` with
    /// `GrB_VALUEEQ_UINT64` into a scratch matrix, then its `nvals`.
    #[must_use]
    pub fn count_value_eq(
        &self,
        value: u64,
    ) -> u64 {
        let out = Self::new(self.nrows(), self.ncols());
        unsafe {
            let info = GrB_Matrix_select_UINT64(
                out.inner(),
                null_mut(),
                null_mut(),
                GrB_VALUEEQ_UINT64,
                self.inner(),
                value,
                null_mut(),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        out.nvals()
    }

    /// Create a new UINT64 matrix (for C-compatible tensor encoding and inline
    /// edge-id storage).
    pub fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        unsafe {
            let mut m: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
            let info = GrB_Matrix_new(m.as_mut_ptr(), GrB_UINT64, nrows, ncols);
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GrB_Matrix_new failed: {info:?}"
            );
            let m = m.assume_init();
            pin_sparse(m);
            Self {
                m: Arc::new(Handle(m)),
                lock: Arc::new(Mutex::new(())),
                has_pending: Arc::new(AtomicBool::new(false)),
                phantom: PhantomData,
            }
        }
    }

    /// Set a UINT64 value at (i, j).
    pub fn set(
        &mut self,
        i: u64,
        j: u64,
        value: u64,
    ) {
        unsafe {
            let info = GrB_Matrix_setElement_UINT64(self.inner(), value, i, j);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    /// Read the UINT64 value at (i, j). Returns `None` if no entry exists.
    #[must_use]
    pub fn get(
        &self,
        i: u64,
        j: u64,
    ) -> Option<u64> {
        unsafe {
            let mut val: MaybeUninit<u64> = MaybeUninit::uninit();
            let info = GrB_Matrix_extractElement_UINT64(val.as_mut_ptr(), self.inner(), i, j);
            if info == GrB_Info::GrB_SUCCESS {
                Some(val.assume_init())
            } else {
                None
            }
        }
    }

    /// UINT64 row-range iterator yielding `(row, col, value)` triples over rows
    /// in `[min_row, max_row]`. Supports `seek` for amortized per-row scans.
    #[must_use]
    pub fn iter(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> Iter<Uint64Extract> {
        Iter::new(self, min_row, max_row)
    }

    /// Bulk-insert UINT64 entries from (row, col, val) arrays. Matrix must be empty and UINT64 typed.
    pub fn build(
        &mut self,
        rows: &[u64],
        cols: &[u64],
        vals: &[u64],
    ) {
        debug_assert_eq!(rows.len(), cols.len());
        debug_assert_eq!(rows.len(), vals.len());
        if rows.is_empty() {
            return;
        }
        let nvals = rows.len() as u64;
        unsafe {
            let info = GrB_Matrix_build_UINT64(
                self.inner(),
                rows.as_ptr(),
                cols.as_ptr(),
                vals.as_ptr(),
                nvals,
                GxB_ANY_UINT64,
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }
}

/// `FP64` matrices, e.g. a weight per node pair. So far only the test kit's oracle uses them:
/// a test graph is a weight matrix.
#[cfg(test)]
impl Matrix<f64> {
    /// Create a new `FP64` matrix.
    #[must_use]
    pub fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        unsafe {
            let mut m: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
            let info = GrB_Matrix_new(m.as_mut_ptr(), GrB_FP64, nrows, ncols);
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GrB_Matrix_new failed: {info:?}"
            );
            let m = m.assume_init();
            pin_sparse(m);
            Self {
                m: Arc::new(Handle(m)),
                lock: Arc::new(Mutex::new(())),
                has_pending: Arc::new(AtomicBool::new(false)),
                phantom: PhantomData,
            }
        }
    }

    /// Set an `FP64` value at `(i, j)`.
    ///
    /// An invalid index (outside the dimensions) or out of memory panics, in
    /// release too: any error `GrB_Info`, one below 0.
    pub fn set(
        &mut self,
        i: u64,
        j: u64,
        value: f64,
    ) {
        unsafe {
            let info = GrB_Matrix_setElement_FP64(self.inner(), value, i, j);
            assert!((info as i32) >= 0, "GrB_Matrix_setElement_FP64: {info:?}");
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    /// Build from coordinate arrays into this (empty) matrix. A repeated `(row, col)` is a bug
    /// in the caller, not something to fold: there is no `dup` operator, so GraphBLAS refuses
    /// it, and this panics, as it does on an index outside the dimensions.
    ///
    /// Both checks hold in release: mismatched lengths would read past an array, and a
    /// refused build would leave the matrix silently empty.
    pub(crate) fn build(
        &mut self,
        rows: &[u64],
        cols: &[u64],
        vals: &[f64],
    ) {
        assert!(rows.len() == cols.len() && rows.len() == vals.len());
        let info = unsafe {
            GrB_Matrix_build_FP64(
                self.inner(),
                rows.as_ptr(),
                cols.as_ptr(),
                vals.as_ptr(),
                rows.len() as u64,
                null_mut(),
            )
        };
        assert_eq!(
            info,
            GrB_Info::GrB_SUCCESS,
            "GrB_Matrix_build_FP64: {info:?}"
        );
        self.has_pending.store(true, Ordering::Relaxed);
    }

    /// Every stored entry as parallel row, column and value arrays (`extractTuples`), in the
    /// order GraphBLAS stores them: row by row, columns ascending.
    #[must_use]
    pub(crate) fn extract_tuples(&self) -> (Vec<u64>, Vec<u64>, Vec<f64>) {
        self.wait();
        let mut count = self.nvals();
        let mut rows = vec![0; count as usize];
        let mut cols = vec![0; count as usize];
        let mut vals = vec![0.0; count as usize];
        let info = unsafe {
            GrB_Matrix_extractTuples_FP64(
                rows.as_mut_ptr(),
                cols.as_mut_ptr(),
                vals.as_mut_ptr(),
                &raw mut count,
                self.inner(),
            )
        };
        assert_eq!(
            info,
            GrB_Info::GrB_SUCCESS,
            "GrB_Matrix_extractTuples_FP64: {info:?}"
        );
        rows.truncate(count as usize);
        cols.truncate(count as usize);
        vals.truncate(count as usize);
        (rows, cols, vals)
    }
}

impl Matrix<bool> {
    pub fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self {
        unsafe {
            let mut m: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
            let info = GrB_Matrix_new(m.as_mut_ptr(), GrB_BOOL, nrows, ncols);
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GrB_Matrix_new failed: {info:?}"
            );
            let m = m.assume_init();
            pin_sparse(m);
            Self {
                m: Arc::new(Handle(m)),
                lock: Arc::new(Mutex::new(())),
                has_pending: Arc::new(AtomicBool::new(false)),
                phantom: PhantomData,
            }
        }
    }

    /// Retrieves the boolean value at the specified position in the matrix.
    /// Returns `None` if the element does not exist.
    ///
    /// # Parameters
    /// - `i`: The row index.
    /// - `j`: The column index.
    ///
    /// # Returns
    /// - `Some(bool)`: The boolean value at the specified position.
    /// - `None`: The element does not exist.
    #[must_use]
    pub fn get(
        &self,
        i: u64,
        j: u64,
    ) -> Option<bool> {
        unsafe {
            let mut m: MaybeUninit<bool> = MaybeUninit::uninit();
            let info = GrB_Matrix_extractElement_BOOL(m.as_mut_ptr(), self.inner(), i, j);
            if info == GrB_Info::GrB_SUCCESS {
                Some(m.assume_init())
            } else {
                None
            }
        }
    }

    pub fn set(
        &mut self,
        i: u64,
        j: u64,
        value: bool,
    ) {
        unsafe {
            let info = GrB_Matrix_setElement_BOOL(self.inner(), value, i, j);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    /// Set every pair in `rows` x `cols` to `true`, in one GraphBLAS call.
    ///
    /// `GrB_Matrix_assign_BOOL` takes the two index lists and forms the
    /// cartesian product itself, so the product is never enumerated on this
    /// side — which is the whole reason for the method. Setting a million
    /// nodes' single label went from a million `setElement` calls to one.
    ///
    /// No mask, no accumulator and no descriptor: assign writes `x` at every
    /// `(i, j)` in `I x J` and leaves everything outside that block alone,
    /// which is the union this wants. Duplicates in either list are harmless —
    /// assigning `true` twice is assigning `true`.
    pub fn assign_product_true(
        &mut self,
        rows: &[u64],
        cols: &[u64],
    ) {
        if rows.is_empty() || cols.is_empty() {
            return;
        }
        unsafe {
            let info = GrB_Matrix_assign_BOOL(
                self.inner(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                true,
                rows.as_ptr(),
                rows.len() as u64,
                cols.as_ptr(),
                cols.len() as u64,
                std::ptr::null_mut(),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    /// Bulk-insert entries from (row, col) arrays. Matrix must be empty.
    /// Uses a single GraphBLAS FFI call instead of N individual setElement
    /// calls; the scalar variant needs no values array and produces an iso
    /// matrix (pattern only, one shared value).
    pub fn build(
        &mut self,
        rows: &[u64],
        cols: &[u64],
    ) {
        debug_assert_eq!(rows.len(), cols.len());
        if rows.is_empty() {
            return;
        }
        let nvals = rows.len() as u64;
        unsafe {
            let mut scalar: GrB_Scalar = null_mut();
            let mut info = GrB_Scalar_new(&raw mut scalar, GrB_BOOL);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            info = GrB_Scalar_setElement_BOOL(scalar, true);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            info =
                GxB_Matrix_build_Scalar(self.inner(), rows.as_ptr(), cols.as_ptr(), scalar, nvals);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            info = GrB_Scalar_free(&raw mut scalar);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    /// Delta-aware matrix-multiply: `self = self * (m, dp, dm)` operating
    /// directly on the base/dp/dm layers of a versioned matrix (or a
    /// `Tensor`'s forward adjacency), mirroring C FalkorDB's `Delta_mxm`.
    ///
    /// Computes `(self * (m + dp))<!(self * dm)>` without first materializing
    /// the merged matrix. In the common read-only case (`dp.nvals() == 0 &&
    /// dm.nvals() == 0`) this is a single `GrB_mxm` against `m`, avoiding
    /// the eWiseAdd that a materialized merge would otherwise pay.
    ///
    /// Accepts layers of any element type: the `ANY_PAIR_BOOL` semiring only
    /// inspects the sparsity pattern, so a UINT64 relationship matrix (inline
    /// edge-id values) traverses identically to a BOOL one.
    pub fn delta_lmxm<TV>(
        &mut self,
        m: &Matrix<TV>,
        dp: &Matrix<TV>,
        dm: &Matrix<bool>,
    ) {
        // The delta layers arrive raw from a shared snapshot and may hold
        // pending work; a GrB op would finish it internally (a mutation),
        // racing other readers on the same handles. Materialize through the
        // mutex-guarded wait first — a no-op (one atomic load) when synced.
        // `m` needs no wait: committed bases are synced at MVCC commit.
        dp.wait();
        dm.wait();
        let dp_nvals = dp.nvals();
        let dm_nvals = dm.nvals();

        if dp_nvals == 0 && dm_nvals == 0 {
            // Hot path: clean snapshot, just self * m
            self.lmxm(m);
            return;
        }

        let nrows = self.nrows();
        let ncols = m.ncols();

        // `self * dm`, counted: how many deleted base edges reach each output.
        // Its pattern is the mask; the counts are what make the mask exact.
        //
        // The mask alone is exact only for a row of `self` with one entry. A
        // row with several (a fused chain's frontier) reaches an output through
        // several base edges, and one of them being deleted masked the output
        // even when another was live: `x→z` deleted and `y→z` live, with `x`
        // and `y` in one row, lost `z`. Since `dm ⊆ m`, an output under the
        // mask is live exactly when `self * m` reaches it by more paths than
        // `self * dm` does, so those are counted at the masked positions only
        // and added back.
        let mut mask: Option<Matrix<u64>> = None;
        let mut restored: Option<Matrix<bool>> = None;
        if dm_nvals > 0 {
            let dm_paths = Matrix::<u64>::new(nrows, ncols);
            unsafe {
                let info = GrB_mxm(
                    dm_paths.inner(),
                    null_mut(),
                    null_mut(),
                    GxB_PLUS_PAIR_UINT64,
                    self.inner(),
                    dm.inner(),
                    null_mut(),
                );
                debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            }
            if dm_paths.nvals() > 0 {
                let m_paths = Matrix::<u64>::new(nrows, ncols);
                let live = Matrix::<bool>::new(nrows, ncols);
                unsafe {
                    let info = GrB_mxm(
                        m_paths.inner(),
                        dm_paths.inner(),
                        null_mut(),
                        GxB_PLUS_PAIR_UINT64,
                        self.inner(),
                        m.inner(),
                        GrB_DESC_S,
                    );
                    debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
                    let info = GrB_Matrix_eWiseMult_BinaryOp(
                        live.inner(),
                        null_mut(),
                        null_mut(),
                        GrB_GT_UINT64,
                        m_paths.inner(),
                        dm_paths.inner(),
                        null_mut(),
                    );
                    debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
                    let info = GrB_Matrix_select_BOOL(
                        live.inner(),
                        null_mut(),
                        null_mut(),
                        GrB_VALUEEQ_BOOL,
                        live.inner(),
                        true,
                        null_mut(),
                    );
                    debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
                }
                if live.nvals() > 0 {
                    restored = Some(live);
                }
                mask = Some(dm_paths);
            }
        }

        let mut accum: Option<Matrix<bool>> = None;
        if dp_nvals > 0 {
            let mut ac = Matrix::<bool>::new(nrows, ncols);
            unsafe {
                let info = GrB_mxm(
                    ac.inner(),
                    null_mut(),
                    null_mut(),
                    GxB_ANY_PAIR_BOOL,
                    self.inner(),
                    dp.inner(),
                    null_mut(),
                );
                debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            }
            if ac.nvals() > 0 {
                accum = Some(ac);
            }
        }

        unsafe {
            let (mask_ptr, desc) = mask
                .as_ref()
                .map_or((null_mut(), null_mut()), |m| (m.inner(), GrB_DESC_RSC));
            let info = GrB_mxm(
                self.inner(),
                mask_ptr,
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                self.inner(),
                m.inner(),
                desc,
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }

        if let Some(live) = restored {
            self.element_wise_add(None, None, Some(&live), None);
        }
        if let Some(ac) = accum {
            self.element_wise_add(None, None, Some(&ac), None);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    #[must_use]
    pub fn iter(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> Iter<BoolExtract> {
        Iter::new(self, min_row, max_row)
    }
}

/// Strategy for extracting values from a GraphBLAS row iterator position.
///
/// # Safety
/// Implementations must only call valid GraphBLAS FFI functions on the provided matrix.
pub trait IterExtract {
    type Item;

    /// Extract the item at the iterator's current position (an O(1) cursor
    /// read — no per-entry matrix lookup).
    ///
    /// # Safety
    /// `it` must be a valid attached `GxB_Iterator` positioned on a valid entry.
    unsafe fn extract(it: GxB_Iterator) -> Self::Item;

    /// `(row, col)` position of an item, for sorted-merge ordering.
    fn pos(item: &Self::Item) -> (u64, u64);
}

/// Extracts `(row, col)` pairs from a boolean matrix.
pub struct BoolExtract;

impl IterExtract for BoolExtract {
    type Item = (u64, u64);

    unsafe fn extract(it: GxB_Iterator) -> Self::Item {
        unsafe {
            let row = GxB_rowIterator_getRowIndex(it);
            let col = GxB_rowIterator_getColIndex(it);
            (row, col)
        }
    }

    fn pos(item: &Self::Item) -> (u64, u64) {
        *item
    }
}

/// Extracts `(row, col, value)` triples from a UINT64 matrix.
pub struct Uint64Extract;

impl IterExtract for Uint64Extract {
    type Item = (u64, u64, u64);

    unsafe fn extract(it: GxB_Iterator) -> Self::Item {
        unsafe {
            let row = GxB_rowIterator_getRowIndex(it);
            let col = GxB_rowIterator_getColIndex(it);
            let val = GxB_Iterator_get_UINT64(it);
            (row, col, val)
        }
    }

    fn pos(item: &Self::Item) -> (u64, u64) {
        (item.0, item.1)
    }
}

pub struct Iter<E: IterExtract = BoolExtract> {
    m: Arc<Handle>,
    /// The underlying GraphBLAS iterator.
    inner: GxB_Iterator,
    /// Indicates whether the iterator is depleted.
    depleted: bool,
    /// The maximum row index for the iterator.
    max_row: u64,
    _extract: PhantomData<E>,
}

unsafe impl<E: IterExtract> Send for Iter<E> {}
unsafe impl<E: IterExtract> Sync for Iter<E> {}

impl<E: IterExtract> Drop for Iter<E> {
    /// Frees the GraphBLAS iterator when the `Iter` is dropped. The matrix
    /// is freed by its [`Handle`] if this was the last reference, after the
    /// iterator attached to it (fields drop after this body).
    fn drop(&mut self) {
        unsafe {
            if !self.inner.is_null() {
                GxB_Iterator_free(&raw mut self.inner);
            }
        }
    }
}

impl<E: IterExtract> Iter<E> {
    /// The matrix this iterator walks.
    fn matrix(&self) -> GrB_Matrix {
        self.m.0
    }

    /// Creates a new iterator for traversing all elements in a matrix.
    ///
    /// # Parameters
    /// - `m`: The matrix to iterate over.
    /// - `min_row`: The minimum row index to start iterating from.
    /// - `max_row`: The maximum row index to stop iterating at.
    #[must_use]
    pub fn new<T>(
        m: &Matrix<T>,
        min_row: u64,
        max_row: u64,
    ) -> Self {
        let mut it = Self::detached(m);
        it.seek(min_row, max_row);
        it
    }

    /// An iterator over `m` with no `GxB_Iterator` behind it yet: it yields
    /// nothing until [`Self::seek`] attaches one.
    ///
    /// `GxB_Iterator_new` + `GxB_rowIterator_attach` + the matching free cost
    /// about 1,700 instructions, which is most of what a single-row scan of a
    /// small matrix pays. A caller that has established the range holds no
    /// entry can skip all of it and still hand back a real iterator — one that
    /// attaches if it is ever re-seeked somewhere the answer differs, so
    /// skipping stays an optimisation rather than a promise the caller has to
    /// keep.
    #[must_use]
    pub fn detached<T>(m: &Matrix<T>) -> Self {
        Self {
            m: m.m.clone(),
            inner: null_mut(),
            depleted: true,
            max_row: 0,
            _extract: PhantomData,
        }
    }

    /// Attach the GraphBLAS iterator if this is still detached.
    fn attach(&mut self) {
        if !self.inner.is_null() {
            return;
        }
        unsafe {
            let mut iter = MaybeUninit::uninit();
            let info = GxB_Iterator_new(iter.as_mut_ptr());
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GxB_Iterator_new failed: {info:?}"
            );
            let iter = iter.assume_init();
            let info = GxB_rowIterator_attach(iter, self.matrix(), null_mut());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            self.inner = iter;
        }
    }
}

impl<E: IterExtract> Iter<E> {
    /// Re-seek an existing iterator to a new row range without re-allocating
    /// the underlying GxB_Iterator. Used by hot-loop callers (e.g.
    /// `CondTraverseOp`) to amortize the iterator allocation across many
    /// per-row scans of the same matrix.
    pub fn seek(
        &mut self,
        min_row: u64,
        max_row: u64,
    ) {
        self.attach();
        unsafe {
            let mut info = GxB_rowIterator_seekRow(self.inner, min_row);
            debug_assert!(
                info == GrB_Info::GrB_SUCCESS
                    || info == GrB_Info::GrB_NO_VALUE
                    || info == GrB_Info::GxB_EXHAUSTED
            );
            while info == GrB_Info::GrB_NO_VALUE
                && GxB_rowIterator_getRowIndex(self.inner) < max_row
            {
                info = GxB_rowIterator_nextRow(self.inner);
            }
            self.max_row = max_row;
            self.depleted =
                info != GrB_Info::GrB_SUCCESS || GxB_rowIterator_getRowIndex(self.inner) > max_row;
        }
    }
}

impl<E: IterExtract> Iterator for Iter<E> {
    type Item = E::Item;

    /// Advances the iterator and returns the next element in the matrix.
    ///
    /// # Returns
    /// - `Some(E::Item)`: The next element in the matrix.
    /// - `None`: The iterator is depleted.
    fn next(&mut self) -> Option<Self::Item> {
        if self.depleted {
            return None;
        }
        unsafe {
            let item = E::extract(self.inner);
            if GxB_rowIterator_nextCol(self.inner) != GrB_Info::GrB_SUCCESS {
                let mut info = GxB_rowIterator_nextRow(self.inner);
                debug_assert!(
                    info == GrB_Info::GrB_SUCCESS
                        || info == GrB_Info::GrB_NO_VALUE
                        || info == GrB_Info::GxB_EXHAUSTED
                );
                while info == GrB_Info::GrB_NO_VALUE
                    && GxB_rowIterator_getRowIndex(self.inner) < self.max_row
                {
                    info = GxB_rowIterator_nextRow(self.inner);
                }
                self.depleted = info != GrB_Info::GrB_SUCCESS
                    || GxB_rowIterator_getRowIndex(self.inner) > self.max_row;
            }
            Some(item)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, VecDeque};

    use super::super::serialization::{Decode, DecodeLimits, Encode, Reader, Writer};
    use super::super::test_init::ensure_init;
    use super::{GrB_Orientation, Matrix, OFF_FORMAT};

    /// The buffers a matrix writes, kept in order so a test can edit one and
    /// replay the stream into [`Matrix::decode`]. A matrix writes only buffers,
    /// plus three scalars per component vector.
    #[derive(Clone)]
    enum Op {
        Unsigned(u64),
        Signed(i64),
        Buffer(Vec<u8>),
    }

    #[derive(Default)]
    struct Tape(Vec<Op>);

    impl Writer for Tape {
        fn write_unsigned(
            &mut self,
            val: u64,
        ) {
            self.0.push(Op::Unsigned(val));
        }
        fn write_signed(
            &mut self,
            val: i64,
        ) {
            self.0.push(Op::Signed(val));
        }
        fn write_double(
            &mut self,
            _: f64,
        ) {
            unreachable!("a matrix writes no doubles");
        }
        fn write_buffer(
            &mut self,
            data: &[u8],
        ) {
            self.0.push(Op::Buffer(data.to_vec()));
        }
    }

    struct Replay(VecDeque<Op>);

    impl Reader for Replay {
        fn read_unsigned(&mut self) -> Result<u64, String> {
            match self.0.pop_front() {
                Some(Op::Unsigned(v)) => Ok(v),
                _ => Err("replay: expected unsigned".to_string()),
            }
        }
        fn read_signed(&mut self) -> Result<i64, String> {
            match self.0.pop_front() {
                Some(Op::Signed(v)) => Ok(v),
                _ => Err("replay: expected signed".to_string()),
            }
        }
        fn read_double(&mut self) -> Result<f64, String> {
            Err("replay: no doubles".to_string())
        }
        fn read_buffer(&mut self) -> Result<Vec<u8>, String> {
            match self.0.pop_front() {
                Some(Op::Buffer(v)) => Ok(v),
                _ => Err("replay: expected buffer".to_string()),
            }
        }
    }

    /// A 64 x 64 matrix of 128 entries, as the stream `encode` writes.
    fn encoded() -> Vec<Op> {
        let mut m = Matrix::<u64>::new(64, 64);
        let (rows, cols): (Vec<u64>, Vec<u64>) =
            (0..64u64).flat_map(|i| [(i, i), (i, (i + 1) % 64)]).unzip();
        let vals: Vec<u64> = (0..128).collect();
        m.build(&rows, &cols, &vals);
        m.wait();
        let mut tape = Tape::default();
        m.encode(&mut tape);
        tape.0
    }

    fn decode_u64(ops: Vec<Op>) -> Result<Matrix<u64>, String> {
        Matrix::<u64>::decode(&mut Replay(ops.into()))
    }

    /// The component vectors follow the container buffer in the order x, h, p,
    /// i, b; each is its array, type name, then three scalars. Index of the
    /// array of component `k`.
    fn array_of(k: usize) -> usize {
        1 + 5 * k
    }

    #[test]
    fn decode_round_trips() {
        ensure_init();
        let m = decode_u64(encoded()).unwrap();
        assert_eq!((m.nrows(), m.ncols(), m.nvals()), (64, 64, 128));
        assert_eq!(m.iter(0, u64::MAX).count(), 128);
    }

    /// `GxB_load_Matrix_from_Container` checks that the arrays are long enough,
    /// not that what is in them is a valid matrix. An index out of range made
    /// `Tensor::decode` read off the end of the value array and crash the server.
    #[test]
    fn decode_refuses_a_corrupt_structure() {
        ensure_init();
        // x, h, p, i, b: the row indices are the fourth vector.
        for (what, k, byte) in [("index array", 3, 0xFF), ("pointer array", 2, 0xFF)] {
            let mut ops = encoded();
            let Op::Buffer(arr) = &mut ops[array_of(k)] else {
                panic!("not a buffer");
            };
            assert!(!arr.is_empty(), "{what} is empty");
            arr.fill(byte);
            let err = decode_u64(ops).map(|_| ()).unwrap_err();
            assert!(
                err.contains("corrupt matrix") || err.contains("GxB_load_Matrix"),
                "{what}: {err}"
            );
        }
    }

    /// A payload may declare column-major storage. Row iterators refuse it, and
    /// `Tensor::decode` iterates every matrix by row, so it is stored row-major.
    #[test]
    fn decode_stores_a_column_major_payload_row_major() {
        ensure_init();
        let mut ops = encoded();
        let Op::Buffer(header) = &mut ops[0] else {
            panic!("not a buffer");
        };
        let at = OFF_FORMAT + 4;
        header[at..at + 4].copy_from_slice(&(GrB_Orientation::GrB_COLMAJOR as i32).to_le_bytes());
        let m = decode_u64(ops).unwrap();
        assert_eq!((m.nrows(), m.ncols(), m.nvals()), (64, 64, 128));
        assert_eq!(m.iter(0, u64::MAX).count(), 128);
    }

    #[test]
    fn decode_refuses_a_header_it_cannot_honour() {
        ensure_init();
        // A bitmap or full matrix is nrows * ncols of storage.
        let mut ops = encoded();
        let Op::Buffer(header) = &mut ops[0] else {
            panic!("not a buffer");
        };
        header[OFF_FORMAT..OFF_FORMAT + 4].copy_from_slice(&8i32.to_le_bytes());
        let err = decode_u64(ops).map(|_| ()).unwrap_err();
        assert!(err.contains("not sparse"), "{err}");

        // Dimensions past what the payload's own counts allow.
        let mut huge = Matrix::<u64>::new(1 << 50, 1 << 50);
        huge.wait();
        let mut tape = Tape::default();
        huge.encode(&mut tape);
        let _scope = DecodeLimits {
            node_bound: 10,
            edge_bound: 10,
        }
        .scope();
        let err = decode_u64(tape.0).map(|_| ()).unwrap_err();
        assert!(err.contains("exceed"), "{err}");
    }

    /// A hypersparse matrix keeps a list of the rows that have entries; a list
    /// out of order or out of range is as unsound as bad indices.
    #[test]
    fn decode_checks_a_hypersparse_matrix() {
        ensure_init();
        let mut m = Matrix::<u64>::new(1 << 40, 1 << 40);
        m.build(&[5, 1 << 30, 1 << 39], &[1, 2, 3], &[10, 11, 12]);
        m.wait();
        let mut tape = Tape::default();
        m.encode(&mut tape);
        let decoded = decode_u64(tape.0.clone()).unwrap();
        assert_eq!(decoded.nvals(), 3);

        // The hyperlist is the second component: swap its first two rows.
        let mut bad = tape.0;
        let Op::Buffer(h) = &mut bad[array_of(1)] else {
            panic!("not a buffer");
        };
        let w = h.len() / 3;
        let (a, b) = h.split_at_mut(w);
        a.swap_with_slice(&mut b[..w]);
        let err = decode_u64(bad).map(|_| ()).unwrap_err();
        assert!(err.contains("corrupt matrix"), "{err}");
    }

    /// A reader of `Matrix<u64>` takes eight bytes per entry from the array.
    #[test]
    fn decode_refuses_a_narrower_element_type() {
        ensure_init();
        let mut m = Matrix::<bool>::new(8, 8);
        m.set(1, 2, true);
        m.wait();
        let mut tape = Tape::default();
        m.encode(&mut tape);
        let err = decode_u64(tape.0).map(|_| ()).unwrap_err();
        assert!(err.contains("narrower"), "{err}");
    }

    /// `grown` must place every entry at its original coordinate, at the new
    /// dims, for every growth shape — including the degenerate no-growth call
    /// and growth in one dimension only, which change the concat tile grid.
    #[test]
    fn grown_preserves_entries_at_every_growth_shape() {
        ensure_init();
        let (r0, c0) = (64u64, 48u64);
        let coords: BTreeSet<(u64, u64)> = (0..r0)
            .flat_map(|i| [(i, (i * 7) % c0), (i, (i * 11 + 3) % c0)])
            .collect();
        let rows: Vec<u64> = coords.iter().map(|&(i, _)| i).collect();
        let cols: Vec<u64> = coords.iter().map(|&(_, j)| j).collect();
        let vals: Vec<u64> = (0..coords.len() as u64).collect();

        let mut src = Matrix::<u64>::new(r0, c0);
        src.build(&rows, &cols, &vals);
        src.wait();

        for (nrows, ncols) in [
            (r0, c0),           // 1x1 grid: a plain copy
            (r0 * 4, c0),       // 2x1 grid: rows only
            (r0, c0 * 4),       // 1x2 grid: cols only
            (r0 * 4, c0 * 4),   // 2x2 grid: both
            (100_000, 100_000), // the capacity-grow shape, sparse -> hyper
        ] {
            let g = src.grown(nrows, ncols);
            g.wait();
            assert_eq!((g.nrows(), g.ncols()), (nrows, ncols));
            assert_eq!(g.nvals(), src.nvals(), "{nrows}x{ncols}: nvals changed");
            let got: BTreeSet<(u64, u64, u64)> = g.iter(0, u64::MAX).collect();
            let want: BTreeSet<(u64, u64, u64)> = rows
                .iter()
                .zip(&cols)
                .zip(&vals)
                .map(|((&i, &j), &v)| (i, j, v))
                .collect();
            assert_eq!(got, want, "{nrows}x{ncols}: entries moved or values lost");
        }
        // The source must be untouched — it is still shared with the snapshot.
        assert_eq!((src.nrows(), src.ncols()), (r0, c0));
        assert_eq!(src.nvals(), coords.len() as u64);
    }

    /// Growing a `bool` layer must keep it a pure pattern: a `u64`-typed concat
    /// output would typecast, and a `false` value reads as absent to the valued
    /// masks the delta layers are used with.
    #[test]
    fn grown_keeps_bool_layers_a_pattern() {
        ensure_init();
        let mut src = Matrix::<bool>::new(32, 32);
        src.build(&[0, 5, 31], &[0, 7, 31]);
        src.wait();
        let g = src.grown(4_096, 4_096);
        g.wait();
        assert_eq!(g.nvals(), 3);
        for (i, j) in [(0u64, 0u64), (5, 7), (31, 31)] {
            assert_eq!(g.get(i, j), Some(true), "({i}, {j}) lost or turned false");
        }
    }

    /// A frontier row holding two nodes that both reach `z` keeps `z` when
    /// only one of the two edges is deleted: the deletion layer masks an
    /// output only if every base path to it is deleted. With node 1 = x,
    /// 2 = y, 3 = z, frontier row 0 = {x, y}, base edges x→z and y→z.
    #[test]
    fn delta_lmxm_keeps_an_output_one_live_edge_still_reaches() {
        ensure_init();
        let n = 8;
        let mut m = Matrix::<bool>::new(n, n);
        m.build(&[1, 2], &[3, 3]);
        m.wait();
        let dp = Matrix::<bool>::new(n, n);
        let reach = |deleted: &[u64]| {
            let mut dm = Matrix::<bool>::new(n, n);
            dm.build(deleted, &vec![3; deleted.len()]);
            dm.wait();
            let mut f = Matrix::<bool>::new(n, n);
            f.build(&[0, 0], &[1, 2]);
            f.wait();
            f.delta_lmxm(&m, &dp, &dm);
            f.wait();
            f.get(0, 3)
        };
        assert_eq!(reach(&[1]), Some(true), "y→z is live");
        assert_eq!(reach(&[1, 2]), None, "both paths deleted");
    }

    #[test]
    #[should_panic(expected = "grown must not shrink")]
    fn grown_rejects_shrinking() {
        ensure_init();
        let src = Matrix::<bool>::new(64, 64);
        let _ = src.grown(32, 64);
    }

    /// `Matrix::<bool>::build` uses `GxB_Matrix_build_Scalar`, which takes no
    /// `dup` operator. Callers do produce duplicate coordinates (adjacency
    /// removal candidates accumulated across relationship types), so pin the
    /// semantics: duplicates collapse into one entry rather than failing.
    #[test]
    fn build_bool_tolerates_duplicate_pairs() {
        ensure_init();
        let mut m = Matrix::<bool>::new(8, 8);
        m.build(&[1, 3, 1, 3, 1], &[2, 4, 2, 4, 2]);
        m.wait();
        assert_eq!(m.nvals(), 2);
        assert_eq!(m.get(1, 2), Some(true));
        assert_eq!(m.get(3, 4), Some(true));
    }

    /// `Matrix::<bool>::build` must produce an **iso** matrix — one value
    /// shared by the whole pattern, not a byte per entry.
    ///
    /// This stopped being automatic at GraphBLAS 10.5.0, which dropped the
    /// post-iso check from the `GrB_*_build` family: "GrB_Matrix_build_* and
    /// GrB_Vector_build_* always build a non-iso matrix"
    /// (`Source/builder/GB_build.c`). An all-`true` value array no longer
    /// collapses on its own. This path keeps its iso form only because it goes
    /// through `GxB_Matrix_build_Scalar`, which the same file documents as
    /// always iso — so the second half of this test is the one that would have
    /// caught the change, and is what makes the first half meaningful rather
    /// than tautological.
    #[test]
    fn build_bool_is_iso_and_grb_build_is_not() {
        use std::ptr::null_mut;

        use super::super::{
            GrB_BOOL, GrB_Info, GrB_Matrix, GrB_Matrix_build_BOOL, GrB_Matrix_free, GrB_Matrix_new,
            GrB_Matrix_wait, GrB_WaitMode, GxB_ANY_BOOL, GxB_Matrix_iso, GxB_Matrix_memoryUsage,
        };

        ensure_init();
        const N: u64 = 4096;
        let rows: Vec<u64> = (0..N).collect();
        let cols: Vec<u64> = (0..N).map(|i| (i * 7) % N).collect();

        let mut scalar_built = Matrix::<bool>::new(N, N);
        scalar_built.build(&rows, &cols);
        scalar_built.wait();

        let mut iso = false;
        let info = unsafe { GxB_Matrix_iso(&raw mut iso, scalar_built.inner()) };
        assert_eq!(info, GrB_Info::GrB_SUCCESS);
        assert!(
            iso,
            "Matrix::<bool>::build is no longer iso — every label and adjacency \
             matrix just grew a byte per entry"
        );

        // The same pattern through `GrB_Matrix_build_BOOL` with an all-true
        // value array: iso before 10.5.0, non-iso from 10.5.0 on.
        let mut raw: GrB_Matrix = null_mut();
        let vals = vec![true; rows.len()];
        let (raw_iso, raw_bytes) = unsafe {
            let info = GrB_Matrix_new(&raw mut raw, GrB_BOOL, N, N);
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
            let info = GrB_Matrix_build_BOOL(
                raw,
                rows.as_ptr(),
                cols.as_ptr(),
                vals.as_ptr(),
                rows.len() as u64,
                GxB_ANY_BOOL,
            );
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
            let info = GrB_Matrix_wait(raw, GrB_WaitMode::GrB_COMPLETE as i32);
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
            let mut raw_iso = false;
            let info = GxB_Matrix_iso(&raw mut raw_iso, raw);
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
            let mut bytes: usize = 0;
            let info = GxB_Matrix_memoryUsage(&raw mut bytes, raw);
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
            let info = GrB_Matrix_free(&raw mut raw);
            assert_eq!(info, GrB_Info::GrB_SUCCESS);
            (raw_iso, bytes)
        };
        assert!(
            !raw_iso,
            "GrB_Matrix_build_BOOL produced an iso matrix — the 10.5.0 post-iso \
             removal has been reverted upstream, so the scalar-build workaround \
             in algo_procedures.rs can go back to a plain value array"
        );
        assert!(
            scalar_built.memory_usage() < raw_bytes,
            "iso build should be the cheaper of the two: {} vs {raw_bytes} bytes",
            scalar_built.memory_usage()
        );
    }

    /// The encoded container header must not carry this process's heap
    /// addresses (its component pointers) into DUMP / GRAPH.COPY payloads.
    #[test]
    fn encode_writes_no_heap_addresses() {
        use super::super::GxB_Container_struct as C;
        use std::mem::offset_of;

        struct FirstBuffer(Option<Vec<u8>>);
        impl super::Writer for FirstBuffer {
            fn write_unsigned(
                &mut self,
                _: u64,
            ) {
            }
            fn write_signed(
                &mut self,
                _: i64,
            ) {
            }
            fn write_double(
                &mut self,
                _: f64,
            ) {
            }
            fn write_buffer(
                &mut self,
                d: &[u8],
            ) {
                self.0.get_or_insert_with(|| d.to_vec());
            }
        }

        ensure_init();
        let mut m = Matrix::<bool>::new(8, 8);
        m.set(1, 2, true);
        m.wait();
        let mut w = FirstBuffer(None);
        super::Encode::encode(&m, &mut w);
        let header = w.0.expect("no container header written");
        // p..matrix_future are contiguous pointers, void_future runs to the end
        for (what, range) in [
            (
                "vector / matrix pointers",
                offset_of!(C, p)..offset_of!(C, iso),
            ),
            ("void pointers", offset_of!(C, void_future)..size_of::<C>()),
        ] {
            assert!(
                header[range.clone()].iter().all(|&b| b == 0),
                "{what} written into the payload: {:02x?}",
                &header[range]
            );
        }
        // encoding unloads and reloads the matrix; it must be intact
        assert_eq!(m.get(1, 2), Some(true));
    }
}
