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
//! [`Matrix`] uses `Arc<GrB_Matrix>` for shared ownership and reference-counted
//! cleanup. A `Mutex` guards operations that require serialization (e.g., `wait`).
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
    serialization::{Decode, Encode, Reader, Writer},
};

/// Size of the `GxB_Container_struct` in bytes.
const CONTAINER_STRUCT_SIZE: usize = std::mem::size_of::<super::GxB_Container_struct>();

use super::vector::Vector;
use super::{
    GrB_BOOL, GrB_BinaryOp, GrB_DESC_C, GrB_DESC_CT0, GrB_DESC_CT0T1, GrB_DESC_CT1, GrB_DESC_R,
    GrB_DESC_RC, GrB_DESC_RCT0, GrB_DESC_RCT0T1, GrB_DESC_RCT1, GrB_DESC_RS, GrB_DESC_RSC,
    GrB_DESC_RSCT0, GrB_DESC_RSCT0T1, GrB_DESC_RSCT1, GrB_DESC_RST0, GrB_DESC_RST0T1,
    GrB_DESC_RST1, GrB_DESC_RT0, GrB_DESC_RT0T1, GrB_DESC_RT1, GrB_DESC_S, GrB_DESC_SC,
    GrB_DESC_SCT0, GrB_DESC_SCT0T1, GrB_DESC_SCT1, GrB_DESC_ST0, GrB_DESC_ST0T1, GrB_DESC_ST1,
    GrB_DESC_T0, GrB_DESC_T0T1, GrB_DESC_T1, GrB_Descriptor, GrB_GLOBAL, GrB_Global_set_INT32,
    GrB_Info, GrB_Matrix, GrB_Matrix_apply, GrB_Matrix_build_BOOL, GrB_Matrix_build_UINT64,
    GrB_Matrix_clear, GrB_Matrix_dup, GrB_Matrix_eWiseAdd_BinaryOp, GrB_Matrix_eWiseMult_Semiring,
    GrB_Matrix_extractElement_BOOL, GrB_Matrix_extractElement_UINT64, GrB_Matrix_free,
    GrB_Matrix_get_INT32, GrB_Matrix_ncols, GrB_Matrix_new, GrB_Matrix_nrows, GrB_Matrix_nvals,
    GrB_Matrix_removeElement, GrB_Matrix_resize, GrB_Matrix_set_INT32, GrB_Matrix_setElement_BOOL,
    GrB_Matrix_setElement_UINT64, GrB_Matrix_wait, GrB_Mode, GrB_Orientation, GrB_SECOND_UINT64,
    GrB_Scalar, GrB_Scalar_free, GrB_Scalar_new, GrB_Scalar_setElement_BOOL, GrB_Type, GrB_UINT64,
    GrB_WaitMode, GrB_finalize, GrB_mxm, GrB_transpose, GxB_ANY_BOOL, GxB_ANY_PAIR_BOOL,
    GxB_ANY_UINT64, GxB_Container_free, GxB_Container_new, GxB_Global_Option_set_INT32,
    GxB_HYPERSPARSE, GxB_Iterator, GxB_Iterator_free, GxB_Iterator_get_UINT64, GxB_Iterator_new,
    GxB_JIT_Control, GxB_Matrix_build_Scalar, GxB_Matrix_fprint, GxB_Matrix_isStoredElement,
    GxB_Matrix_memoryUsage, GxB_Matrix_type, GxB_NTHREADS, GxB_ONE_BOOL, GxB_Option_Field,
    GxB_Print_Level, GxB_SPARSE, GxB_init, GxB_load_Matrix_from_Container, GxB_rowIterator_attach,
    GxB_rowIterator_getColIndex, GxB_rowIterator_getRowIndex, GxB_rowIterator_kount,
    GxB_rowIterator_nextCol, GxB_rowIterator_nextRow, GxB_rowIterator_seekRow,
    GxB_unload_Matrix_into_Container,
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
        //     graphblas.sh) are used for hot ops; RUN additionally permits
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
        //     a shipped binary. Used exclusively by gen_prejit.sh to
        //     populate ~/.SuiteSparse/GrBx.y.z/c/ with the .c kernel
        //     sources we then check in as the next generation of vendored
        //     PreJIT (see graphblas.sh harvest mode).
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

/// A wrapper around a GraphBLAS matrix.
///
/// The type parameter `T` is a compile-time tag for the element type the matrix
/// carries (`bool` for pure structure / presence, `u64` for valued matrices such
/// as inline edge ids). It is a zero-sized [`PhantomData`] marker — every
/// `Matrix<T>` has identical layout regardless of `T` — so it only documents and
/// type-checks intent; it does not change the runtime representation.
pub struct Matrix<T> {
    /// The underlying GraphBLAS matrix.
    m: Arc<GrB_Matrix>,
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

impl<T> Drop for Matrix<T> {
    fn drop(&mut self) {
        if let Some(m) = Arc::get_mut(&mut self.m) {
            unsafe {
                let info = GrB_Matrix_free(m);
                // debug_assert in Drop: panicking while unwinding aborts the
                // process. A GrB_*_free failure is logically a leak, not
                // state corruption — surface in debug, swallow in release.
                debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            }
        }
    }
}

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

impl<T> Decode<19> for Matrix<T> {
    fn decode(r: &mut dyn Reader) -> Result<Self, String> {
        let container_bytes = r.read_buffer()?;

        // Validate container size before copying
        if container_bytes.len() < CONTAINER_STRUCT_SIZE {
            return Err(format!(
                "container buffer too small: {} bytes < {} bytes required",
                container_bytes.len(),
                CONTAINER_STRUCT_SIZE
            ));
        }

        unsafe {
            let mut container: MaybeUninit<super::GxB_Container> = MaybeUninit::uninit();
            let info = GxB_Container_new(container.as_mut_ptr());
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GxB_Container_new failed: {info:?}"
            );
            let container = container.assume_init();

            // Copy struct data into the allocated container
            std::ptr::copy_nonoverlapping(
                container_bytes.as_ptr(),
                container.cast::<u8>(),
                CONTAINER_STRUCT_SIZE,
            );

            // Nullify vector/matrix pointers (will be populated below)
            (*container).x = null_mut();
            (*container).h = null_mut();
            (*container).b = null_mut();
            (*container).i = null_mut();
            (*container).p = null_mut();
            (*container).Y = null_mut();

            // Read and load 5 vectors: x, h, p, i, b
            (*container).x = ManuallyDrop::new(Vector::<bool>::decode(r)?).ptr();
            (*container).h = ManuallyDrop::new(Vector::<bool>::decode(r)?).ptr();
            (*container).p = ManuallyDrop::new(Vector::<bool>::decode(r)?).ptr();
            (*container).i = ManuallyDrop::new(Vector::<bool>::decode(r)?).ptr();
            (*container).b = ManuallyDrop::new(Vector::<bool>::decode(r)?).ptr();

            // Create matrix and load from container
            let mut m: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
            let info = GrB_Matrix_new(m.as_mut_ptr(), GrB_BOOL, 0, 0);
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GrB_Matrix_new failed: {info:?}"
            );
            let m = m.assume_init();
            pin_sparse(m);

            let info = GxB_load_Matrix_from_Container(m, container, null_mut());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);

            // The hyper-hash (Y) was nullified above and is not serialized, so
            // a hypersparse matrix comes back with GxB_WILL_WAIT set. Rebuild
            // it now so `pending()` reflects real pending work (no-op for
            // non-hypersparse matrices).
            let info = GrB_Matrix_wait(m, GrB_WaitMode::GrB_MATERIALIZE as _);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);

            let mut c = container;
            let info = GxB_Container_free(&raw mut c);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);

            Ok(Self {
                m: Arc::new(m),
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

            // Write container struct bytes
            let container_bytes =
                std::slice::from_raw_parts(container.cast::<u8>(), CONTAINER_STRUCT_SIZE);
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
                *self.m,
                GxB_HYPERSPARSE as i32,
                GxB_Option_Field::GxB_SPARSITY_CONTROL as _,
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            // Deltas also opt out of the hyper hash: GrB_Matrix_wait
            // (MATERIALIZE) rebuilds A->Y from scratch on every commit, an
            // O(nvec) sort that grows with the accumulating delta — measured
            // as the dominant cost of small repeated creates. Without A->Y,
            // lookups binary-search A->h, which is fine at delta sizes.
            let info = GrB_Matrix_set_INT32(*self.m, 0, GxB_Option_Field::GxB_HYPER_HASH as _);
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
                *self.m,
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
            let info = GxB_rowIterator_attach(it, *self.m, null_mut());
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
            let info = GxB_Matrix_type(type_.as_mut_ptr(), *self.m);
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
                m: Arc::new(m),
                lock: Arc::new(Mutex::new(())),
                has_pending: Arc::new(AtomicBool::new(true)),
                phantom: PhantomData,
            };
            let info = GrB_transpose(*transpose.m, null_mut(), null_mut(), *self.m, null_mut());
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
        *self.m
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
        unsafe { GxB_Matrix_isStoredElement(*self.m, i, j) == GrB_Info::GrB_SUCCESS }
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
                *t.m,
                null_mut(),
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                *self.m,
                *b.m,
                null_mut(),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        t.nvals()
    }

    #[must_use]
    pub fn pending(&self) -> bool {
        unsafe {
            let mut pending = MaybeUninit::uninit();
            let info = GrB_Matrix_get_INT32(
                *self.m,
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

    /// The integer widths GraphBLAS v10 picked for this matrix's index and
    /// offset arrays. Diagnostic hook for #2430: the width is chosen per matrix
    /// from its dimensions and can differ between two matrices holding the same
    /// content, which changes what a row search costs.
    #[cfg(test)]
    pub fn integer_bits_for_test(&self) -> (i32, i32, i32) {
        let mut row = 0i32;
        let mut col = 0i32;
        let mut off = 0i32;
        unsafe {
            GrB_Matrix_get_INT32(
                *self.m,
                &raw mut row,
                GxB_Option_Field::GxB_ROWINDEX_INTEGER_BITS as _,
            );
            GrB_Matrix_get_INT32(
                *self.m,
                &raw mut col,
                GxB_Option_Field::GxB_COLINDEX_INTEGER_BITS as _,
            );
            GrB_Matrix_get_INT32(
                *self.m,
                &raw mut off,
                GxB_Option_Field::GxB_OFFSET_INTEGER_BITS as _,
            );
        }
        (row, col, off)
    }

    /// Call `GrB_Matrix_wait(MATERIALIZE)` unconditionally, bypassing the
    /// `has_pending` short-circuit. Diagnostic hook for #2430: `has_pending`
    /// tracks pending *tuples*, and a hypersparse matrix can also be waiting on
    /// its hyper-hash, which is a different condition entirely.
    #[cfg(test)]
    pub fn force_materialize_for_test(&self) {
        let lock = self.lock.lock();
        unsafe {
            let info = GrB_Matrix_wait(*self.m, GrB_WaitMode::GrB_MATERIALIZE as _);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        drop(lock);
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
            let info = GrB_Matrix_wait(*self.m, GrB_WaitMode::GrB_MATERIALIZE as _);
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
            let info = GxB_Matrix_memoryUsage(&raw mut usage, *self.m);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            usage
        }
    }

    pub fn clear(&mut self) {
        unsafe {
            let info = GrB_Matrix_clear(*self.m);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(false, Ordering::Relaxed);
    }

    pub fn remove_all<U>(
        &mut self,
        b: &Matrix<U>,
    ) {
        unsafe {
            let info = GrB_transpose(*self.m, *b.m, null_mut(), *self.m, GrB_DESC_RCT0);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    pub fn select(
        &mut self,
        mask: &Matrix<bool>,
        a: &Self,
    ) {
        unsafe {
            let info = GrB_transpose(*self.m, *mask.m, null_mut(), *a.m, GrB_DESC_RCT0);
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
                *self.m,
                mask.map_or(null_mut(), |m| *m.m),
                null_mut(),
                T::add_op(),
                a.map_or(*self.m, |a| *a.m),
                b.map_or(*self.m, |b| *b.m),
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
                *self.m,
                mask.map_or(null_mut(), |m| *m.m),
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                a.map_or(*self.m, |a| *a.m),
                b.map_or(*self.m, |b| *b.m),
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
                *self.m,
                mask.map_or(null_mut(), |m| *m.m),
                GxB_ANY_BOOL,
                GxB_ONE_BOOL,
                *a.m,
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
                *self.m,
                null_mut(),
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                *self.m,
                *b.m,
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
                *self.m,
                null_mut(),
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                *b.m,
                *self.m,
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
            let info = GrB_Matrix_nrows(&raw mut nrows, *self.m);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            nrows
        }
    }

    #[must_use]
    pub fn ncols(&self) -> u64 {
        unsafe {
            let mut ncols = 0u64;
            let info = GrB_Matrix_ncols(&raw mut ncols, *self.m);
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
            let info = GrB_Matrix_resize(*self.m, nrows, ncols);
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
                *self.m,
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
            let info = GrB_Matrix_nvals(&raw mut nvals, *self.m);
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
            let info = GrB_Matrix_removeElement(*self.m, i, j);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    pub fn print(
        &self,
        level: GxB_Print_Level,
    ) {
        unsafe {
            let info = GxB_Matrix_fprint(*self.m, null_mut(), level as _, null_mut());
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
            m: Arc::new(unsafe {
                let mut m: MaybeUninit<GrB_Matrix> = MaybeUninit::uninit();
                let info = GrB_Matrix_dup(m.as_mut_ptr(), *self.m);
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
                    *self.m,
                    &raw mut hyper_hash,
                    GxB_Option_Field::GxB_HYPER_HASH as _,
                );
                debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
                if hyper_hash == 0 {
                    let info = GrB_Matrix_set_INT32(m, 0, GxB_Option_Field::GxB_HYPER_HASH as _);
                    debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
                }
                m
            }),
            lock: Arc::new(Mutex::new(())),
            has_pending: Arc::new(AtomicBool::new(dup_pending)),
            phantom: PhantomData,
        }
    }
}

impl Matrix<u64> {
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
                m: Arc::new(m),
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
            let info = GrB_Matrix_setElement_UINT64(*self.m, value, i, j);
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
            let info = GrB_Matrix_extractElement_UINT64(val.as_mut_ptr(), *self.m, i, j);
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
                *self.m,
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
                m: Arc::new(m),
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
            let info = GrB_Matrix_extractElement_BOOL(m.as_mut_ptr(), *self.m, i, j);
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
            let info = GrB_Matrix_setElement_BOOL(*self.m, value, i, j);
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
            info = GxB_Matrix_build_Scalar(*self.m, rows.as_ptr(), cols.as_ptr(), scalar, nvals);
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

        let mut mask: Option<Matrix<bool>> = None;
        if dm_nvals > 0 {
            let mut mk = Matrix::<bool>::new(nrows, ncols);
            unsafe {
                let info = GrB_mxm(
                    *mk.m,
                    null_mut(),
                    null_mut(),
                    GxB_ANY_PAIR_BOOL,
                    *self.m,
                    *dm.m,
                    null_mut(),
                );
                debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            }
            if mk.nvals() > 0 {
                mask = Some(mk);
            }
        }

        let mut accum: Option<Matrix<bool>> = None;
        if dp_nvals > 0 {
            let mut ac = Matrix::<bool>::new(nrows, ncols);
            unsafe {
                let info = GrB_mxm(
                    *ac.m,
                    null_mut(),
                    null_mut(),
                    GxB_ANY_PAIR_BOOL,
                    *self.m,
                    *dp.m,
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
                .map_or((null_mut(), null_mut()), |m| (*m.m, GrB_DESC_RSC));
            let info = GrB_mxm(
                *self.m,
                mask_ptr,
                null_mut(),
                GxB_ANY_PAIR_BOOL,
                *self.m,
                *m.m,
                desc,
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
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
    m: Arc<GrB_Matrix>,
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
    /// Frees the GraphBLAS iterator when the `Iter` is dropped.
    fn drop(&mut self) {
        unsafe {
            if let Some(m) = Arc::get_mut(&mut self.m) {
                let info = GrB_Matrix_free(m);
                // debug_assert: don't panic in Drop (see Matrix::drop above).
                debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            }
            GxB_Iterator_free(&raw mut self.inner);
        }
    }
}

impl<E: IterExtract> Iter<E> {
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
        unsafe {
            let mut iter = MaybeUninit::uninit();
            let info = GxB_Iterator_new(iter.as_mut_ptr());
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GxB_Iterator_new failed: {info:?}"
            );
            let iter = iter.assume_init();
            let info = GxB_rowIterator_attach(iter, *m.m, null_mut());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            let mut info = GxB_rowIterator_seekRow(iter, min_row);
            debug_assert!(
                info == GrB_Info::GrB_SUCCESS
                    || info == GrB_Info::GrB_NO_VALUE
                    || info == GrB_Info::GxB_EXHAUSTED
            );
            while info == GrB_Info::GrB_NO_VALUE && GxB_rowIterator_getRowIndex(iter) < max_row {
                info = GxB_rowIterator_nextRow(iter);
            }
            Self {
                m: m.m.clone(),
                inner: iter,
                depleted: info != GrB_Info::GrB_SUCCESS
                    || GxB_rowIterator_getRowIndex(iter) > max_row,
                max_row,
                _extract: PhantomData,
            }
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
    use std::collections::BTreeSet;

    use super::super::test_init::ensure_init;
    use super::Matrix;

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
        let info = unsafe { GxB_Matrix_iso(&raw mut iso, *scalar_built.m) };
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
}
