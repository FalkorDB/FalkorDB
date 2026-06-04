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
    GrB_BOOL, GrB_DESC_C, GrB_DESC_CT0, GrB_DESC_CT0T1, GrB_DESC_CT1, GrB_DESC_R, GrB_DESC_RC,
    GrB_DESC_RCT0, GrB_DESC_RCT0T1, GrB_DESC_RCT1, GrB_DESC_RS, GrB_DESC_RSC, GrB_DESC_RSCT0,
    GrB_DESC_RSCT0T1, GrB_DESC_RSCT1, GrB_DESC_RST0, GrB_DESC_RST0T1, GrB_DESC_RST1, GrB_DESC_RT0,
    GrB_DESC_RT0T1, GrB_DESC_RT1, GrB_DESC_S, GrB_DESC_SC, GrB_DESC_SCT0, GrB_DESC_SCT0T1,
    GrB_DESC_SCT1, GrB_DESC_ST0, GrB_DESC_ST0T1, GrB_DESC_ST1, GrB_DESC_T0, GrB_DESC_T0T1,
    GrB_DESC_T1, GrB_Descriptor, GrB_GLOBAL, GrB_Global_set_INT32, GrB_Info, GrB_Matrix,
    GrB_Matrix_build_BOOL, GrB_Matrix_build_UINT64, GrB_Matrix_clear, GrB_Matrix_dup,
    GrB_Matrix_eWiseAdd_Semiring, GrB_Matrix_eWiseMult_Semiring, GrB_Matrix_extractElement_BOOL,
    GrB_Matrix_extractElement_UINT64, GrB_Matrix_extractTuples_BOOL, GrB_Matrix_free,
    GrB_Matrix_get_INT32, GrB_Matrix_ncols, GrB_Matrix_new, GrB_Matrix_nrows, GrB_Matrix_nvals,
    GrB_Matrix_removeElement, GrB_Matrix_resize, GrB_Matrix_setElement_BOOL,
    GrB_Matrix_setElement_UINT64, GrB_Matrix_wait, GrB_Mode, GrB_UINT64, GrB_WaitMode,
    GrB_finalize, GrB_mxm, GrB_transpose, GxB_ANY_BOOL, GxB_ANY_PAIR_BOOL, GxB_ANY_UINT64,
    GxB_Container_free, GxB_Container_new, GxB_Global_Option_set_INT32, GxB_Iterator,
    GxB_Iterator_free, GxB_Iterator_new, GxB_JIT_Control, GxB_Matrix_fprint,
    GxB_Matrix_memoryUsage, GxB_Matrix_type, GxB_NTHREADS, GxB_Option_Field, GxB_Print_Level,
    GxB_init, GxB_load_Matrix_from_Container, GxB_rowIterator_attach, GxB_rowIterator_getColIndex,
    GxB_rowIterator_getRowIndex, GxB_rowIterator_nextCol, GxB_rowIterator_nextRow,
    GxB_rowIterator_seekRow, GxB_unload_Matrix_into_Container,
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
        //     +6% to +87% across the test_bench.py suite; aligning with
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

/// A trait for querying and modifying the size of a matrix.
pub trait Size {
    /// Returns the number of rows in the matrix.
    fn nrows(&self) -> u64;

    /// Returns the number of columns in the matrix.
    fn ncols(&self) -> u64;

    /// Resizes the matrix to the specified number of rows and columns.
    ///
    /// # Parameters
    /// - `nrows`: The new number of rows.
    /// - `ncols`: The new number of columns.
    fn resize(
        &mut self,
        nrows: u64,
        ncols: u64,
    );

    /// Returns the number of non-zero values in the matrix.
    fn nvals(&self) -> u64;
}

/// A trait for retrieving elements from a matrix.
pub trait Get {
    /// Retrieves the element at the specified row and column.
    /// Returns `None` if the element does not exist.
    ///
    /// # Parameters
    /// - `i`: The row index.
    /// - `j`: The column index.
    ///
    /// # Returns
    /// - `Some(T)`: The element at the specified position.
    /// - `None`: The element does not exist.
    fn get(
        &self,
        i: u64,
        j: u64,
    ) -> Option<bool>;
}

/// A trait for setting elements in a matrix.
pub trait Set {
    /// Sets the element at the specified row and column to the given value.
    ///
    /// # Parameters
    /// - `i`: The row index.
    /// - `j`: The column index.
    /// - `value`: The value to set.
    fn set(
        &mut self,
        i: u64,
        j: u64,
        value: bool,
    );
}

/// A trait for removing elements from a matrix.
pub trait Remove {
    /// Removes the element at the specified row and column.
    ///
    /// # Parameters
    /// - `i`: The row index.
    /// - `j`: The column index.
    fn remove(
        &mut self,
        i: u64,
        j: u64,
    );
}

pub trait Transpose {
    /// Transposes the matrix.
    #[must_use]
    fn transpose(&self) -> Self;
}

pub trait MaskedElementWiseAdd {
    fn element_wise_add(
        &mut self,
        mask: Option<&Matrix>,
        a: Option<&Self>,
        b: Option<&Self>,
        descriptor: Option<Descriptor>,
    );
}

impl MaskedElementWiseAdd for Matrix {
    fn element_wise_add(
        &mut self,
        mask: Option<&Matrix>,
        a: Option<&Self>,
        b: Option<&Self>,
        descriptor: Option<Descriptor>,
    ) {
        unsafe {
            let info = GrB_Matrix_eWiseAdd_Semiring(
                *self.m,
                mask.map_or(null_mut(), |m| *m.m),
                GxB_ANY_BOOL,
                GxB_ANY_PAIR_BOOL,
                a.map_or(*self.m, |a| *a.m),
                b.map_or(*self.m, |b| *b.m),
                descriptor.map_or(null_mut(), std::convert::Into::into),
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }
}

pub trait MaskedElementWiseMultiply {
    fn element_wise_multiply(
        &mut self,
        mask: Option<&Matrix>,
        a: Option<&Self>,
        b: Option<&Self>,
        descriptor: Option<Descriptor>,
    );
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

impl MaskedElementWiseMultiply for Matrix {
    fn element_wise_multiply(
        &mut self,
        mask: Option<&Self>,
        a: Option<&Self>,
        b: Option<&Self>,
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
}

pub trait MxM {
    /// Multiplies two matrices and stores the result in the current matrix.
    ///
    /// # Parameters
    /// - `b`: The matrix to multiply with.
    fn lmxm(
        &mut self,
        b: &Self,
    );

    fn rmxm(
        &mut self,
        b: &Self,
    );
}

impl MxM for Matrix {
    fn lmxm(
        &mut self,
        b: &Self,
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

    fn rmxm(
        &mut self,
        b: &Self,
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
}

impl Matrix {
    /// Delta-aware matrix-multiply: `self = self * vm` operating directly on
    /// the versioned matrix's base/dp/dm components, mirroring C FalkorDB's
    /// `Delta_mxm`.
    ///
    /// Computes `(self * (m + dp))<!(self * dm)>` without first materializing
    /// the merged matrix. In the common read-only case (`dp.nvals() == 0 &&
    /// dm.nvals() == 0`) this is a single `GrB_mxm` against `vm.m()`, avoiding
    /// the eWiseAdd that `to_matrix()` would otherwise pay.
    pub fn delta_lmxm(
        &mut self,
        vm: &super::versioned_matrix::VersionedMatrix,
    ) {
        let dp = vm.dp();
        let dm = vm.dm();
        let dp_nvals = dp.nvals();
        let dm_nvals = dm.nvals();

        if dp_nvals == 0 && dm_nvals == 0 {
            // Hot path: clean snapshot, just self * vm.m()
            self.lmxm(vm.m());
            return;
        }

        let nrows = self.nrows();
        let ncols = vm.ncols();

        let mut mask: Option<Matrix> = None;
        if dm_nvals > 0 {
            let mut mk = Matrix::new(nrows, ncols);
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

        let mut accum: Option<Matrix> = None;
        if dp_nvals > 0 {
            let mut ac = Matrix::new(nrows, ncols);
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
                *vm.m().m,
                desc,
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }

        if let Some(ac) = accum {
            self.element_wise_add(None, None, Some(&ac), None);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }
}

/// A wrapper around a GraphBLAS boolean matrix.
#[derive(Clone)]
pub struct Matrix {
    /// The underlying GraphBLAS matrix.
    m: Arc<GrB_Matrix>,
    lock: Arc<Mutex<()>>,
    /// Set to `true` by every mutating op; `wait()` short-circuits when `false`
    /// to skip the lock + GrB_Matrix_wait FFI under read-heavy contention.
    has_pending: Arc<AtomicBool>,
}

unsafe impl Send for Matrix {}
unsafe impl Sync for Matrix {}

impl Drop for Matrix {
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

impl Decode<19> for Matrix {
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

            let info = GxB_load_Matrix_from_Container(m, container, null_mut());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);

            let mut c = container;
            let info = GxB_Container_free(&raw mut c);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);

            Ok(Self {
                m: Arc::new(m),
                lock: Arc::new(Mutex::new(())),
                has_pending: Arc::new(AtomicBool::new(false)),
            })
        }
    }
}

impl Encode<19> for Matrix {
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

impl Matrix {
    /// Returns the raw GrB_Matrix handle for FFI calls (e.g. LAGraph).
    /// The caller must NOT free the returned handle.
    #[must_use]
    pub fn inner(&self) -> GrB_Matrix {
        *self.m
    }

    /// Iterate entries as `(row, col, value)` UINT64 triples.
    ///
    /// Used when loading C-produced relation matrices where single-edge
    /// entries store the edge ID as a UINT64 value.
    #[must_use]
    pub fn uint64_iter(&self) -> Iter<Uint64Extract> {
        Iter::new(self, 0, u64::MAX)
    }

    /// Returns true if this matrix has UINT64 element type.
    ///
    /// C-produced relation matrices store edge IDs as UINT64, while
    /// Rust-produced ones use BOOL.
    #[must_use]
    pub fn is_uint64(&self) -> bool {
        unsafe {
            let mut t: MaybeUninit<super::GrB_Type> = MaybeUninit::uninit();
            let info = GxB_Matrix_type(t.as_mut_ptr(), *self.m);
            assert_eq!(
                info,
                GrB_Info::GrB_SUCCESS,
                "GxB_Matrix_type failed: {info:?}"
            );
            t.assume_init() == GrB_UINT64
        }
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

    pub fn remove_all(
        &mut self,
        b: &Self,
    ) {
        unsafe {
            let info = GrB_transpose(*self.m, *b.m, null_mut(), *self.m, GrB_DESC_RCT0);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    pub fn select(
        &mut self,
        mask: &Matrix,
        a: &Matrix,
    ) {
        unsafe {
            let info = GrB_transpose(*self.m, *mask.m, null_mut(), *a.m, GrB_DESC_RCT0);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }
}

impl Size for Matrix {
    fn nrows(&self) -> u64 {
        unsafe {
            let mut nrows = 0u64;
            let info = GrB_Matrix_nrows(&raw mut nrows, *self.m);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            nrows
        }
    }

    fn ncols(&self) -> u64 {
        unsafe {
            let mut ncols = 0u64;
            let info = GrB_Matrix_ncols(&raw mut ncols, *self.m);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            ncols
        }
    }

    fn resize(
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

    fn nvals(&self) -> u64 {
        unsafe {
            let mut nvals = 0u64;
            let info = GrB_Matrix_nvals(&raw mut nvals, *self.m);
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
            nvals
        }
    }
}

pub trait New {
    fn new(
        nrows: u64,
        ncols: u64,
    ) -> Self;
}

impl New for Matrix {
    fn new(
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
            Self {
                m: Arc::new(m.assume_init()),
                lock: Arc::new(Mutex::new(())),
                has_pending: Arc::new(AtomicBool::new(false)),
            }
        }
    }
}

pub trait Dup<T> {
    fn dup(&self) -> T;
}

impl Dup<Self> for Matrix {
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
                m.assume_init()
            }),
            lock: Arc::new(Mutex::new(())),
            has_pending: Arc::new(AtomicBool::new(dup_pending)),
        }
    }
}

impl Matrix {
    /// Create a new UINT64 matrix (for C-compatible tensor encoding).
    #[must_use]
    pub fn new_uint64(
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
            Self {
                m: Arc::new(m.assume_init()),
                lock: Arc::new(Mutex::new(())),
                has_pending: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    /// Set a UINT64 value at (i, j).
    pub fn set_uint64(
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

    #[must_use]
    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(
        &self,
        min_row: u64,
        max_row: u64,
    ) -> Iter {
        Iter::new(self, min_row, max_row)
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

impl Remove for Matrix {
    fn remove(
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
}

impl Get for Matrix {
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
    fn get(
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
}

impl Set for Matrix {
    fn set(
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
}

impl Matrix {
    /// Bulk-insert entries from (row, col) arrays. Matrix must be empty.
    /// Uses a single GraphBLAS FFI call instead of N individual setElement calls.
    pub fn build_bool(
        &mut self,
        rows: &[u64],
        cols: &[u64],
    ) {
        debug_assert_eq!(rows.len(), cols.len());
        if rows.is_empty() {
            return;
        }
        let nvals = rows.len() as u64;
        let vals = vec![true; rows.len()];
        unsafe {
            let info = GrB_Matrix_build_BOOL(
                *self.m,
                rows.as_ptr(),
                cols.as_ptr(),
                vals.as_ptr(),
                nvals,
                GxB_ANY_BOOL,
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        self.has_pending.store(true, Ordering::Relaxed);
    }

    /// Bulk-insert UINT64 entries from (row, col, val) arrays. Matrix must be empty and UINT64 typed.
    pub fn build_uint64(
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

    /// Bulk-extract all (row, col) entries from a boolean matrix.
    #[must_use]
    pub fn extract_tuples_bool(&self) -> (Vec<u64>, Vec<u64>) {
        let mut nvals = self.nvals();
        if nvals == 0 {
            return (Vec::new(), Vec::new());
        }
        let mut rows = vec![0u64; nvals as usize];
        let mut cols = vec![0u64; nvals as usize];
        let mut vals = vec![false; nvals as usize];
        unsafe {
            let info = GrB_Matrix_extractTuples_BOOL(
                rows.as_mut_ptr(),
                cols.as_mut_ptr(),
                vals.as_mut_ptr(),
                &raw mut nvals,
                *self.m,
            );
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        rows.truncate(nvals as usize);
        cols.truncate(nvals as usize);
        (rows, cols)
    }
}

impl Transpose for Matrix
where
    Self: New,
{
    /// Transposes the matrix.
    ///
    /// # Returns
    /// A new matrix that is the transpose of the original.
    fn transpose(&self) -> Self {
        let transpose = Self::new(self.ncols(), self.nrows());
        unsafe {
            let info = GrB_transpose(*transpose.m, null_mut(), null_mut(), *self.m, null_mut());
            debug_assert_eq!(info, GrB_Info::GrB_SUCCESS);
        }
        transpose.has_pending.store(true, Ordering::Relaxed);
        transpose
    }
}

/// Strategy for extracting values from a GraphBLAS row iterator position.
///
/// # Safety
/// Implementations must only call valid GraphBLAS FFI functions on the provided matrix.
pub trait IterExtract {
    type Item;

    /// Extract the item from the current iterator position.
    ///
    /// # Safety
    /// `m` must be a valid `GrB_Matrix` and the iterator must be positioned on a valid entry.
    unsafe fn extract(
        m: GrB_Matrix,
        row: u64,
        col: u64,
    ) -> Self::Item;
}

/// Extracts `(row, col)` pairs from a boolean matrix.
pub struct BoolExtract;

impl IterExtract for BoolExtract {
    type Item = (u64, u64);

    unsafe fn extract(
        _m: GrB_Matrix,
        row: u64,
        col: u64,
    ) -> Self::Item {
        (row, col)
    }
}

/// Extracts `(row, col, value)` triples from a UINT64 matrix.
pub struct Uint64Extract;

impl IterExtract for Uint64Extract {
    type Item = (u64, u64, u64);

    unsafe fn extract(
        m: GrB_Matrix,
        row: u64,
        col: u64,
    ) -> Self::Item {
        let mut val: u64 = 0;
        unsafe { GrB_Matrix_extractElement_UINT64(&raw mut val, m, row, col) };
        (row, col, val)
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
    pub fn new(
        m: &Matrix,
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
            let row = GxB_rowIterator_getRowIndex(self.inner);
            let col = GxB_rowIterator_getColIndex(self.inner);
            let item = E::extract(*self.m, row, col);
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
