//! GraphBLAS FFI bindings and safe Rust wrappers for sparse linear algebra.
//!
//! This module is the foundation for FalkorDB's graph storage layer. It provides
//! raw C bindings to SuiteSparse:GraphBLAS (the `graphblas-sys` crate, which
//! also links the library) and safe Rust abstractions built on top of them.
//! LAGraph's bindings come from the `lagraph-sys` crate, which links LAGraph.
//!
//! ## Module Layout
//!
//! ```text
//! graphblas/
//!   mod.rs                  <-- Re-exports the FFI bindings of the `graphblas-sys` crate
//!   matrix.rs               <-- Safe wrapper around GrB_Matrix (boolean sparse matrix)
//!   vector.rs               <-- Safe wrapper around GrB_Vector (boolean sparse vector)
//!   tensor.rs               <-- 3D sparse structure for multi-edge storage
//!   versioned_matrix.rs     <-- Copy-on-write matrix with MVCC delta tracking
//! ```
//!
//! ## How the Graph Uses GraphBLAS
//!
//! ```text
//!   Node IDs:       0   1   2   3   4
//!                 +---+---+---+---+---+
//!   Adjacency  0 |   | 1 |   |   |   |   Node 0 --> Node 1
//!   Matrix     1 |   |   | 1 |   |   |   Node 1 --> Node 2
//!   (sparse)   2 |   |   |   |   |   |
//!              3 | 1 |   |   |   | 1 |   Node 3 --> Node 0, Node 3 --> Node 4
//!              4 |   |   |   |   |   |
//!                 +---+---+---+---+---+
//!
//!   Only non-zero entries are stored in memory (sparse format).
//!   Each relationship type gets its own matrix.
//!   Each label gets its own matrix (diagonal: node_id -> node_id).
//! ```
//!
//! ## Initialization
//!
//! Call [`matrix::init`] before using any GraphBLAS operations. Custom memory
//! allocators can be provided to integrate with Redis memory tracking.

#[cfg(test)]
mod block_scaling_bench;
#[cfg(test)]
mod degree_bench;
#[cfg(test)]
mod fold_cost_bench;
pub mod matrix;
#[cfg(test)]
mod me_delta_bench;
pub mod serialization;
pub mod tensor;
pub mod vector;
pub mod versioned_matrix;

/// The instruction counter the benches in this directory measure with.
///
/// One declaration, shared, for the same reason as [`test_init`]: benches that
/// disagreed on a field offset would report numbers that cannot be compared.
#[cfg(test)]
pub(crate) mod instr {
    /// Running instruction total for this process, or `None` where the platform
    /// has no cheap equivalent.
    ///
    /// macOS exposes `proc_pid_rusage(RUSAGE_INFO_V4)` to any caller for its own
    /// pid with no privileges; `ri_instructions` is `u64` field 29 of the struct
    /// body, which starts after the 16-byte `ri_uuid`. Same offsets as
    /// `bench/src/falkorbench/counters.py::read_rusage`, so engine-level and
    /// data-structure-level numbers are on one scale.
    #[cfg(target_os = "macos")]
    pub fn read_instr() -> Option<u64> {
        const RUSAGE_INFO_V4: i32 = 4;
        const RI_INSTRUCTIONS_OFF: usize = 16 + 29 * 8;

        unsafe extern "C" {
            fn proc_pid_rusage(
                pid: i32,
                flavor: i32,
                buffer: *mut u8,
            ) -> i32;
            fn getpid() -> i32;
        }
        let mut buf = [0u8; 1024];
        // SAFETY: `buf` is 1024 bytes, far larger than `struct rusage_info_v4`
        // (~0x150 bytes), and the kernel writes at most that flavor's size.
        if unsafe { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, buf.as_mut_ptr()) } != 0 {
            return None;
        }
        Some(u64::from_le_bytes(
            buf[RI_INSTRUCTIONS_OFF..RI_INSTRUCTIONS_OFF + 8]
                .try_into()
                .expect("8 bytes"),
        ))
    }

    #[cfg(not(target_os = "macos"))]
    pub fn read_instr() -> Option<u64> {
        None
    }
}

/// Process-wide GraphBLAS initialization for unit tests. `GrB_init` may only
/// be called once per process, so every `#[cfg(test)]` module must go through
/// this shared guard rather than owning its own `Once`.
#[cfg(test)]
pub(crate) mod test_init {
    use std::sync::{Once, mpsc};
    use std::thread;

    static INIT: Once = Once::new();

    unsafe extern "C" {
        fn malloc(size: usize) -> *mut std::ffi::c_void;
        fn calloc(
            n: usize,
            size: usize,
        ) -> *mut std::ffi::c_void;
        fn realloc(
            p: *mut std::ffi::c_void,
            size: usize,
        ) -> *mut std::ffi::c_void;
        fn free(p: *mut std::ffi::c_void);
        fn omp_get_max_threads() -> i32;
    }

    pub fn ensure_init() {
        INIT.call_once(|| {
            // libtest runs every test on its own freshly spawned thread. With
            // libomp linked statically (the Linux toolchain image and CI;
            // macOS resolves it dynamically, which is why this never
            // reproduces there), the OpenMP runtime is torn down when the
            // thread that started it exits. Every later parallel region then
            // segfaults: a JIT kernel lookup in `__kmpc_set_lock`, or any
            // kernel that really runs on several threads. The shipped module
            // never hits this: redis-server's main thread outlives every
            // query. So GraphBLAS is started here on a thread that never
            // exits, and the first OpenMP call binds the runtime to it.
            let (started, wait) = mpsc::channel();
            thread::spawn(move || {
                super::matrix::init(Some(malloc), Some(calloc), Some(realloc), Some(free)).unwrap();
                // SAFETY: a plain query of the OpenMP runtime.
                unsafe { omp_get_max_threads() };
                started.send(()).unwrap();
                loop {
                    thread::park();
                }
            });
            wait.recv().unwrap();
            // `GxB_JIT_OFF` keeps tests on generic kernels rather than the
            // vendored PreJIT ones: a correctness-equivalent path, and these
            // are correctness tests.
            let info = unsafe {
                super::GrB_Global_set_INT32(
                    super::GrB_GLOBAL,
                    super::GxB_JIT_Control::GxB_JIT_OFF as i32,
                    super::GxB_Option_Field::GxB_JIT_C_CONTROL as _,
                )
            };
            assert_eq!(info, super::GrB_Info::GrB_SUCCESS, "JIT_OFF failed");
        });
    }
}

/// The raw bindings: every `GrB_*` and `GxB_*` symbol, from the one crate
/// that generates them and links the library.
pub use graphblas_sys::*;
