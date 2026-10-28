//! The CSR export hands GraphBLAS's own arrays over to Rust, which shares its
//! allocator (`matrix::init`'s contract): each exported `Vec` must be a block
//! GraphBLAS allocated, freed exactly once, by Rust's global allocator, and
//! never by GraphBLAS.
//!
//! Its own test binary: it installs a logging allocator both into GraphBLAS
//! and as Rust's global allocator, and `GrB_init` happens once per process.

#![cfg(unix)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use graph::graph::graphblas::matrix::{self, Matrix};
use graph::graph::graphblas::{
    GrB_Info, GrB_Matrix_set_INT32, GrB_Orientation, GxB_Option_Field, GxB_SPARSE,
};

/// A fixed-size, allocation-free log of addresses, appended to from inside
/// the allocators while [`RECORDING`] is set.
struct Log {
    len: AtomicUsize,
    slots: [AtomicUsize; Log::CAP],
}

impl Log {
    const CAP: usize = 1 << 16;

    const fn new() -> Self {
        Self {
            len: AtomicUsize::new(0),
            slots: [const { AtomicUsize::new(0) }; Self::CAP],
        }
    }

    fn push(
        &self,
        p: *mut c_void,
    ) {
        if p.is_null() || !RECORDING.load(Ordering::SeqCst) {
            return;
        }
        let k = self.len.fetch_add(1, Ordering::SeqCst);
        if k < Self::CAP {
            self.slots[k].store(p as usize, Ordering::SeqCst);
        }
    }

    fn clear(&self) {
        self.len.store(0, Ordering::SeqCst);
    }

    fn count(
        &self,
        p: *const c_void,
    ) -> usize {
        let len = self.len.load(Ordering::SeqCst);
        assert!(len <= Self::CAP, "log overflowed");
        self.slots[..len]
            .iter()
            .filter(|s| s.load(Ordering::SeqCst) == p as usize)
            .count()
    }
}

static RECORDING: AtomicBool = AtomicBool::new(false);
/// Blocks GraphBLAS's malloc/calloc/realloc returned, and those its free freed.
static GRB_ALLOC: Log = Log::new();
static GRB_FREE: Log = Log::new();
/// Blocks Rust's global allocator freed, and the bytes it allocated.
static RUST_FREE: Log = Log::new();
static RUST_ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

struct Logging;

unsafe impl GlobalAlloc for Logging {
    unsafe fn alloc(
        &self,
        layout: Layout,
    ) -> *mut u8 {
        if RECORDING.load(Ordering::SeqCst) {
            RUST_ALLOC_BYTES.fetch_add(layout.size(), Ordering::SeqCst);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(
        &self,
        ptr: *mut u8,
        layout: Layout,
    ) {
        RUST_FREE.push(ptr.cast());
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Logging = Logging;

unsafe extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn calloc(
        n: usize,
        size: usize,
    ) -> *mut c_void;
    fn realloc(
        p: *mut c_void,
        size: usize,
    ) -> *mut c_void;
    fn free(p: *mut c_void);
}

unsafe extern "C" fn logging_malloc(n: usize) -> *mut c_void {
    let p = unsafe { malloc(n) };
    GRB_ALLOC.push(p);
    p
}

unsafe extern "C" fn logging_calloc(
    n: usize,
    size: usize,
) -> *mut c_void {
    let p = unsafe { calloc(n, size) };
    GRB_ALLOC.push(p);
    p
}

unsafe extern "C" fn logging_realloc(
    p: *mut c_void,
    n: usize,
) -> *mut c_void {
    let q = unsafe { realloc(p, n) };
    if !q.is_null() && q != p {
        GRB_FREE.push(p);
        GRB_ALLOC.push(q);
    }
    q
}

unsafe extern "C" fn logging_free(p: *mut c_void) {
    GRB_FREE.push(p);
    unsafe { free(p) }
}

/// A CSR-ready `n x n` FP64 matrix with distinct values, so it is not iso:
/// sparse, row-major, 32-bit offsets and column indices. `Matrix::prepare_csr`
/// is private to the crate, so this sets the raw fields.
fn weights(n: u64) -> Matrix<f64> {
    let mut m = Matrix::<f64>::new(n, n);
    for (value, field) in [
        (GxB_SPARSE as i32, GxB_Option_Field::GxB_SPARSITY_CONTROL),
        (
            GrB_Orientation::GrB_ROWMAJOR as i32,
            GxB_Option_Field::GrB_STORAGE_ORIENTATION_HINT,
        ),
        (32, GxB_Option_Field::GxB_OFFSET_INTEGER_HINT),
        (32, GxB_Option_Field::GxB_COLINDEX_INTEGER_HINT),
    ] {
        let info = unsafe { GrB_Matrix_set_INT32(m.inner(), value, field as _) };
        assert_eq!(info, GrB_Info::GrB_SUCCESS);
    }
    for r in 0..n {
        for k in 1..4 {
            m.set(r, (r * 7 + k * 13) % n, (r * 4 + k) as f64);
        }
    }
    m.wait();
    m
}

/// Whether `p` is a block GraphBLAS allocated and has not freed since
/// [`start`]. Both sides use libc malloc, so an address alone can be one that
/// GraphBLAS freed and Rust then got back.
fn grb_live(p: *const c_void) -> bool {
    match GRB_ALLOC.count(p) - GRB_FREE.count(p) {
        0 => false,
        1 => true,
        n => panic!("{n} live GraphBLAS blocks at one address"),
    }
}

/// Restart the free logs: from here on, any free of a live array counts.
fn watch_frees() {
    GRB_FREE.clear();
    RUST_FREE.clear();
}

fn start() {
    for log in [&GRB_ALLOC, &GRB_FREE, &RUST_FREE] {
        log.clear();
    }
    RUST_ALLOC_BYTES.store(0, Ordering::SeqCst);
    RECORDING.store(true, Ordering::SeqCst);
}

fn stop() {
    RECORDING.store(false, Ordering::SeqCst);
}

#[test]
fn adopted_arrays_come_from_graphblas_and_rust_frees_them_once() {
    // SAFETY: GraphBLAS allocates with libc malloc/free, and `Logging`
    // frees through `System`, whose `dealloc` is `free(ptr)` on unix.
    unsafe {
        matrix::init(
            Some(logging_malloc),
            Some(logging_calloc),
            Some(logging_realloc),
            Some(logging_free),
        )
    }
    .unwrap();
    const N: u64 = 2000;

    start();
    let mut m = weights(N);
    let before = RUST_ALLOC_BYTES.load(Ordering::SeqCst);
    let csr = m.take_csr::<u32, u32, f64>().expect("exports");
    let during = RUST_ALLOC_BYTES.load(Ordering::SeqCst) - before;
    assert_eq!(csr.row_ptr.len(), N as usize + 1);
    assert_eq!(csr.col_idx.len(), csr.vals.len());
    let arrays = [
        csr.row_ptr.as_ptr().cast::<c_void>(),
        csr.col_idx.as_ptr().cast(),
        csr.vals.as_ptr().cast(),
    ];
    assert!(during < 1024, "the export allocated {during} bytes in Rust");
    for p in arrays {
        assert!(grb_live(p), "each array is a block GraphBLAS allocated");
    }
    // The arrays stay live until `csr` drops, so no other block can take
    // their addresses in between: every free logged from here is theirs.
    watch_frees();
    drop(m);
    for p in arrays {
        assert_eq!(GRB_FREE.count(p), 0, "GraphBLAS freed an adopted array");
        assert_eq!(RUST_FREE.count(p), 0, "freed while still exported");
    }
    drop(csr);
    stop();
    for p in arrays {
        assert_eq!(GRB_FREE.count(p), 0, "GraphBLAS freed an adopted array");
        assert_eq!(RUST_FREE.count(p), 1, "Rust frees each array exactly once");
    }
}
