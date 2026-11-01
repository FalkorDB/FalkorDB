//! `Matrix` clones share one `GrB_Matrix`; whichever clone (or iterator) goes
//! last must free it, exactly once, even when clones are dropped on different
//! threads at the same moment.
//!
//! Its own test binary: it installs a counting allocator into GraphBLAS, and
//! `GrB_init` happens once per process.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

use graph::graph::graphblas::matrix::{self, Matrix};

static LIVE: AtomicI64 = AtomicI64::new(0);

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

unsafe extern "C" fn counting_malloc(n: usize) -> *mut c_void {
    let p = unsafe { malloc(n) };
    if !p.is_null() {
        LIVE.fetch_add(1, Ordering::Relaxed);
    }
    p
}

unsafe extern "C" fn counting_calloc(
    n: usize,
    size: usize,
) -> *mut c_void {
    let p = unsafe { calloc(n, size) };
    if !p.is_null() {
        LIVE.fetch_add(1, Ordering::Relaxed);
    }
    p
}

unsafe extern "C" fn counting_realloc(
    p: *mut c_void,
    n: usize,
) -> *mut c_void {
    let q = unsafe { realloc(p, n) };
    if p.is_null() && !q.is_null() {
        LIVE.fetch_add(1, Ordering::Relaxed);
    }
    q
}

unsafe extern "C" fn counting_free(p: *mut c_void) {
    if !p.is_null() {
        LIVE.fetch_sub(1, Ordering::Relaxed);
    }
    unsafe { free(p) }
}

#[test]
fn concurrently_dropped_clones_free_the_matrix_once() {
    matrix::init(
        Some(counting_malloc),
        Some(counting_calloc),
        Some(counting_realloc),
        Some(counting_free),
    )
    .unwrap();

    // A leak shows up as live GraphBLAS blocks left behind; a double free
    // would crash. The window is a few instructions wide, so use many pairs
    // and release both threads of each pair together.
    const PAIRS: usize = 100_000;
    let before = LIVE.load(Ordering::SeqCst);
    let (mine, theirs): (Vec<_>, Vec<_>) = (0..PAIRS)
        .map(|i| {
            let m = Matrix::<bool>::new(4, 4);
            let c = m.clone();
            // every third pair races an iterator against a clone instead
            if i % 3 == 0 {
                let it = c.iter(0, 3);
                drop(c);
                (m, Err(it))
            } else {
                (m, Ok(c))
            }
        })
        .unzip();

    let arrived = Arc::new(AtomicUsize::new(0));
    let rendezvous = |a: &AtomicUsize, i: usize| {
        a.fetch_add(1, Ordering::SeqCst);
        while a.load(Ordering::SeqCst) < 2 * (i + 1) {
            std::hint::spin_loop();
        }
    };
    let other = arrived.clone();
    let h = std::thread::spawn(move || {
        for (i, c) in theirs.into_iter().enumerate() {
            rendezvous(&other, i);
            drop(c);
        }
    });
    for (i, m) in mine.into_iter().enumerate() {
        rendezvous(&arrived, i);
        drop(m);
    }
    h.join().unwrap();

    let left = LIVE.load(Ordering::SeqCst) - before;
    assert_eq!(left, 0, "{left} GraphBLAS blocks never freed");
}
