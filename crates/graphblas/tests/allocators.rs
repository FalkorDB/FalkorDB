//! `init` through custom allocators, in a process of its own: every byte
//! GraphBLAS allocates goes through the functions handed to `GxB_init`.

use std::{
    os::raw::c_void,
    sync::atomic::{AtomicUsize, Ordering},
};

use graphblas::*;

static MALLOCS: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);

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

unsafe extern "C" fn counted_malloc(size: usize) -> *mut c_void {
    MALLOCS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: a plain call.
    unsafe { malloc(size) }
}

unsafe extern "C" fn counted_calloc(
    n: usize,
    size: usize,
) -> *mut c_void {
    MALLOCS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: a plain call.
    unsafe { calloc(n, size) }
}

unsafe extern "C" fn counted_realloc(
    p: *mut c_void,
    size: usize,
) -> *mut c_void {
    // SAFETY: `p` came from `counted_malloc` / `counted_calloc`, which use
    // the same allocator.
    unsafe { realloc(p, size) }
}

unsafe extern "C" fn counted_free(p: *mut c_void) {
    FREES.fetch_add(1, Ordering::Relaxed);
    // SAFETY: as in `counted_realloc`.
    unsafe { free(p) }
}

#[test]
fn init_with_allocators_routes_every_allocation() {
    // SAFETY: the four functions count and then forward to libc's, which
    // are one heap, thread-safe, and there for the life of the process.
    let allocators = unsafe {
        Allocators::new(
            counted_malloc,
            counted_calloc,
            counted_realloc,
            counted_free,
        )
    };
    init(Some(allocators)).unwrap();
    set_jit(JitControl::Off).unwrap();
    let after_init = MALLOCS.load(Ordering::Relaxed);
    assert!(after_init > 0, "GrB_init itself allocates");

    {
        let mut a = Matrix::<f64>::new(100, 100).unwrap();
        for i in 0..100 {
            a.set_element(i, i, 1.0).unwrap();
        }
        a.wait(WaitMode::Materialize).unwrap();
        let mut c = Matrix::<f64>::new(100, 100).unwrap();
        mxm(
            &mut c,
            Mask::NONE,
            None,
            &PLUS_TIMES_SEMIRING_FP64,
            &a,
            &a,
            Desc::NONE,
        )
        .unwrap();
        assert_eq!(c.nvals(), 100);
    }
    let after_work = MALLOCS.load(Ordering::Relaxed);
    assert!(
        after_work > after_init,
        "the matrices allocated through the hook"
    );
    assert!(FREES.load(Ordering::Relaxed) > 0, "and freed through it");
}
