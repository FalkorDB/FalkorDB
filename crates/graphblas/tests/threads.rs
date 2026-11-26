//! The thread count, in a process of its own: it is a global setting.

use std::os::raw::c_int;

unsafe extern "C" {
    fn omp_get_max_threads() -> c_int;
}

#[test]
fn set_nthreads_also_tells_openmp() {
    graphblas::init(None).expect("GrB_init");
    for n in [3, 1] {
        graphblas::set_nthreads(n).unwrap();
        // SAFETY: a plain query of the OpenMP runtime.
        assert_eq!(unsafe { omp_get_max_threads() }, n);
    }
}
