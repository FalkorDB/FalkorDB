//! The crate before `init`, in a process that never calls it.

use graphblas::*;

#[test]
fn constructors_fail_before_init() {
    assert!(!is_initialized());
    assert!(Matrix::<bool>::new(1, 1).is_err());
    assert!(set_nthreads(1).is_err());
}

#[test]
#[should_panic(expected = "GxB_ANY_PAIR_BOOL")]
fn reading_a_builtin_before_init_panics() {
    let _ = Lazy::force(&ANY_PAIR_BOOL);
}
