//! The crate before `init`, in a process that never calls it.

use graphblas::*;

#[test]
fn settings_fail_before_init() {
    assert!(!is_initialized());
    assert!(set_nthreads(1).is_err());
}
