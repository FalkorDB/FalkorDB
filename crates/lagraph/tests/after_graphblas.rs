//! `lagraph::init` after `graphblas::init`, in a process of its own.

use graphblas::ErrorKind;
use lagraph::Status;

#[test]
fn lagraph_refuses_once_graphblas_started_alone() {
    graphblas::init(None).unwrap();
    let late = lagraph::init(None).unwrap_err();
    assert_eq!(late.status, Status::GraphBlas(ErrorKind::InvalidValue));
    assert!(!lagraph::is_initialized());
    assert!(graphblas::is_initialized(), "GraphBLAS is still up");
}
