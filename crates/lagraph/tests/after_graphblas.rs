//! `lagraph::init` after `graphblas::init`, in a process of its own.

use graphblas::{ErrorKind, Matrix};
use lagraph::{Graph, Kind, Status, minimum_spanning_forest};

#[test]
fn lagraph_refuses_once_graphblas_started_alone() {
    graphblas::init(None).unwrap();
    let late = lagraph::init(None).unwrap_err();
    assert_eq!(late.status, Status::GraphBlas(ErrorKind::InvalidValue));
    assert!(!lagraph::is_initialized());
    assert!(graphblas::is_initialized(), "GraphBLAS is still up");

    let a = Matrix::<bool>::new(2, 2).unwrap();
    let unready = Graph::new(a, Kind::Directed).unwrap_err();
    assert_eq!(unready.status, Status::GraphBlas(ErrorKind::Panic));
    assert!(unready.message.contains("lagraph::init"), "{unready}");

    // The one algorithm that takes a matrix and no graph asks as well.
    let weights = Matrix::<i64>::new(2, 2).unwrap();
    let unready = minimum_spanning_forest(false, &weights, true).unwrap_err();
    assert_eq!(unready.status, Status::GraphBlas(ErrorKind::Panic));
}
