//! Initialisation, in a process of its own.

use graphblas::{ErrorKind, Matrix};
use lagraph::{Graph, Kind, Status};

#[test]
fn lagraph_starts_graphblas_and_only_once() {
    // Before `init`, a graph cannot be made: the matrix cannot either.
    assert!(Matrix::<bool>::new(2, 2).is_err());
    assert!(!lagraph::is_initialized());

    lagraph::init(None).unwrap();
    assert!(lagraph::is_initialized());
    assert!(graphblas::is_initialized(), "LAGr_Init started GraphBLAS");

    let again = lagraph::init(None).unwrap_err();
    assert_eq!(again.status, Status::GraphBlas(ErrorKind::InvalidValue));
    let graphblas_again = graphblas::init(None).unwrap_err();
    assert_eq!(graphblas_again.kind, ErrorKind::InvalidValue);

    let a = Matrix::<bool>::new(2, 2).unwrap();
    Graph::new(a, Kind::Directed).unwrap();
}
