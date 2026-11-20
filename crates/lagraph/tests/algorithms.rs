//! The graph type and the algorithms, against the real libraries.
//!
//! `GrB_init` and `LAGraph_Init` are process-wide, so every test goes
//! through [`setup`]; the tests that need a process of their own
//! (`init` itself, use before it) are separate files.

#![allow(
    clippy::float_cmp,
    reason = "the expected values are small whole numbers, which an f64 holds exactly"
)]

use std::{
    sync::{Once, mpsc},
    thread,
};

use graphblas::{ErrorKind, GrB_Index, JitControl, Matrix, Orientation, Vector};
use lagraph::{Graph, Kind, Status, betweenness, connected_components, page_rank};
use lagraph_sys as sys;

fn setup() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // On a thread that never ends: see `graphblas::init`.
        let (started, wait) = mpsc::channel();
        thread::spawn(move || {
            lagraph::init(None).expect("LAGraph_Init");
            graphblas::set_jit(JitControl::Off).expect("JIT off");
            started.send(()).expect("the test is waiting");
            loop {
                thread::park();
            }
        });
        wait.recv().expect("the libraries started");
    });
}

/// An `n`-node pattern matrix; an undirected one stores both directions.
fn adjacency(
    n: GrB_Index,
    edges: &[(GrB_Index, GrB_Index)],
    kind: Kind,
) -> Matrix<bool> {
    let mut a = Matrix::new(n, n).unwrap();
    for &(i, j) in edges {
        a.set_element(i, j, true).unwrap();
        if kind == Kind::Undirected {
            a.set_element(j, i, true).unwrap();
        }
    }
    a
}

fn graph(
    n: GrB_Index,
    edges: &[(GrB_Index, GrB_Index)],
    kind: Kind,
) -> Graph<bool> {
    Graph::new(adjacency(n, edges, kind), kind).unwrap()
}

/// Every node's value, by node id; the vector must be full.
fn dense<T: graphblas::GrbType>(v: &Vector<T>) -> Vec<T> {
    let (ids, values) = v.extract_tuples().unwrap();
    assert_eq!(ids, (0..v.size()).collect::<Vec<_>>(), "one value per node");
    values
}

fn close(
    actual: f64,
    expected: f64,
) -> bool {
    (actual - expected).abs() < 1e-4
}

/// FalkorDB/FalkorDB#3349: GraphBLAS holds a `1 x 1` matrix by column, and
/// LAGraph then rejects the graph, and the transpose it caches itself, with
/// `LAGRAPH_INVALID_GRAPH`.
#[test]
fn a_one_node_graph_runs_every_algorithm() {
    setup();
    let lone = adjacency(1, &[], Kind::Directed);
    assert_eq!(
        lone.orientation(),
        Orientation::ByCol,
        "what LAGraph rejects"
    );

    let mut directed = Graph::new(lone, Kind::Directed).unwrap();
    assert_eq!(directed.adjacency().orientation(), Orientation::ByRow);
    directed.cache_transpose().unwrap();
    directed.cache_out_degree().unwrap();
    directed
        .check()
        .expect("the cached transpose is held by row too");

    let (rank, _) = page_rank(&directed, 0.85, 1e-4, 100).unwrap();
    assert!(close(f64::from(dense(&rank)[0]), 1.0));

    let centrality = betweenness(&directed, &[0]).unwrap();
    assert_eq!(centrality.extract_element(0).unwrap().unwrap_or(0.0), 0.0);

    let mut undirected = graph(1, &[], Kind::Undirected);
    assert_eq!(dense(&connected_components(&mut undirected).unwrap()), [0]);
}

#[test]
fn page_rank_says_which_cached_property_is_missing() {
    setup();
    let mut cycle = graph(3, &[(0, 1), (1, 2), (2, 0)], Kind::Directed);

    let error = page_rank(&cycle, 0.85, 1e-4, 100).unwrap_err();
    assert_eq!(error.status, Status::NotCached);
    assert!(error.message.contains("G->AT is required"), "{error}");

    cycle.cache_transpose().unwrap();
    let error = page_rank(&cycle, 0.85, 1e-4, 100).unwrap_err();
    assert_eq!(error.status, Status::NotCached);
    assert!(
        error.message.contains("G->out_degree is required"),
        "{error}"
    );

    cycle.cache_out_degree().unwrap();
    let (rank, iterations) = page_rank(&cycle, 0.85, 1e-4, 100).unwrap();
    assert!(iterations >= 1);
    for score in dense(&rank) {
        assert!(close(f64::from(score), 1.0 / 3.0), "{score}");
    }
}

#[test]
fn page_rank_reports_hitting_the_iteration_limit() {
    setup();
    let mut path = graph(4, &[(0, 1), (1, 2), (2, 3)], Kind::Directed);
    path.cache_transpose().unwrap();
    path.cache_out_degree().unwrap();

    let error = page_rank(&path, 0.85, 1e-12, 1).unwrap_err();
    assert_eq!(error.status, Status::ConvergenceFailure);
    assert_ne!(error.message, "");
}

#[test]
fn betweenness_counts_the_paths_through_a_node() {
    setup();
    let mut path = graph(3, &[(0, 1), (1, 2)], Kind::Directed);
    path.cache_transpose().unwrap();

    // Only 0 -> 1 -> 2 passes through another node.
    let centrality = betweenness(&path, &[0, 1, 2]).unwrap();
    let score = |node| centrality.extract_element(node).unwrap().unwrap_or(0.0);
    assert_eq!([score(0), score(1), score(2)], [0.0, 1.0, 0.0]);

    // LAGraph itself answers 1.0 for every node here.
    let error = betweenness(&path, &[]).unwrap_err();
    assert_eq!(error.status, Status::GraphBlas(ErrorKind::InvalidValue));

    let error = betweenness(&path, &[7]).unwrap_err();
    assert_eq!(error.status, Status::GraphBlas(ErrorKind::InvalidIndex));
    assert!(error.message.contains("invalid source node"), "{error}");
}

#[test]
fn connected_components_needs_a_symmetric_graph() {
    setup();
    let mut undirected = graph(5, &[(0, 1), (2, 3)], Kind::Undirected);
    let ids = dense(&connected_components(&mut undirected).unwrap());
    assert_eq!(ids[0], ids[1]);
    assert_eq!(ids[2], ids[3]);
    assert_ne!(ids[0], ids[2]);
    assert_ne!(ids[4], ids[0]);
    assert_ne!(ids[4], ids[2]);

    // The same edges as a directed graph: symmetric, but LAGraph does not
    // know it until it is asked to look.
    let symmetric = adjacency(5, &[(0, 1), (2, 3)], Kind::Undirected);
    let mut directed = Graph::new(symmetric, Kind::Directed).unwrap();
    assert_eq!(directed.is_symmetric_structure(), None);
    let error = connected_components(&mut directed).unwrap_err();
    assert_eq!(error.status, Status::SymmetricStructureRequired);
    assert!(error.message.contains("symmetric"), "{error}");

    assert!(directed.cache_symmetric_structure().unwrap());
    assert!(directed.has_transpose(), "looking computed the transpose");
    directed.check().unwrap();
    assert_eq!(dense(&connected_components(&mut directed).unwrap()), ids);

    let mut one_way = graph(2, &[(0, 1)], Kind::Directed);
    assert!(!one_way.cache_symmetric_structure().unwrap());
    assert_eq!(one_way.is_symmetric_structure(), Some(false));
}

/// Above 1024 nodes and an average degree of 8, LAGraph moves the adjacency
/// matrix's arrays out to sample them and moves them back at the end.
#[test]
fn connected_components_gives_a_large_matrix_back_intact() {
    const HALF: GrB_Index = 1500;
    const REACH: GrB_Index = 6;
    setup();
    // Two rings of `HALF` nodes, each node joined to its next `REACH`.
    let mut edges = Vec::new();
    for ring in 0..2 {
        for i in 0..HALF {
            for step in 1..=REACH {
                edges.push((ring * HALF + i, ring * HALF + (i + step) % HALF));
            }
        }
    }
    let rings = graph(2 * HALF, &edges, Kind::Undirected);
    let stored = rings.adjacency().nvals();
    assert!(stored > 8 * 2 * HALF, "dense enough to be sampled");

    // The engine's matrices are pinned to "sparse or hypersparse"; LAGraph
    // re-pins the matrix to sparse alone while it samples it.
    let either = (sys::GxB_SPARSE | sys::GxB_HYPERSPARSE).cast_signed();
    let field = sys::GxB_Option_Field::GxB_SPARSITY_CONTROL as i32;
    let sparsity_control = |a: &Matrix<bool>| {
        let mut control = 0;
        // SAFETY: a live matrix, a valid out-pointer and an integer field.
        let info = unsafe { sys::GrB_Matrix_get_INT32(a.as_raw(), &raw mut control, field) };
        assert_eq!(info, sys::GrB_Info::GrB_SUCCESS);
        control
    };
    let mut pinned = rings.into_matrix();
    // SAFETY: a live matrix, an integer field and a value it accepts.
    let info = unsafe { sys::GrB_Matrix_set_INT32(pinned.as_raw(), either, field) };
    assert_eq!(info, sys::GrB_Info::GrB_SUCCESS);
    pinned.wait(graphblas::WaitMode::Materialize).unwrap();
    let mut rings = Graph::new(pinned, Kind::Undirected).unwrap();

    let ids = dense(&connected_components(&mut rings).unwrap());
    assert_eq!(
        sparsity_control(rings.adjacency()),
        either,
        "the format setting is put back"
    );
    let (first_ring, second_ring) = ids.split_at(usize::try_from(HALF).unwrap());
    assert!(first_ring.iter().all(|&id| id == first_ring[0]));
    assert!(second_ring.iter().all(|&id| id == second_ring[0]));
    assert_ne!(first_ring[0], second_ring[0]);

    assert_eq!(rings.adjacency().nvals(), stored);
    rings.check().unwrap();
    let a = rings.into_matrix();
    assert_eq!(a.extract_element(0, REACH).unwrap(), Some(true));
    assert_eq!(a.extract_element(HALF, HALF + 1).unwrap(), Some(true));
    assert_eq!(a.extract_element(0, HALF).unwrap(), None);
}

#[test]
fn asking_an_undirected_graph_for_its_transpose_is_not_an_error() {
    setup();
    let mut pair = graph(2, &[(0, 1)], Kind::Undirected);
    // LAGraph answers "not needed", a positive status: the matrix is its
    // own transpose. Nothing is cached and nothing fails.
    pair.cache_transpose().unwrap();
    assert!(!pair.has_transpose());
    pair.cache_in_degree().unwrap();
}

#[test]
fn betweenness_of_a_graph_with_no_nodes_is_empty() {
    setup();
    // What the engine gets for a label no node carries.
    let mut nobody = graph(0, &[], Kind::Directed);
    nobody.cache_transpose().unwrap();
    let centrality = betweenness(&nobody, &[]).unwrap();
    assert_eq!((centrality.size(), centrality.nvals()), (0, 0));
}

#[test]
fn a_matrix_that_is_not_square_is_refused_when_the_graph_is_built() {
    setup();
    let wide = Matrix::<bool>::new(2, 3).unwrap();
    let error = Graph::new(wide, Kind::Directed).unwrap_err();
    assert_eq!(error.status, Status::InvalidGraph);
    assert!(
        error.message.contains("adjacency matrix must be square"),
        "{error}"
    );
}

#[test]
fn into_matrix_returns_the_adjacency_matrix() {
    setup();
    let mut g = graph(3, &[(0, 1), (1, 2)], Kind::Directed);
    g.cache_transpose().unwrap();
    g.cache_out_degree().unwrap();
    g.cache_in_degree().unwrap();
    assert_eq!(g.kind(), Kind::Directed);

    let mut a = g.into_matrix();
    assert_eq!(a.nvals(), 2);
    a.set_element(2, 0, true).unwrap();
    assert_eq!(a.nvals(), 3, "writable again");
}

#[test]
fn the_smallest_weight_is_cached_for_a_weighted_graph() {
    setup();
    let mut weights = Matrix::<f64>::new(3, 3).unwrap();
    weights.set_element(0, 1, 2.5).unwrap();
    weights.set_element(1, 2, 0.5).unwrap();
    let mut g = Graph::new(weights, Kind::Directed).unwrap();
    g.cache_min_weight().unwrap();
    g.check().unwrap();
}
