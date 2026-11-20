//! The experimental algorithms, against the real libraries.

#![allow(
    clippy::float_cmp,
    reason = "the expected values are small whole numbers, which an f64 holds exactly"
)]

use std::{
    sync::{Once, mpsc},
    thread,
};

use graphblas::{ErrorKind, GrB_Index, GrbType, JitControl, Matrix, Vector};
use lagraph::{
    FlowOutputs, Graph, Kind, Status, Traversal, breadth_first_search_extended,
    community_detection_label_propagation, harmonic_centrality, max_flow, minimum_spanning_forest,
};

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

/// An `n`-node matrix from `(from, to, weight)`.
fn weighted<T: GrbType>(
    n: GrB_Index,
    edges: &[(GrB_Index, GrB_Index, T)],
) -> Matrix<T> {
    let mut a = Matrix::new(n, n).unwrap();
    for &(i, j, weight) in edges {
        a.set_element(i, j, weight).unwrap();
    }
    a
}

/// The same edges in both directions.
fn symmetric<T: GrbType>(
    n: GrB_Index,
    edges: &[(GrB_Index, GrB_Index, T)],
) -> Matrix<T> {
    let mut a = weighted(n, edges);
    for &(i, j, weight) in edges {
        a.set_element(j, i, weight).unwrap();
    }
    a
}

fn pattern(
    n: GrB_Index,
    edges: &[(GrB_Index, GrB_Index)],
) -> Matrix<bool> {
    weighted(
        n,
        &edges.iter().map(|&(i, j)| (i, j, true)).collect::<Vec<_>>(),
    )
}

/// The stored entries as `(index, value)`, in index order.
fn stored<T: GrbType>(v: &Vector<T>) -> Vec<(GrB_Index, T)> {
    let (ids, values) = v.extract_tuples().unwrap();
    let mut pairs: Vec<_> = ids.into_iter().zip(values).collect();
    pairs.sort_by_key(|&(id, _)| id);
    pairs
}

#[test]
fn breadth_first_search_gives_levels_and_parents() {
    setup();
    // 0 -> 1 -> 2 -> 3, and 4 out of reach.
    let path = Graph::new(pattern(5, &[(0, 1), (1, 2), (2, 3)]), Kind::Directed).unwrap();

    let all = breadth_first_search_extended(Traversal::Both, &path, 0, None, None, false).unwrap();
    assert_eq!(
        stored(&all.level.unwrap()),
        [(0, 0), (1, 1), (2, 2), (3, 3)]
    );
    assert_eq!(
        stored(&all.parent.unwrap()),
        [(0, 0), (1, 0), (2, 1), (3, 2)]
    );

    let near =
        breadth_first_search_extended(Traversal::Levels, &path, 0, Some(1), None, false).unwrap();
    assert_eq!(stored(&near.level.unwrap()), [(0, 0), (1, 1)]);
    assert!(near.parent.is_none(), "not asked for");

    let until =
        breadth_first_search_extended(Traversal::Parents, &path, 0, None, Some(2), false).unwrap();
    assert!(until.level.is_none());
    assert_eq!(stored(&until.parent.unwrap()), [(0, 0), (1, 0), (2, 1)]);

    let error =
        breadth_first_search_extended(Traversal::Both, &path, 9, None, None, false).unwrap_err();
    assert_eq!(error.status, Status::GraphBlas(ErrorKind::InvalidIndex));
    assert!(error.message.contains("invalid source node"), "{error}");
}

#[test]
fn breadth_first_search_runs_on_one_node() {
    setup();
    let lone = Graph::new(pattern(1, &[]), Kind::Directed).unwrap();
    let found = breadth_first_search_extended(Traversal::Both, &lone, 0, None, None, true).unwrap();
    assert_eq!(stored(&found.level.unwrap()), [(0, 0)]);
    assert_eq!(stored(&found.parent.unwrap()), [(0, 0)]);
}

#[test]
fn harmonic_centrality_scores_the_listed_nodes() {
    setup();
    // 0 -> 1 -> 2: node 0 reaches 1 at distance 1 and 2 at distance 2.
    let path = Graph::new(pattern(3, &[(0, 1), (1, 2)]), Kind::Directed).unwrap();
    let mut everyone = Vector::<bool>::new(3).unwrap();
    for node in 0..3 {
        everyone.set_element(node, true).unwrap();
    }

    let found = harmonic_centrality(true, &path, &everyone).unwrap();
    let scores = stored(&found.scores);
    let nodes: Vec<_> = scores.iter().map(|&(node, _)| node).collect();
    assert_eq!(nodes, [0, 1, 2], "one score per listed node");
    // HyperLogLog estimates; exact values are 1.5, 1 and 0.
    for ((node, score), exact) in scores.iter().zip([1.5, 1.0, 0.0]) {
        assert!(
            (score - exact).abs() < 0.1,
            "node {node}: {score} vs {exact}"
        );
    }
    assert_eq!(found.reachable.expect("asked for").size(), 3);

    let found = harmonic_centrality(false, &path, &everyone).unwrap();
    assert!(found.reachable.is_none());

    let mut negative = Vector::<i64>::new(3).unwrap();
    negative.set_element(0, -1).unwrap();
    let error = harmonic_centrality(false, &path, &negative).unwrap_err();
    assert_eq!(error.status, Status::GraphBlas(ErrorKind::InvalidValue));
    assert!(error.message.contains("Negative node weights"), "{error}");
}

#[test]
fn harmonic_centrality_checks_the_weights_and_copes_with_nobody() {
    setup();
    let path = Graph::new(pattern(3, &[(0, 1), (1, 2)]), Kind::Directed).unwrap();

    // No node takes part: LAGraph returns early, with one output missing.
    let nobody = Vector::<bool>::new(3).unwrap();
    let found = harmonic_centrality(true, &path, &nobody).unwrap();
    assert_eq!(found.scores.nvals(), 0);
    let reachable = found.reachable.expect("asked for");
    assert_eq!((reachable.size(), reachable.nvals()), (3, 0));

    // LAGraph would take this vector and size the scores to it.
    let mut too_long = Vector::<bool>::new(8).unwrap();
    too_long.set_element(0, true).unwrap();
    let error = harmonic_centrality(false, &path, &too_long).unwrap_err();
    assert_eq!(
        error.status,
        Status::GraphBlas(ErrorKind::DimensionMismatch)
    );
}

#[test]
fn max_flow_on_one_node_fails_inside_lagraph() {
    setup();
    // Not what anyone wants, and pinned so that an upstream fix is noticed:
    // LAGraph searches a 1-by-1 matrix it builds itself, which GraphBLAS
    // holds by column, and its own graph check refuses that.
    let mut lone = Graph::new(weighted(1, &[(0, 0, 1.0)]), Kind::Directed).unwrap();
    lone.cache_min_weight().unwrap();
    lone.cache_transpose().unwrap();
    let error = max_flow(FlowOutputs::default(), &lone, 0, 0).unwrap_err();
    assert_eq!(error.status, Status::InvalidGraph);

    // Source and sink being the same node is not the reason.
    let mut pair = Graph::new(weighted(2, &[(0, 1, 1.0)]), Kind::Directed).unwrap();
    pair.cache_min_weight().unwrap();
    pair.cache_transpose().unwrap();
    assert_eq!(
        max_flow(FlowOutputs::default(), &pair, 0, 0).unwrap().value,
        0.0
    );
}

#[test]
fn label_propagation_separates_two_triangles() {
    setup();
    let edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
    let both: Vec<_> = edges.iter().map(|&(i, j)| (i, j, true)).collect();
    let triangles = Graph::new(symmetric(6, &both), Kind::Undirected).unwrap();

    let labels = community_detection_label_propagation(&triangles, 10).unwrap();
    let labels: Vec<u64> = stored(&labels)
        .into_iter()
        .map(|(_, label)| label)
        .collect();
    assert_eq!(
        labels,
        [0, 0, 0, 3, 3, 3],
        "the smallest id of each triangle"
    );
}

#[test]
fn minimum_spanning_forest_keeps_the_light_edges() {
    setup();
    // A triangle 0-1-2 whose heaviest edge is 0-2, and a separate edge 3-4.
    let edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0), (3, 4, 7.0)];
    let a = symmetric(5, &edges);

    let forest = minimum_spanning_forest(true, &a, false).unwrap();
    let (_, _, weights) = forest.edges.extract_tuples().unwrap();
    assert_eq!(weights.len(), 3, "five nodes in two trees");
    assert_eq!(weights.iter().sum::<f64>(), 10.0);

    let ids: Vec<u64> = stored(&forest.component.unwrap())
        .into_iter()
        .map(|(_, id)| id)
        .collect();
    assert_eq!(ids[0], ids[1]);
    assert_eq!(ids[1], ids[2]);
    assert_eq!(ids[3], ids[4]);
    assert_ne!(ids[0], ids[3]);

    // One direction only: `sanitize` supplies the other. Integer weights
    // come back as `i64`.
    let one_way = weighted::<u32>(3, &[(0, 1, 4), (1, 2, 6), (0, 2, 9)]);
    let forest = minimum_spanning_forest(false, &one_way, true).unwrap();
    assert!(forest.component.is_none());
    let (_, _, weights): (_, _, Vec<i64>) = forest.edges.extract_tuples().unwrap();
    assert_eq!(weights.iter().sum::<i64>(), 10);

    let wide = Matrix::<f64>::new(2, 3).unwrap();
    let error = minimum_spanning_forest(false, &wide, false).unwrap_err();
    assert_eq!(
        error.status,
        Status::GraphBlas(ErrorKind::DimensionMismatch)
    );
}

#[test]
fn max_flow_needs_its_cached_properties_and_finds_the_flow() {
    setup();
    // Capacity 5 leaves the source and capacity 5 enters the sink.
    let capacities = [
        (0, 1, 3.0),
        (0, 2, 2.0),
        (1, 2, 1.0),
        (1, 3, 2.0),
        (2, 3, 3.0),
    ];
    let mut network = Graph::new(weighted(4, &capacities), Kind::Directed).unwrap();
    let none = FlowOutputs::default();

    let error = max_flow(none, &network, 0, 3).unwrap_err();
    assert_eq!(
        error.status,
        Status::GraphBlas(ErrorKind::UninitializedObject)
    );

    network.cache_min_weight().unwrap();
    let error = max_flow(none, &network, 0, 3).unwrap_err();
    assert_eq!(error.status, Status::NotCached);
    assert!(error.message.contains("G->AT is required"), "{error}");

    network.cache_transpose().unwrap();
    let found = max_flow(none, &network, 0, 3).unwrap();
    assert_eq!(found.value, 5.0);
    assert!(found.flow.is_none() && found.residual.is_none());

    let with_flow = FlowOutputs {
        flow: true,
        residual: false,
    };
    let found = max_flow(with_flow, &network, 0, 3).unwrap();
    assert_eq!(found.value, 5.0);
    let flow = found.flow.unwrap();
    let out_of_source: f64 = (1..4)
        .filter_map(|node| flow.extract_element(0, node).unwrap())
        .sum();
    assert_eq!(out_of_source, 5.0);
    assert!(found.residual.is_none());

    let with_residual = FlowOutputs {
        flow: false,
        residual: true,
    };
    let found = max_flow(with_residual, &network, 0, 3).unwrap();
    assert_eq!(found.value, 5.0);
    assert_eq!(found.residual.unwrap().nrows(), 4);

    // At the pinned revision, building the flow matrix frees the operator
    // the residual is built with. When LAGraph fixes that, this starts to
    // pass: drop the note on `FlowOutputs` and assert both outputs here.
    let both = FlowOutputs {
        flow: true,
        residual: true,
    };
    let error = max_flow(both, &network, 0, 3).unwrap_err();
    assert_eq!(error.status, Status::GraphBlas(ErrorKind::NullPointer));

    let error = max_flow(none, &network, 0, 9).unwrap_err();
    assert_eq!(error.status, Status::GraphBlas(ErrorKind::InvalidValue));
}
