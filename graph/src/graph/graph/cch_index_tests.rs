//! The CCH indexes as `Graph` holds them: create, drop and lookup through
//! `Graph`'s CCH methods, on hand-held entries and on real graphs built without Cypher.

use std::sync::Arc;

use super::{Graph, TypeId};
use crate::graph::graphblas::matrix::Matrix;
use crate::graph::graphblas::test_init;
use crate::index::falkordb::cch::testkit::{
    Decoys, Materialized, assert_path_valid, dijkstra, grid, materialize, node_count,
    random_sparse, to_input,
};
use crate::index::falkordb::cch::{Cch, CchError, CchIndex, CchIndexDef};
use crate::index::falkordb::versioned::{BuildState, VersionedIndex};

/// Checked before the input is built, so it needs no graph data and no graph input.
#[test]
fn creating_a_held_index_again_is_refused() {
    let mut g = graph("cch_exists");
    let a = CchIndexDef::new(vec![TypeId(2), TypeId(1)], 0);
    hold(&mut g, &a);
    let same_set = CchIndexDef::new(vec![TypeId(1), TypeId(2), TypeId(1)], 0);
    assert_eq!(g.create_cch_index(same_set), Err(CchError::AlreadyExists));
    assert_eq!(
        g.cch_indexes().count(),
        1,
        "the refused create changed nothing"
    );
}

/// Hands a hierarchy straight to the index instead of calling `create_cch_index`, so these
/// tests need no graph input.
fn hold(
    g: &mut Graph,
    def: &CchIndexDef,
) -> Arc<Cch> {
    let cch = Arc::new(Cch::build(&to_input(&grid(2, 2))).expect("build"));
    g.falkordb_index
        .add_cch(
            CchIndex {
                def: def.clone(),
                cch: Arc::clone(&cch),
            },
            BuildState::Ready,
        )
        .expect("not held yet");
    cch
}

/// `GRAPH.MEMORY USAGE` counts a held hierarchy and its type list, and stops counting it once
/// dropped.
#[test]
fn memory_usage_counts_each_held_hierarchy() {
    let mut g = graph("cch_memory");
    let before = g.falkordb_index.memory_usage();
    let a = CchIndexDef::new(vec![TypeId(1), TypeId(2)], 0);
    let cch = hold(&mut g, &a);
    assert_eq!(
        g.falkordb_index.memory_usage() - before,
        cch.memory_usage() + 2 * size_of::<TypeId>()
    );
    assert_eq!(g.drop_cch_index(&a), Ok(()));
    assert_eq!(g.falkordb_index.memory_usage(), before);
}

fn graph(name: &str) -> Graph {
    test_init::ensure_init();
    Graph::new(16, 16, 1, 0, name)
}

/// Nothing builds an index in the background yet, so this sets the state by hand: a
/// building entry is held and listed, and refuses a second create, but is not served.
#[test]
fn a_building_index_is_held_but_not_served() {
    let mut g = graph("cch_building");
    let a = CchIndexDef::new(vec![TypeId(1)], 0);
    hold(&mut g, &a);
    CchIndex::entry_mut(&mut g.falkordb_index, &a)
        .expect("held")
        .state = BuildState::building(());

    assert!(g.cch_index(&a).is_none(), "a building index cannot answer");
    assert_eq!(g.cch_indexes().collect::<Vec<_>>(), vec![&a]);
    assert_eq!(g.create_cch_index(a.clone()), Err(CchError::AlreadyExists));
    assert_eq!(g.drop_cch_index(&a), Ok(()));
}

/// Lookup, listing and drop by definition: the same type under two weights is two
/// indexes, and dropping one leaves the other.
#[test]
fn drop_removes_only_the_named_index() {
    let mut g = graph("cch_drop");
    let a = CchIndexDef::new(vec![TypeId(1)], 0);
    let b = CchIndexDef::new(vec![TypeId(1)], 1);
    assert_eq!(g.cch_indexes().count(), 0);
    assert!(g.cch_index(&a).is_none());
    let held = hold(&mut g, &a);
    hold(&mut g, &b);
    assert!(Arc::ptr_eq(&g.cch_index(&a).expect("held").cch, &held));
    assert!(
        g.cch_index(&CchIndexDef::new(vec![TypeId(1), TypeId(2)], 0))
            .is_none(),
        "a different rel set is a different index"
    );
    assert_eq!(g.cch_indexes().collect::<Vec<_>>(), vec![&a, &b]);

    assert_eq!(g.drop_cch_index(&a), Ok(()));
    assert!(g.cch_index(&a).is_none());
    assert!(g.cch_index(&b).is_some());
    assert_eq!(g.cch_indexes().collect::<Vec<_>>(), vec![&b]);
    assert_eq!(
        g.drop_cch_index(&a),
        Err(CchError::NotFound),
        "already dropped"
    );
}

/// MVCC: a version taken before a drop keeps the hierarchy; the drop only affects the
/// version it ran on. Likewise the version a reader holds does not grow an index because
/// a writer created one on the next version, and throwing the next version away (the
/// rollback path) leaves the reader's version exactly as it was.
#[test]
fn a_create_or_drop_on_a_new_version_leaves_the_old_one_as_it_was() {
    let mut old = graph("cch_mvcc_drop");
    let a = CchIndexDef::new(vec![TypeId(3)], 0);
    let held = hold(&mut old, &a);

    let mut next = old.new_version();
    assert_eq!(next.drop_cch_index(&a), Ok(()));
    assert!(next.cch_index(&a).is_none());
    let kept = old.cch_index(&a).expect("old version still holds it");
    assert!(Arc::ptr_eq(&kept.cch, &held));

    let m = materialize(
        &random_sparse(10, 25, 4, 3),
        Decoys::default(),
        "cch_mvcc_create",
    );
    let (reader, def) = (m.graph, m.def);
    let mut writer = reader.new_version();
    writer.create_cch_index(def.clone()).expect("create");
    assert!(writer.cch_index(&def).is_some());
    assert!(
        reader.cch_index(&def).is_none(),
        "reader's version is untouched"
    );
    drop(writer); // rollback
    assert_eq!(reader.cch_indexes().count(), 0);
}

// ---- end to end: a real Graph, built without Cypher -------------------------------

/// Every ordered pair of real nodes: weight against the kit's Dijkstra, nodes against
/// its path checker. Node ids past the oracle's are capacity slots and are skipped.
fn assert_queries_match(
    cch: &Cch,
    oracle: &Matrix<f64>,
) {
    let n = node_count(oracle) as u64;
    assert!(
        cch.node_count() as u64 >= n,
        "the hierarchy spans every node"
    );
    for s in 0..n {
        let truth = dijkstra(oracle, s);
        for t in 0..n {
            match cch.query(s, t) {
                Some((w, nodes)) => {
                    assert_eq!(w, truth[t as usize], "{s}->{t} weight");
                    assert_path_valid(oracle, s, t, w, &nodes);
                }
                None => assert!(truth[t as usize].is_infinite(), "{s}->{t} missed"),
            }
        }
    }
}

/// A materialized graph for `oracle` with `decoys`, its index created.
fn indexed(
    oracle: &Matrix<f64>,
    decoys: Decoys,
    name: &str,
) -> Materialized {
    let mut m = materialize(oracle, decoys, name);
    m.graph.create_cch_index(m.def.clone()).expect("create");
    m
}

/// The whole stack: `Graph` -> input -> Phase 1/2 -> query. Parallel edges on both indexed
/// types, ties across them, closed edges, self-loops and a third, unindexed relationship type
/// whose edges are cheaper than all: none may be used, and between two nodes the cheapest
/// edge over the covered types wins.
#[test]
fn a_created_index_answers_like_dijkstra_on_the_real_graph() {
    let decoys = Decoys {
        heavier: 2,
        closed: true,
        unindexed: true,
        self_loops: true,
        ties: true,
    };
    for seed in 0..4u64 {
        let oracle = random_sparse(24, 70, 6, seed);
        let m = indexed(&oracle, decoys, &format!("cch_e2e_{seed}"));
        let idx = m.graph.cch_index(&m.def).expect("held after create");
        assert_queries_match(&idx.cch, &oracle);
    }
}
