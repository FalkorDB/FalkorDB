//! The CCH indexes as `Graph` holds them: create, drop and lookup through
//! `Graph`'s CCH methods, on hand-held entries and on real graphs built without Cypher.

use std::sync::Arc;

use super::{Graph, TypeId};
use crate::graph::graphblas::matrix::Matrix;
use crate::graph::graphblas::test_init;
use crate::index::falkordb::cch::testkit::{
    Decoys, Materialized, assert_path_valid, dijkstra, grid, materialize, node_count,
    random_sparse, to_input, weights,
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
        .add_cch(CchIndex::built(def.clone(), Arc::clone(&cch)));
    cch
}

/// `GRAPH.MEMORY USAGE` counts each held hierarchy, at least its two weight arrays (one
/// `f64` per arc and direction) and both copies of its type list (the entry's and the map
/// key's), and stops counting it once dropped. Two indexes over the same graph and as many
/// types count the same, each on its own.
#[test]
fn memory_usage_counts_each_held_hierarchy() {
    let mut g = graph("cch_memory");
    let before = g.falkordb_index.memory_usage();
    let a = CchIndexDef::new(vec![TypeId(1), TypeId(2)], 0);
    let b = CchIndexDef::new(vec![TypeId(3), TypeId(4)], 0);

    let cch = hold(&mut g, &a);
    let one = g.falkordb_index.memory_usage() - before;
    let arc_count = cch.topology().arc_count();
    assert!(arc_count > 0, "the fixture has arcs to count");
    assert!(
        one >= 2 * arc_count * size_of::<f64>(),
        "{one} bytes cannot hold both weight arrays of {arc_count} arcs"
    );
    assert_eq!(
        one,
        cch.memory_usage() + 2 * size_of_val(a.rel_types()),
        "the hierarchy and both copies of the type list"
    );
    hold(&mut g, &b);
    assert_eq!(g.falkordb_index.memory_usage() - before, 2 * one);

    assert_eq!(g.drop_cch_index(&a), Ok(()));
    assert_eq!(g.falkordb_index.memory_usage() - before, one);
    assert_eq!(g.drop_cch_index(&b), Ok(()));
    assert_eq!(g.falkordb_index.memory_usage(), before);
}

/// A create that fails holds nothing: an unknown relationship type, an unknown weight
/// attribute, and a negative weight on an indexed relationship each leave the version with
/// no index and its memory usage as it was.
#[test]
fn a_failed_create_holds_nothing() {
    let m = materialize(
        &weights(3, &[(0, 1, 2.0), (1, 2, -1.0)]),
        Decoys::default(),
        "cch_failed_create",
    );
    let mut g = m.graph;
    let unknown_type = CchIndexDef::new(vec![TypeId(99)], 0);
    let unknown_attribute = CchIndexDef::new(m.def.rel_types().to_vec(), u16::MAX);
    let before = g.falkordb_index.memory_usage();
    for (def, refused) in [
        (unknown_type, CchError::UnknownRelType),
        (unknown_attribute, CchError::UnknownAttribute),
        (m.def, CchError::InvalidWeight),
    ] {
        assert_eq!(g.create_cch_index(def), Err(refused));
        assert_eq!(g.cch_indexes().count(), 0);
        assert_eq!(g.falkordb_index.memory_usage(), before);
    }
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
    assert!(Arc::ptr_eq(
        &g.cch_index(&a).expect("held").cch().expect("built"),
        &held
    ));
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
    assert!(Arc::ptr_eq(kept.cch().expect("built"), &held));

    let m = materialize(
        &random_sparse(10, 25, 4, 3),
        Decoys::default(),
        "cch_mvcc_create",
    );
    let (reader, def) = (m.graph, m.def);
    let mut writer = reader.new_version();
    writer.build_cch_index_now(def.clone()).expect("create");
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
    m.graph.build_cch_index_now(m.def.clone()).expect("create");
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
        assert_queries_match(idx.cch().expect("built"), &oracle);
    }
}

/// The background build: a `CREATE` holds the index `Building`, and the index builder's one batch
/// builds the hierarchy off the lock and adopts it through `commit_indexer_batch`, as it does a
/// column's batches. A test runs the job's batch itself ([`Job::step`]).
mod background_build {
    use std::sync::Arc;

    use roaring::RoaringTreemap;
    use rustc_hash::FxHashMap;

    use super::super::falkordb_index_tests::{TestHost, builder_on, job_on};
    use super::{Graph, TypeId, assert_queries_match};
    use crate::graph::graphblas::matrix::Matrix;
    use crate::graph::mvcc_graph::MvccGraph;
    use crate::index::falkordb::cch::testkit::{INDEXED, RealEdge, index_def, real_graph, weights};
    use crate::index::falkordb::cch::{CchError, CchIndexDef, Hierarchy, maint};
    use crate::index::falkordb::online_build::{Job, Sizing, Step};
    use crate::index::falkordb::versioned::{BuildState, IndexId};
    use crate::index::index_docs::IndexDocs;
    use crate::runtime::value::Value;

    /// `0 -> 1 -> 2` at 1.0 each and `0 -> 2` at 5.0 on `T0`, `2 -> 0` at 7.0 on `T1`; edge ids
    /// in that order.
    const EDGES: [RealEdge; 4] = [
        (0, 1, Some(1.0), INDEXED[0]),
        (1, 2, Some(1.0), INDEXED[0]),
        (0, 2, Some(5.0), INDEXED[0]),
        (2, 0, Some(7.0), INDEXED[1]),
    ];

    /// The oracle of [`EDGES`], with `0 -> 1` weighing `first`.
    fn oracle(first: f64) -> Matrix<f64> {
        weights(3, &[(0, 1, first), (1, 2, 1.0), (0, 2, 5.0), (2, 0, 7.0)])
    }

    /// [`EDGES`] in a graph behind a test host, node 3 with no edge, with no index yet, the
    /// index over them, and its weight attribute.
    fn host(name: &str) -> (Arc<TestHost>, CchIndexDef, u16) {
        let (graph, attr) = real_graph(4, &EDGES, name);
        let def = index_def(&graph, attr);
        (TestHost::new(MvccGraph::from_graph(graph)), def, attr)
    }

    /// `CREATE CCH INDEX` as a writer commits it.
    fn create(
        host: &TestHost,
        def: &CchIndexDef,
    ) {
        host.write(|g| g.create_cch_index(def.clone()).expect("create"));
    }

    /// The job that builds the index's published `Building` generation.
    fn job(
        host: &Arc<TestHost>,
        def: &CchIndexDef,
    ) -> Job {
        job_on(host, &IndexId::Cch(def.clone()), Sizing::fixed(1))
    }

    /// [`job`], whose batch runs `while_reading` on its version once it has built the
    /// hierarchy, before its commit.
    fn job_running(
        host: &Arc<TestHost>,
        def: &CchIndexDef,
        while_reading: impl FnOnce(&Graph) + Send + 'static,
    ) -> Job {
        let builder = builder_on(host, Sizing::fixed(1));
        *builder.while_reading.lock() = Some(Box::new(while_reading));
        let generation = host
            .mvcc
            .read()
            .falkordb_index
            .cch_entry(def)
            .expect("held")
            .generation;
        Job::new(
            builder,
            Arc::downgrade(&host.mvcc),
            IndexId::Cch(def.clone()),
            generation,
        )
    }

    /// The published `Building` generation of the index over `def`.
    fn generation_of(
        host: &TestHost,
        def: &CchIndexDef,
    ) -> u64 {
        host.mvcc
            .read()
            .falkordb_index
            .cch_entry(def)
            .expect("held")
            .generation
    }

    /// Create a `T0` edge `src -> dst` at `weight`, flushed as a writer's unit of work is; its id.
    fn add_edge(
        g: &mut Graph,
        attr: u16,
        src: u64,
        dst: u64,
        weight: f64,
    ) -> u64 {
        let mut docs = IndexDocs::default();
        let ids = g
            .relationship_id_space()
            .reserve(1, &RoaringTreemap::new())
            .expect("reserved");
        g.create_relationships_bulk(
            &Arc::new(String::from(INDEXED[0])),
            &[src],
            &[dst],
            &ids,
            &mut docs,
        )
        .expect("creatable");
        let weights = FxHashMap::from_iter([(ids[0], vec![(attr, Value::Float(weight))])]);
        g.set_relationships_attributes(&weights, &mut docs)
            .expect("valid");
        g.apply_cch_writes(&mut docs).expect("flushed");
        ids[0]
    }

    /// Set edge `id`'s weight to `weight`, flushed as a writer's unit of work is.
    fn set_weight(
        g: &mut Graph,
        attr: u16,
        id: u64,
        weight: f64,
    ) {
        let attrs = FxHashMap::from_iter([(id, vec![(attr, Value::Float(weight))])]);
        let mut docs = IndexDocs::default();
        g.set_relationships_attributes(&attrs, &mut docs)
            .expect("a valid weight");
        g.apply_cch_writes(&mut docs).expect("flushed");
    }

    /// Set edge 0's weight to `weight`, as a writer's unit of work does.
    fn set_first_weight(
        g: &mut Graph,
        attr: u16,
        weight: f64,
    ) {
        let attrs = FxHashMap::from_iter([(0, vec![(attr, Value::Float(weight))])]);
        let mut docs = IndexDocs::default();
        g.set_relationships_attributes(&attrs, &mut docs)
            .expect("a valid weight");
        g.apply_cch_writes(&mut docs).expect("flushed");
    }

    /// The commit that brings the `Building` index in hands it to the builder, under its
    /// generation; a later commit does not hand it over again, and an RDB load's synchronous
    /// build brings it in `Ready`, which is never handed over.
    #[test]
    fn a_commit_hands_the_builder_a_created_index_once() {
        let (host, def, _) = host("cch_bg_handed");
        let before = host.mvcc.read();
        let mut created = before.new_version();
        created.create_cch_index(def.clone()).expect("create");
        let generation = created
            .falkordb_index
            .cch_entry(&def)
            .expect("held")
            .generation;
        assert_eq!(
            created.indexes_to_build_since(&before),
            vec![(IndexId::Cch(def.clone()), generation)]
        );
        let later = created.new_version();
        assert!(later.indexes_to_build_since(&created).is_empty());

        let mut loaded = before.new_version();
        loaded.build_cch_index_now(def).expect("build");
        assert!(loaded.indexes_to_build_since(&before).is_empty());
    }

    /// A write the built hierarchy cannot take (an edge between nodes it does not join) is not
    /// built in the write path: the entry goes back to `Building` under a new generation, the
    /// commit hands it to the builder, and the builder's batch adopts a hierarchy with the edge.
    #[test]
    fn a_structural_write_hands_the_rebuild_to_the_builder() {
        let (host, def, attr) = host("cch_bg_structural");
        host.write(|g| g.build_cch_index_now(def.clone()).expect("build"));
        let before = host.mvcc.read();
        let generation = before
            .falkordb_index
            .cch_entry(&def)
            .expect("held")
            .generation;
        host.write(|g| {
            let mut docs = IndexDocs::default();
            let ids = g
                .relationship_id_space()
                .reserve(1, &RoaringTreemap::new())
                .expect("reserved");
            g.create_relationships_bulk(
                &Arc::new(String::from(INDEXED[0])),
                &[0],
                &[3],
                &ids,
                &mut docs,
            )
            .expect("creatable");
            let weights = FxHashMap::from_iter([(ids[0], vec![(attr, Value::Float(0.5))])]);
            g.set_relationships_attributes(&weights, &mut docs)
                .expect("valid");
            g.apply_cch_writes(&mut docs).expect("flushed");
        });
        let published = host.mvcc.read();
        let entry = published.falkordb_index.cch_entry(&def).expect("held");
        assert!(entry.is_building(), "no rebuild in the write");
        assert_ne!(entry.generation, generation);
        assert!(
            before.cch_index(&def).is_some(),
            "the version before keeps its hierarchy"
        );
        assert_eq!(job(&host, &def).step(), Step::Done);
        let adopted = host.mvcc.read();
        let cch = adopted
            .cch_index(&def)
            .expect("built")
            .cch()
            .expect("built");
        assert_eq!(cch.query(0, 3).map(|(weight, _)| weight), Some(0.5));
    }

    /// A write while the index is `Building` records the ids of the edges it wrote in the
    /// index's written set, which its commit publishes for the build's commit to catch up on.
    #[test]
    fn a_write_while_building_records_its_edge_ids() {
        let (host, def, attr) = host("cch_bg_records");
        create(&host, &def);
        host.write(|g| {
            let mut docs = IndexDocs::default();
            let weights = FxHashMap::from_iter([(1, vec![(attr, Value::Float(3.0))])]);
            g.set_relationships_attributes(&weights, &mut docs)
                .expect("valid");
            g.apply_cch_writes(&mut docs).expect("flushed");
        });
        let published = host.mvcc.read();
        let entry = published.falkordb_index.cch_entry(&def).expect("held");
        let BuildState::Building { written, .. } = &entry.state else {
            panic!("still building");
        };
        assert!(written.since_last_indexer_step.contains(1));
        assert!(written.by_this_version.is_empty(), "published");
    }

    /// A create holds the index `Building` with no hierarchy, and its commit leaves it so: the
    /// builder's one batch builds it, then the index answers like Dijkstra.
    #[test]
    fn a_create_returns_building_and_one_batch_builds_it() {
        let (host, def, _) = host("cch_bg_create");
        create(&host, &def);
        let published = host.mvcc.read();
        let entry = published.falkordb_index.cch_entry(&def).expect("held");
        assert!(entry.is_building());
        assert!(matches!(entry.index.hierarchy, Hierarchy::Pending));
        assert!(
            published.cch_index(&def).is_none(),
            "nothing to answer with yet"
        );

        assert_eq!(job(&host, &def).step(), Step::Done);
        let published = host.mvcc.read();
        let index = published.cch_index(&def).expect("built");
        assert_queries_match(index.cch().expect("built"), &oracle(1.0));
    }

    /// A weight written while the build reads is in the hierarchy its commit adopts: the commit
    /// catches up on the edges writers wrote, scoped.
    #[test]
    fn a_weight_written_during_the_build_is_caught_up() {
        let (host, def, attr) = host("cch_bg_write");
        create(&host, &def);
        let writer = Arc::clone(&host);
        let mut job = job_running(&host, &def, move |_| {
            writer.write(|g| set_first_weight(g, attr, 10.0));
        });
        assert_eq!(job.step(), Step::Done);
        let published = host.mvcc.read();
        let cch = published
            .cch_index(&def)
            .expect("built")
            .cch()
            .expect("built");
        assert_queries_match(cch, &oracle(10.0));
    }

    /// An edge written while the build reads that the new hierarchy does not join starts the
    /// build again, under the same generation: the same job's next batch reads the newer
    /// version and adopts a hierarchy with the edge.
    #[test]
    fn a_structural_write_during_the_build_starts_it_again() {
        let (host, def, attr) = host("cch_bg_restart");
        create(&host, &def);
        let generation = generation_of(&host, &def);
        let writer = Arc::clone(&host);
        let mut job = job_running(&host, &def, move |_| {
            writer.write(|g| {
                add_edge(g, attr, 0, 3, 0.5);
            });
        });
        assert_eq!(job.step(), Step::More, "built again");
        assert_eq!(generation_of(&host, &def), generation);
        assert_eq!(job.step(), Step::Done);
        let published = host.mvcc.read();
        let cch = published
            .cch_index(&def)
            .expect("built")
            .cch()
            .expect("built");
        assert_eq!(cch.query(0, 3).map(|(weight, _)| weight), Some(0.5));
    }

    /// Writers that overtake every build with a change it cannot take: the `MAX_REDOS`-th (5th)
    /// build in a row overtaken that way gives the index up as failed, after four restarts,
    /// with that reason, and no job runs for it.
    #[test]
    fn a_build_overtaken_too_often_is_failed() {
        let (host, def, attr) = host("cch_bg_redos");
        create(&host, &def);
        let mut steps = Vec::new();
        for round in 0..6 {
            let writer = Arc::clone(&host);
            let mut job = job_running(&host, &def, move |_| {
                writer.write(|g| {
                    // A new node, joined to node 0: never in the hierarchy being built.
                    let node = g
                        .node_id_space()
                        .reserve(1, &RoaringTreemap::new())
                        .expect("reserved")[0];
                    g.create_nodes(&RoaringTreemap::from_iter([node]))
                        .expect("creatable");
                    add_edge(g, attr, 0, node, 1.0 + f64::from(round));
                });
            });
            let step = job.step();
            steps.push(step);
            if step == Step::Done {
                break;
            }
        }
        assert_eq!(
            steps,
            [Step::More, Step::More, Step::More, Step::More, Step::Done]
        );
        let published = host.mvcc.read();
        let entry = published.falkordb_index.cch_entry(&def).expect("held");
        assert!(!entry.is_building());
        assert_eq!(
            entry.index.failure(),
            Some("structural writes kept overtaking its build")
        );
    }

    /// A writer that forked while the index was building, wrote a weight, and escalates after
    /// the adoption: its rebase takes its write into the adopted hierarchy, so what it commits
    /// answers with its own write. One that wrote an edge the hierarchy cannot take puts the
    /// index back to `Building` under a new generation instead, for the builder.
    #[test]
    fn a_writer_rebased_onto_an_adoption_takes_its_writes() {
        for structural in [false, true] {
            let (host, def, attr) = host(&format!("cch_bg_rebase_writes_{structural}"));
            create(&host, &def);
            let mut fork = host.mvcc.fork();
            if structural {
                add_edge(&mut fork, attr, 0, 3, 0.5);
            } else {
                set_first_weight(&mut fork, attr, 10.0);
            }
            assert_eq!(job(&host, &def).step(), Step::Done);
            let adopted = host
                .mvcc
                .read()
                .falkordb_index
                .cch_entry(&def)
                .expect("held")
                .generation;
            host.mvcc.rebase_onto_indexer_batches(&mut fork);
            host.mvcc.commit(fork).expect("the writer commits");
            let published = host.mvcc.read();
            let entry = published.falkordb_index.cch_entry(&def).expect("held");
            if structural {
                assert!(entry.is_building() && entry.generation != adopted);
                assert_eq!(job(&host, &def).step(), Step::Done);
                let built = host.mvcc.read();
                let cch = built.cch_index(&def).expect("built").cch().expect("built");
                assert_eq!(cch.query(0, 3).map(|(weight, _)| weight), Some(0.5));
            } else {
                let cch = published
                    .cch_index(&def)
                    .expect("built")
                    .cch()
                    .expect("built");
                assert_queries_match(cch, &oracle(10.0));
            }
        }
    }

    /// A writer whose rebase onto an adoption fails to take its writes (a resource limit, a bug)
    /// does not fail the healthy index: the index goes back to `Building` under a new
    /// generation, the builder builds it again, and the result has the writer's write.
    #[test]
    fn a_rebase_that_cannot_take_its_writes_restarts_the_build() {
        let (host, def, attr) = host("cch_bg_rebase_error");
        create(&host, &def);
        let mut fork = host.mvcc.fork();
        set_first_weight(&mut fork, attr, 10.0);
        assert_eq!(job(&host, &def).step(), Step::Done);
        let adopted = host
            .mvcc
            .read()
            .falkordb_index
            .cch_entry(&def)
            .expect("held")
            .generation;
        maint::fail_next_update(CchError::Internal("injected"));
        host.mvcc.rebase_onto_indexer_batches(&mut fork);
        host.mvcc.commit(fork).expect("the writer commits");
        let published = host.mvcc.read();
        let entry = published.falkordb_index.cch_entry(&def).expect("held");
        assert!(entry.is_building() && entry.generation != adopted);
        assert_eq!(entry.index.failure(), None);
        assert_eq!(job(&host, &def).step(), Step::Done);
        let built = host.mvcc.read();
        let cch = built.cch_index(&def).expect("built").cch().expect("built");
        assert_queries_match(cch, &oracle(10.0));
    }

    /// Random writes, weight changes and new edges among the existing nodes, before a build
    /// reads, while it reads, or on the built index, round after round: each published version
    /// that holds the index built answers like Dijkstra on its own graph. A new edge between
    /// nodes the hierarchy does not join sends the index back to the builder.
    #[test]
    fn writes_interleaved_with_builds_keep_every_published_version_exact() {
        use parking_lot::Mutex;
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        /// The `T0` edges as the graph holds them, `(src, dst, weight, id)`.
        type Mirror = Vec<(u64, u64, f64, u64)>;

        fn write(
            state: &Mutex<(SmallRng, Mirror)>,
            attr: u16,
            g: &mut Graph,
        ) {
            let mut guard = state.lock();
            let (rng, mirror) = &mut *guard;
            let weight = f64::from(rng.random_range(1..9_u32));
            if rng.random_range(0..2) == 0 {
                let at = rng.random_range(0..mirror.len());
                set_weight(g, attr, mirror[at].3, weight);
                mirror[at].2 = weight;
            } else {
                let (u, v) = (rng.random_range(0..4_u64), rng.random_range(0..4_u64));
                if u != v {
                    let id = add_edge(g, attr, u, v, weight);
                    mirror.push((u, v, weight, id));
                }
            }
        }

        for seed in 0..8_u64 {
            let (host, def, attr) = host(&format!("cch_bg_interleave_{seed}"));
            let mirror: Mirror = EDGES
                .iter()
                .enumerate()
                .filter(|(_, edge)| edge.3 == INDEXED[0])
                .map(|(id, edge)| (edge.0, edge.1, edge.2.unwrap_or(1.0), id as u64))
                .collect();
            let state = Arc::new(Mutex::new((SmallRng::seed_from_u64(seed), mirror)));
            create(&host, &def);
            for _ in 0..6 {
                let building = host
                    .mvcc
                    .read()
                    .falkordb_index
                    .cch_entry(&def)
                    .expect("held")
                    .is_building();
                let during = state.lock().0.random_range(0..2) == 0;
                if !building {
                    host.write(|g| write(&state, attr, g));
                } else if during {
                    let (writer, state) = (Arc::clone(&host), Arc::clone(&state));
                    let mut job = job_running(&host, &def, move |_| {
                        writer.write(|g| write(&state, attr, g));
                    });
                    while job.step() == Step::More {}
                } else {
                    host.write(|g| write(&state, attr, g));
                    let mut job = job(&host, &def);
                    while job.step() == Step::More {}
                }
                let published = host.mvcc.read();
                let Some(index) = published.cch_index(&def) else {
                    continue;
                };
                // `T1`'s edge, which the index also spans, then `T0`'s.
                let mut cheapest: std::collections::BTreeMap<(u64, u64), f64> =
                    std::collections::BTreeMap::from([((2, 0), 7.0)]);
                for &(u, v, weight, _) in &state.lock().1 {
                    let pair = cheapest.entry((u, v)).or_insert(weight);
                    *pair = pair.min(weight);
                }
                let mut edges: Vec<(u64, u64, f64)> =
                    cheapest.into_iter().map(|((u, v), w)| (u, v, w)).collect();
                assert_queries_match(index.cch().expect("built"), &weights(4, &edges));
            }
        }
    }

    /// A drop while the build reads, or a drop and a create again: the batch's commit publishes
    /// nothing for the dropped generation, and the new one is built by a job of its own.
    #[test]
    fn a_drop_or_a_recreate_during_the_build_discards_it() {
        for recreate in [false, true] {
            let (host, def, _) = host(&format!("cch_bg_drop_{recreate}"));
            create(&host, &def);
            let generation = host
                .mvcc
                .read()
                .falkordb_index
                .cch_entry(&def)
                .expect("held")
                .generation;
            let (writer, dropped) = (Arc::clone(&host), def.clone());
            let mut held_build = job_running(&host, &def, move |_| {
                writer.write(|g| {
                    g.drop_cch_index(&dropped).expect("held");
                    if recreate {
                        g.create_cch_index(dropped.clone()).expect("create");
                    }
                });
            });
            assert_eq!(held_build.step(), Step::Cancelled, "recreate: {recreate}");
            let published = host.mvcc.read();
            match published.falkordb_index.cch_entry(&def) {
                None => assert!(!recreate),
                Some(entry) => {
                    assert!(recreate);
                    assert!(entry.is_building() && entry.generation != generation);
                }
            }
            if recreate {
                assert!(published.cch_index(&def).is_none());
                assert_eq!(job(&host, &def).step(), Step::Done);
                assert!(host.mvcc.read().cch_index(&def).is_some());
            }
        }
    }

    /// A build that fails ends the build all the same: the entry is `Ready` in the builder's
    /// sense, holds the reason and no hierarchy, and no job runs for it again. Here the index
    /// names a type the graph does not have, which `CREATE` would refuse; held directly, its build
    /// fails on it.
    #[test]
    fn a_failed_build_is_held_with_its_reason() {
        let (host, _, _) = host("cch_bg_failed");
        let def = CchIndexDef::new(vec![TypeId(42)], 0);
        host.write(|g| g.falkordb_index.add_cch_building(def.clone()));
        assert_eq!(job(&host, &def).step(), Step::Done);
        let published = host.mvcc.read();
        let entry = published.falkordb_index.cch_entry(&def).expect("held");
        assert!(!entry.is_building());
        assert_eq!(entry.index.failure(), Some("unknown relationship type"));
        assert!(published.cch_index(&def).is_none());
    }

    /// A writer that forked before the build was adopted, and escalates after: its fork takes
    /// the published copy, the adopted hierarchy, so its commit does not put the index back to
    /// `Building`.
    #[test]
    fn a_writer_forked_before_the_adoption_takes_the_adopted_hierarchy() {
        let (host, def, _) = host("cch_bg_rebase");
        create(&host, &def);
        let mut fork = host.mvcc.fork();
        assert_eq!(job(&host, &def).step(), Step::Done);
        assert!(
            fork.falkordb_index
                .cch_entry(&def)
                .expect("held")
                .is_building()
        );
        assert!(!host.mvcc.is_stale(&fork), "a batch keeps the data version");
        host.mvcc.rebase_onto_indexer_batches(&mut fork);
        let adopted = Arc::clone(
            host.mvcc
                .read()
                .cch_index(&def)
                .expect("built")
                .cch()
                .expect("built"),
        );
        let rebased = fork.cch_index(&def).expect("rebased").cch().expect("built");
        assert!(Arc::ptr_eq(rebased, &adopted));
        host.mvcc.commit(fork).expect("the writer commits");
        assert!(host.mvcc.read().cch_index(&def).is_some());
    }
}
