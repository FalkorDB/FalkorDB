//! The native index as the graph maintains it: each test drives a real `Graph` write path
//! and checks the index column it should have changed. The build's tests drive indexer batches
//! between writes.

use super::{Graph, NodeId, RelationshipId};
use crate::entity_type::EntityType;
use crate::graph::mvcc_graph::{Claim, MvccGraph, PANIC_IN_INDEXER_COMMIT};
use crate::index::Field;
use crate::index::falkordb::encode::encode_numeric;
use crate::index::falkordb::falkordb_index::{BuildCursor, IndexKey, NumericAnswer};
use crate::index::falkordb::graph_writes;
use crate::index::falkordb::versioned::{
    ALLOW_BATCH_RACE, BuildState, CATCH_UP_BY, CatchUp, Dropped, IndexId, IndexerCommit,
    PreparedBatch, REFRESH_COST,
};
use crate::index::index_docs::IndexDocs;
use crate::index::{IndexQuery, IndexType};
use crate::runtime::orderset::OrderSet;
use crate::runtime::value::Value;
use atomic_refcell::AtomicRefCell;
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::CString;
use std::sync::Arc;

/// Run one graph write with its own [`IndexDocs`] and apply its native-index changes straight
/// away, the way each write call changed the index before they were batched. Tests that check
/// the batch itself build their own `IndexDocs` instead.
macro_rules! applied {
    ($g:expr, |$w:ident| $call:expr) => {{
        let mut $w = IndexDocs::default();
        let result = $call;
        $g.apply_index_writes(&mut $w);
        result
    }};
}

/// Reserve `n` node ids the way one query's `CreateOp` does: open the
/// graph's batch, then ask the space for ids this query does not hold yet.
fn reserve_nodes(
    g: &mut Graph,
    n: usize,
) -> Vec<u64> {
    g.open_id_batches().expect("a consistent space");
    g.node_id_space()
        .reserve(n, &RoaringTreemap::new())
        .expect("reserved")
}

/// Run indexer batches of at most `entities` nodes or edges on `g` itself until no column is
/// `Building`. Returns how many batches it took.
fn build_to_end(
    g: &mut Graph,
    entities: usize,
) -> usize {
    let mut batches = 0;
    while g.falkordb_index.has_building_columns() {
        graph_writes::run_indexer_batch(g, entities);
        batches += 1;
        assert!(batches < 100_000, "the build never finishes");
    }
    batches
}

/// `attr = v`, as the read path asks it.
fn eq(
    attr: &Arc<String>,
    v: i64,
) -> IndexQuery<Value> {
    IndexQuery::Equal {
        key: attr.clone(),
        value: Value::Int(v),
    }
}

/// `Graph::new` builds GraphBLAS matrices, which need GraphBLAS initialized
/// first (done at Redis module-load in production, never from a bare unit
/// test).
///
/// This MUST go through the crate-wide guard rather than its own `Once`:
/// GraphBLAS may be initialized exactly once per process, so a second `Once`
/// gets `GrB_INVALID_VALUE`, and when the loser is
/// `graphblas::test_init::ensure_init` its `unwrap` panics inside `call_once`,
/// poisoning that `Once` for every later GraphBLAS test.
fn ensure_graphblas() {
    crate::graph::graphblas::test_init::ensure_init();
}

/// Folded-roots MVCC: `new_version` forks the FalkorDB index copy-on-write,
/// so mutating the writer's version leaves the committed version — the
/// snapshot a reader may still hold — untouched, riding one version bump.
#[test]
fn new_version_isolates_the_falkordb_index() {
    ensure_graphblas();
    let label = Arc::new("Person".to_string());
    let attr = Arc::new("age".to_string());

    let mut committed = Graph::new(64, 64, 10, 1, "t");
    committed
        .falkordb_index
        .create_numeric(EntityType::Node, &label, &attr);
    committed
        .falkordb_index
        .numeric_mut(EntityType::Node, &label, &attr)
        .unwrap()
        .add(&Value::Int(30), 1);

    // A writer forks the next version and indexes another node.
    let mut writer = committed.new_version();
    writer
        .falkordb_index
        .numeric_mut(EntityType::Node, &label, &attr)
        .unwrap()
        .add(&Value::Int(40), 2);

    let all = |g: &Graph| -> Vec<u64> {
        g.falkordb_index
            .numeric(EntityType::Node, &label, &attr)
            .unwrap()
            .range(None, None, true, true)
            .collect()
    };
    assert_eq!(all(&committed), vec![1], "committed snapshot is untouched");
    assert_eq!(all(&writer), vec![1, 2], "writer sees its own write");
    assert_eq!(writer.version, committed.version + 1);
}

/// CREATE INDEX builds the column from real graph state: three
/// `:Person {age}` nodes, then a range scan returns the right ids in
/// `(value, id)` order. The column is built by one indexer batch over every node
/// (its `get_label_matrix` → `get_node_attribute_by_idx` reads), so it is
/// `Ready` and serves reads when CREATE INDEX returns. Drives the native half of
/// `Graph::create_index` directly, bypassing the RediSearch FFI (uninitialised
/// in a unit test).
#[test]
fn create_index_builds_the_column_from_live_nodes() {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let label = Arc::new("Person".to_string());
    let attr = Arc::new("age".to_string());

    // Register the label and activate three nodes (ids 0,1,2 on a fresh graph).
    let lid = g.get_label_id_mut("Person");
    let ids: Vec<u64> = reserve_nodes(&mut g, 3);
    let mut set = RoaringTreemap::new();
    for &id in &ids {
        set.insert(id);
    }
    g.create_nodes(&set)
        .expect("freshly reserved ids are creatable");

    // Assign the :Person label to all three.
    let label_cols: Vec<u64> = vec![lid.0 as u64; ids.len()];
    applied!(g, |w| g.set_nodes_labels_bulk(
        &ids,
        &label_cols,
        &mut w,
        false
    ));

    // age = 10, 20, 30.
    let aid = g.get_or_create_node_attr_id(&attr);
    let mut attrs: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
    for (i, &id) in ids.iter().enumerate() {
        attrs.insert(id, vec![(aid, Value::Int(((i + 1) * 10) as i64))]);
    }
    applied!(g, |w| g.set_nodes_attributes(&attrs, &mut w)).unwrap();

    // Build the index numeric column from the live nodes.
    graph_writes::create_index(
        &mut g,
        &IndexType::Range,
        EntityType::Node,
        &label,
        std::slice::from_ref(&attr),
    );
    assert_eq!(
        g.falkordb_index
            .column_state(EntityType::Node, &label, &attr),
        Some(BuildState::Ready)
    );
    assert!(
        matches!(
            g.falkordb_index
                .query_numeric(EntityType::Node, &label, &eq(&attr, 20)),
            Some(NumericAnswer::Rows(_))
        ),
        "a Ready column serves reads"
    );

    let scan = |lo: Option<i64>, hi: Option<i64>| -> Vec<u64> {
        let lo = lo.map(Value::Int);
        let hi = hi.map(Value::Int);
        g.falkordb_index
            .numeric(EntityType::Node, &label, &attr)
            .unwrap()
            .range(lo.as_ref(), hi.as_ref(), true, true)
            .collect()
    };
    // [15, 35] → age 20 (id 1), age 30 (id 2), in (value, id) order.
    assert_eq!(scan(Some(15), Some(35)), vec![ids[1], ids[2]]);
    // Unbounded → all three, value-ordered.
    assert_eq!(scan(None, None), vec![ids[0], ids[1], ids[2]]);
}

/// RDB load creates each native column empty and `Ready`, and its populate rebuilds the column
/// once every entity is in. Afterwards each column is `Ready` and holds what the graph's values
/// say, for nodes and for edges.
#[test]
fn rdb_load_populate_builds_each_column() {
    let (mut g, label, v, ty, w) = scan_fixture();
    let range = |attr: &Arc<String>| -> HashMap<Arc<String>, Vec<Arc<Field>>> {
        let field = Field::new(CString::new(attr.as_str()).unwrap(), IndexType::Range, None);
        HashMap::from([(attr.clone(), vec![Arc::new(field)])])
    };
    for (entity, l, a) in [
        (EntityType::Node, &label, &v),
        (EntityType::Relationship, &ty, &w),
    ] {
        g.falkordb_index
            .create_unpopulated(&IndexType::Range, entity, l, std::slice::from_ref(a));
        graph_writes::populate(&mut g, entity, l, &range(a));
        assert_eq!(
            g.falkordb_index.column_state(entity, l, a),
            Some(BuildState::Ready),
            "{entity:?}"
        );
    }
    let node_values: Vec<i64> = (0..10).map(|i| i * 10).collect();
    let nodes = one_pass(&g, EntityType::Node, &label, &v, &node_values);
    assert_eq!(nodes.1.len(), 9, "ten nodes, one deleted");
    assert_eq!(
        contents(&g, EntityType::Node, &label, &v, &node_values),
        nodes
    );
    let edge_values: Vec<i64> = (0..8).collect();
    let edges = one_pass(&g, EntityType::Relationship, &ty, &w, &edge_values);
    assert_eq!(edges.1.len(), 8);
    assert_eq!(
        contents(&g, EntityType::Relationship, &ty, &w, &edge_values),
        edges
    );
}

/// The edge counterpart: CREATE INDEX on a relationship type builds its column from the live
/// edges, a multi-edge pair included, in one indexer batch dispatched to the edge column, so it is
/// `Ready` and serves reads when CREATE INDEX returns, and holds what a one-pass build does.
#[test]
fn create_index_builds_the_column_from_live_edges() {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let ty = Arc::new("R".to_string());
    let attr = Arc::new("w".to_string());
    let aid = g.get_or_create_rel_attr_id(&attr);
    let nodes = reserve_nodes(&mut g, 3);
    g.create_nodes(&nodes.iter().copied().collect())
        .expect("freshly reserved ids are creatable");
    // a->b twice (one multi-edge pair), a->c and c->b; w = 1..=4.
    let (a, b, c) = (nodes[0], nodes[1], nodes[2]);
    let (srcs, dsts) = ([a, a, a, c], [b, b, c, b]);
    let eids: Vec<u64> = g
        .relationship_id_space()
        .reserve(srcs.len(), &RoaringTreemap::new())
        .expect("reserved");
    g.create_relationships_bulk(&ty, &srcs, &dsts, &eids)
        .expect("freshly reserved ids are creatable");
    let attrs: FxHashMap<u64, Vec<(u16, Value)>> = eids
        .iter()
        .zip(1..)
        .map(|(&eid, w)| (eid, vec![(aid, Value::Int(w))]))
        .collect();
    applied!(g, |w| g.import_relationship_attrs(&attrs, &mut w));

    graph_writes::create_index(
        &mut g,
        &IndexType::Range,
        EntityType::Relationship,
        &ty,
        std::slice::from_ref(&attr),
    );
    assert_eq!(
        g.falkordb_index
            .column_state(EntityType::Relationship, &ty, &attr),
        Some(BuildState::Ready)
    );
    assert_eq!(
        g.falkordb_index
            .query_numeric(EntityType::Relationship, &ty, &eq(&attr, 2))
            .map(|answer| match answer {
                NumericAnswer::Rows(rows) => rows.collect::<Vec<u64>>(),
                NumericAnswer::NotReady(_) => panic!("a Ready column answered NotReady"),
            }),
        Some(vec![eids[1]]),
        "a Ready column serves reads"
    );
    let values = [1, 2, 3, 4];
    assert_eq!(
        contents(&g, EntityType::Relationship, &ty, &attr, &values),
        one_pass(&g, EntityType::Relationship, &ty, &attr, &values)
    );
    assert_eq!(
        contents(&g, EntityType::Relationship, &ty, &attr, &values).1,
        eids
    );
}

// --- Write-path maintenance ---

/// A graph with a index on `(:Person, v)` and `n` labeled Person nodes
/// (ids `0..n`), no attrs yet.
fn graph_with_index(n: usize) -> (Graph, Arc<String>, Arc<String>, Vec<u64>) {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let label = Arc::new("Person".to_string());
    let attr = Arc::new("v".to_string());
    g.falkordb_index
        .create_numeric(EntityType::Node, &label, &attr);
    let lid = g.get_label_id_mut("Person");
    let ids: Vec<u64> = reserve_nodes(&mut g, n);
    let mut set = RoaringTreemap::new();
    for &id in &ids {
        set.insert(id);
    }
    g.create_nodes(&set)
        .expect("freshly reserved ids are creatable");
    let cols = vec![lid.0 as u64; ids.len()];
    applied!(g, |w| g.set_nodes_labels_bulk(&ids, &cols, &mut w, false));
    (g, label, attr, ids)
}

/// Drive the existing-node SET path with one `(id, attr) = value`.
fn set_attr(
    g: &mut Graph,
    id: u64,
    attr: &Arc<String>,
    value: Value,
) {
    let aid = g.get_or_create_node_attr_id(attr);
    let mut attrs = FxHashMap::default();
    attrs.insert(id, vec![(aid, value)]);
    let mut w = IndexDocs::default();
    g.set_nodes_attributes(&attrs, &mut w).unwrap();
    g.apply_index_writes(&mut w);
}

fn range_scan(
    g: &Graph,
    label: &Arc<String>,
    attr: &Arc<String>,
    lo: i64,
    hi: i64,
) -> Vec<u64> {
    g.falkordb_index
        .numeric(EntityType::Node, label, attr)
        .unwrap()
        .range(Some(&Value::Int(lo)), Some(&Value::Int(hi)), true, true)
        .collect()
}

/// UPDATE must remove the old tuple, not only add the new one.
#[test]
fn update_removes_the_old_tuple() {
    let (mut g, label, attr, ids) = graph_with_index(1);
    set_attr(&mut g, ids[0], &attr, Value::Int(5));
    assert_eq!(range_scan(&g, &label, &attr, 4, 6), vec![ids[0]]);
    set_attr(&mut g, ids[0], &attr, Value::Int(9)); // 5 -> 9
    assert!(
        range_scan(&g, &label, &attr, 4, 6).is_empty(),
        "the stale value 5 must not surface after the update"
    );
    assert_eq!(range_scan(&g, &label, &attr, 8, 10), vec![ids[0]]);
}

/// SET x = null drops the entry (remove old, add nothing).
#[test]
fn set_null_removes_from_index() {
    let (mut g, label, attr, ids) = graph_with_index(1);
    set_attr(&mut g, ids[0], &attr, Value::Int(5));
    set_attr(&mut g, ids[0], &attr, Value::Null);
    assert!(range_scan(&g, &label, &attr, 0, 100).is_empty());
}

/// DELETE removes the node's tuple, leaving the others.
#[test]
fn delete_removes_from_index() {
    let (mut g, label, attr, ids) = graph_with_index(2);
    set_attr(&mut g, ids[0], &attr, Value::Int(5));
    set_attr(&mut g, ids[1], &attr, Value::Int(6));
    let mut del = RoaringTreemap::new();
    del.insert(ids[0]);
    applied!(g, |w| g.delete_nodes(&del, &mut w)).unwrap();
    assert_eq!(range_scan(&g, &label, &attr, 0, 100), vec![ids[1]]);
}

/// Eager removal at delete means a reused id (same value) has no stale duplicate — the id-reuse
/// correctness the delete model relies on.
#[test]
fn delete_then_reuse_id_and_value_stays_correct() {
    let (mut g, label, attr, ids) = graph_with_index(1);
    set_attr(&mut g, ids[0], &attr, Value::Int(5));
    let mut del = RoaringTreemap::new();
    del.insert(ids[0]);
    applied!(g, |w| g.delete_nodes(&del, &mut w)).unwrap();
    assert!(range_scan(&g, &label, &attr, 4, 6).is_empty());

    // Reclaim the freed id for a fresh node with the same value.
    let reused: Vec<u64> = reserve_nodes(&mut g, 1);
    assert_eq!(reused[0], ids[0], "the deleted id should be reclaimed");
    let mut set = RoaringTreemap::new();
    set.insert(reused[0]);
    g.create_nodes(&set)
        .expect("freshly reserved ids are creatable");
    let lid = g.get_label_id_mut("Person");
    applied!(g, |w| g.set_nodes_labels_bulk(
        &reused,
        &[lid.0 as u64],
        &mut w,
        false
    ));
    set_attr(&mut g, reused[0], &attr, Value::Int(5));

    // Exactly one entry — the reused node — no resurrected duplicate.
    assert_eq!(range_scan(&g, &label, &attr, 4, 6), vec![reused[0]]);
}

/// A graph with the `(:Person, v)` index and one node that is NOT labeled `:Person`.
fn unlabeled_graph_with_index() -> (Graph, Arc<String>, Arc<String>, u64) {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let label = Arc::new("Person".to_string());
    let attr = Arc::new("v".to_string());
    g.falkordb_index
        .create_numeric(EntityType::Node, &label, &attr);
    let _lid = g.get_label_id_mut("Person"); // register the label matrix
    let ids: Vec<u64> = reserve_nodes(&mut g, 1);
    let mut set = RoaringTreemap::new();
    set.insert(ids[0]);
    g.create_nodes(&set)
        .expect("freshly reserved ids are creatable");
    (g, label, attr, ids[0])
}

/// SET :Label indexes the node's already-present attrs.
#[test]
fn set_label_indexes_existing_attrs() {
    let (mut g, label, attr, id) = unlabeled_graph_with_index();
    set_attr(&mut g, id, &attr, Value::Int(5));
    assert!(
        range_scan(&g, &label, &attr, 4, 6).is_empty(),
        "no :Person label yet ⇒ not indexed"
    );
    let lid = g.get_label_id_mut("Person");
    applied!(g, |w| g.set_nodes_labels_bulk(
        &[id],
        &[lid.0 as u64],
        &mut w,
        false
    ));
    assert_eq!(
        range_scan(&g, &label, &attr, 4, 6),
        vec![id],
        "SET :Person must index the existing attr"
    );
}

/// REMOVE :Label drops the tuple, and no phantom resurrects when the id is reused with a
/// different value.
#[test]
fn remove_label_drops_tuple_and_no_resurrect_on_reuse() {
    let (mut g, label, attr, ids) = graph_with_index(1);
    set_attr(&mut g, ids[0], &attr, Value::Int(5));
    assert_eq!(range_scan(&g, &label, &attr, 4, 6), vec![ids[0]]);

    // REMOVE n:Person — the (:Person, v) tuple must be gone (without this hook it orphans).
    let lid = g.get_label_id_mut("Person");
    applied!(g, |w| g.remove_nodes_labels(
        &[ids[0]],
        &[lid.0 as u64],
        &mut w
    ));
    assert!(
        range_scan(&g, &label, &attr, 4, 6).is_empty(),
        "REMOVE :Person must drop the tuple"
    );

    // Delete the node and reuse its id for a fresh :Person with a DIFFERENT value.
    let mut del = RoaringTreemap::new();
    del.insert(ids[0]);
    applied!(g, |w| g.delete_nodes(&del, &mut w)).unwrap();
    let reused: Vec<u64> = reserve_nodes(&mut g, 1);
    assert_eq!(reused[0], ids[0]);
    let mut set = RoaringTreemap::new();
    set.insert(reused[0]);
    g.create_nodes(&set)
        .expect("freshly reserved ids are creatable");
    applied!(g, |w| g.set_nodes_labels_bulk(
        &reused,
        &[lid.0 as u64],
        &mut w,
        false
    ));
    set_attr(&mut g, reused[0], &attr, Value::Int(99));

    assert!(
        range_scan(&g, &label, &attr, 4, 6).is_empty(),
        "the old value 5 must not resurrect through the reused id"
    );
    assert_eq!(range_scan(&g, &label, &attr, 98, 100), vec![reused[0]]);
}

/// Every edge write path keeps the `(:R, w)` column in step: create, update, explicit delete,
/// and the cascade a node delete triggers (`delete_implicit_edges`). The cascade reaches the
/// index through a different graph path from the explicit delete, so each needs its own check.
#[test]
fn edge_writes_maintain_the_index() {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let ty = Arc::new("R".to_string());
    let attr = Arc::new("w".to_string());
    g.falkordb_index
        .create_numeric(EntityType::Relationship, &ty, &attr);
    let aid = g.get_or_create_rel_attr_id(&attr);

    // Two nodes, a and b, and three a->b edges with w = 10, 20, 30.
    let nodes = reserve_nodes(&mut g, 2);
    g.create_nodes(&nodes.iter().copied().collect())
        .expect("freshly reserved ids are creatable");
    let (a, b) = (nodes[0], nodes[1]);
    let ids: Vec<u64> = g
        .relationship_id_space()
        .reserve(3, &RoaringTreemap::new())
        .expect("reserved");
    g.create_relationships_bulk(&ty, &[a; 3], &[b; 3], &ids)
        .expect("freshly reserved ids are creatable");
    let mut attrs: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
    for (i, &id) in ids.iter().enumerate() {
        attrs.insert(id, vec![(aid, Value::Int(((i as i64) + 1) * 10))]);
    }
    applied!(g, |w| g.import_relationship_attrs(&attrs, &mut w));

    let scan = |g: &Graph, lo: i64, hi: i64| -> Vec<u64> {
        g.falkordb_index
            .numeric(EntityType::Relationship, &ty, &attr)
            .unwrap()
            .range(Some(&Value::Int(lo)), Some(&Value::Int(hi)), true, true)
            .collect()
    };
    assert_eq!(scan(&g, 0, 100), ids, "create indexes every edge");

    // Update: 20 -> 25 must remove 20, not only add 25.
    let mut update: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
    update.insert(ids[1], vec![(aid, Value::Int(25))]);
    applied!(g, |w| g.set_relationships_attributes(&update, &mut w)).unwrap();
    assert!(scan(&g, 20, 20).is_empty(), "the old value is gone");
    assert_eq!(scan(&g, 25, 25), vec![ids[1]]);

    // Explicit delete.
    let deleted: RoaringTreemap = std::iter::once(ids[0]).collect();
    applied!(g, |w| g.delete_relationships(&deleted, &mut w)).unwrap();
    assert_eq!(scan(&g, 0, 100), vec![ids[1], ids[2]]);

    // Deleting `a` cascades to its two remaining edges.
    let deleted_nodes: RoaringTreemap = std::iter::once(a).collect();
    g.open_id_batches().expect("a consistent space");
    applied!(g, |w| g.delete_nodes(&deleted_nodes, &mut w)).unwrap();
    applied!(g, |w| g.delete_implicit_edges(
        &deleted_nodes,
        &RoaringTreemap::new(),
        &mut w,
    ))
    .unwrap();
    assert!(
        scan(&g, 0, 100).is_empty(),
        "the cascade removes what the explicit delete did not"
    );
}

// --- Indexer batches ---

/// `:Person` nodes carrying `attr` with `values`, set before any index exists, so indexer batches are
/// their only way into the column.
fn people_with(
    attr: &Arc<String>,
    values: &[i64],
) -> (Graph, Arc<String>, Vec<u64>) {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let label = Arc::new("Person".to_string());
    let lid = g.get_label_id_mut("Person");
    let ids = reserve_nodes(&mut g, values.len());
    g.create_nodes(&ids.iter().copied().collect())
        .expect("freshly reserved ids are creatable");
    applied!(g, |w| g.set_nodes_labels_bulk(
        &ids,
        &vec![lid.0 as u64; ids.len()],
        &mut w,
        false
    ));
    for (&id, &v) in ids.iter().zip(values) {
        set_attr(&mut g, id, attr, Value::Int(v));
    }
    (g, label, ids)
}

/// One more `:Person` node with `attr = value`, created through the write path.
fn add_person(
    g: &mut Graph,
    attr: &Arc<String>,
    value: i64,
) -> u64 {
    let lid = g.get_label_id_mut("Person");
    let id = reserve_nodes(g, 1)[0];
    g.create_nodes(&std::iter::once(id).collect())
        .expect("freshly reserved ids are creatable");
    applied!(g, |w| g.set_nodes_labels_bulk(
        &[id],
        &[lid.0 as u64],
        &mut w,
        false
    ));
    set_attr(g, id, attr, Value::Int(value));
    id
}

/// Column `(:Person, attr)` created `Building`, with no batch taken yet: what `CREATE INDEX` does
/// before its indexer batches.
fn create_index_on(
    g: &mut Graph,
    label: &Arc<String>,
    attr: &Arc<String>,
) {
    g.falkordb_index
        .create_building(EntityType::Node, label, attr);
}

/// DROP INDEX on `(:Person, attr)`, the native half: what `Graph::drop_index` does to this index
/// once RediSearch has confirmed the drop.
fn drop_index_on(
    g: &mut Graph,
    label: &Arc<String>,
    attr: &Arc<String>,
) {
    g.falkordb_index.drop_index(
        &IndexType::Range,
        EntityType::Node,
        label,
        std::slice::from_ref(attr),
    );
}

/// The cursor of column `(entity, label, attr)`, or `None` once it is `Ready`.
fn cursor(
    g: &Graph,
    entity: EntityType,
    label: &Arc<String>,
    attr: &Arc<String>,
) -> Option<BuildCursor> {
    match g
        .falkordb_index
        .column_state(entity, label, attr)
        .expect("the column exists")
    {
        BuildState::Building { next } => Some(next),
        BuildState::Ready => None,
    }
}

/// Column `(entity, label, attr)` as the ids it answers for each of `values`, plus all its ids in
/// value order: enough to tell two columns apart over the values a test uses.
fn contents(
    g: &Graph,
    entity: EntityType,
    label: &Arc<String>,
    attr: &Arc<String>,
    values: &[i64],
) -> (Vec<(i64, Vec<u64>)>, Vec<u64>) {
    let idx = g
        .falkordb_index
        .numeric(entity, label, attr)
        .expect("the column exists");
    let per_value = values
        .iter()
        .map(|&v| {
            let v_ref = Value::Int(v);
            (
                v,
                idx.range(Some(&v_ref), Some(&v_ref), true, true).collect(),
            )
        })
        .collect();
    (per_value, idx.range(None, None, true, true).collect())
}

/// What column `(entity, label, attr)` must hold for `g` as it stands now, in the shape
/// [`contents`] reports: worked out from the graph's values here, sharing no code with the
/// indexer batches. Every live `label` node, or `label`-typed edge, with a numeric `attr`, as its
/// `(key, id)` tuple, in tuple order.
fn one_pass(
    g: &Graph,
    entity: EntityType,
    label: &Arc<String>,
    attr: &Arc<String>,
    values: &[i64],
) -> (Vec<(i64, Vec<u64>)>, Vec<u64>) {
    let mut tuples: Vec<(u64, u64)> = Vec::new();
    match entity {
        EntityType::Node => {
            if let (Some(lm), Some(idx)) =
                (g.get_label_matrix(label), g.get_node_attribute_id(attr))
            {
                for (n, _) in lm.iter(0, u64::MAX) {
                    let value = g.get_node_attribute_by_idx(NodeId::from(n), idx as u16);
                    if let Some(key) = value.as_ref().and_then(encode_numeric) {
                        tuples.push((key, n));
                    }
                }
            }
        }
        EntityType::Relationship => {
            if let (Some(t), Some(idx)) = (
                g.get_relationship_matrix(label),
                g.get_relationship_attribute_id(attr),
            ) {
                for (_, _, eid) in t.iter(0, u64::MAX, false) {
                    let value =
                        g.get_relationship_attribute_by_idx(RelationshipId::from(eid), idx as u16);
                    if let Some(key) = value.as_ref().and_then(encode_numeric) {
                        tuples.push((key, eid));
                    }
                }
            }
        }
    }
    tuples.sort_unstable();
    tuples.dedup();
    let per_value = values
        .iter()
        .map(|&v| {
            let key = encode_numeric(&Value::Int(v)).expect("an integer encodes");
            (
                v,
                tuples
                    .iter()
                    .filter(|&&(k, _)| k == key)
                    .map(|&(_, id)| id)
                    .collect(),
            )
        })
        .collect();
    (per_value, tuples.into_iter().map(|(_, id)| id).collect())
}

/// A build in small batches, with writes landing between the batches both on nodes the batches
/// already indexed and on nodes they have not reached, ends with exactly the column a one-pass
/// build of the final graph has: nothing deleted or re-valued survives, and nothing is missed. A
/// write reaches the column only below the cursor; one at or above it is left to the batch that
/// reaches the node, which reads its value then.
#[test]
fn batched_build_matches_a_one_pass_build_under_concurrent_node_writes() {
    let attr = Arc::new("v".to_string());
    let (mut g, label, ids) = people_with(&attr, &[10, 20, 30, 40, 50, 60]);
    create_index_on(&mut g, &label, &attr);

    // The first batch indexes ids[0] and ids[1], and stops at ids[2].
    assert!(graph_writes::run_indexer_batch(&mut g, 2));
    assert_eq!(
        cursor(&g, EntityType::Node, &label, &attr),
        Some(BuildCursor::Node(NodeId::from(ids[2])))
    );
    assert_eq!(range_scan(&g, &label, &attr, 0, 100), vec![ids[0], ids[1]]);

    // Writes between batches. Behind the cursor, ids[1] is deleted and ids[0] goes 10 -> 15. Ahead
    // of it, ids[3] is deleted and ids[4] goes 50 -> 55. And a new node gets 70.
    let mut del = RoaringTreemap::new();
    del.insert(ids[1]);
    del.insert(ids[3]);
    applied!(g, |w| g.delete_nodes(&del, &mut w)).unwrap();
    set_attr(&mut g, ids[0], &attr, Value::Int(15));
    set_attr(&mut g, ids[4], &attr, Value::Int(55));
    let created = add_person(&mut g, &attr, 70);
    // The new node takes the lowest freed id, ids[1], which is behind the cursor too.
    assert_eq!(created, ids[1]);
    assert_eq!(
        range_scan(&g, &label, &attr, 0, 100),
        vec![ids[0], created],
        "the writes behind the cursor are in; the ones ahead of it wait for their batch"
    );

    build_to_end(&mut g, 2);
    let values = [10, 15, 20, 30, 40, 50, 55, 60, 70];
    assert_eq!(
        contents(&g, EntityType::Node, &label, &attr, &values),
        one_pass(&g, EntityType::Node, &label, &attr, &values)
    );
    assert_eq!(
        range_scan(&g, &label, &attr, 0, 100),
        vec![ids[0], ids[2], ids[4], ids[5], created]
    );
}

/// The edge counterpart: batches stop inside a source's edges, finishing the multi-edge pair they
/// stop in, and writes land between them behind and ahead of the cursor, placed by their `(src,
/// dst)` pair. That includes a new edge of the source being resumed, at a destination the batches
/// have already passed, which only the write path indexes, and a deleted edge behind the cursor,
/// whose endpoints are gone by the time its removal is applied.
#[test]
fn batched_build_matches_a_one_pass_build_under_concurrent_edge_writes() {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let ty = Arc::new("R".to_string());
    let attr = Arc::new("w".to_string());
    let aid = g.get_or_create_rel_attr_id(&attr);
    let nodes = reserve_nodes(&mut g, 4);
    g.create_nodes(&nodes.iter().copied().collect())
        .expect("freshly reserved ids are creatable");
    let (a, b, c, d) = (nodes[0], nodes[1], nodes[2], nodes[3]);

    // a->b three times (one multi-edge pair), then a->c, a->d, b->c and c->d; w = 1..=7.
    let (srcs, dsts) = ([a, a, a, a, a, b, c], [b, b, b, c, d, c, d]);
    let eids: Vec<u64> = g
        .relationship_id_space()
        .reserve(srcs.len(), &RoaringTreemap::new())
        .expect("reserved");
    g.create_relationships_bulk(&ty, &srcs, &dsts, &eids)
        .expect("freshly reserved ids are creatable");
    let with_w = |pairs: &[(u64, i64)]| -> FxHashMap<u64, Vec<(u16, Value)>> {
        pairs
            .iter()
            .map(|&(eid, w)| (eid, vec![(aid, Value::Int(w))]))
            .collect()
    };
    let initial: Vec<(u64, i64)> = eids.iter().copied().zip(1..).collect();
    applied!(g, |w| g
        .import_relationship_attrs(&with_w(&initial), &mut w));

    g.falkordb_index
        .create_building(EntityType::Relationship, &ty, &attr);

    // A two-edge batch stops inside the a->b pair, so it finishes the pair: three edges, and the
    // next batch resumes at a->c.
    assert!(graph_writes::run_indexer_batch(&mut g, 2));
    assert_eq!(
        cursor(&g, EntityType::Relationship, &ty, &attr),
        Some(BuildCursor::Edge {
            src: NodeId::from(a),
            dst: NodeId::from(c)
        })
    );
    let held = |g: &Graph| contents(g, EntityType::Relationship, &ty, &attr, &[]).1;
    assert_eq!(held(&g), eids[..3].to_vec());

    // Writes between batches: behind the cursor, delete the second a->b edge and add a fourth a->b
    // edge with w = 8 (a destination the batches have passed); ahead of it, a->d goes 5 -> 55 and
    // b->c is deleted.
    let mut del = RoaringTreemap::new();
    del.insert(eids[1]);
    del.insert(eids[5]);
    applied!(g, |w| g.delete_relationships(&del, &mut w)).unwrap();
    applied!(g, |w| g
        .set_relationships_attributes(&with_w(&[(eids[4], 55)]), &mut w))
    .unwrap();
    let added = g
        .relationship_id_space()
        .reserve(1, &RoaringTreemap::new())
        .expect("reserved")[0];
    g.create_relationships_bulk(&ty, &[a], &[b], &[added])
        .expect("freshly reserved ids are creatable");
    applied!(g, |w| g
        .import_relationship_attrs(&with_w(&[(added, 8)]), &mut w));
    assert_eq!(
        held(&g),
        vec![eids[0], eids[2], added],
        "w = 1, 3, 8: the writes behind the cursor are in, the deleted edge's removal included"
    );

    build_to_end(&mut g, 2);
    let values = [1, 2, 3, 4, 5, 6, 7, 8, 55];
    assert_eq!(
        contents(&g, EntityType::Relationship, &ty, &attr, &values),
        one_pass(&g, EntityType::Relationship, &ty, &attr, &values)
    );
    assert_eq!(
        held(&g),
        vec![eids[0], eids[2], eids[3], eids[6], added, eids[4]],
        "w = 1, 3, 4, 7, 8, 55"
    );
}

/// An edge write is placed by its `(src, dst)` pair, source first, the order batches read edges
/// in. With the cursor at 1->3, an edge 2->1 is ahead of it even though its destination is lower,
/// so a write to it waits for its batch; 1->0 is behind, so a write to it is applied at once.
#[test]
fn an_edge_write_is_placed_by_its_source_then_its_destination() {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let ty = Arc::new("R".to_string());
    let attr = Arc::new("w".to_string());
    let aid = g.get_or_create_rel_attr_id(&attr);
    let nodes = reserve_nodes(&mut g, 4);
    g.create_nodes(&nodes.iter().copied().collect())
        .expect("freshly reserved ids are creatable");
    // In batch order: 0->2, 1->0, 1->3, 2->1; w = 1..=4.
    let (srcs, dsts) = (
        [nodes[0], nodes[1], nodes[1], nodes[2]],
        [nodes[2], nodes[0], nodes[3], nodes[1]],
    );
    let eids: Vec<u64> = g
        .relationship_id_space()
        .reserve(srcs.len(), &RoaringTreemap::new())
        .expect("reserved");
    g.create_relationships_bulk(&ty, &srcs, &dsts, &eids)
        .expect("freshly reserved ids are creatable");
    let with_w = |pairs: &[(u64, i64)]| -> FxHashMap<u64, Vec<(u16, Value)>> {
        pairs
            .iter()
            .map(|&(eid, w)| (eid, vec![(aid, Value::Int(w))]))
            .collect()
    };
    let initial: Vec<(u64, i64)> = eids.iter().copied().zip(1..).collect();
    applied!(g, |w| g
        .import_relationship_attrs(&with_w(&initial), &mut w));
    g.falkordb_index
        .create_building(EntityType::Relationship, &ty, &attr);

    // Two edges, 0->2 and 1->0; the next batch resumes at 1->3.
    assert!(graph_writes::run_indexer_batch(&mut g, 2));
    assert_eq!(
        cursor(&g, EntityType::Relationship, &ty, &attr),
        Some(BuildCursor::Edge {
            src: NodeId::from(nodes[1]),
            dst: NodeId::from(nodes[3])
        })
    );

    // 1->0 goes 2 -> 20 (behind the cursor); 2->1 goes 4 -> 40 (ahead of it).
    applied!(g, |w| g.set_relationships_attributes(
        &with_w(&[(eids[1], 20), (eids[3], 40)]),
        &mut w
    ))
    .unwrap();
    let values = [1, 2, 3, 4, 20, 40];
    let (per_value, _) = contents(&g, EntityType::Relationship, &ty, &attr, &values);
    let at = |v: i64| per_value.iter().find(|(x, _)| *x == v).unwrap().1.clone();
    assert_eq!(at(20), vec![eids[1]], "1->0 is behind the cursor");
    assert!(at(40).is_empty(), "2->1 is ahead of the cursor");

    build_to_end(&mut g, 2);
    assert_eq!(
        contents(&g, EntityType::Relationship, &ty, &attr, &values),
        one_pass(&g, EntityType::Relationship, &ty, &attr, &values)
    );
}

/// One unit of work deletes an edge behind the cursor and creates an edge that reuses its id ahead
/// of the cursor, as a replica applying a buffer of a C master's writes can (C reuses a freed id
/// at once). When the removal is applied, the id's place is the new edge's, ahead of the cursor,
/// yet the column must lose the deleted edge's tuple and hold nothing for the new edge until its
/// batch: with a value other than the deleted edge's, and with the same value, where the unit's
/// removal and addition of that one tuple collapse to the addition.
#[test]
fn an_edge_id_reused_ahead_of_the_cursor_leaves_no_tuple_behind() {
    for new_w in [9, 1] {
        ensure_graphblas();
        let mut g = Graph::new(64, 64, 10, 1, "t");
        let ty = Arc::new("R".to_string());
        let attr = Arc::new("w".to_string());
        let aid = g.get_or_create_rel_attr_id(&attr);
        let nodes = reserve_nodes(&mut g, 4);
        g.create_nodes(&nodes.iter().copied().collect())
            .expect("freshly reserved ids are creatable");
        let (a, b, c, d) = (nodes[0], nodes[1], nodes[2], nodes[3]);
        // In batch order: a->b, a->c, c->d; w = 1..=3.
        let (srcs, dsts) = ([a, a, c], [b, c, d]);
        let eids: Vec<u64> = g
            .relationship_id_space()
            .reserve(srcs.len(), &RoaringTreemap::new())
            .expect("reserved");
        g.create_relationships_bulk(&ty, &srcs, &dsts, &eids)
            .expect("freshly reserved ids are creatable");
        let with_w = |pairs: &[(u64, i64)]| -> FxHashMap<u64, Vec<(u16, Value)>> {
            pairs
                .iter()
                .map(|&(eid, w)| (eid, vec![(aid, Value::Int(w))]))
                .collect()
        };
        let initial: Vec<(u64, i64)> = eids.iter().copied().zip(1..).collect();
        applied!(g, |w| g
            .import_relationship_attrs(&with_w(&initial), &mut w));
        g.falkordb_index
            .create_building(EntityType::Relationship, &ty, &attr);

        // One edge, a->b; the next batch resumes at a->c.
        assert!(graph_writes::run_indexer_batch(&mut g, 1));
        assert_eq!(
            cursor(&g, EntityType::Relationship, &ty, &attr),
            Some(BuildCursor::Edge {
                src: NodeId::from(a),
                dst: NodeId::from(c)
            })
        );
        let held = |g: &Graph| contents(g, EntityType::Relationship, &ty, &attr, &[]).1;
        assert_eq!(held(&g), vec![eids[0]]);

        // One unit of work: delete a->b, then create d->a under its id, as a replica creates an
        // edge under the id its master gave it (`effects/v3/apply.rs`).
        let mut w = IndexDocs::default();
        let mut del = RoaringTreemap::new();
        del.insert(eids[0]);
        g.delete_relationships(&del, &mut w).unwrap();
        let reused = eids[0];
        g.create_relationships_bulk(&ty, &[d], &[a], &[reused])
            .expect("a freed id is creatable");
        g.import_relationship_attrs(&with_w(&[(reused, new_w)]), &mut w);
        g.apply_index_writes(&mut w);
        assert!(
            held(&g).is_empty(),
            "w = {new_w}: a->b's tuple is out, and d->a, ahead of the cursor, waits for its batch"
        );

        build_to_end(&mut g, 1);
        let values = [1, 2, 3, new_w];
        assert_eq!(
            contents(&g, EntityType::Relationship, &ty, &attr, &values),
            one_pass(&g, EntityType::Relationship, &ty, &attr, &values),
            "w = {new_w}"
        );
    }
}

/// `Person` nodes 0..10 with `v = 10 * id`, node 4 deleted (a gap in the rows), and `R` edges with
/// several per source node and one multi-edge pair, so batches stop inside and between rows and
/// inside the pair. Both columns are created, and `Building`.
fn scan_fixture() -> (Graph, Arc<String>, Arc<String>, Arc<String>, Arc<String>) {
    let v = Arc::new("v".to_string());
    let (mut g, label, ids) = people_with(&v, &(0..10).map(|i| i * 10).collect::<Vec<_>>());
    let ty = Arc::new("R".to_string());
    let w = Arc::new("w".to_string());
    let aid = g.get_or_create_rel_attr_id(&w);
    // Five edges out of node 1 (three of them 1->2), two out of node 7, one out of node 8.
    let (srcs, dsts) = ([1u64, 1, 1, 1, 1, 7, 7, 8], [2u64, 2, 2, 3, 9, 8, 9, 9]);
    let eids: Vec<u64> = g
        .relationship_id_space()
        .reserve(srcs.len(), &RoaringTreemap::new())
        .expect("reserved");
    g.create_relationships_bulk(&ty, &srcs, &dsts, &eids)
        .expect("freshly reserved ids are creatable");
    let mut attrs: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
    for (i, &eid) in eids.iter().enumerate() {
        attrs.insert(eid, vec![(aid, Value::Int(i as i64))]);
    }
    applied!(g, |w| g.import_relationship_attrs(&attrs, &mut w));
    let mut del = RoaringTreemap::new();
    del.insert(ids[4]);
    applied!(g, |w| g.delete_nodes(&del, &mut w)).unwrap();
    for (entity, l, a) in [
        (EntityType::Node, &label, &v),
        (EntityType::Relationship, &ty, &w),
    ] {
        g.falkordb_index.create_building(entity, l, a);
    }
    (g, label, v, ty, w)
}

/// Batches of any size ends with the column a one-pass build has: nothing dropped or
/// doubled where a batch stops, for nodes (with a gap in the ids) and for edges (with batches stopping
/// inside rows and inside a multi-edge pair).
#[test]
fn any_batch_size_builds_the_one_pass_column() {
    let node_values: Vec<i64> = (0..10).map(|i| i * 10).collect();
    let edge_values: Vec<i64> = (0..8).collect();
    for batch in [1usize, 2, 3, 7, 64] {
        let (mut g, label, v, ty, w) = scan_fixture();
        let nodes = one_pass(&g, EntityType::Node, &label, &v, &node_values);
        let edges = one_pass(&g, EntityType::Relationship, &ty, &w, &edge_values);
        assert_eq!(nodes.1.len(), 9, "ten nodes, one deleted");
        assert_eq!(edges.1.len(), 8);

        build_to_end(&mut g, batch);
        assert_eq!(
            contents(&g, EntityType::Node, &label, &v, &node_values),
            nodes,
            "nodes, {batch} per batch"
        );
        assert_eq!(
            contents(&g, EntityType::Relationship, &ty, &w, &edge_values),
            edges,
            "edges, {batch} per batch"
        );
    }
}

/// While a column is building, a read of it scans, and the scan keeps what the built column will
/// answer: at every cursor a build passes through, for nodes and for edges, for each shape a
/// column serves. The planner drops the predicate it hands an index scan, so nothing else checks
/// the scan's rows; a scan that kept them all would return every entity of the label. `= null`
/// holds for nothing, and is answered with no rows rather than declined.
#[test]
fn a_building_columns_scan_keeps_what_the_built_column_answers() {
    let (mut g, label, v, ty, w) = scan_fixture();
    let int = Value::Int;
    let range = |key: &Arc<String>, min: Option<i64>, max: Option<i64>, inclusive: bool| {
        IndexQuery::Range {
            key: key.clone(),
            min: min.map(int),
            max: max.map(int),
            include_min: inclusive,
            include_max: inclusive,
        }
    };
    let in_list = |key: &Arc<String>, values: &[Value]| IndexQuery::InList {
        key: key.clone(),
        list: Value::List(Arc::new(values.iter().cloned().collect())),
    };
    let equal = |key: &Arc<String>, value: Value| IndexQuery::Equal {
        key: key.clone(),
        value,
    };
    // Node `v` is `10 * id` over ids 0..10 with node 4 deleted; edge `w` is 0..8.
    let node_queries = [
        eq(&v, 30),
        eq(&v, 40),
        eq(&v, 35),
        equal(&v, Value::Float(30.0)),
        equal(&v, Value::Null),
        range(&v, Some(20), Some(60), true),
        range(&v, Some(20), Some(60), false),
        range(&v, Some(50), None, true),
        range(&v, None, Some(30), false),
        in_list(&v, &[int(10), int(70), int(70), int(1000)]),
        IndexQuery::Or(vec![
            eq(&v, 0),
            in_list(&v, &[int(90)]),
            equal(&v, Value::Null),
        ]),
    ];
    let edge_queries = [
        eq(&w, 3),
        equal(&w, Value::Null),
        range(&w, Some(2), Some(5), true),
        in_list(&w, &[int(0), int(7)]),
        IndexQuery::Or(vec![eq(&w, 1), eq(&w, 7)]),
        IndexQuery::Or(vec![eq(&w, 1), equal(&w, Value::Null)]),
    ];
    let labels: OrderSet<Arc<String>> = std::iter::once(label.clone()).collect();
    let sorted = |mut ids: Vec<u64>| {
        ids.sort_unstable();
        ids
    };

    // What each scan kept, at every cursor: `(edge?, query, kept)`.
    let mut kept: Vec<(bool, usize, Vec<u64>)> = Vec::new();
    loop {
        for (i, q) in node_queries.iter().enumerate() {
            if let Some(NumericAnswer::NotReady(recheck)) = g.query_index_numeric_nodes(&label, q) {
                let attr = g
                    .get_node_attribute_id(recheck.attribute())
                    .expect("registered") as u16;
                let ids = g
                    .get_nodes(&labels, 0)
                    .filter(|&id| g.node_admitted(id, attr, &recheck))
                    .map(u64::from)
                    .collect();
                kept.push((false, i, sorted(ids)));
            }
        }
        for (i, q) in edge_queries.iter().enumerate() {
            if let Some(NumericAnswer::NotReady(recheck)) = g.query_index_numeric_edges(&ty, q) {
                let attr = g
                    .get_relationship_attribute_id(recheck.attribute())
                    .expect("registered") as u16;
                let ids = g
                    .get_all_edges(&ty)
                    .into_iter()
                    .filter(|&(_, _, edge)| g.relationship_admitted(edge, attr, &recheck))
                    .map(|(_, _, edge)| u64::from(edge))
                    .collect();
                kept.push((true, i, sorted(ids)));
            }
        }
        if !g.falkordb_index.has_building_columns() {
            break;
        }
        graph_writes::run_indexer_batch(&mut g, 2);
    }
    for edges in [false, true] {
        let queries = if edges {
            edge_queries.len()
        } else {
            node_queries.len()
        };
        for i in 0..queries {
            assert!(
                kept.iter()
                    .filter(|&&(e, j, _)| e == edges && j == i)
                    .count()
                    >= 2,
                "edges={edges}, query {i}: scanned at more than one cursor"
            );
        }
    }

    for (edges, i, scanned) in kept {
        let built = if edges {
            match g.query_index_numeric_edges(&ty, &edge_queries[i]) {
                Some(NumericAnswer::Rows(rows)) => {
                    rows.map(|(_, _, edge)| u64::from(edge)).collect()
                }
                _ => panic!("edge query {i}: a built column serves it"),
            }
        } else {
            match g.query_index_numeric_nodes(&label, &node_queries[i]) {
                Some(NumericAnswer::Rows(rows)) => rows.map(u64::from).collect(),
                _ => panic!("node query {i}: a built column serves it"),
            }
        };
        assert_eq!(scanned, sorted(built), "edges={edges}, query {i}");
    }
}

/// A node batch reads its size and no more: after each batch the column holds exactly the nodes
/// read so far, all of them below the cursor, so nine nodes take exactly `ceil(9 / batch)` batches,
/// the last of which flips the column `Ready`. A batch that read further than it moved the cursor
/// would leave nodes at or above the cursor in the column.
#[test]
fn a_node_batch_reads_at_most_its_size() {
    let attr = Arc::new("v".to_string());
    for (batch, batches) in [(1usize, 9usize), (2, 5), (4, 3), (9, 1), (10, 1)] {
        let (mut g, label, ids) = people_with(&attr, &(1..=9).collect::<Vec<_>>());
        create_index_on(&mut g, &label, &attr);
        for k in 1..=batches {
            let read = (k * batch).min(ids.len());
            assert_eq!(
                graph_writes::run_indexer_batch(&mut g, batch),
                read < ids.len(),
                "batch {batch}, #{k}: Building until the last"
            );
            assert_eq!(
                range_scan(&g, &label, &attr, 0, 100),
                ids[..read].to_vec(),
                "batch {batch}, #{k}: the column holds exactly the nodes read so far"
            );
            assert_eq!(
                cursor(&g, EntityType::Node, &label, &attr),
                ids.get(read).map(|&id| BuildCursor::Node(NodeId::from(id))),
                "batch {batch}, #{k}: the cursor is past them"
            );
        }
    }
}

/// A batch of size zero reads nothing, for a node column and for an edge column: the column stays
/// empty and `Building`, and the build that follows still reaches every entity. C's populate reads
/// nothing for a zero batch either (`indexed < batch_size || multi`, `src/index/index_construct.c`).
#[test]
fn a_batch_of_size_zero_reads_nothing() {
    for entity in [EntityType::Node, EntityType::Relationship] {
        let (mut g, label, v, ty, w) = scan_fixture();
        let (l, a, other, values): (_, _, _, Vec<i64>) = match entity {
            EntityType::Node => (
                &label,
                &v,
                (EntityType::Relationship, &ty, &w),
                (0..10).map(|i| i * 10).collect(),
            ),
            EntityType::Relationship => (&ty, &w, (EntityType::Node, &label, &v), (0..8).collect()),
        };
        g.falkordb_index.drop_column(other.0, other.1, other.2);

        assert!(
            graph_writes::run_indexer_batch(&mut g, 0),
            "{entity:?}: still Building"
        );
        assert!(
            contents(&g, entity, l, a, &[]).1.is_empty(),
            "{entity:?}: nothing read"
        );

        build_to_end(&mut g, 2);
        assert_eq!(
            contents(&g, entity, l, a, &values),
            one_pass(&g, entity, l, a, &values),
            "{entity:?}"
        );
    }
}

/// The same unit of work against a `Building` column, on a node the batches already indexed and on
/// one they have not reached: neither the old value nor the SET's value may survive the DELETE.
#[test]
fn set_then_delete_in_one_batch_during_a_build() {
    let attr = Arc::new("v".to_string());
    let (mut g, label, ids) = people_with(&attr, &[1, 2, 3]);
    create_index_on(&mut g, &label, &attr);
    assert!(graph_writes::run_indexer_batch(&mut g, 1));
    assert_eq!(range_scan(&g, &label, &attr, 0, 10), vec![ids[0]]);

    // SET ids[0].v = 5, ids[2].v = 5, then DELETE both, as `Pending::commit` orders them.
    let aid = g.get_or_create_node_attr_id(&attr);
    let mut attrs = FxHashMap::default();
    attrs.insert(ids[0], vec![(aid, Value::Int(5))]);
    attrs.insert(ids[2], vec![(aid, Value::Int(5))]);
    let mut writes = IndexDocs::default();
    g.set_nodes_attributes(&attrs, &mut writes).unwrap();
    let mut del = RoaringTreemap::new();
    del.insert(ids[0]);
    del.insert(ids[2]);
    g.delete_nodes(&del, &mut writes).unwrap();
    g.apply_index_writes(&mut writes);

    build_to_end(&mut g, 1);
    assert_eq!(
        range_scan(&g, &label, &attr, 0, 10),
        vec![ids[1]],
        "neither deleted node's old value nor its SET value survives"
    );
}

// --- DROP and CREATE INDEX while a column is building ---

/// DROP INDEX while the column is building takes everything already indexed with it — what the
/// batches put there lives in the column — and leaves no batch anything to do.
/// Writes after the drop stage nothing.
#[test]
fn dropping_a_building_column_leaves_nothing_behind() {
    let attr = Arc::new("v".to_string());
    let (mut g, label, ids) = people_with(&attr, &[10, 20]);
    create_index_on(&mut g, &label, &attr);
    assert!(graph_writes::run_indexer_batch(&mut g, 1));
    set_attr(&mut g, ids[0], &attr, Value::Int(30));

    drop_index_on(&mut g, &label, &attr);
    assert!(!g.falkordb_index.has_building_columns());
    assert!(
        g.falkordb_index
            .numeric(EntityType::Node, &label, &attr)
            .is_none()
    );
    assert!(
        !graph_writes::run_indexer_batch(&mut g, usize::MAX),
        "nothing left to build"
    );
    assert!(
        g.falkordb_index
            .numeric(EntityType::Node, &label, &attr)
            .is_none()
    );

    // Indexing stopped: a later write to the attribute stages nothing.
    let aid = g.get_or_create_node_attr_id(&attr);
    let mut attrs = FxHashMap::default();
    attrs.insert(ids[1], vec![(aid, Value::Int(40))]);
    let mut writes = IndexDocs::default();
    g.set_nodes_attributes(&attrs, &mut writes).unwrap();
    assert!(writes.native.is_empty(), "no column, so nothing to index");
}

/// DROP then CREATE INDEX on the same column mid-build: the new column starts from the first node,
/// so a value that changed while no index existed is indexed as it is now, and the old one is gone.
#[test]
fn recreating_a_building_column_restarts_its_build() {
    let attr = Arc::new("v".to_string());
    let (mut g, label, ids) = people_with(&attr, &[10, 20]);
    create_index_on(&mut g, &label, &attr);
    assert!(graph_writes::run_indexer_batch(&mut g, 1));

    drop_index_on(&mut g, &label, &attr);
    set_attr(&mut g, ids[0], &attr, Value::Int(15)); // while no index exists
    create_index_on(&mut g, &label, &attr);
    assert_eq!(
        cursor(&g, EntityType::Node, &label, &attr),
        Some(BuildCursor::new(EntityType::Node)),
        "the re-created column starts over"
    );

    build_to_end(&mut g, 1);
    assert_eq!(range_scan(&g, &label, &attr, 10, 10), Vec::<u64>::new());
    assert_eq!(range_scan(&g, &label, &attr, 15, 15), vec![ids[0]]);
    assert_eq!(range_scan(&g, &label, &attr, 20, 20), vec![ids[1]]);
}

/// CREATE INDEX on a second attribute of the same label while the first is building adds a column
/// of its own, starting from the first node, and leaves the first column's cursor alone; each
/// column ends up with its own values.
#[test]
fn indexing_a_second_attribute_leaves_the_first_build_alone() {
    let v = Arc::new("v".to_string());
    let w = Arc::new("w".to_string());
    let (mut g, label, ids) = people_with(&v, &[10, 20]);
    for (&id, value) in ids.iter().zip([1, 2]) {
        set_attr(&mut g, id, &w, Value::Int(value));
    }
    create_index_on(&mut g, &label, &v);
    assert!(graph_writes::run_indexer_batch(&mut g, 1));
    let v_cursor = cursor(&g, EntityType::Node, &label, &v);
    assert_eq!(v_cursor, Some(BuildCursor::Node(NodeId::from(ids[1]))));

    create_index_on(&mut g, &label, &w);
    assert_eq!(
        cursor(&g, EntityType::Node, &label, &w),
        Some(BuildCursor::new(EntityType::Node))
    );
    assert_eq!(
        cursor(&g, EntityType::Node, &label, &v),
        v_cursor,
        "the first column's build is untouched"
    );

    build_to_end(&mut g, 1);
    assert_eq!(range_scan(&g, &label, &v, 0, 100), vec![ids[0], ids[1]]);
    assert_eq!(range_scan(&g, &label, &w, 2, 2), vec![ids[1]]);
}

// --- The replica's write paths ---
//
// A replica applies effects through its own `Graph` entry points, not the query path's. Each
// test drives one of them the way `effects::v3::apply` does and checks the replica's column.

/// CREATE_NODE then UPDATE_NODE, as the replica applies them: labels through
/// `set_node_labels_product`, values through `set_nodes_attributes_rows_of_labels`. The update
/// must remove the old value as well as add the new one.
#[test]
fn replica_node_rows_maintain_the_index() {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let label = Arc::new("Person".to_string());
    let attr = Arc::new("v".to_string());
    g.falkordb_index
        .create_numeric(EntityType::Node, &label, &attr);
    let lid = g.get_label_id_mut("Person").0 as u64;
    let aid = g.get_or_create_node_attr_id(&attr);

    // CREATE_NODE: two nodes, v = 5 and 6.
    let ids = reserve_nodes(&mut g, 2);
    g.create_nodes(&ids.iter().copied().collect())
        .expect("freshly reserved ids are creatable");
    applied!(g, |w| g.set_node_labels_product(&ids, &[lid], &mut w, true));
    applied!(g, |w| g.set_nodes_attributes_rows_of_labels(
        &ids,
        &[lid],
        &[aid],
        &[Value::Int(5), Value::Int(6)],
        &mut w,
    ))
    .unwrap();
    assert_eq!(
        range_scan(&g, &label, &attr, 0, 100),
        ids,
        "created nodes are indexed"
    );

    // UPDATE_NODE: the first node's v goes 5 -> 9.
    applied!(g, |w| g.set_nodes_attributes_rows_of_labels(
        &ids[..1],
        &[lid],
        &[aid],
        &[Value::Int(9)],
        &mut w,
    ))
    .unwrap();
    assert!(
        range_scan(&g, &label, &attr, 5, 5).is_empty(),
        "the old value is gone"
    );
    assert_eq!(range_scan(&g, &label, &attr, 9, 9), vec![ids[0]]);
}

/// SET_LABELS on a node that already has attributes, as the replica applies it.
#[test]
fn replica_set_labels_indexes_existing_attrs() {
    let (mut g, label, attr, id) = unlabeled_graph_with_index();
    set_attr(&mut g, id, &attr, Value::Int(5));
    let lid = g.get_label_id_mut("Person").0 as u64;
    applied!(g, |w| g.set_node_labels_product(
        &[id],
        &[lid],
        &mut w,
        false
    ));
    assert_eq!(range_scan(&g, &label, &attr, 4, 6), vec![id]);
}

/// UPDATE_EDGE, as the replica applies it: one record is one type, through
/// `set_relationships_attributes_of_type`.
#[test]
fn replica_edge_update_of_type_maintains_the_index() {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let ty = Arc::new("R".to_string());
    let attr = Arc::new("w".to_string());
    g.falkordb_index
        .create_numeric(EntityType::Relationship, &ty, &attr);
    let aid = g.get_or_create_rel_attr_id(&attr);
    let nodes = reserve_nodes(&mut g, 2);
    g.create_nodes(&nodes.iter().copied().collect())
        .expect("freshly reserved ids are creatable");
    let ids: Vec<u64> = g
        .relationship_id_space()
        .reserve(1, &RoaringTreemap::new())
        .expect("reserved");
    g.create_relationships_bulk(&ty, &nodes[..1], &nodes[1..], &ids)
        .expect("freshly reserved ids are creatable");
    let type_id = g.get_type_id_mut("R");

    let attrs_of = |w: i64| -> FxHashMap<u64, Vec<(u16, Value)>> {
        std::iter::once((ids[0], vec![(aid, Value::Int(w))])).collect()
    };
    applied!(g, |w| g.set_relationships_attributes_of_type(
        type_id,
        &attrs_of(10),
        &mut w
    ))
    .unwrap();
    applied!(g, |w| g.set_relationships_attributes_of_type(
        type_id,
        &attrs_of(20),
        &mut w
    ))
    .unwrap();

    let scan = |lo: i64, hi: i64| -> Vec<u64> {
        g.falkordb_index
            .numeric(EntityType::Relationship, &ty, &attr)
            .unwrap()
            .range(Some(&Value::Int(lo)), Some(&Value::Int(hi)), true, true)
            .collect()
    };
    assert!(scan(10, 10).is_empty(), "the old value is gone");
    assert_eq!(scan(20, 20), ids);
}

// --- One buffer per unit of work ---

/// SET then DELETE of one node inside one unit of work. `Pending::commit` runs its SETs before
/// its DELETEs and stages both into one buffer, so the batch holds `add (5, n)` from the SET and
/// then `remove (5, n)` from the DELETE. Applying all removes and then all adds would leave the
/// deleted node in the column, and it would come back as a wrong row once the id is reused. The
/// buffer keeps each entry's last change, so the remove wins.
#[test]
fn set_then_delete_in_one_batch_leaves_no_entry() {
    let (mut g, label, attr, ids) = graph_with_index(2);
    set_attr(&mut g, ids[0], &attr, Value::Int(1));
    set_attr(&mut g, ids[1], &attr, Value::Int(2));

    // One unit of work, in `Pending::commit`'s order: SET n.v = 5, then DELETE n.
    let aid = g.get_or_create_node_attr_id(&attr);
    let mut attrs = FxHashMap::default();
    attrs.insert(ids[0], vec![(aid, Value::Int(5))]);
    let mut writes = IndexDocs::default();
    g.set_nodes_attributes(&attrs, &mut writes).unwrap();
    let mut del = RoaringTreemap::new();
    del.insert(ids[0]);
    g.delete_nodes(&del, &mut writes).unwrap();
    g.apply_index_writes(&mut writes);

    assert!(
        range_scan(&g, &label, &attr, 5, 5).is_empty(),
        "the deleted node's new value must not be indexed"
    );
    assert!(
        range_scan(&g, &label, &attr, 1, 1).is_empty(),
        "nor its old one"
    );
    assert_eq!(
        range_scan(&g, &label, &attr, 0, 10),
        vec![ids[1]],
        "the untouched node is all that is left"
    );
}

// --- Indexer batch commits ---

/// An MVCC graph of `:Person` nodes with `v` = each of `values`, and a `Building` column on
/// `(:Person, v)`.
fn mvcc_building(values: &[i64]) -> (MvccGraph, Arc<String>, Arc<String>, Vec<u64>) {
    let attr = Arc::new("v".to_string());
    let (mut g, label, ids) = people_with(&attr, values);
    create_index_on(&mut g, &label, &attr);
    (MvccGraph::from_graph(g), label, attr, ids)
}

/// An indexer batch of at most `budget` entities of index `id`, applied to a new build version of
/// `mvcc`'s published one, as the indexer's turn prepares it: the version and the batch.
fn prepare(
    mvcc: &MvccGraph,
    id: &IndexId,
    budget: usize,
) -> (Arc<AtomicRefCell<Graph>>, PreparedBatch) {
    let version = Arc::new(AtomicRefCell::new(mvcc.read().borrow().new_build_version()));
    let batch = version
        .borrow_mut()
        .indexer_prepare(id, budget)
        .expect("the index is Building");
    (version, batch)
}

/// One writer's unit of work on `mvcc`: claim the write slot, `write` into the fork, commit it.
fn commit_write(
    mvcc: &MvccGraph,
    write: impl FnOnce(&mut Graph),
) {
    let fork = mvcc.write().expect("the write slot is free");
    write(&mut fork.borrow_mut());
    mvcc.commit(fork).expect("the write commits");
}

/// With nothing committed since its version was made, an indexer batch publishes that version
/// itself. The batch changes no data, so the data version stays, and the slot is free after.
#[test]
fn an_indexer_batch_with_nothing_committed_since_publishes_its_version() {
    let (mvcc, label, attr, ids) = mvcc_building(&[10, 20, 30, 40]);
    let id = IndexId::Column(EntityType::Node, (label.clone(), attr.clone()));
    let data_version = mvcc.read().borrow().version;

    let (prepared, batch) = prepare(&mvcc, &id, 2);
    assert_eq!(
        mvcc.commit_indexer_batch(&prepared, &batch),
        Some(IndexerCommit::Same)
    );
    let published = mvcc.read();
    assert!(Arc::ptr_eq(&published, &prepared), "its own version");
    let published = published.borrow();
    assert_eq!(published.version, data_version, "the data version stays");
    assert_eq!(
        cursor(&published, EntityType::Node, &label, &attr),
        Some(BuildCursor::Node(NodeId::from(ids[2])))
    );
    assert_eq!(
        range_scan(&published, &label, &attr, 0, 100),
        ids[..2].to_vec()
    );
    assert!(mvcc.write().is_some(), "the write slot is free");
    mvcc.rollback();
}

/// When only another index's batch committed since, an indexer batch puts its column into a new
/// build version of the published one: both batches' progress is kept.
#[test]
fn an_indexer_batch_after_another_columns_batch_swaps_its_column_in() {
    let (g, label, v, ty, w) = scan_fixture();
    let mvcc = MvccGraph::from_graph(g);
    let nodes = IndexId::Column(EntityType::Node, (label.clone(), v.clone()));
    let edges = IndexId::Column(EntityType::Relationship, (ty.clone(), w.clone()));

    let (node_version, node_batch) = prepare(&mvcc, &nodes, 3);
    let (edge_version, edge_batch) = prepare(&mvcc, &edges, 2);
    assert_eq!(
        mvcc.commit_indexer_batch(&edge_version, &edge_batch),
        Some(IndexerCommit::Same)
    );
    assert_eq!(
        mvcc.commit_indexer_batch(&node_version, &node_batch),
        Some(IndexerCommit::Swap)
    );

    let published = mvcc.read();
    assert!(!Arc::ptr_eq(&published, &node_version));
    let published = published.borrow();
    for (entity, l, a, version) in [
        (EntityType::Node, &label, &v, &node_version),
        (EntityType::Relationship, &ty, &w, &edge_version),
    ] {
        let version = version.borrow();
        assert_eq!(
            cursor(&published, entity, l, a),
            cursor(&version, entity, l, a),
            "{entity:?}"
        );
        assert_eq!(
            contents(&published, entity, l, a, &[]),
            contents(&version, entity, l, a, &[]),
            "{entity:?}"
        );
        assert!(cursor(&published, entity, l, a).is_some(), "{entity:?}");
    }
}

/// A reader keeps its borrow of the published version across an indexer batch's commit, however
/// the batch goes in. The commit only reads the published version and publishes a version of its
/// own; a `borrow_mut` of the published one would meet the reader's borrow, and `AtomicRefCell`
/// would panic.
#[test]
fn an_indexer_batch_commit_leaves_a_borrowed_published_version_alone() {
    // Same, and CatchUp after a writer: one column.
    for overtaken in [false, true] {
        let (mvcc, label, attr, ids) = mvcc_building(&[10, 20, 30, 40]);
        let id = IndexId::Column(EntityType::Node, (label.clone(), attr.clone()));
        let (prepared, batch) = prepare(&mvcc, &id, 2);
        if overtaken {
            commit_write(&mvcc, |g| set_attr(g, ids[0], &attr, Value::Int(15)));
        }
        let held = mvcc.read();
        let reader = held.borrow();
        let before = cursor(&reader, EntityType::Node, &label, &attr);

        let committed = mvcc.commit_indexer_batch(&prepared, &batch);
        assert!(
            matches!(
                (overtaken, &committed),
                (false, Some(IndexerCommit::Same)) | (true, Some(IndexerCommit::CatchUp(_)))
            ),
            "{committed:?}"
        );
        assert_eq!(cursor(&reader, EntityType::Node, &label, &attr), before);
        assert_ne!(
            cursor(&mvcc.read().borrow(), EntityType::Node, &label, &attr),
            before,
            "the batch's progress is published"
        );
    }

    // Swap: another column's batch committed first.
    let (g, label, v, ty, w) = scan_fixture();
    let mvcc = MvccGraph::from_graph(g);
    let nodes = IndexId::Column(EntityType::Node, (label.clone(), v.clone()));
    let edges = IndexId::Column(EntityType::Relationship, (ty, w));
    let (node_version, node_batch) = prepare(&mvcc, &nodes, 3);
    let (edge_version, edge_batch) = prepare(&mvcc, &edges, 2);
    assert_eq!(
        mvcc.commit_indexer_batch(&edge_version, &edge_batch),
        Some(IndexerCommit::Same)
    );
    let held = mvcc.read();
    let reader = held.borrow();
    let before = cursor(&reader, EntityType::Node, &label, &v);
    assert_eq!(
        mvcc.commit_indexer_batch(&node_version, &node_batch),
        Some(IndexerCommit::Swap)
    );
    assert_eq!(cursor(&reader, EntityType::Node, &label, &v), before);
}

/// An indexer batch whose column a writer dropped, or dropped and created again, since its
/// version was made publishes nothing. A column created again starts its build over: its next
/// batch reads from the first node.
#[test]
fn an_indexer_batch_on_a_dropped_or_recreated_column_publishes_nothing() {
    for recreate in [false, true] {
        let (mvcc, label, attr, ids) = mvcc_building(&[10, 20, 30]);
        let id = IndexId::Column(EntityType::Node, (label.clone(), attr.clone()));
        let (prepared, batch) = prepare(&mvcc, &id, 2);
        commit_write(&mvcc, |g| {
            drop_index_on(g, &label, &attr);
            if recreate {
                create_index_on(g, &label, &attr);
            }
        });

        let before = mvcc.read();
        assert_eq!(
            mvcc.commit_indexer_batch(&prepared, &batch),
            Some(IndexerCommit::Dropped(Dropped::Gone)),
            "recreate {recreate}"
        );
        assert!(Arc::ptr_eq(&mvcc.read(), &before), "nothing published");
        assert!(mvcc.write().is_some(), "the write slot is free");
        mvcc.rollback();

        if recreate {
            let (prepared, batch) = prepare(&mvcc, &id, 2);
            assert_eq!(
                mvcc.commit_indexer_batch(&prepared, &batch),
                Some(IndexerCommit::Same)
            );
            assert_eq!(
                range_scan(&mvcc.read().borrow(), &label, &attr, 0, 100),
                ids[..2].to_vec(),
                "the build starts over"
            );
        }
    }
}

/// An indexer batch read on a version writers have since moved past catches up on what they
/// changed in its column, and is published: the batch's progress with the writers' values. A
/// writer's change below the batch's new cursor is in the column; one at or above it is not, since
/// the next batch reads it. Both ways of catching up, refreshing and inserting, give that.
#[test]
fn an_indexer_batch_after_a_writer_catches_up_on_its_changes() {
    for how in [CatchUp::Refresh, CatchUp::Insert] {
        let (mvcc, label, attr, ids) = mvcc_building(&[10, 20, 30, 40]);
        let id = IndexId::Column(EntityType::Node, (label.clone(), attr.clone()));
        let (prepared, batch) = prepare(&mvcc, &id, 2);
        commit_write(&mvcc, |g| {
            set_attr(g, ids[0], &attr, Value::Int(15));
            set_attr(g, ids[2], &attr, Value::Int(35));
        });
        let data_version = mvcc.read().borrow().version;

        CATCH_UP_BY.with(|by| by.set(Some(how)));
        let committed = mvcc.commit_indexer_batch(&prepared, &batch);
        CATCH_UP_BY.with(|by| by.set(None));
        assert_eq!(committed, Some(IndexerCommit::CatchUp(how)));
        let published = mvcc.read();
        let published = published.borrow();
        assert_eq!(
            published.version, data_version,
            "{how:?}: the writer's data"
        );
        assert_eq!(
            cursor(&published, EntityType::Node, &label, &attr),
            Some(BuildCursor::Node(NodeId::from(ids[2]))),
            "{how:?}: the batch's progress"
        );
        assert_eq!(
            range_scan(&published, &label, &attr, 0, 100),
            vec![ids[0], ids[1]],
            "{how:?}: nothing at or above the cursor"
        );
        assert_eq!(
            range_scan(&published, &label, &attr, 15, 15),
            vec![ids[0]],
            "{how:?}: the writer's value"
        );
        assert!(
            range_scan(&published, &label, &attr, 10, 10).is_empty(),
            "{how:?}"
        );
    }
}

/// A batch that writers committed after refreshes what they changed when that costs less than
/// inserting the batch, an entity costing [`REFRESH_COST`] tuples, and inserts the batch otherwise.
#[test]
fn a_batch_refreshes_writers_changes_only_when_that_costs_less_than_inserting_it() {
    let budget = 2 * REFRESH_COST as usize;
    for (written, how) in [(1, CatchUp::Refresh), (3, CatchUp::Insert)] {
        let values: Vec<i64> = (0..=budget as i64).collect();
        let (mvcc, label, attr, ids) = mvcc_building(&values);
        let id = IndexId::Column(EntityType::Node, (label.clone(), attr.clone()));
        let (prepared, batch) = prepare(&mvcc, &id, budget);
        commit_write(&mvcc, |g| {
            for &id in &ids[..written] {
                set_attr(g, id, &attr, Value::Int(1000));
            }
        });
        assert_eq!(
            mvcc.commit_indexer_batch(&prepared, &batch),
            Some(IndexerCommit::CatchUp(how)),
            "{written} written against a batch of {budget}"
        );
        assert_eq!(
            range_scan(&mvcc.read().borrow(), &label, &attr, 1000, 1000),
            ids[..written].to_vec()
        );
    }
}

/// A claim needs the fork mutably only to rebase it: with no indexer batch committed since it was
/// made, a fork can be claimed while it is borrowed.
#[test]
fn a_claim_with_nothing_to_rebase_leaves_the_fork_alone() {
    let (mvcc, _, _, _) = mvcc_building(&[10, 20, 30]);
    let fork = mvcc.fork();
    let reading = fork.borrow();
    assert_eq!(mvcc.claim(&fork), Ok(()));
    drop(reading);
    mvcc.rollback();
}

/// A claimant's fork is never borrowed. One that is, when indexer batches overtook it, is a bug a
/// debug build stops on, rather than a write the host runs again for as long as it lasts.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "a fork is claimed while it is borrowed")]
fn a_borrowed_fork_to_rebase_stops_a_debug_build() {
    let (mvcc, label, attr, _) = mvcc_building(&[10, 20, 30]);
    let id = IndexId::Column(EntityType::Node, (label.clone(), attr.clone()));
    let fork = mvcc.fork();
    let (prepared, batch) = prepare(&mvcc, &id, 2);
    mvcc.commit_indexer_batch(&prepared, &batch);
    let _reading = fork.borrow();
    let _ = mvcc.claim(&fork);
}

/// A writer whose fork an indexer batch overtook on a `Building` column claims and commits over
/// it: the column keeps the batch's progress, and the writer's changes below its cursor are in it.
/// A change at or above the cursor is not, since a later batch reads it.
#[test]
fn a_writer_overtaken_by_an_indexer_batch_keeps_its_progress() {
    let (mvcc, label, attr, ids) = mvcc_building(&[10, 20, 30, 40]);
    let id = IndexId::Column(EntityType::Node, (label.clone(), attr.clone()));
    let fork = mvcc.fork();
    set_attr(&mut fork.borrow_mut(), ids[0], &attr, Value::Int(15));
    set_attr(&mut fork.borrow_mut(), ids[3], &attr, Value::Int(45));
    let (prepared, batch) = prepare(&mvcc, &id, 2);
    assert_eq!(
        mvcc.commit_indexer_batch(&prepared, &batch),
        Some(IndexerCommit::Same)
    );

    assert_eq!(mvcc.claim(&fork), Ok(()));
    mvcc.commit(fork).expect("the write commits");
    let published = mvcc.read();
    let published = published.borrow();
    assert_eq!(
        cursor(&published, EntityType::Node, &label, &attr),
        Some(BuildCursor::Node(NodeId::from(ids[2]))),
        "the batch's progress"
    );
    assert_eq!(
        range_scan(&published, &label, &attr, 0, 100),
        vec![ids[0], ids[1]],
        "nothing at or above the cursor"
    );
    assert_eq!(
        range_scan(&published, &label, &attr, 15, 15),
        vec![ids[0]],
        "the writer's value"
    );
}

/// Two indexer batches read from one version on the same column: the first publishes, the second
/// finds the first in its way and publishes nothing. One chain of batches per column rules this
/// out, so it is reported; here the report is allowed, to check what is published.
#[test]
fn of_two_indexer_batches_on_one_column_only_the_first_publishes() {
    let (mvcc, label, attr, _) = mvcc_building(&[10, 20, 30]);
    let id = IndexId::Column(EntityType::Node, (label.clone(), attr.clone()));
    let (first, first_batch) = prepare(&mvcc, &id, 1);
    let (second, second_batch) = prepare(&mvcc, &id, 2);
    assert_eq!(
        mvcc.commit_indexer_batch(&first, &first_batch),
        Some(IndexerCommit::Same)
    );
    ALLOW_BATCH_RACE.with(|allow| allow.set(true));
    let second_commit = mvcc.commit_indexer_batch(&second, &second_batch);
    ALLOW_BATCH_RACE.with(|allow| allow.set(false));
    assert_eq!(
        second_commit,
        Some(IndexerCommit::Dropped(Dropped::AnotherBatch))
    );
    assert!(
        Arc::ptr_eq(&mvcc.read(), &first),
        "the first batch's version"
    );
}

/// A debug build stops on two indexer batches racing on one column.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "two indexer batches raced")]
fn a_race_of_two_indexer_batches_stops_a_debug_build() {
    let (mvcc, label, attr, _) = mvcc_building(&[10, 20, 30]);
    let id = IndexId::Column(EntityType::Node, (label.clone(), attr.clone()));
    let (first, first_batch) = prepare(&mvcc, &id, 1);
    let (second, second_batch) = prepare(&mvcc, &id, 1);
    mvcc.commit_indexer_batch(&first, &first_batch);
    mvcc.commit_indexer_batch(&second, &second_batch);
}

/// An indexer batch's commit keeps the data version, so a writer's fork made before it can still
/// be claimed.
#[test]
fn a_fork_made_before_an_indexer_batch_commit_can_still_be_claimed() {
    let (mvcc, label, attr, _) = mvcc_building(&[10, 20, 30]);
    let id = IndexId::Column(EntityType::Node, (label.clone(), attr.clone()));
    let fork = mvcc.fork();
    let (prepared, batch) = prepare(&mvcc, &id, 2);
    assert_eq!(
        mvcc.commit_indexer_batch(&prepared, &batch),
        Some(IndexerCommit::Same)
    );
    assert_eq!(mvcc.claim(&fork), Ok(()));
    mvcc.rollback();
}

/// While a writer holds the write slot, an indexer batch's commit takes nothing and publishes
/// nothing.
#[test]
fn an_indexer_batch_does_not_commit_while_a_writer_holds_the_slot() {
    let (mvcc, label, attr, _) = mvcc_building(&[10, 20, 30]);
    let id = IndexId::Column(EntityType::Node, (label.clone(), attr.clone()));
    let before = mvcc.read();
    let writer = mvcc.write().expect("the write slot is free");
    let (prepared, batch) = prepare(&mvcc, &id, 2);
    assert_eq!(mvcc.commit_indexer_batch(&prepared, &batch), None);
    assert!(Arc::ptr_eq(&mvcc.read(), &before), "nothing published");
    drop(writer);
    mvcc.rollback();
}

/// An indexer batch's commit that unwinds, inside the commit itself, leaves the write slot free:
/// the next writer can claim it.
#[test]
fn an_indexer_batch_commit_that_unwinds_releases_the_write_slot() {
    let (mvcc, label, attr, _) = mvcc_building(&[10, 20, 30]);
    let id = IndexId::Column(EntityType::Node, (label.clone(), attr.clone()));
    let before = mvcc.read();
    let (prepared, batch) = prepare(&mvcc, &id, 2);
    PANIC_IN_INDEXER_COMMIT.with(|panic| panic.set(true));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mvcc.commit_indexer_batch(&prepared, &batch)
    }));
    PANIC_IN_INDEXER_COMMIT.with(|panic| panic.set(false));
    assert!(unwound.is_err(), "the commit unwound");
    assert!(Arc::ptr_eq(&mvcc.read(), &before), "nothing published");
    assert!(mvcc.write().is_some(), "the write slot is free");
    mvcc.rollback();
}

/// Delete node `id`, in a unit of work of its own.
fn delete_node(
    g: &mut Graph,
    id: u64,
) {
    g.open_id_batches().expect("a consistent space");
    let mut del = RoaringTreemap::new();
    del.insert(id);
    applied!(g, |w| g.delete_nodes(&del, &mut w)).expect("deleted");
}

/// Move node `id` from `:Person` to `:Other`.
fn relabel_to_other(
    g: &mut Graph,
    id: u64,
) {
    let person = g.get_label_id_mut("Person").0 as u64;
    let other = g.get_label_id_mut("Other").0 as u64;
    applied!(g, |d| g.remove_nodes_labels(&[id], &[person], &mut d));
    applied!(g, |d| g.set_nodes_labels_bulk(
        &[id],
        &[other],
        &mut d,
        false
    ));
}

fn attr_named(name: &str) -> Arc<String> {
    Arc::new(name.to_string())
}

/// Every order in which `actors` actors each do two things, the first before the second: each
/// order lists, for each event, the actor whose turn it is.
fn orders(actors: usize) -> Vec<Vec<usize>> {
    fn extend(
        left: &mut [u8],
        order: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if left.iter().all(|&n| n == 0) {
            out.push(order.clone());
            return;
        }
        for actor in 0..left.len() {
            if left[actor] > 0 {
                left[actor] -= 1;
                order.push(actor);
                extend(left, order, out);
                order.pop();
                left[actor] += 1;
            }
        }
    }
    let mut out = Vec::new();
    extend(&mut vec![2; actors], &mut Vec::new(), &mut out);
    out
}

/// The `:Person` and `:Other` nodes of a version: whether each is a `:Person`, and its `v` and `w`.
type Model = BTreeMap<u64, (bool, Option<i64>, Option<i64>)>;

fn model_of(g: &Graph) -> Model {
    let value = |n: u64, attr: &str| {
        let idx = g.get_node_attribute_id(&attr_named(attr))?;
        match g.get_node_attribute_by_idx(NodeId::from(n), idx as u16)? {
            Value::Int(x) => Some(x),
            _ => None,
        }
    };
    let mut model = Model::new();
    for (label, person) in [("Person", true), ("Other", false)] {
        if let Some(m) = g.get_label_matrix(&attr_named(label)) {
            for (n, _) in m.iter(0, u64::MAX) {
                model.insert(n, (person, value(n, "v"), value(n, "w")));
            }
        }
    }
    model
}

/// The rules on [`BuildState`], checked on a published version: each column on `attrs` holds
/// exactly the tuples its state says, from the version's own data. A `Ready` column holds every
/// live `:Person` node's, a `Building` one those of the nodes below its cursor, and nothing else.
fn check_rules(
    g: &Graph,
    label: &Arc<String>,
    attrs: [&Arc<String>; 2],
    what: &str,
) {
    let model = model_of(g);
    for (i, attr) in attrs.into_iter().enumerate() {
        let Some(state) = g.falkordb_index.column_state(EntityType::Node, label, attr) else {
            continue;
        };
        let expected: BTreeSet<(u64, u64)> = model
            .iter()
            .filter_map(|(&id, &(person, v, w))| {
                let value = if i == 0 { v } else { w };
                let key = encode_numeric(&Value::Int(value?))?;
                let below = match state {
                    BuildState::Ready => true,
                    BuildState::Building {
                        next: BuildCursor::Node(next),
                    } => id < u64::from(next),
                    BuildState::Building { .. } => unreachable!("a node column"),
                };
                (person && below).then_some((key, id))
            })
            .collect();
        let held: BTreeSet<(u64, u64)> = g
            .falkordb_index
            .numeric(EntityType::Node, label, attr)
            .expect("a numeric column")
            .tuples()
            .into_iter()
            .collect();
        assert_eq!(held, expected, "{what}: {attr} ({state:?})");
    }
}

/// How far the build of column `(label, attr)` of `g` got: its cursor, `u64::MAX` once `Ready`, or
/// `None` if there is no such column.
fn build_progress(
    g: &Graph,
    label: &Arc<String>,
    attr: &Arc<String>,
) -> Option<u64> {
    match g
        .falkordb_index
        .column_state(EntityType::Node, label, attr)?
    {
        BuildState::Ready => Some(u64::MAX),
        BuildState::Building {
            next: BuildCursor::Node(next),
        } => Some(u64::from(next)),
        BuildState::Building { .. } => unreachable!("a node column"),
    }
}

/// What one writer of [`every_order_of_two_writers_and_two_batches_keeps_both_columns_right`] does
/// to its fork, given the nodes' ids.
type OrderedWrite = fn(&mut Graph, &[u64]);

/// Every order of two writers and two indexer batches, each starting (a fork; a batch prepared on
/// a new build version) and later committing, on two `Building` columns: for pairs of writes the
/// rules must get right (re-values and deletes in a batch and past its cursor, a create that may
/// reuse a freed id, a label moved off, the index dropped and created again), with the two
/// batches on different columns and on the same one. After every commit the published version
/// keeps the rules on [`BuildState`] ([`check_rules`]): a writer's commit publishes its own data
/// and a batch's changes none; no commit moves a column's build back, unless its writer created
/// the column again; and a batch that is published moves its own column on. So do both columns
/// once built to the end. Counts how the commits went, so an outcome no order reached fails the
/// test.
#[test]
fn every_order_of_two_writers_and_two_batches_keeps_both_columns_right() {
    const NODES: usize = 12;
    const BATCH: usize = 5;
    let (v, w) = (attr_named("v"), attr_named("w"));
    let (mut template, label, ids) = people_with(&v, &(0..NODES as i64).collect::<Vec<_>>());
    for (i, &id) in ids.iter().enumerate() {
        set_attr(&mut template, id, &w, Value::Int(100 + i as i64));
    }
    create_index_on(&mut template, &label, &v);
    create_index_on(&mut template, &label, &w);

    let writes: [(&str, OrderedWrite, OrderedWrite); 5] = [
        (
            "re-value in the batch, delete past the cursor",
            |g, ids| set_attr(g, ids[1], &attr_named("v"), Value::Int(51)),
            |g, ids| delete_node(g, ids[7]),
        ),
        (
            "delete in the batch, create (maybe reusing its id)",
            |g, ids| delete_node(g, ids[2]),
            |g, _| {
                add_person(g, &attr_named("v"), 52);
            },
        ),
        (
            "label moved off in the batch, re-value past the cursor",
            |g, ids| relabel_to_other(g, ids[3]),
            |g, ids| set_attr(g, ids[8], &attr_named("w"), Value::Int(158)),
        ),
        (
            "index dropped and created again, re-value in the batch",
            |g, _| {
                let (label, v) = (attr_named("Person"), attr_named("v"));
                drop_index_on(g, &label, &v);
                create_index_on(g, &label, &v);
            },
            |g, ids| {
                set_attr(g, ids[0], &attr_named("v"), Value::Int(54));
                set_attr(g, ids[0], &attr_named("w"), Value::Int(154));
            },
        ),
        (
            "re-value then delete past the cursor, re-value and move a label off",
            |g, ids| {
                set_attr(g, ids[11], &attr_named("w"), Value::Int(161));
                delete_node(g, ids[11]);
            },
            |g, ids| {
                set_attr(g, ids[4], &attr_named("v"), Value::Int(55));
                relabel_to_other(g, ids[9]);
            },
        ),
    ];

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let orders = orders(4);
    ALLOW_BATCH_RACE.with(|allow| allow.set(true));
    for how in [CatchUp::Refresh, CatchUp::Insert] {
        CATCH_UP_BY.with(|by| by.set(Some(how)));
        for (name, first, second) in writes {
            for batches_on in [[&v, &w], [&v, &v]] {
                for order in &orders {
                    let mvcc = MvccGraph::from_graph(template.new_version());
                    let mut forks: [Option<(Arc<AtomicRefCell<Graph>>, Arc<AtomicRefCell<Graph>>)>;
                        2] = [None, None];
                    let mut batches: [Option<(Arc<AtomicRefCell<Graph>>, PreparedBatch)>; 2] =
                        [None, None];
                    let mut started = [false; 4];
                    for (event, &actor) in order.iter().enumerate() {
                        let what = format!(
                            "{name}; {how:?}; batches on {}, {}; order {order:?}, event {event}",
                            batches_on[0], batches_on[1]
                        );
                        let before = {
                            let g = mvcc.read();
                            let g = g.borrow();
                            [
                                build_progress(&g, &label, &v),
                                build_progress(&g, &label, &w),
                            ]
                        };
                        let starting = !std::mem::replace(&mut started[actor], true);
                        match (actor, starting) {
                            (0 | 1, true) => {
                                let fork = mvcc.fork();
                                let write = if actor == 0 { first } else { second };
                                write(&mut fork.borrow_mut(), &ids);
                                forks[actor] = Some((fork, mvcc.read()));
                            }
                            (0 | 1, false) => {
                                let (fork, base) = forks[actor].take().expect("started");
                                match mvcc.claim(&fork) {
                                    Ok(()) => {
                                        let over_batches = !Arc::ptr_eq(&mvcc.read(), &base);
                                        let data = model_of(&fork.borrow());
                                        mvcc.commit(fork)
                                            .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                                        *counts
                                            .entry(if over_batches {
                                                "writer over batches"
                                            } else {
                                                "writer claimed"
                                            })
                                            .or_default() += 1;
                                        let published = mvcc.read();
                                        let published = published.borrow();
                                        assert_eq!(model_of(&published), data, "{what}: its data");
                                        let recreated_v =
                                            actor == 0 && name.starts_with("index dropped");
                                        for (i, attr) in [&v, &w].into_iter().enumerate() {
                                            if !(recreated_v && i == 0) {
                                                assert!(
                                                    build_progress(&published, &label, attr)
                                                        >= before[i],
                                                    "{what}: {attr}'s build moved back"
                                                );
                                            }
                                        }
                                        check_rules(&published, &label, [&v, &w], &what);
                                    }
                                    Err(Claim::Stale) => {
                                        *counts.entry("writer stale").or_default() += 1;
                                    }
                                    Err(e) => panic!("{what}: {e:?}"),
                                }
                            }
                            (_, true) => {
                                let id = IndexId::Column(
                                    EntityType::Node,
                                    (label.clone(), batches_on[actor - 2].clone()),
                                );
                                let prepared = Arc::new(AtomicRefCell::new(
                                    mvcc.read().borrow().new_build_version(),
                                ));
                                let batch = prepared.borrow_mut().indexer_prepare(&id, BATCH);
                                batches[actor - 2] = batch.map(|batch| (prepared, batch));
                            }
                            (_, false) => {
                                let Some((prepared, batch)) = batches[actor - 2].take() else {
                                    continue;
                                };
                                let data = model_of(&mvcc.read().borrow());
                                let how = mvcc
                                    .commit_indexer_batch(&prepared, &batch)
                                    .unwrap_or_else(|| panic!("{what}: refused"));
                                *counts
                                    .entry(match how {
                                        IndexerCommit::Same => "batch same",
                                        IndexerCommit::Swap => "batch swapped",
                                        IndexerCommit::CatchUp(CatchUp::Refresh) => {
                                            "batch caught up, refreshing"
                                        }
                                        IndexerCommit::CatchUp(CatchUp::Insert) => {
                                            "batch caught up, inserting"
                                        }
                                        IndexerCommit::Dropped(Dropped::Gone) => {
                                            "batch dropped, gone"
                                        }
                                        IndexerCommit::Dropped(Dropped::AnotherBatch) => {
                                            "batch dropped, another batch"
                                        }
                                    })
                                    .or_default() += 1;
                                let published = mvcc.read();
                                let published = published.borrow();
                                assert_eq!(model_of(&published), data, "{what}: no data changed");
                                for (i, attr) in [&v, &w].into_iter().enumerate() {
                                    assert!(
                                        build_progress(&published, &label, attr) >= before[i],
                                        "{what}: {attr}'s build moved back"
                                    );
                                }
                                if !matches!(how, IndexerCommit::Dropped(_)) {
                                    let i = usize::from(batches_on[actor - 2].as_str() == "w");
                                    assert!(
                                        build_progress(&published, &label, batches_on[actor - 2])
                                            > before[i],
                                        "{what}: the batch's column did not move on"
                                    );
                                }
                                check_rules(&published, &label, [&v, &w], &what);
                            }
                        }
                    }
                    // Then build both columns to the end.
                    for attr in [&v, &w] {
                        let id = IndexId::Column(EntityType::Node, (label.clone(), attr.clone()));
                        loop {
                            let prepared = Arc::new(AtomicRefCell::new(
                                mvcc.read().borrow().new_build_version(),
                            ));
                            let Some(batch) = prepared.borrow_mut().indexer_prepare(&id, BATCH)
                            else {
                                break;
                            };
                            assert_eq!(
                                mvcc.commit_indexer_batch(&prepared, &batch),
                                Some(IndexerCommit::Same)
                            );
                        }
                    }
                    let what = format!(
                        "{name}; batches on {}, {}; order {order:?}, built",
                        batches_on[0], batches_on[1]
                    );
                    check_rules(&mvcc.read().borrow(), &label, [&v, &w], &what);
                }
            }
        }
    }
    CATCH_UP_BY.with(|by| by.set(None));
    ALLOW_BATCH_RACE.with(|allow| allow.set(false));
    for outcome in [
        "writer claimed",
        "writer over batches",
        "writer stale",
        "batch same",
        "batch swapped",
        "batch caught up, refreshing",
        "batch caught up, inserting",
        "batch dropped, gone",
        "batch dropped, another batch",
    ] {
        assert!(
            counts.get(outcome).copied().unwrap_or(0) > 0,
            "no order reached \"{outcome}\": {counts:?}"
        );
    }
}

/// Delete node `id` and the edges on it, in the order a commit does.
fn delete_node_with_edges(
    g: &mut Graph,
    id: u64,
) {
    delete_node(g, id);
    let del: RoaringTreemap = [id].into_iter().collect();
    applied!(g, |w| g.delete_implicit_edges(
        &del,
        &RoaringTreemap::new(),
        &mut w
    ))
    .expect("deleted");
}

/// `ty`-typed edges from each of `srcs` to the matching `dsts`, with `attr` = `first`,
/// `first + 1`, ...: their ids.
fn add_edges(
    g: &mut Graph,
    ty: &Arc<String>,
    attr: &Arc<String>,
    srcs: &[u64],
    dsts: &[u64],
    first: i64,
) -> Vec<u64> {
    g.open_id_batches().expect("a consistent space");
    let aid = g.get_or_create_rel_attr_id(attr);
    let eids: Vec<u64> = g
        .relationship_id_space()
        .reserve(srcs.len(), &RoaringTreemap::new())
        .expect("reserved");
    g.create_relationships_bulk(ty, srcs, dsts, &eids)
        .expect("freshly reserved ids are creatable");
    let values: FxHashMap<u64, Vec<(u16, Value)>> = eids
        .iter()
        .zip(first..)
        .map(|(&eid, w)| (eid, vec![(aid, Value::Int(w))]))
        .collect();
    applied!(g, |d| g.import_relationship_attrs(&values, &mut d));
    eids
}

/// Set edge `eid`'s `attr` to `value`, through the write path.
fn set_edge_attr(
    g: &mut Graph,
    eid: u64,
    attr: &Arc<String>,
    value: i64,
) {
    g.open_id_batches().expect("a consistent space");
    let aid = g.get_or_create_rel_attr_id(attr);
    let set: FxHashMap<u64, Vec<(u16, Value)>> = [(eid, vec![(aid, Value::Int(value))])]
        .into_iter()
        .collect();
    applied!(g, |d| g.set_relationships_attributes(&set, &mut d)).expect("set");
}

/// Column `key` of `entity` checked against `g`'s own data, by the rules on [`BuildState`]: it
/// holds exactly the current tuples of every `label` node, or `label`-typed edge, if `Ready`, or
/// of those `g` places below its cursor if `Building`, and nothing else. Reads `g` entity by
/// entity, not through the build's own scans. A column `g` does not have passes.
fn check_column(
    g: &Graph,
    entity: EntityType,
    (label, attr): &IndexKey,
    what: &str,
) {
    let Some(state) = g.falkordb_index.column_state(entity, label, attr) else {
        return;
    };
    let mut entities: Vec<(u64, Option<Value>)> = Vec::new();
    match entity {
        EntityType::Node => {
            let idx = g.get_node_attribute_id(attr);
            if let Some(m) = g.get_label_matrix(label) {
                for (n, _) in m.iter(0, u64::MAX) {
                    let value = idx
                        .and_then(|idx| g.get_node_attribute_by_idx(NodeId::from(n), idx as u16));
                    entities.push((n, value));
                }
            }
        }
        EntityType::Relationship => {
            let idx = g.get_relationship_attribute_id(attr);
            if let Some(t) = g.get_relationship_matrix(label) {
                for (_, _, eid) in t.iter(0, u64::MAX, false) {
                    let value = idx.and_then(|idx| {
                        g.get_relationship_attribute_by_idx(RelationshipId::from(eid), idx as u16)
                    });
                    entities.push((eid, value));
                }
            }
        }
    }
    let expected: BTreeSet<(u64, u64)> = entities
        .into_iter()
        .filter(|&(id, _)| match state {
            BuildState::Ready => true,
            BuildState::Building { next } => g
                .build_place(entity, id)
                .is_some_and(|at| at.precedes(next)),
        })
        .filter_map(|(id, value)| Some((encode_numeric(&value?)?, id)))
        .collect();
    let held: BTreeSet<(u64, u64)> = g
        .falkordb_index
        .numeric(entity, label, attr)
        .expect("a numeric column")
        .tuples()
        .into_iter()
        .collect();
    assert_eq!(held, expected, "{what}: {label}.{attr} ({state:?})");
}

/// Writers and indexer batches on real threads, on three columns at once (two node columns and an
/// edge column), with one indexer thread per column, as the builder runs one chain of batches per
/// column, and three writer threads that fork, write and claim, or write holding the slot:
/// re-values, creates, deletes (a node's edges with it), labels moved off, edges added and
/// re-valued, and now and then a column dropped and created again, `Building`. Until commits
/// publish by atomic swap, `MvccGraph` commits through `&mut self`, so the threads share it behind
/// a test lock: forks and reads take its read side, claims and commits its write side, and a batch
/// is read with no lock held. After every commit the published version keeps the rules on
/// [`BuildState`] against its own data ([`check_column`]), and every column ends `Ready`, as a
/// one-pass build of the final data. It runs until every way a commit can go was seen, so a run
/// in which the threads never raced fails.
#[test]
fn writers_and_indexer_batches_on_threads_keep_every_column_right() {
    use parking_lot::{Mutex, RwLock};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    const NODES: i64 = 400;
    const VALUES: i64 = 10_000;
    const BATCH: usize = 7;
    const SEEN: [&str; 9] = [
        "writer claimed",
        "writer over batches",
        "writer stale",
        "writer holding the slot",
        "batch same",
        "batch swapped",
        "batch caught up, refreshing",
        "batch caught up, inserting",
        "batch dropped, gone",
    ];
    let (v, w, x, ty) = (
        attr_named("v"),
        attr_named("w"),
        attr_named("x"),
        attr_named("R"),
    );
    let (mut g, label, ids) = people_with(&v, &(0..NODES).collect::<Vec<_>>());
    for (i, &id) in ids.iter().enumerate() {
        set_attr(&mut g, id, &w, Value::Int(1000 + i as i64));
    }
    add_edges(&mut g, &ty, &x, &ids[..ids.len() - 1], &ids[1..], 0);
    let columns: Vec<(EntityType, IndexKey)> = vec![
        (EntityType::Node, (Arc::clone(&label), Arc::clone(&v))),
        (EntityType::Node, (Arc::clone(&label), Arc::clone(&w))),
        (EntityType::Relationship, (Arc::clone(&ty), Arc::clone(&x))),
    ];
    for (entity, (l, a)) in &columns {
        g.falkordb_index.create_building(*entity, l, a);
    }
    let mvcc = RwLock::new(MvccGraph::from_graph(g));

    let counts: Mutex<BTreeMap<&str, usize>> = Mutex::new(BTreeMap::new());
    let count = |how: &'static str| *counts.lock().entry(how).or_default() += 1;
    let seen_all = || {
        let counts = counts.lock();
        SEEN.iter()
            .all(|how| counts.get(how).copied().unwrap_or(0) > 0)
    };
    let check = |what: &str| {
        let published = mvcc.read().read();
        let g = published.borrow();
        for (entity, key) in &columns {
            check_column(&g, *entity, key, what);
        }
    };
    // A random write on `g`, by `op`: re-create a column, delete a node and its edges, create a
    // node, move one off `:Person`, add an edge, re-value an edge or a node. Half the time a node
    // it picks is one of those just past a node column's cursor, which its next batches read.
    let random_write = |g: &mut Graph, op: u32, rng: &mut StdRng| {
        let mut people: Vec<u64> = g
            .get_label_matrix(&label)
            .map(|m| m.iter(0, u64::MAX).map(|(n, _)| n).collect())
            .unwrap_or_default();
        let attr = if rng.random_bool(0.5) { &v } else { &w };
        if rng.random_bool(0.5)
            && let Some(BuildState::Building {
                next: BuildCursor::Node(next),
            }) = g
                .falkordb_index
                .column_state(EntityType::Node, &label, attr)
        {
            let near: Vec<u64> = people
                .iter()
                .copied()
                .filter(|&id| id >= u64::from(next))
                .take(2 * BATCH)
                .collect();
            if !near.is_empty() {
                people = near;
            }
        }
        let edges: Vec<u64> = g
            .get_relationship_matrix(&ty)
            .map(|t| t.iter(0, u64::MAX, false).map(|(_, _, e)| e).collect())
            .unwrap_or_default();
        let value = rng.random_range(0..VALUES);
        match op {
            0..=3 => {
                let (entity, (l, a)) = &columns[rng.random_range(0..columns.len())];
                g.falkordb_index.drop_column(*entity, l, a);
                g.falkordb_index.create_building(*entity, l, a);
            }
            4..=11 if !people.is_empty() => {
                delete_node_with_edges(g, people[rng.random_range(0..people.len())]);
            }
            12..=19 => {
                let id = add_person(g, &v, value);
                set_attr(g, id, &w, Value::Int(rng.random_range(0..VALUES)));
            }
            20..=25 if !people.is_empty() => {
                relabel_to_other(g, people[rng.random_range(0..people.len())]);
            }
            26..=35 if people.len() >= 2 => {
                let src = people[rng.random_range(0..people.len())];
                let dst = people[rng.random_range(0..people.len())];
                add_edges(g, &ty, &x, &[src], &[dst], value);
            }
            36..=55 if !edges.is_empty() => {
                set_edge_attr(g, edges[rng.random_range(0..edges.len())], &x, value);
            }
            _ if !people.is_empty() => {
                set_attr(
                    g,
                    people[rng.random_range(0..people.len())],
                    attr,
                    Value::Int(value),
                );
            }
            _ => {}
        }
    };

    let writers_left = AtomicUsize::new(3);
    let started = Instant::now();
    thread::scope(|scope| {
        for (seed, (entity, key)) in (100..).zip(&columns) {
            let (mvcc, count, check, writers_left) = (&mvcc, &count, &check, &writers_left);
            scope.spawn(move || {
                let mut rng = StdRng::seed_from_u64(seed);
                let id = IndexId::Column(*entity, key.clone());
                loop {
                    // Before the batch reads: once no writer is left, a column that is not
                    // `Building` stays so.
                    let done = writers_left.load(Ordering::Acquire) == 0;
                    let prepared = Arc::new(AtomicRefCell::new(
                        mvcc.read().read().borrow().new_build_version(),
                    ));
                    let Some(batch) = prepared.borrow_mut().indexer_prepare(&id, BATCH) else {
                        if done {
                            break;
                        }
                        thread::yield_now();
                        continue;
                    };
                    // A batch's read takes milliseconds at full size; this one's, microseconds.
                    // The wait gives writers the time to commit while it is taken.
                    thread::sleep(Duration::from_micros(rng.random_range(0..200)));
                    let how = mvcc
                        .write()
                        .commit_indexer_batch(&prepared, &batch)
                        .expect("the slot is free under the lock");
                    count(match how {
                        IndexerCommit::Same => "batch same",
                        IndexerCommit::Swap => "batch swapped",
                        IndexerCommit::CatchUp(CatchUp::Refresh) => "batch caught up, refreshing",
                        IndexerCommit::CatchUp(CatchUp::Insert) => "batch caught up, inserting",
                        IndexerCommit::Dropped(Dropped::Gone) => "batch dropped, gone",
                        IndexerCommit::Dropped(Dropped::AnotherBatch) => {
                            "batch dropped, another batch"
                        }
                    });
                    check(&format!("a batch on {}.{} ({how:?})", key.0, key.1));
                }
            });
        }
        for seed in 0..3 {
            let (mvcc, count, check, writers_left, random_write, seen_all) = (
                &mvcc,
                &count,
                &check,
                &writers_left,
                &random_write,
                &seen_all,
            );
            scope.spawn(move || {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut writes = 0;
                while writes < 150 || (!seen_all() && started.elapsed() < Duration::from_secs(30)) {
                    writes += 1;
                    let op = rng.random_range(0..100);
                    if rng.random_bool(0.3) {
                        let mvcc = mvcc.write();
                        let version = mvcc.write().expect("the slot is free under the lock");
                        random_write(&mut version.borrow_mut(), op, &mut rng);
                        mvcc.commit(version).expect("the writer commits");
                        drop(mvcc);
                        count("writer holding the slot");
                        check("a write holding the slot");
                        continue;
                    }
                    loop {
                        let (base, fork) = {
                            let mvcc = mvcc.read();
                            (mvcc.read(), mvcc.fork())
                        };
                        random_write(&mut fork.borrow_mut(), op, &mut rng);
                        // As a batch's, so batches commit while a fork waits to claim.
                        thread::sleep(Duration::from_micros(rng.random_range(0..200)));
                        let mvcc = mvcc.write();
                        let over_batches = !Arc::ptr_eq(&mvcc.read(), &base);
                        match mvcc.claim(&fork) {
                            Ok(()) => {
                                mvcc.commit(fork).expect("the writer commits");
                                drop(mvcc);
                                count(if over_batches {
                                    "writer over batches"
                                } else {
                                    "writer claimed"
                                });
                                check("a claimed write");
                                break;
                            }
                            Err(Claim::Stale) => count("writer stale"),
                            Err(e) => panic!("{e:?}"),
                        }
                    }
                }
                writers_left.fetch_sub(1, Ordering::Release);
            });
        }
    });

    let counts = counts.into_inner();
    for how in SEEN {
        assert!(
            counts.get(how).copied().unwrap_or(0) > 0,
            "never seen: {how}; {counts:?} in {:?}",
            started.elapsed()
        );
    }
    let published = mvcc.read().read();
    let g = published.borrow();
    for (entity, key) in &columns {
        check_column(&g, *entity, key, "built");
        assert_eq!(
            g.falkordb_index.column_state(*entity, &key.0, &key.1),
            Some(BuildState::Ready),
            "{}.{} built",
            key.0,
            key.1
        );
        assert_eq!(
            contents(&g, *entity, &key.0, &key.1, &[]),
            one_pass(&g, *entity, &key.0, &key.1, &[]),
            "{}.{} against a one-pass build",
            key.0,
            key.1
        );
    }
}
