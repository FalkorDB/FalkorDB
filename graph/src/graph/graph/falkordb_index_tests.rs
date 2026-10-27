//! The native index as the graph maintains it: each test drives a real `Graph` write path
//! and checks the index column it should have changed. The build's tests drive indexer batches
//! between writes.

use super::{Graph, NodeId};
use crate::entity_type::EntityType;
use crate::index::falkordb::falkordb_index::{BuildCursor, ColumnState};
use crate::index::falkordb::graph_writes;
use crate::index::index_docs::IndexDocs;
use crate::index::{IndexQuery, IndexType};
use crate::runtime::value::Value;
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;
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
        Some(ColumnState::Ready)
    );
    assert!(
        g.falkordb_index
            .query_numeric(EntityType::Node, &label, &eq(&attr, 20))
            .is_some(),
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
        ColumnState::Building { next } => Some(next),
        ColumnState::Ready => None,
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

/// What a one-pass build of `g` as it stands now gives column `(entity, label, attr)`: the column
/// rebuilt on a copy of `g` by the RDB-load populate, which collects every value and bulk-builds,
/// and shares no code with the indexer batches.
fn one_pass(
    g: &Graph,
    entity: EntityType,
    label: &Arc<String>,
    attr: &Arc<String>,
    values: &[i64],
) -> (Vec<(i64, Vec<u64>)>, Vec<u64>) {
    let mut copy = g.new_version();
    graph_writes::build_column(&mut copy, entity, label, attr);
    contents(&copy, entity, label, attr, values)
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

/// A node batch reads at most its size: nine nodes take exactly `ceil(9 / batch)` batches, the last
/// of which flips the column `Ready`.
#[test]
fn a_node_batch_reads_at_most_its_size() {
    let attr = Arc::new("v".to_string());
    for (batch, batches) in [(1usize, 9usize), (2, 5), (4, 3), (9, 1), (10, 1)] {
        let (mut g, label, _) = people_with(&attr, &(1..=9).collect::<Vec<_>>());
        create_index_on(&mut g, &label, &attr);
        assert_eq!(build_to_end(&mut g, batch), batches, "batch {batch}");
        assert_eq!(range_scan(&g, &label, &attr, 0, 100).len(), 9);
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
