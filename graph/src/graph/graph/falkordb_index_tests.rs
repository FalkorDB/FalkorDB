//! The native index as the graph maintains it: each test drives a real `Graph` write path
//! and checks the index column it should have changed.

use super::Graph;
use crate::entity_type::EntityType;
use crate::index::IndexType;
use crate::index::falkordb::graph_writes;
use crate::runtime::value::Value;
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;
use std::sync::Arc;

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

/// CREATE INDEX bulk-builds the column from real graph state: three
/// `:Person {age}` nodes, then a range scan returns the right ids in
/// `(value, id)` order. Drives the native half of `Graph::create_index`
/// (`graph_writes::create_index` and its `get_label_matrix` →
/// `get_node_attribute_by_idx` reads) directly, bypassing the RediSearch FFI
/// (uninitialised in a unit test).
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
    g.set_nodes_labels_bulk(&ids, &label_cols, &mut FxHashMap::default(), false);

    // age = 10, 20, 30.
    let aid = g.get_or_create_node_attr_id(&attr);
    let mut attrs: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
    for (i, &id) in ids.iter().enumerate() {
        attrs.insert(id, vec![(aid, Value::Int(((i + 1) * 10) as i64))]);
    }
    g.set_nodes_attributes(&attrs, &mut FxHashMap::default())
        .unwrap();

    // Build the index numeric column from the live nodes.
    graph_writes::create_index(
        &mut g,
        &IndexType::Range,
        EntityType::Node,
        &label,
        std::slice::from_ref(&attr),
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
    g.set_nodes_labels_bulk(&ids, &cols, &mut FxHashMap::default(), false);
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
    g.set_nodes_attributes(&attrs, &mut FxHashMap::default())
        .unwrap();
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
    g.delete_nodes(&del, &mut FxHashMap::default()).unwrap();
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
    g.delete_nodes(&del, &mut FxHashMap::default()).unwrap();
    assert!(range_scan(&g, &label, &attr, 4, 6).is_empty());

    // Reclaim the freed id for a fresh node with the same value.
    let reused: Vec<u64> = reserve_nodes(&mut g, 1);
    assert_eq!(reused[0], ids[0], "the deleted id should be reclaimed");
    let mut set = RoaringTreemap::new();
    set.insert(reused[0]);
    g.create_nodes(&set)
        .expect("freshly reserved ids are creatable");
    let lid = g.get_label_id_mut("Person");
    g.set_nodes_labels_bulk(&reused, &[lid.0 as u64], &mut FxHashMap::default(), false);
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
    g.set_nodes_labels_bulk(&[id], &[lid.0 as u64], &mut FxHashMap::default(), false);
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
    g.remove_nodes_labels(&[ids[0]], &[lid.0 as u64], &mut FxHashMap::default());
    assert!(
        range_scan(&g, &label, &attr, 4, 6).is_empty(),
        "REMOVE :Person must drop the tuple"
    );

    // Delete the node and reuse its id for a fresh :Person with a DIFFERENT value.
    let mut del = RoaringTreemap::new();
    del.insert(ids[0]);
    g.delete_nodes(&del, &mut FxHashMap::default()).unwrap();
    let reused: Vec<u64> = reserve_nodes(&mut g, 1);
    assert_eq!(reused[0], ids[0]);
    let mut set = RoaringTreemap::new();
    set.insert(reused[0]);
    g.create_nodes(&set)
        .expect("freshly reserved ids are creatable");
    g.set_nodes_labels_bulk(&reused, &[lid.0 as u64], &mut FxHashMap::default(), false);
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
    g.import_relationship_attrs(&attrs, &mut FxHashMap::default());

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
    g.set_relationships_attributes(&update, &mut FxHashMap::default())
        .unwrap();
    assert!(scan(&g, 20, 20).is_empty(), "the old value is gone");
    assert_eq!(scan(&g, 25, 25), vec![ids[1]]);

    // Explicit delete.
    let deleted: RoaringTreemap = std::iter::once(ids[0]).collect();
    g.delete_relationships(&deleted, &mut FxHashMap::default())
        .unwrap();
    assert_eq!(scan(&g, 0, 100), vec![ids[1], ids[2]]);

    // Deleting `a` cascades to its two remaining edges.
    let deleted_nodes: RoaringTreemap = std::iter::once(a).collect();
    g.open_id_batches().expect("a consistent space");
    g.delete_nodes(&deleted_nodes, &mut FxHashMap::default())
        .unwrap();
    g.delete_implicit_edges(
        &deleted_nodes,
        &RoaringTreemap::new(),
        &mut FxHashMap::default(),
    )
    .unwrap();
    assert!(
        scan(&g, 0, 100).is_empty(),
        "the cascade removes what the explicit delete did not"
    );
}
