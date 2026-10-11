//! The native index as the graph maintains it: each test drives a real `Graph` write path
//! and checks the index column it should have changed.

use super::Graph;
use crate::entity_type::EntityType;
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

/// Start an online build the way the module's controller does: read the column's epoch from
/// the work list, then BASE from the graph as it stands now (the job's snapshot).
fn begin_build(
    g: &Graph,
    entity: EntityType,
    label: &Arc<String>,
    attr: &Arc<String>,
) -> (u64, Vec<(u64, u64)>) {
    let epoch = graph_writes::building_columns(g)
        .into_iter()
        .find(|(e, l, a, _)| *e == entity && l == label && a == attr)
        .map(|(.., epoch)| epoch)
        .expect("the column is Building");
    let base = graph_writes::base_encoded(g, entity, label, attr).expect("the column exists");
    (epoch, base)
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

/// CREATE INDEX builds the column from real graph state: three `:Person {age}`
/// nodes, then a range scan returns the right ids in `(value, id)` order. The
/// column starts `Building` and serves no reads; the background job's two steps
/// (`base_encoded` from a snapshot, then `install_base`) fill it from the
/// `get_label_matrix` → `get_node_attribute_by_idx` reads and flip it `Ready`.
/// Drives the native half of `Graph::create_index` directly, bypassing the
/// RediSearch FFI (uninitialised in a unit test).
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

    // CREATE INDEX marks the column `Building` and scans nothing.
    graph_writes::create_index(
        &mut g,
        &IndexType::Range,
        EntityType::Node,
        &label,
        std::slice::from_ref(&attr),
    );
    let (epoch, base) = begin_build(&g, EntityType::Node, &label, &attr);
    assert!(
        g.falkordb_index
            .query_numeric(EntityType::Node, &label, &eq(&attr, 20))
            .is_none(),
        "a Building column serves no reads"
    );

    // The background job's install builds the column from the live nodes and flips it `Ready`.
    assert!(graph_writes::install_base(
        &mut g,
        EntityType::Node,
        &label,
        &attr,
        epoch,
        base
    ));
    assert!(graph_writes::building_columns(&g).is_empty());

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

// --- Online (background) build ---

/// Online-build reconciliation: CREATE INDEX marks the column `Building` and the job reads BASE,
/// then a DELETE and a SET land while the build is pending, then BASE installs. TOMB, filled by
/// the write hooks' removes, must keep the deleted and re-valued nodes' stale BASE entries out
/// (no resurrection), while DELTA carries the new value. Drives the real `delete_nodes` and
/// `set_nodes_attributes` hooks and the real install.
#[test]
fn online_build_install_validates_against_concurrent_writes() {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let label = Arc::new("Person".to_string());
    let attr = Arc::new("v".to_string());
    let lid = g.get_label_id_mut("Person");
    let ids: Vec<u64> = reserve_nodes(&mut g, 6);
    g.create_nodes(&ids.iter().copied().collect())
        .expect("freshly reserved ids are creatable");
    let cols = vec![lid.0 as u64; ids.len()];
    applied!(g, |w| g.set_nodes_labels_bulk(&ids, &cols, &mut w, false));
    // 10, 20, 30, 40, 50, 60, set before the index exists, so BASE is their only source.
    for (i, &id) in ids.iter().enumerate() {
        set_attr(&mut g, id, &attr, Value::Int(((i as i64) + 1) * 10));
    }

    graph_writes::create_index(
        &mut g,
        &IndexType::Range,
        EntityType::Node,
        &label,
        std::slice::from_ref(&attr),
    );
    let (epoch, base) = begin_build(&g, EntityType::Node, &label, &attr);
    assert_eq!(base.len(), 6);

    // Writes that land during the build: delete id1 (20) and id3 (40), update id4 50 -> 55.
    let mut del = RoaringTreemap::new();
    del.insert(ids[1]);
    del.insert(ids[3]);
    applied!(g, |w| g.delete_nodes(&del, &mut w)).unwrap();
    set_attr(&mut g, ids[4], &attr, Value::Int(55));

    // One install: TOMB is subtracted from the stale BASE before DELTA is replayed.
    assert!(graph_writes::install_base(
        &mut g,
        EntityType::Node,
        &label,
        &attr,
        epoch,
        base
    ));

    // Survivors, value-ordered: id0 (10), id2 (30), id4 (55), id5 (60).
    let all: Vec<u64> = g
        .falkordb_index
        .numeric(EntityType::Node, &label, &attr)
        .unwrap()
        .range(None, None, true, true)
        .collect();
    assert_eq!(all, vec![ids[0], ids[2], ids[4], ids[5]]);
    assert!(
        range_scan(&g, &label, &attr, 20, 20).is_empty(),
        "deleted id1 must not resurrect"
    );
    assert!(
        range_scan(&g, &label, &attr, 40, 40).is_empty(),
        "deleted id3 must not resurrect"
    );
    assert!(
        range_scan(&g, &label, &attr, 50, 50).is_empty(),
        "updated-away value 50 must not resurrect"
    );
    assert_eq!(
        range_scan(&g, &label, &attr, 55, 55),
        vec![ids[4]],
        "id4 has its new value"
    );
}

/// The edge mirror of the node test above: a DELETE and a SET on edges land while the build is
/// pending, and the install must keep the deleted and re-valued edges' stale BASE entries out.
/// Drives the real `delete_relationships` and `set_relationships_attributes` hooks.
#[test]
fn online_build_edge_install_validates_against_concurrent_writes() {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let ty = Arc::new("R".to_string());
    let attr = Arc::new("w".to_string());
    let aid = g.get_or_create_rel_attr_id(&attr);

    // Six a->b edges with w = 10..60, set before the index exists, so BASE is their only source.
    let nodes = reserve_nodes(&mut g, 2);
    g.create_nodes(&nodes.iter().copied().collect())
        .expect("freshly reserved ids are creatable");
    let (a, b) = (nodes[0], nodes[1]);
    let ids: Vec<u64> = g
        .relationship_id_space()
        .reserve(6, &RoaringTreemap::new())
        .expect("reserved");
    g.create_relationships_bulk(&ty, &[a; 6], &[b; 6], &ids)
        .expect("freshly reserved ids are creatable");
    let mut attrs: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
    for (i, &id) in ids.iter().enumerate() {
        attrs.insert(id, vec![(aid, Value::Int(((i as i64) + 1) * 10))]);
    }
    applied!(g, |w| g.import_relationship_attrs(&attrs, &mut w));

    graph_writes::create_index(
        &mut g,
        &IndexType::Range,
        EntityType::Relationship,
        &ty,
        std::slice::from_ref(&attr),
    );
    let (epoch, base) = begin_build(&g, EntityType::Relationship, &ty, &attr);
    assert_eq!(base.len(), 6);

    // Writes that land during the build: delete id1 (20) and id3 (40), update id4 50 -> 55.
    let mut del = RoaringTreemap::new();
    del.insert(ids[1]);
    del.insert(ids[3]);
    applied!(g, |w| g.delete_relationships(&del, &mut w)).unwrap();
    let mut upd: FxHashMap<u64, Vec<(u16, Value)>> = FxHashMap::default();
    upd.insert(ids[4], vec![(aid, Value::Int(55))]);
    applied!(g, |w| g.set_relationships_attributes(&upd, &mut w)).unwrap();

    assert!(graph_writes::install_base(
        &mut g,
        EntityType::Relationship,
        &ty,
        &attr,
        epoch,
        base
    ));

    // Survivors, value-ordered: id0 (10), id2 (30), id4 (55), id5 (60).
    let point = |v: i64| -> Vec<u64> {
        g.falkordb_index
            .numeric(EntityType::Relationship, &ty, &attr)
            .unwrap()
            .range(Some(&Value::Int(v)), Some(&Value::Int(v)), true, true)
            .collect()
    };
    let all: Vec<u64> = g
        .falkordb_index
        .numeric(EntityType::Relationship, &ty, &attr)
        .unwrap()
        .range(None, None, true, true)
        .collect();
    assert_eq!(all, vec![ids[0], ids[2], ids[4], ids[5]]);
    assert!(point(20).is_empty(), "deleted edge id1 must not resurrect");
    assert!(point(40).is_empty(), "deleted edge id3 must not resurrect");
    assert!(
        point(50).is_empty(),
        "updated-away value 50 must not resurrect"
    );
    assert_eq!(point(55), vec![ids[4]], "edge id4 has its new value");
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

/// The same unit of work against a `Building` column: the batch must still feed TOMB, so the
/// install drops the deleted node's BASE entry, and must not leave the SET's value in DELTA.
#[test]
fn set_then_delete_in_one_batch_during_a_build() {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let label = Arc::new("Person".to_string());
    let attr = Arc::new("v".to_string());
    let lid = g.get_label_id_mut("Person");
    let ids: Vec<u64> = reserve_nodes(&mut g, 2);
    g.create_nodes(&ids.iter().copied().collect())
        .expect("freshly reserved ids are creatable");
    applied!(g, |w| g.set_nodes_labels_bulk(
        &ids,
        &[lid.0 as u64; 2],
        &mut w,
        false
    ));
    set_attr(&mut g, ids[0], &attr, Value::Int(1));
    set_attr(&mut g, ids[1], &attr, Value::Int(2));

    graph_writes::create_index(
        &mut g,
        &IndexType::Range,
        EntityType::Node,
        &label,
        std::slice::from_ref(&attr),
    );
    let (epoch, base) = begin_build(&g, EntityType::Node, &label, &attr);

    let aid = g.get_or_create_node_attr_id(&attr);
    let mut attrs = FxHashMap::default();
    attrs.insert(ids[0], vec![(aid, Value::Int(5))]);
    let mut writes = IndexDocs::default();
    g.set_nodes_attributes(&attrs, &mut writes).unwrap();
    let mut del = RoaringTreemap::new();
    del.insert(ids[0]);
    g.delete_nodes(&del, &mut writes).unwrap();
    g.apply_index_writes(&mut writes);

    assert!(graph_writes::install_base(
        &mut g,
        EntityType::Node,
        &label,
        &attr,
        epoch,
        base
    ));
    assert_eq!(
        range_scan(&g, &label, &attr, 0, 10),
        vec![ids[1]],
        "neither the deleted node's BASE value nor its SET value survives the install"
    );
}

// --- Build identity ---

/// A graph deleted and re-created under the same name builds the same column under a different
/// epoch. The build registry keys in-flight builds by graph name, column and epoch, so with a
/// per-graph counter the new graph's first build had the old graph's key and waited for the old
/// job to finish before it could be dispatched.
#[test]
fn a_recreated_graph_never_reuses_a_build_epoch() {
    ensure_graphblas();
    let label = Arc::new("Person".to_string());
    let attr = Arc::new("v".to_string());
    let epoch_of_first_build = || {
        let mut g = Graph::new(64, 64, 10, 1, "same-name");
        graph_writes::create_index(
            &mut g,
            &IndexType::Range,
            EntityType::Node,
            &label,
            std::slice::from_ref(&attr),
        );
        let builds = graph_writes::building_columns(&g);
        assert_eq!(builds.len(), 1);
        builds[0].3
    };
    let old = epoch_of_first_build();
    let new = epoch_of_first_build();
    assert_ne!(
        old, new,
        "the re-created graph's build must not share the old one's epoch"
    );
}

// --- DROP and CREATE INDEX while a column is building ---

/// `:Person` nodes carrying `attr` with `values`, set before any index exists, so a build's BASE
/// is their only source.
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

/// CREATE INDEX on `(:Person, attr)`, the native half.
fn create_index_on(
    g: &mut Graph,
    label: &Arc<String>,
    attr: &Arc<String>,
) {
    graph_writes::create_index(
        g,
        &IndexType::Range,
        EntityType::Node,
        label,
        std::slice::from_ref(attr),
    );
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

/// DROP INDEX while the column is building takes everything already indexed with it — DELTA and
/// TOMB live in the column — and leaves the job nothing to install into. BASE lives only in the
/// job, which drops it when its install comes back empty-handed. Writes after the drop stage
/// nothing.
#[test]
fn dropping_a_building_column_leaves_nothing_behind() {
    let attr = Arc::new("v".to_string());
    let (mut g, label, ids) = people_with(&attr, &[10, 20]);
    create_index_on(&mut g, &label, &attr);
    let (epoch, base) = begin_build(&g, EntityType::Node, &label, &attr);
    assert_eq!(base.len(), 2);
    // A write during the build lands in DELTA (30) and TOMB (10).
    set_attr(&mut g, ids[0], &attr, Value::Int(30));

    drop_index_on(&mut g, &label, &attr);
    assert!(graph_writes::building_columns(&g).is_empty());
    assert!(
        g.falkordb_index
            .numeric(EntityType::Node, &label, &attr)
            .is_none()
    );

    // The running job's install finds no column, installs nothing and tells the caller not to
    // commit; a job dispatched after the drop finds nothing to scan.
    assert!(!graph_writes::install_base(
        &mut g,
        EntityType::Node,
        &label,
        &attr,
        epoch,
        base
    ));
    assert!(
        g.falkordb_index
            .numeric(EntityType::Node, &label, &attr)
            .is_none()
    );
    assert!(graph_writes::base_encoded(&g, EntityType::Node, &label, &attr).is_none());

    // Indexing stopped: a later write to the attribute stages nothing.
    let aid = g.get_or_create_node_attr_id(&attr);
    let mut attrs = FxHashMap::default();
    attrs.insert(ids[1], vec![(aid, Value::Int(40))]);
    let mut writes = IndexDocs::default();
    g.set_nodes_attributes(&attrs, &mut writes).unwrap();
    assert!(writes.native.is_empty(), "no column, so nothing to index");
}

/// DROP then CREATE INDEX on the same column while the first build is still running: the new
/// column has a new epoch, so the old job cannot install its stale BASE into it, and the new
/// build installs the current values.
#[test]
fn recreating_a_building_column_ignores_the_old_build() {
    let attr = Arc::new("v".to_string());
    let (mut g, label, ids) = people_with(&attr, &[10, 20]);
    create_index_on(&mut g, &label, &attr);
    let (old_epoch, old_base) = begin_build(&g, EntityType::Node, &label, &attr);

    drop_index_on(&mut g, &label, &attr);
    set_attr(&mut g, ids[0], &attr, Value::Int(15)); // while no index exists
    create_index_on(&mut g, &label, &attr);
    let (new_epoch, new_base) = begin_build(&g, EntityType::Node, &label, &attr);
    assert_ne!(old_epoch, new_epoch);

    assert!(
        !graph_writes::install_base(&mut g, EntityType::Node, &label, &attr, old_epoch, old_base),
        "the old job must not install into the re-created column"
    );
    assert_eq!(
        graph_writes::building_columns(&g).len(),
        1,
        "still Building, waiting for its own job"
    );

    assert!(graph_writes::install_base(
        &mut g,
        EntityType::Node,
        &label,
        &attr,
        new_epoch,
        new_base
    ));
    assert_eq!(range_scan(&g, &label, &attr, 10, 10), Vec::<u64>::new());
    assert_eq!(range_scan(&g, &label, &attr, 15, 15), vec![ids[0]]);
    assert_eq!(range_scan(&g, &label, &attr, 20, 20), vec![ids[1]]);
}

/// CREATE INDEX on a second attribute of the same label while the first is building adds a
/// column of its own and leaves the first build alone: same epoch, still installable, and each
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
    let (v_epoch, v_base) = begin_build(&g, EntityType::Node, &label, &v);

    create_index_on(&mut g, &label, &w);
    let (w_epoch, w_base) = begin_build(&g, EntityType::Node, &label, &w);
    assert_eq!(
        begin_build(&g, EntityType::Node, &label, &v).0,
        v_epoch,
        "the first column's build is untouched"
    );
    assert_ne!(v_epoch, w_epoch);

    assert!(graph_writes::install_base(
        &mut g,
        EntityType::Node,
        &label,
        &v,
        v_epoch,
        v_base
    ));
    assert!(graph_writes::install_base(
        &mut g,
        EntityType::Node,
        &label,
        &w,
        w_epoch,
        w_base
    ));
    assert!(graph_writes::building_columns(&g).is_empty());
    assert_eq!(range_scan(&g, &label, &v, 0, 100), vec![ids[0], ids[1]]);
    assert_eq!(range_scan(&g, &label, &w, 2, 2), vec![ids[1]]);
}

// --- Background BASE scan of a published version ---

/// A published-looking graph: `Person` nodes 0..10 with `v = 10 * id`, node 4 deleted (a gap in
/// the rows), and `R` edges with several per source node, so chunk boundaries fall inside and
/// between rows.
fn scan_fixture() -> (Graph, Arc<String>, Arc<String>, Arc<String>, Arc<String>) {
    let v = Arc::new("v".to_string());
    let (mut g, label, ids) = people_with(&v, &(0..10).map(|i| i * 10).collect::<Vec<_>>());
    let ty = Arc::new("R".to_string());
    let w = Arc::new("w".to_string());
    let aid = g.get_or_create_rel_attr_id(&w);
    // Three edges out of node 1, two out of node 7, one out of node 8.
    let (srcs, dsts) = ([1u64, 1, 1, 7, 7, 8], [2u64, 3, 9, 8, 9, 9]);
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
        graph_writes::create_index(
            &mut g,
            &IndexType::Range,
            entity,
            l,
            std::slice::from_ref(a),
        );
    }
    (g, label, v, ty, w)
}

/// The chunked scan returns exactly what the one-pass scan returns, whatever the chunk size:
/// nothing dropped or doubled at a chunk boundary, for nodes and for edges.
#[test]
fn chunked_base_scan_matches_the_one_pass_scan() {
    let (g, label, v, ty, w) = scan_fixture();
    let whole_nodes = graph_writes::base_encoded(&g, EntityType::Node, &label, &v).unwrap();
    let whole_edges = graph_writes::base_encoded(&g, EntityType::Relationship, &ty, &w).unwrap();
    assert_eq!(whole_nodes.len(), 9, "ten nodes, one deleted");
    assert_eq!(whole_edges.len(), 6);
    let snapshot = atomic_refcell::AtomicRefCell::new(g);
    for chunk_rows in [1u64, 2, 3, 7, 64, graph_writes::BASE_SCAN_CHUNK_ROWS] {
        let nodes = graph_writes::base_encoded_in_chunks(
            &snapshot,
            EntityType::Node,
            &label,
            &v,
            chunk_rows,
            || (),
        );
        let edges = graph_writes::base_encoded_in_chunks(
            &snapshot,
            EntityType::Relationship,
            &ty,
            &w,
            chunk_rows,
            || (),
        );
        assert_eq!(
            nodes.as_ref(),
            Some(&whole_nodes),
            "nodes, {chunk_rows} rows per chunk"
        );
        assert_eq!(
            edges.as_ref(),
            Some(&whole_edges),
            "edges, {chunk_rows} rows per chunk"
        );
    }
}

/// Each chunk borrows the snapshot only while the guard `hold` returned is alive: when the guard
/// is dropped, nothing borrows the snapshot any more. In production the guard is the graph's
/// read lock, which is what keeps the scan from colliding with the write path's mutable borrow
/// of a published version.
#[test]
fn chunked_base_scan_borrows_only_under_its_guard() {
    struct Guard<'a> {
        snapshot: &'a atomic_refcell::AtomicRefCell<Graph>,
        released: &'a std::cell::Cell<usize>,
    }
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            assert!(
                self.snapshot.try_borrow_mut().is_ok(),
                "the snapshot is still borrowed when the guard is released"
            );
            self.released.set(self.released.get() + 1);
        }
    }
    let (g, label, v, ..) = scan_fixture();
    let snapshot = atomic_refcell::AtomicRefCell::new(g);
    let released = std::cell::Cell::new(0);
    let base =
        graph_writes::base_encoded_in_chunks(&snapshot, EntityType::Node, &label, &v, 3, || {
            assert!(
                snapshot.try_borrow_mut().is_ok(),
                "the snapshot is borrowed before the guard is taken"
            );
            Guard {
                snapshot: &snapshot,
                released: &released,
            }
        });
    assert_eq!(base.map(|b| b.len()), Some(9));
    assert!(
        released.get() >= 4,
        "ten rows in chunks of three take four holds"
    );
}
