//! The native index as the graph maintains it: each test drives a real `Graph` write path
//! and checks the index column it should have changed. The online build's tests drive build
//! steps between writes, and its turns through `MvccGraph` against writers that fork and claim.

use super::{Graph, NodeId};
use crate::entity_type::EntityType;
use crate::graph::mvcc_graph::{BuildCommit, Claim, MvccGraph};
use crate::index::falkordb::falkordb_index::{BuildCursor, ColumnState};
use crate::index::falkordb::graph_writes;
use crate::index::falkordb::online_build::{self, BuildHost, Turn};
use crate::index::index_docs::IndexDocs;
use crate::index::{IndexQuery, IndexType};
use crate::runtime::value::Value;
use atomic_refcell::AtomicRefCell;
use parking_lot::RwLock;
use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

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

/// Run build steps of at most `entities` nodes or edges on `g` itself until no column is
/// `Building`. Returns how many steps it took.
fn build_to_end(
    g: &mut Graph,
    entities: usize,
) -> usize {
    let mut steps = 0;
    while g.falkordb_index.has_building_columns() {
        graph_writes::build_step(g, entities);
        steps += 1;
        assert!(steps < 100_000, "the build never finishes");
    }
    steps
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
/// column starts `Building` and serves no reads; a build step fills it from the
/// `get_label_matrix` → `get_node_attribute_by_idx` reads and flips it `Ready`.
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
    assert!(g.falkordb_index.has_building_columns());
    assert!(
        g.falkordb_index
            .query_numeric(EntityType::Node, &label, &eq(&attr, 20))
            .is_none(),
        "a Building column serves no reads"
    );

    // Three nodes fit one step, which indexes them and flips the column `Ready`.
    assert!(!graph_writes::build_step(
        &mut g,
        graph_writes::BUILD_STEP_ENTITIES
    ));
    assert!(!g.falkordb_index.has_building_columns());
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

// --- Online (background) build ---

/// `:Person` nodes carrying `attr` with `values`, set before any index exists, so build steps are
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
/// re-created on a copy of `g` and built in one step with no batch limit.
fn one_pass(
    g: &Graph,
    entity: EntityType,
    label: &Arc<String>,
    attr: &Arc<String>,
    values: &[i64],
) -> (Vec<(i64, Vec<u64>)>, Vec<u64>) {
    let mut copy = g.new_version();
    let attrs = std::slice::from_ref(attr);
    copy.falkordb_index
        .drop_index(&IndexType::Range, entity, label, attrs);
    graph_writes::create_index(&mut copy, &IndexType::Range, entity, label, attrs);
    build_to_end(&mut copy, usize::MAX);
    contents(&copy, entity, label, attr, values)
}

/// A build in small steps, with writes landing between the steps both on nodes the steps already
/// indexed and on nodes they have not reached, ends with exactly the column a one-pass build of the
/// final graph has: nothing deleted or re-valued survives, and nothing is missed. Writes between
/// steps need only the write path: a step reads current values, and skips the entities written
/// since the last step, whose tuples the writes already set.
#[test]
fn stepped_build_matches_a_one_pass_build_under_concurrent_node_writes() {
    let attr = Arc::new("v".to_string());
    let (mut g, label, ids) = people_with(&attr, &[10, 20, 30, 40, 50, 60]);
    create_index_on(&mut g, &label, &attr);

    // The first step indexes ids[0] and ids[1], and stops at ids[2].
    assert!(graph_writes::build_step(&mut g, 2));
    assert_eq!(
        cursor(&g, EntityType::Node, &label, &attr),
        Some(BuildCursor::Node(NodeId::from(ids[2])))
    );
    assert_eq!(range_scan(&g, &label, &attr, 0, 100), vec![ids[0], ids[1]]);

    // Writes between steps. Behind the cursor, ids[1] is deleted and ids[0] goes 10 -> 15. Ahead
    // of it, ids[3] is deleted and ids[4] goes 50 -> 55. And a new node gets 70.
    let mut del = RoaringTreemap::new();
    del.insert(ids[1]);
    del.insert(ids[3]);
    applied!(g, |w| g.delete_nodes(&del, &mut w)).unwrap();
    set_attr(&mut g, ids[0], &attr, Value::Int(15));
    set_attr(&mut g, ids[4], &attr, Value::Int(55));
    let created = add_person(&mut g, &attr, 70);

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

/// The edge counterpart: steps stop inside a source's edges, finishing the multi-edge pair they stop
/// in, and writes land between them behind and ahead of the cursor — including a new edge of the
/// source being resumed, at a destination the steps have already passed, which only the write path
/// indexes.
#[test]
fn stepped_build_matches_a_one_pass_build_under_concurrent_edge_writes() {
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

    graph_writes::create_index(
        &mut g,
        &IndexType::Range,
        EntityType::Relationship,
        &ty,
        std::slice::from_ref(&attr),
    );

    // A two-edge step stops inside the a->b pair, so it finishes the pair: three edges, and the
    // next step resumes at a->c.
    assert!(graph_writes::build_step(&mut g, 2));
    assert_eq!(
        cursor(&g, EntityType::Relationship, &ty, &attr),
        Some(BuildCursor::Edge {
            src: NodeId::from(a),
            dst: NodeId::from(c)
        })
    );
    let held = |g: &Graph| contents(g, EntityType::Relationship, &ty, &attr, &[]).1;
    assert_eq!(held(&g), eids[..3].to_vec());

    // Writes between steps: behind the cursor, delete the second a->b edge and add a fourth a->b
    // edge with w = 8 (a destination the steps have passed); ahead of it, a->d goes 5 -> 55 and
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

/// `Person` nodes 0..10 with `v = 10 * id`, node 4 deleted (a gap in the rows), and `R` edges with
/// several per source node and one multi-edge pair, so steps stop inside and between rows and
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

/// Stepping with any batch size ends with the column a one-pass build has: nothing dropped or
/// doubled where a step stops, for nodes (with a gap in the ids) and for edges (with steps stopping
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
            "nodes, {batch} per step"
        );
        assert_eq!(
            contents(&g, EntityType::Relationship, &ty, &w, &edge_values),
            edges,
            "edges, {batch} per step"
        );
    }
}

/// A node step reads at most its batch: nine nodes take exactly `ceil(9 / batch)` steps, the last
/// of which flips the column `Ready`.
#[test]
fn a_node_step_reads_at_most_its_batch() {
    let attr = Arc::new("v".to_string());
    for (batch, steps) in [(1usize, 9usize), (2, 5), (4, 3), (9, 1), (10, 1)] {
        let (mut g, label, _) = people_with(&attr, &(1..=9).collect::<Vec<_>>());
        create_index_on(&mut g, &label, &attr);
        assert_eq!(build_to_end(&mut g, batch), steps, "batch {batch}");
        assert_eq!(range_scan(&g, &label, &attr, 0, 100).len(), 9);
    }
}

/// The same unit of work against a `Building` column, on a node the steps already indexed and on
/// one they have not reached: neither the old value nor the SET's value may survive the DELETE.
#[test]
fn set_then_delete_in_one_batch_during_a_build() {
    let attr = Arc::new("v".to_string());
    let (mut g, label, ids) = people_with(&attr, &[1, 2, 3]);
    create_index_on(&mut g, &label, &attr);
    assert!(graph_writes::build_step(&mut g, 1));
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
/// steps and the writes put there both live in the column — and leaves no step anything to do.
/// Writes after the drop stage nothing.
#[test]
fn dropping_a_building_column_leaves_nothing_behind() {
    let attr = Arc::new("v".to_string());
    let (mut g, label, ids) = people_with(&attr, &[10, 20]);
    create_index_on(&mut g, &label, &attr);
    assert!(graph_writes::build_step(&mut g, 1));
    set_attr(&mut g, ids[0], &attr, Value::Int(30));

    drop_index_on(&mut g, &label, &attr);
    assert!(!g.falkordb_index.has_building_columns());
    assert!(
        g.falkordb_index
            .numeric(EntityType::Node, &label, &attr)
            .is_none()
    );
    assert!(
        !graph_writes::build_step(&mut g, graph_writes::BUILD_STEP_ENTITIES),
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
    assert!(graph_writes::build_step(&mut g, 1));

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
    assert!(graph_writes::build_step(&mut g, 1));
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

// --- The builder: turns, and merging with writers ---

/// A host for driving build turns in a test: a plain lock around the coordinator. `racing` holds
/// writes that each commit, one after another, at the start of the next `publish` — the way writers
/// commit while a turn reads its batch. `refuse` makes `publish` refuse, as for a deleted graph.
struct TestHost {
    mvcc: RwLock<MvccGraph>,
    racing: Mutex<Vec<GraphWrite>>,
    refuse: AtomicBool,
}

type GraphWrite = Box<dyn FnOnce(&mut Graph) + Send>;

impl TestHost {
    /// `:Person` nodes with `attr = values`, and a committed `CREATE INDEX` on them.
    fn building(
        attr: &Arc<String>,
        values: &[i64],
    ) -> (Arc<Self>, Arc<String>, Vec<u64>) {
        let (g, label, ids) = people_with(attr, values);
        let host = Self::new(g);
        host.write(|g| create_index_on(g, &label, attr));
        (host, label, ids)
    }

    fn new(g: Graph) -> Arc<Self> {
        Arc::new(Self {
            mvcc: RwLock::new(MvccGraph::from_graph(g)),
            racing: Mutex::new(Vec::new()),
            refuse: AtomicBool::new(false),
        })
    }

    /// A writer commits `write` while the next turn reads its batch.
    fn race_next_turn(
        &self,
        write: impl FnOnce(&mut Graph) + Send + 'static,
    ) {
        self.racing.lock().unwrap().push(Box::new(write));
    }

    /// One build turn with batches of 10.
    fn turn(&self) -> Turn {
        online_build::run_turn_within(self, 10)
    }

    /// Turns until the graph has nothing left to build.
    fn build_to_end(&self) {
        for _ in 0..10_000 {
            match self.turn() {
                Turn::Idle => return,
                Turn::Committed { .. } => {}
                other => panic!("unexpected turn {other:?}"),
            }
        }
        panic!("the build never finishes");
    }

    /// A write as Redis's main thread makes one: take the slot, then commit.
    fn write(
        &self,
        f: impl FnOnce(&mut Graph),
    ) {
        let mut mvcc = self.mvcc.write();
        let version = mvcc.write().expect("the slot is free");
        f(&mut version.borrow_mut());
        mvcc.commit(version).expect("commits");
    }

    /// A queued write starts: a private version, without the slot.
    fn fork(&self) -> Arc<AtomicRefCell<Graph>> {
        self.mvcc.read().fork()
    }

    /// The queued write finishes: claim the slot for its version, then commit it.
    fn finish(
        &self,
        fork: Arc<AtomicRefCell<Graph>>,
    ) -> Result<(), Claim> {
        let mut mvcc = self.mvcc.write();
        mvcc.claim(&fork)?;
        mvcc.commit(fork).expect("the writer commits");
        Ok(())
    }

    fn published(&self) -> Arc<AtomicRefCell<Graph>> {
        self.mvcc.read().read()
    }

    /// The column's ids in value order, from the published version.
    fn indexed(
        &self,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Vec<u64> {
        range_scan(&self.published().borrow(), label, attr, i64::MIN, i64::MAX)
    }

    fn cursor(
        &self,
        label: &Arc<String>,
        attr: &Arc<String>,
    ) -> Option<BuildCursor> {
        cursor(&self.published().borrow(), EntityType::Node, label, attr)
    }

    /// The published node column over `values`, next to what a one-pass build of the published
    /// graph gives it.
    fn against_one_pass(
        &self,
        label: &Arc<String>,
        attr: &Arc<String>,
        values: &[i64],
    ) -> (
        (Vec<(i64, Vec<u64>)>, Vec<u64>),
        (Vec<(i64, Vec<u64>)>, Vec<u64>),
    ) {
        self.against_one_pass_of(EntityType::Node, label, attr, values)
    }

    fn against_one_pass_of(
        &self,
        entity: EntityType,
        label: &Arc<String>,
        attr: &Arc<String>,
        values: &[i64],
    ) -> (
        (Vec<(i64, Vec<u64>)>, Vec<u64>),
        (Vec<(i64, Vec<u64>)>, Vec<u64>),
    ) {
        let published = self.published();
        let g = published.borrow();
        (
            contents(&g, entity, label, attr, values),
            one_pass(&g, entity, label, attr, values),
        )
    }

    /// Whether the write slot is free: taking it succeeds, and it is given back.
    fn slot_free(&self) -> bool {
        let mvcc = self.mvcc.read();
        let free = mvcc.write().is_some();
        if free {
            mvcc.rollback();
        }
        free
    }
}

impl BuildHost for TestHost {
    fn read(
        &self,
        f: &mut dyn FnMut(&MvccGraph),
    ) -> bool {
        f(&self.mvcc.read());
        true
    }

    fn publish(
        &self,
        f: &mut dyn FnMut(&mut MvccGraph),
    ) -> bool {
        if self.refuse.load(Ordering::Relaxed) {
            return false;
        }
        let racing = std::mem::take(&mut *self.racing.lock().unwrap());
        for write in racing {
            self.write(write);
        }
        f(&mut self.mvcc.write());
        true
    }
}

/// A [`TestHost`] held weakly, as the Redis host holds its graph.
struct WeakHost(Weak<TestHost>);

impl BuildHost for WeakHost {
    fn read(
        &self,
        f: &mut dyn FnMut(&MvccGraph),
    ) -> bool {
        self.0.upgrade().is_some_and(|host| host.read(f))
    }

    fn publish(
        &self,
        f: &mut dyn FnMut(&mut MvccGraph),
    ) -> bool {
        self.0.upgrade().is_some_and(|host| host.publish(f))
    }
}

/// Delete node `id` through the write path.
fn delete_node(
    g: &mut Graph,
    id: u64,
) {
    g.open_id_batches().expect("a consistent space");
    let mut del = RoaringTreemap::new();
    del.insert(id);
    applied!(g, |w| g.delete_nodes(&del, &mut w)).expect("deleted");
}

/// The writes a merge has to get right, on `:Person` nodes valued 1..=25 whose first build batch
/// is `ids[0..10]`: one entity in the batch re-valued, one in it deleted, one ahead of it
/// re-valued, and one created.
fn the_racing_write(
    ids: &[u64],
    attr: &Arc<String>,
) -> impl FnOnce(&mut Graph) + Send + 'static {
    let (ids, attr) = (ids.to_vec(), Arc::clone(attr));
    move |g: &mut Graph| {
        set_attr(g, ids[3], &attr, Value::Int(103));
        delete_node(g, ids[5]);
        set_attr(g, ids[20], &attr, Value::Int(120));
        add_person(g, &attr, 200);
    }
}

/// Every value the racing write and the nodes it starts from can hold.
fn racing_values() -> Vec<i64> {
    (1..=25).chain([103, 120, 200]).collect()
}

/// Turns of at most 10 nodes build a 25-node column in three commits, then find nothing to do.
#[test]
fn turns_build_a_column_and_go_idle() {
    let attr = Arc::new("v".to_string());
    let (host, label, ids) = TestHost::building(&attr, &(1..=25).collect::<Vec<_>>());
    assert_eq!(host.turn(), Turn::Committed { more: true });
    assert_eq!(host.turn(), Turn::Committed { more: true });
    assert_eq!(host.turn(), Turn::Committed { more: false });
    assert_eq!(host.turn(), Turn::Idle);
    assert_eq!(host.indexed(&label, &attr), ids);
    assert_eq!(host.cursor(&label, &attr), None, "Ready");
}

/// A queued write forked before a build step committed, and claims after it. Its version
/// is stale only because of the step, so the claim succeeds instead of running the write again,
/// and the published column keeps the step's progress and every change the write made — including
/// removing what the step indexed of the entities the write re-valued and deleted.
#[test]
fn a_write_claims_over_build_steps_and_keeps_their_progress() {
    let attr = Arc::new("v".to_string());
    let (host, label, ids) = TestHost::building(&attr, &(1..=25).collect::<Vec<_>>());

    let fork = host.fork();
    the_racing_write(&ids, &attr)(&mut fork.borrow_mut());
    assert_eq!(host.turn(), Turn::Committed { more: true });
    assert_eq!(
        host.finish(fork),
        Ok(()),
        "not stale: only a build step committed"
    );

    assert_eq!(
        host.cursor(&label, &attr),
        Some(BuildCursor::Node(NodeId::from(ids[10]))),
        "the step's progress survived the write"
    );
    let published = host.published();
    let g = published.borrow();
    assert!(
        range_scan(&g, &label, &attr, 4, 4).is_empty(),
        "the step indexed ids[3] at 4; the write re-valued it"
    );
    assert!(
        range_scan(&g, &label, &attr, 6, 6).is_empty(),
        "the step indexed ids[5] at 6; the write deleted it"
    );
    assert_eq!(range_scan(&g, &label, &attr, 103, 103), vec![ids[3]]);
    drop(g);

    host.build_to_end();
    let (got, want) = host.against_one_pass(&label, &attr, &racing_values());
    assert_eq!(got, want);
}

/// A write commits while a build step is taken. The step is not dropped: it is taken again on what
/// the write published, so it indexes the values the write left.
#[test]
fn a_build_step_that_raced_a_write_is_taken_again_on_its_version() {
    let attr = Arc::new("v".to_string());
    let (host, label, ids) = TestHost::building(&attr, &(1..=25).collect::<Vec<_>>());

    host.race_next_turn(the_racing_write(&ids, &attr));
    assert_eq!(host.turn(), Turn::Committed { more: true });
    assert_eq!(
        host.cursor(&label, &attr),
        Some(BuildCursor::Node(NodeId::from(ids[10]))),
        "the step was taken again, not dropped"
    );
    let published = host.published();
    let g = published.borrow();
    assert!(
        range_scan(&g, &label, &attr, 4, 4).is_empty(),
        "the first try read ids[3] at 4 before the write re-valued it"
    );
    assert!(
        range_scan(&g, &label, &attr, 6, 6).is_empty(),
        "the first try read ids[5] before the write deleted it"
    );
    assert_eq!(range_scan(&g, &label, &attr, 1, 3), ids[0..3].to_vec());
    assert_eq!(range_scan(&g, &label, &attr, 103, 103), vec![ids[3]]);
    drop(g);

    host.build_to_end();
    let (got, want) = host.against_one_pass(&label, &attr, &racing_values());
    assert_eq!(got, want);
}

/// Several writers commit while one step is taken: the step is taken again on a version that holds
/// all of their writes.
#[test]
fn a_build_step_is_taken_again_over_several_writers() {
    let attr = Arc::new("v".to_string());
    let (host, label, ids) = TestHost::building(&attr, &(1..=25).collect::<Vec<_>>());
    let (first, second) = (ids.clone(), ids.clone());
    let (a1, a2) = (Arc::clone(&attr), Arc::clone(&attr));
    host.race_next_turn(move |g| set_attr(g, first[3], &a1, Value::Int(103)));
    host.race_next_turn(move |g| {
        set_attr(g, second[3], &a2, Value::Int(104));
        delete_node(g, second[5]);
        set_attr(g, second[8], &a2, Value::Int(108));
    });
    assert_eq!(host.turn(), Turn::Committed { more: true });

    let published = host.published();
    let g = published.borrow();
    for stale in [4, 6, 9, 103] {
        assert!(
            range_scan(&g, &label, &attr, stale, stale).is_empty(),
            "no tuple at {stale}"
        );
    }
    assert_eq!(range_scan(&g, &label, &attr, 104, 104), vec![ids[3]]);
    assert_eq!(range_scan(&g, &label, &attr, 108, 108), vec![ids[8]]);
    assert_eq!(
        g.get_node_attribute_by_idx(
            NodeId::from(ids[3]),
            g.get_node_attribute_id(&attr).expect("v") as u16
        ),
        Some(Value::Int(104)),
        "both writers' data is in the version the batch went into"
    );
    drop(g);

    host.build_to_end();
    let values: Vec<i64> = (1..=25).chain([103, 104, 108]).collect();
    let (got, want) = host.against_one_pass(&label, &attr, &values);
    assert_eq!(got, want);
}

/// A write that committed graph data between a queued write's fork and its claim still makes the
/// fork stale, with build steps committed around it: the queued write matched on data that
/// changed, and runs again.
#[test]
fn a_write_between_fork_and_claim_still_makes_it_stale() {
    let attr = Arc::new("v".to_string());
    let (host, _, ids) = TestHost::building(&attr, &(1..=25).collect::<Vec<_>>());

    let fork = host.fork();
    assert_eq!(host.turn(), Turn::Committed { more: true });
    host.write(|g| set_attr(g, ids[2], &attr, Value::Int(51)));
    assert_eq!(host.turn(), Turn::Committed { more: true });
    assert_eq!(host.finish(fork), Err(Claim::Stale));
    assert!(host.slot_free(), "a stale claim leaves the slot free");
}

/// A step taken on a column that a write dropped, or dropped and re-created, while the step ran
/// never reaches the column that replaced it: the step is taken again on what the write published.
/// A re-created column builds from the start, and a dropped one stays dropped.
#[test]
fn a_step_on_a_dropped_or_recreated_column_is_taken_again_on_the_writers_version() {
    let attr = Arc::new("v".to_string());
    let (host, label, ids) = TestHost::building(&attr, &(1..=25).collect::<Vec<_>>());

    let recreate = {
        let (label, attr, first) = (Arc::clone(&label), Arc::clone(&attr), ids[0]);
        move |g: &mut Graph| {
            drop_index_on(g, &label, &attr);
            create_index_on(g, &label, &attr);
            set_attr(g, first, &attr, Value::Int(100));
        }
    };
    host.race_next_turn(recreate);
    assert_eq!(host.turn(), Turn::Committed { more: true });
    assert_eq!(
        host.cursor(&label, &attr),
        Some(BuildCursor::Node(NodeId::from(ids[10]))),
        "the re-created column's first step"
    );
    let mut values: Vec<i64> = (1..=25).collect();
    values[0] = 100;
    let indexed: Vec<u64> = ids[1..10].iter().copied().chain([ids[0]]).collect();
    assert_eq!(
        host.indexed(&label, &attr),
        indexed,
        "read from the writer's version: ids[0] at 100"
    );

    host.build_to_end();
    let (got, want) = host.against_one_pass(&label, &attr, &values);
    assert_eq!(got, want);

    // On a fresh graph, a column dropped while a step is taken stays dropped.
    let (host, label, _) = TestHost::building(&attr, &(1..=25).collect::<Vec<_>>());
    let drop_it = {
        let (label, attr) = (Arc::clone(&label), Arc::clone(&attr));
        move |g: &mut Graph| drop_index_on(g, &label, &attr)
    };
    host.race_next_turn(drop_it);
    assert_eq!(host.turn(), Turn::Idle, "nothing left to build");
    assert!(
        host.published()
            .borrow()
            .falkordb_index
            .numeric(EntityType::Node, &label, &attr)
            .is_none(),
        "the dropped column stays dropped"
    );
}

/// A queued write that re-created the column claims over build steps of the old one: its column is
/// its own, not the old one's progress, and builds from the start.
#[test]
fn a_write_that_recreated_the_column_keeps_its_own() {
    let attr = Arc::new("v".to_string());
    let (host, label, _) = TestHost::building(&attr, &(1..=25).collect::<Vec<_>>());

    let fork = host.fork();
    drop_index_on(&mut fork.borrow_mut(), &label, &attr);
    create_index_on(&mut fork.borrow_mut(), &label, &attr);
    assert_eq!(host.turn(), Turn::Committed { more: true });
    assert_eq!(host.finish(fork), Ok(()));
    assert_eq!(
        host.cursor(&label, &attr),
        Some(BuildCursor::new(EntityType::Node))
    );
    assert!(host.indexed(&label, &attr).is_empty());

    host.build_to_end();
    let values: Vec<i64> = (1..=25).collect();
    let (got, want) = host.against_one_pass(&label, &attr, &values);
    assert_eq!(got, want);
}

/// What a random write did to the graph's data, to replay on a [`Model`] once it commits.
#[derive(Clone, Debug)]
enum Change {
    Delete(u64),
    Create(u64, Arc<String>, i64),
    Relabel(u64, bool),
    Set(u64, Arc<String>, i64),
}

/// The `:Person` and `:Other` nodes as the committed writes left them: whether each is a
/// `:Person`, and its `v` and `w`.
type Model = BTreeMap<u64, (bool, Option<i64>, Option<i64>)>;

fn replay(
    model: &mut Model,
    changes: &[Change],
) {
    for change in changes {
        match change {
            Change::Delete(id) => {
                model.remove(id);
            }
            Change::Create(id, attr, value) => {
                let value = Some(*value);
                let (v, w) = if attr.as_str() == "v" {
                    (value, None)
                } else {
                    (None, value)
                };
                model.insert(*id, (true, v, w));
            }
            Change::Relabel(id, person) => {
                model.get_mut(id).expect("a live node").0 = *person;
            }
            Change::Set(id, attr, value) => {
                let node = model.get_mut(id).expect("a live node");
                if attr.as_str() == "v" {
                    node.1 = Some(*value);
                } else {
                    node.2 = Some(*value);
                }
            }
        }
    }
}

/// `g`'s `:Person` and `:Other` nodes, as a [`Model`].
fn model_of(g: &Graph) -> Model {
    let value = |n: u64, attr: &str| {
        let idx = g.get_node_attribute_id(&Arc::new(attr.to_string()))?;
        match g.get_node_attribute_by_idx(NodeId::from(n), idx as u16)? {
            Value::Int(x) => Some(x),
            _ => None,
        }
    };
    let mut model = Model::new();
    for (label, person) in [("Person", true), ("Other", false)] {
        if let Some(m) = g.get_label_matrix(&Arc::new(label.to_string())) {
            for (n, _) in m.iter(0, u64::MAX) {
                model.insert(n, (person, value(n, "v"), value(n, "w")));
            }
        }
    }
    model
}

/// Writes and build steps in random order — writers forking, writing and claiming across build
/// commits, writers committing while a batch is read, labels taken off and put back, the index
/// re-created now and then, on two columns of one label built one after the other — always end
/// with the data the committed writes made, and with the columns a one-pass build of it has; and a
/// writer's claim fails only when another writer committed since its fork. Counts how often each
/// merge ran, so a run that never reached one fails.
#[test]
fn random_interleavings_of_writes_and_build_steps_match_a_one_pass_build() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    const VALUES: i64 = 40;
    let (mut writes_over_steps, mut steps_over_writes) = (0, 0);
    for seed in 0..40 {
        let mut rng = StdRng::seed_from_u64(seed);
        let v = Arc::new("v".to_string());
        let w = Arc::new("w".to_string());
        let (host, label, ids) =
            TestHost::building(&v, &(0..150).map(|i| i % VALUES).collect::<Vec<_>>());
        host.write(|g| {
            for (i, &id) in ids.iter().enumerate() {
                set_attr(g, id, &w, Value::Int((i as i64 * 7) % VALUES));
            }
            create_index_on(g, &label, &w);
        });
        let mut model: Model = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let i = i as i64;
                (id, (true, Some(i % VALUES), Some((i * 7) % VALUES)))
            })
            .collect();
        // A writer in flight: its fork, whether another writer committed since it was made, and
        // the changes it made so far.
        let mut queued: Option<(Arc<AtomicRefCell<Graph>>, bool, Vec<Change>)> = None;
        let building = || {
            host.published()
                .borrow()
                .falkordb_index
                .has_building_columns()
        };

        // A random write: re-value, delete or create a `:Person` node, move one between `:Person`
        // and `:Other`; rarely, re-create one of the indexes. Returns what it changed.
        let random_write = {
            let (label, v, w) = (Arc::clone(&label), Arc::clone(&v), Arc::clone(&w));
            move |g: &mut Graph, rng: &mut StdRng| -> Vec<Change> {
                let members = |g: &Graph, label: &str| -> Vec<u64> {
                    g.get_label_matrix(&Arc::new(label.to_string()))
                        .map(|m| m.iter(0, u64::MAX).map(|(n, _)| n).collect())
                        .unwrap_or_default()
                };
                let people = members(g, "Person");
                let others = members(g, "Other");
                let attr = if rng.random_bool(0.5) { &v } else { &w };
                let person = g.get_label_id_mut("Person").0 as u64;
                let other = g.get_label_id_mut("Other").0 as u64;
                match rng.random_range(0..24) {
                    0 => {
                        drop_index_on(g, &label, attr);
                        create_index_on(g, &label, attr);
                        vec![]
                    }
                    1..=3 if !people.is_empty() => {
                        let id = people[rng.random_range(0..people.len())];
                        delete_node(g, id);
                        vec![Change::Delete(id)]
                    }
                    4..=6 => {
                        let value = rng.random_range(0..VALUES);
                        let id = add_person(g, attr, value);
                        vec![Change::Create(id, Arc::clone(attr), value)]
                    }
                    7..=9 if !people.is_empty() => {
                        let id = people[rng.random_range(0..people.len())];
                        applied!(g, |d| g.remove_nodes_labels(&[id], &[person], &mut d));
                        applied!(g, |d| g.set_nodes_labels_bulk(
                            &[id],
                            &[other],
                            &mut d,
                            false
                        ));
                        vec![Change::Relabel(id, false)]
                    }
                    10..=12 if !others.is_empty() => {
                        let id = others[rng.random_range(0..others.len())];
                        applied!(g, |d| g.remove_nodes_labels(&[id], &[other], &mut d));
                        applied!(g, |d| g.set_nodes_labels_bulk(
                            &[id],
                            &[person],
                            &mut d,
                            false
                        ));
                        vec![Change::Relabel(id, true)]
                    }
                    _ if !people.is_empty() => {
                        let id = people[rng.random_range(0..people.len())];
                        let value = rng.random_range(0..VALUES);
                        set_attr(g, id, attr, Value::Int(value));
                        vec![Change::Set(id, Arc::clone(attr), value)]
                    }
                    _ => vec![],
                }
            }
        };

        for _ in 0..300 {
            match rng.random_range(0..6) {
                0 if queued.is_none() => queued = Some((host.fork(), false, vec![])),
                1 => {
                    if let Some((fork, _, changes)) = &mut queued {
                        changes.extend(random_write(&mut fork.borrow_mut(), &mut rng));
                    }
                }
                2 => {
                    if let Some((fork, data_changed, changes)) = queued.take() {
                        let stale = host.published().borrow().version >= fork.borrow().version;
                        let claim = host.finish(fork);
                        assert_eq!(
                            claim.is_err(),
                            data_changed,
                            "seed {seed}: a claim fails exactly when another writer committed"
                        );
                        if claim.is_ok() {
                            replay(&mut model, &changes);
                            if stale {
                                writes_over_steps += 1;
                            }
                        }
                    }
                }
                3 => {
                    let mut write_rng = StdRng::seed_from_u64(rng.random());
                    let mut changes = vec![];
                    host.write(|g| changes = random_write(g, &mut write_rng));
                    replay(&mut model, &changes);
                    if let Some((_, data_changed, _)) = &mut queued {
                        *data_changed = true;
                    }
                }
                4 => {
                    let mut write_rng = StdRng::seed_from_u64(rng.random());
                    let write = random_write.clone();
                    let changes = Arc::new(Mutex::new(vec![]));
                    let made = Arc::clone(&changes);
                    let was_building = building();
                    host.race_next_turn(move |g| {
                        *made.lock().unwrap() = write(g, &mut write_rng);
                    });
                    let turn = online_build::run_turn_within(host.as_ref(), 5);
                    assert!(
                        matches!(turn, Turn::Committed { .. } | Turn::Idle),
                        "seed {seed}: {turn:?}"
                    );
                    // The write ran, and committed, if the turn reached its commit.
                    let ran = host.racing.lock().unwrap().is_empty();
                    host.racing.lock().unwrap().clear();
                    if ran {
                        replay(&mut model, &changes.lock().unwrap());
                        if let Some((_, data_changed, _)) = &mut queued {
                            *data_changed = true;
                        }
                        if was_building && matches!(turn, Turn::Committed { .. }) {
                            steps_over_writes += 1;
                        }
                    }
                }
                _ => {
                    let turn = online_build::run_turn_within(host.as_ref(), 5);
                    assert!(
                        matches!(turn, Turn::Committed { .. } | Turn::Idle),
                        "seed {seed}: {turn:?}"
                    );
                }
            }
        }
        if let Some((fork, data_changed, changes)) = queued.take() {
            let claim = host.finish(fork);
            assert_eq!(claim.is_err(), data_changed, "seed {seed}");
            if claim.is_ok() {
                replay(&mut model, &changes);
            }
        }
        host.build_to_end();
        assert_eq!(
            model_of(&host.published().borrow()),
            model,
            "seed {seed}: the graph holds what the committed writes made"
        );
        let values: Vec<i64> = (0..VALUES).collect();
        for attr in [&v, &w] {
            let (got, want) = host.against_one_pass(&label, attr, &values);
            assert_eq!(got, want, "seed {seed}, column {attr}");
        }
    }
    assert!(
        writes_over_steps >= 50 && steps_over_writes >= 50,
        "each merge ran often: {writes_over_steps} writes over steps, {steps_over_writes} steps \
         over writes"
    );
    eprintln!("{writes_over_steps} writes over steps, {steps_over_writes} steps over writes");
}

/// `:R` edges `n[i] -> n[i + 1]` with `w = i + 1`, 30 of them, and a committed `CREATE INDEX` on
/// `(:R, w)`. Returns the host, the type, the attribute, the nodes and the edge ids.
fn edges_building() -> (Arc<TestHost>, Arc<String>, Arc<String>, Vec<u64>, Vec<u64>) {
    ensure_graphblas();
    let mut g = Graph::new(64, 64, 10, 1, "t");
    let ty = Arc::new("R".to_string());
    let attr = Arc::new("w".to_string());
    let nodes = reserve_nodes(&mut g, 31);
    g.create_nodes(&nodes.iter().copied().collect())
        .expect("freshly reserved ids are creatable");
    let eids = add_edges(&mut g, &ty, &attr, &nodes[..30], &nodes[1..], 1);
    let host = TestHost::new(g);
    host.write(|g| {
        graph_writes::create_index(
            g,
            &IndexType::Range,
            EntityType::Relationship,
            &ty,
            std::slice::from_ref(&attr),
        );
    });
    (host, ty, attr, nodes, eids)
}

/// `:ty` edges `srcs[i] -> dsts[i]` with `attr = first + i`, through the write path.
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

/// The edge writes a merge has to get right, on the chain [`edges_building`] makes, whose first
/// build batch is `eids[0..10]`: one edge in the batch re-valued, one in it deleted, one ahead of
/// it re-valued, and one created from a source the batch covers.
fn the_racing_edge_write(
    nodes: &[u64],
    eids: &[u64],
    ty: &Arc<String>,
    attr: &Arc<String>,
) -> impl FnOnce(&mut Graph) + Send + 'static {
    let (nodes, eids, ty, attr) = (
        nodes.to_vec(),
        eids.to_vec(),
        Arc::clone(ty),
        Arc::clone(attr),
    );
    move |g: &mut Graph| {
        g.open_id_batches().expect("a consistent space");
        let aid = g.get_or_create_rel_attr_id(&attr);
        let set: FxHashMap<u64, Vec<(u16, Value)>> = [(eids[3], 103), (eids[20], 120)]
            .into_iter()
            .map(|(eid, w)| (eid, vec![(aid, Value::Int(w))]))
            .collect();
        applied!(g, |d| g.set_relationships_attributes(&set, &mut d)).expect("set");
        let mut del = RoaringTreemap::new();
        del.insert(eids[5]);
        applied!(g, |d| g.delete_relationships(&del, &mut d)).expect("deleted");
        add_edges(g, &ty, &attr, &[nodes[0]], &[nodes[2]], 200);
    }
}

/// Both orders on an edge column: a queued write forked before a build step and claiming after it
/// (writer over step), and a write committing while a step is taken (step over writer).
#[test]
fn both_orders_hold_on_an_edge_column() {
    let values: Vec<i64> = (1..=30).chain([103, 120, 200]).collect();

    let (host, ty, attr, nodes, eids) = edges_building();
    let fork = host.fork();
    the_racing_edge_write(&nodes, &eids, &ty, &attr)(&mut fork.borrow_mut());
    assert_eq!(host.turn(), Turn::Committed { more: true });
    assert_eq!(host.finish(fork), Ok(()), "only a build step committed");
    host.build_to_end();
    let (got, want) = host.against_one_pass_of(EntityType::Relationship, &ty, &attr, &values);
    assert_eq!(got, want, "writer over step");

    let (host, ty, attr, nodes, eids) = edges_building();
    host.race_next_turn(the_racing_edge_write(&nodes, &eids, &ty, &attr));
    assert_eq!(host.turn(), Turn::Committed { more: true });
    host.build_to_end();
    let (got, want) = host.against_one_pass_of(EntityType::Relationship, &ty, &attr, &values);
    assert_eq!(got, want, "step over writer");
}

/// A version keeps its changes to a building column until it is published: a writer's fork, an
/// inline writer's version alike. A fork that merges over a build commit replays them there and
/// keeps none.
#[test]
fn a_version_keeps_its_change_log_until_published() {
    let attr = Arc::new("v".to_string());
    let (host, _, ids) = TestHost::building(&attr, &(1..=25).collect::<Vec<_>>());
    let logged = |version: &Arc<AtomicRefCell<Graph>>| version.borrow().falkordb_index.logged();

    {
        let mvcc = host.mvcc.read();
        let inline = mvcc.write().expect("free");
        set_attr(&mut inline.borrow_mut(), ids[0], &attr, Value::Int(50));
        assert!(logged(&inline) > 0, "an inline writer's version");
        mvcc.rollback();
    }

    let fork = host.fork();
    set_attr(&mut fork.borrow_mut(), ids[0], &attr, Value::Int(50));
    assert!(logged(&fork) > 0, "a writer's fork");
    host.mvcc.read().claim(&fork).expect("not stale");
    assert!(logged(&fork) > 0, "kept through a plain claim");
    host.mvcc.write().commit(fork).expect("commits");
    assert_eq!(logged(&host.published()), 0, "dropped once published");

    let fork = host.fork();
    set_attr(&mut fork.borrow_mut(), ids[1], &attr, Value::Int(51));
    assert_eq!(host.turn(), Turn::Committed { more: true });
    host.mvcc
        .read()
        .claim(&fork)
        .expect("stale only over a build commit");
    assert_eq!(logged(&fork), 0, "replayed by the merge");
    host.mvcc.write().commit(fork).expect("commits");
}

/// A commit queues its graph for the builder only when it starts a build: a `CREATE INDEX` on a
/// graph with no `Building` column. The builder keeps the graph until its build is done, so a write
/// during the build, and each build step's own commit, queue nothing; a `CREATE INDEX` after the
/// build is done queues it again.
///
/// The pool is global to the test binary and never started here, so this reads how many graphs
/// wait in its queue before and after. It is the only unit test that registers a build host.
#[test]
fn a_commit_queues_its_graph_only_when_it_starts_a_build() {
    let attr = Arc::new("v".to_string());
    let (g, label, ids) = people_with(&attr, &(1..=25).collect::<Vec<_>>());
    let host = TestHost::new(g);
    // Weakly, as the Redis host holds its graph, so the queued host does not keep it alive.
    host.mvcc
        .read()
        .set_build_host(Arc::new(WeakHost(Arc::downgrade(&host))));
    let queued = online_build::pending();

    host.write(|g| create_index_on(g, &label, &attr));
    assert_eq!(
        online_build::pending(),
        queued + 1,
        "CREATE INDEX starts a build"
    );
    host.write(|g| set_attr(g, ids[0], &attr, Value::Int(100)));
    assert_eq!(
        online_build::pending(),
        queued + 1,
        "a write during the build"
    );
    host.build_to_end();
    assert_eq!(
        online_build::pending(),
        queued + 1,
        "the build steps' own commits, the last one included"
    );
    host.write(|g| create_index_on(g, &label, &Arc::new("w".to_string())));
    assert_eq!(
        online_build::pending(),
        queued + 2,
        "a CREATE INDEX after the build is done starts another"
    );
}

/// A host that refuses to publish (the graph was deleted) ends the turn without committing.
#[test]
fn a_refused_publish_commits_nothing() {
    let attr = Arc::new("v".to_string());
    let (host, label, _) = TestHost::building(&attr, &[1, 2, 3]);
    host.refuse.store(true, Ordering::Relaxed);
    assert_eq!(host.turn(), Turn::Gone);
    assert!(host.indexed(&label, &attr).is_empty());
}

/// What a turn does before it publishes: take a build step of at most `nodes` nodes on a new
/// version of `published`, holding no lock.
fn prepare_turn(
    published: &AtomicRefCell<Graph>,
    nodes: usize,
) -> Arc<AtomicRefCell<Graph>> {
    let prepared = Arc::new(AtomicRefCell::new(published.borrow().new_version()));
    assert!(prepared.borrow_mut().step_index_build(nodes));
    prepared
}

/// The step a commit takes again if a writer made its prepared version stale.
fn step_of(nodes: usize) -> impl FnOnce(&mut Graph) -> bool {
    move |g| g.step_index_build(nodes)
}

/// A turn that nothing committed against publishes the version it prepared with no lock held,
/// so its step never holds the host's locks; one a writer committed against takes its step again
/// on the writer's version instead, reading the values it holds.
#[test]
fn a_turn_publishes_its_prepared_version_only_if_nothing_committed_since() {
    let attr = Arc::new("v".to_string());
    let values: Vec<i64> = (1..=25).collect();
    let (host, label, ids) = TestHost::building(&attr, &values);

    let prepared = prepare_turn(&host.published(), 10);
    let commit = host.mvcc.write().commit_index_build(&prepared, step_of(10));
    assert_eq!(commit, Some(BuildCommit::Prepared));
    assert!(
        Arc::ptr_eq(&host.published(), &prepared),
        "the prepared version itself"
    );

    let prepared = prepare_turn(&host.published(), 10);
    host.write(|g| set_attr(g, ids[13], &attr, Value::Int(113)));
    let commit = host.mvcc.write().commit_index_build(&prepared, step_of(10));
    assert_eq!(commit, Some(BuildCommit::Redone));
    assert!(
        !Arc::ptr_eq(&host.published(), &prepared),
        "not a version older than the write"
    );
    assert!(
        range_scan(&host.published().borrow(), &label, &attr, 14, 14).is_empty(),
        "ids[13] holds 113: the step taken again does not index the 14 the first try read"
    );

    host.build_to_end();
    let mut all = values.clone();
    all.push(113);
    let (got, want) = host.against_one_pass(&label, &attr, &all);
    assert_eq!(got, want);
}

/// Two turns at once on one graph — which the pool never runs — both prepared on one version: the
/// second's commit finds the first one's published and takes its step again on it, so it indexes
/// the next batch, at the values a write between them left.
#[test]
fn a_second_turn_takes_its_step_again_on_the_first_ones_version() {
    let attr = Arc::new("v".to_string());
    let values: Vec<i64> = (1..=25).collect();
    let (host, label, ids) = TestHost::building(&attr, &values);
    let published = host.published();
    let turns = [prepare_turn(&published, 10), prepare_turn(&published, 10)];
    drop(published);
    host.write(|g| set_attr(g, ids[3], &attr, Value::Int(103)));
    let commits: Vec<_> = turns
        .iter()
        .map(|prepared| host.mvcc.write().commit_index_build(prepared, step_of(10)))
        .collect();
    assert_eq!(
        commits,
        [Some(BuildCommit::Redone), Some(BuildCommit::Redone)]
    );
    assert_eq!(
        host.cursor(&label, &attr),
        Some(BuildCursor::Node(NodeId::from(ids[20]))),
        "two batches, not the same one twice"
    );
    assert!(
        range_scan(&host.published().borrow(), &label, &attr, 4, 4).is_empty(),
        "ids[3] holds 103: neither step indexes the 4 the first tries read"
    );
    host.build_to_end();
    let mut all = values.clone();
    all.push(103);
    let (got, want) = host.against_one_pass(&label, &attr, &all);
    assert_eq!(got, want);
}

/// A host that records, before its locks are let go, whether the version a commit replaced is
/// still alive, and keeps a weak handle to it.
struct FreeingHost {
    host: Arc<TestHost>,
    replaced: Mutex<Vec<(Weak<AtomicRefCell<Graph>>, bool)>>,
}

impl BuildHost for FreeingHost {
    fn read(
        &self,
        f: &mut dyn FnMut(&MvccGraph),
    ) -> bool {
        self.host.read(f)
    }

    fn publish(
        &self,
        f: &mut dyn FnMut(&mut MvccGraph),
    ) -> bool {
        self.host.publish(&mut |mvcc| {
            let replaced = Arc::downgrade(&mvcc.read());
            f(mvcc);
            let alive = replaced.upgrade().is_some();
            self.replaced.lock().unwrap().push((replaced, alive));
        })
    }
}

/// Freeing the version a commit replaces frees every index node the step copied rather than
/// shared, which can take milliseconds: the turn holds it until the host has let go of its locks,
/// whether the commit published the prepared version or took the step again on a writer's.
#[test]
fn a_turn_frees_the_version_its_commit_replaces_after_the_host_lets_go() {
    let attr = Arc::new("v".to_string());
    let values: Vec<i64> = (1..=25).collect();
    let (host, _, ids) = TestHost::building(&attr, &values);
    let freeing = FreeingHost {
        host: Arc::clone(&host),
        replaced: Mutex::new(Vec::new()),
    };
    assert_eq!(
        online_build::run_turn_within(&freeing, 10),
        Turn::Committed { more: true }
    );
    host.race_next_turn(the_racing_write(&ids, &attr));
    assert_eq!(
        online_build::run_turn_within(&freeing, 10),
        Turn::Committed { more: true }
    );
    let replaced = freeing.replaced.into_inner().unwrap();
    assert_eq!(replaced.len(), 2);
    for (n, (version, alive)) in replaced.iter().enumerate() {
        assert!(alive, "turn {n}: freed under the host's locks");
        assert!(
            version.upgrade().is_none(),
            "turn {n}: freed once the turn ended"
        );
    }
}

/// A fork made on another coordinator — one replaced since — is never claimed, even if this one
/// has committed only build steps since its own data was loaded.
#[test]
fn a_fork_of_another_coordinator_is_never_claimed() {
    let attr = Arc::new("v".to_string());
    let (old, _, ids) = TestHost::building(&attr, &(1..=25).collect::<Vec<_>>());
    let fork = old.fork();
    set_attr(&mut fork.borrow_mut(), ids[0], &attr, Value::Int(999));
    let (loaded, _, _) = people_with(&attr, &[7]);
    let mvcc = MvccGraph::from_graph(loaded);
    assert_eq!(mvcc.claim(&fork), Err(Claim::Stale));
}
