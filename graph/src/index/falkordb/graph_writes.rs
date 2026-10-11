//! How graph writes reach the native index.
//!
//! [`Graph`] calls one function here per write it makes: which entities changed, and how. This
//! module decides what that means for the index: which `(label, attr)` columns an entity belongs
//! to, which values come out and which go in, and how a new column is filled from the live
//! graph. The graph knows nothing about columns, and the index never keeps a reference to the
//! graph.
//!
//! Each write function only reads the graph. It works out which entries come out and which go
//! in, and returns them as [`IndexChanges`]; it keeps no state. The graph's write method that
//! called it stages them for its unit of work, whose owner applies them once, where it publishes
//! RediSearch's documents (see [`pending_writes`](super::pending_writes)), so the index changes
//! once per unit of work and not once per write call.
//!
//! The graph calls each function **before** it changes anything itself, so the old values,
//! labels and types are still there to read: the B-tree is keyed by `(value, id)`, so removing an
//! entry needs its exact value. The buffer is applied after the graph writes, to the same private
//! version. A write that is refused afterwards drops the whole private version, index included,
//! and nothing was published.
//!
//! INVARIANT: every graph path that changes an indexed attribute, a node's labels, or an
//! entity's existence must call a function here. A path that skips one leaves a stale entry
//! behind, and it comes back as a wrong row once its id is reused. The edge read's endpoint
//! lookup hides only edges that are deleted and not yet reused.

use std::collections::HashMap;
use std::sync::Arc;

use atomic_refcell::AtomicRefCell;

use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;

use crate::entity_type::EntityType;
use crate::graph::graph::{Graph, LabelId, NodeId, RelationshipId, TypeId};
use crate::index::{Field, IndexType};
use crate::runtime::value::Value;

use super::encode::encode_numeric;
use super::pending_writes::IndexChanges;

// ---- Nodes ----

/// SET on existing nodes: remove each written attribute's old value and add the new one, under
/// every label the node carries.
///
/// Called before the graph overwrites the attributes, while the old value is still readable.
/// Several SETs of one attribute in a transaction reach here already collapsed to
/// `(first old, final new)`.
pub fn node_attrs_set(
    g: &Graph,
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
) -> IndexChanges {
    if !g.falkordb_index.has_columns(EntityType::Node) {
        return IndexChanges::new(EntityType::Node);
    }
    let mut changes = IndexChanges::new(EntityType::Node);
    let mut labels = Vec::new();
    for (&id, written) in attrs {
        node_label_ids_into(g, id, &mut labels);
        for (attr_id, new_value) in written {
            let Some(attr) = g.node_attr_name(*attr_id) else {
                continue;
            };
            // Hoisted out of the label loop: the old value does not depend on the label.
            let old = g.get_node_attribute_by_idx(NodeId::from(id), *attr_id);
            for label_id in &labels {
                let label = &g.get_labels()[label_id.0];
                if let Some(old) = &old {
                    changes.remove(&g.falkordb_index, label, &attr, old, id);
                }
                changes.add(&g.falkordb_index, label, &attr, new_value, id);
            }
        }
    }
    changes
}

/// The replica's CREATE_NODE and UPDATE_NODE: node `ids[i]` takes
/// `rows[i * attr_ids.len() + j]` for attribute `attr_ids[j]`, under the labels in `label_ids`.
/// As in [`node_attrs_set`], called before the overwrite.
///
/// The labels are the record's, not read from the label matrix: the record states the labels
/// the primary indexed under, and that is what this index must hold too. A created node has no
/// old values, so it only adds.
pub fn node_rows_set(
    g: &Graph,
    ids: &[u64],
    label_ids: &[u64],
    attr_ids: &[u16],
    rows: &[Value],
) -> IndexChanges {
    if !g.falkordb_index.has_columns(EntityType::Node) || attr_ids.is_empty() {
        return IndexChanges::new(EntityType::Node);
    }
    let mut changes = IndexChanges::new(EntityType::Node);
    let attrs: Vec<Option<Arc<String>>> = attr_ids.iter().map(|&a| g.node_attr_name(a)).collect();
    for (&id, row) in ids.iter().zip(rows.chunks_exact(attr_ids.len())) {
        for ((&attr_id, attr), new_value) in attr_ids.iter().zip(&attrs).zip(row) {
            let Some(attr) = attr else {
                continue;
            };
            let old = g.get_node_attribute_by_idx(NodeId::from(id), attr_id);
            for &label_id in label_ids {
                let label = &g.get_labels()[label_id as usize];
                if let Some(old) = &old {
                    changes.remove(&g.falkordb_index, label, attr, old, id);
                }
                changes.add(&g.falkordb_index, label, attr, new_value, id);
            }
        }
    }
    changes
}

/// Nodes created in this transaction: every attribute is an add, under the labels in
/// `new_labels`.
///
/// Labels come from `new_labels` and never from the label matrix. Reading the matrix would force
/// the delta merge that the graph hands the labels over to avoid.
pub fn new_node_attrs(
    g: &Graph,
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
    new_labels: &FxHashMap<u64, Vec<u64>>,
) -> IndexChanges {
    if !g.falkordb_index.has_columns(EntityType::Node) {
        return IndexChanges::new(EntityType::Node);
    }
    let mut changes = IndexChanges::new(EntityType::Node);
    for (&id, written) in attrs {
        let Some(label_ids) = new_labels.get(&id) else {
            continue; // no labels this transaction, so nothing routes to a column
        };
        for (attr_id, value) in written {
            let Some(attr) = g.node_attr_name(*attr_id) else {
                continue;
            };
            for &label_id in label_ids {
                let label = &g.get_labels()[label_id as usize];
                changes.add(&g.falkordb_index, label, &attr, value, id);
            }
        }
    }
    changes
}

/// Bulk-loaded nodes (`GRAPH.BULK`): every attribute is an add, under `label_ids`, the label set
/// the bulk token applies to every row. As in [`new_node_attrs`], the label matrix is not read.
pub fn bulk_node_attrs(
    g: &Graph,
    rows: &[(u64, Vec<(u16, Value)>)],
    label_ids: &[LabelId],
) -> IndexChanges {
    if !g.falkordb_index.has_columns(EntityType::Node) {
        return IndexChanges::new(EntityType::Node);
    }
    let mut changes = IndexChanges::new(EntityType::Node);
    for (id, written) in rows {
        for (attr_id, value) in written {
            let Some(attr) = g.node_attr_name(*attr_id) else {
                continue;
            };
            for label_id in label_ids {
                let label = &g.get_labels()[label_id.0];
                changes.add(&g.falkordb_index, label, &attr, value, *id);
            }
        }
    }
    changes
}

/// SET `:Label`: each `(label_rows[i], label_cols[i])` node gains a label, so its current
/// attributes go into that label's columns.
///
/// A node created in this transaction has no attributes yet (they are imported after its
/// labels), so it adds nothing here. [`new_node_attrs`] indexes it.
pub fn labels_added(
    g: &Graph,
    label_rows: &[u64],
    label_cols: &[u64],
) -> IndexChanges {
    if !g.falkordb_index.has_columns(EntityType::Node) {
        return IndexChanges::new(EntityType::Node);
    }
    let mut changes = IndexChanges::new(EntityType::Node);
    for (&id, &label_id) in label_rows.iter().zip(label_cols) {
        add_node_under_label(&mut changes, g, id, label_id);
    }
    changes
}

/// The replica's label add: every node in `ids` gains every label in `label_ids`. As
/// [`labels_added`], with the pairs left as a product.
pub fn labels_product_added(
    g: &Graph,
    ids: &[u64],
    label_ids: &[u64],
) -> IndexChanges {
    if !g.falkordb_index.has_columns(EntityType::Node) {
        return IndexChanges::new(EntityType::Node);
    }
    let mut changes = IndexChanges::new(EntityType::Node);
    for &label_id in label_ids {
        for &id in ids {
            add_node_under_label(&mut changes, g, id, label_id);
        }
    }
    changes
}

/// REMOVE `:Label`: each `(label_rows[i], label_cols[i])` node loses a label, so its attributes
/// come out of that label's columns. Without this the entries orphan, and come back as wrong
/// rows when the id is reused.
pub fn labels_removed(
    g: &Graph,
    label_rows: &[u64],
    label_cols: &[u64],
) -> IndexChanges {
    if !g.falkordb_index.has_columns(EntityType::Node) {
        return IndexChanges::new(EntityType::Node);
    }
    let mut changes = IndexChanges::new(EntityType::Node);
    for (&id, &label_id) in label_rows.iter().zip(label_cols) {
        remove_node_under_label(&mut changes, g, id, label_id);
    }
    changes
}

/// DELETE: remove every indexed value of each deleted node. Called while its labels and
/// attributes are still there.
pub fn nodes_deleted(
    g: &Graph,
    ids: &RoaringTreemap,
) -> IndexChanges {
    if !g.falkordb_index.has_columns(EntityType::Node) {
        return IndexChanges::new(EntityType::Node);
    }
    let mut changes = IndexChanges::new(EntityType::Node);
    let mut labels = Vec::new();
    for id in ids {
        node_label_ids_into(g, id, &mut labels);
        for (attr, value) in g.get_node_all_attrs(NodeId::from(id)) {
            for label_id in &labels {
                let label = &g.get_labels()[label_id.0];
                changes.remove(&g.falkordb_index, label, &attr, &value, id);
            }
        }
    }
    changes
}

// ---- Relationships ----
//
// An edge column stores `(value, edge_id)`. The read recovers `(src, dst)` from the graph's own
// `edge_id → (src, dst)` reverse index, so nothing here tracks endpoints. An edge has exactly one
// type, fixed at creation, so unlike a node's labels there is no type-change path. Routing is a
// single `(type, attr)` lookup.

/// SET on existing edges: remove each written attribute's old value and add the new one. Called
/// before the graph overwrites the attributes, as for [`node_attrs_set`].
pub fn edge_attrs_set(
    g: &Graph,
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
) -> IndexChanges {
    edge_attrs_set_with(g, attrs, None)
}

/// [`edge_attrs_set`] for edges all of type `type_id`: the replica's UPDATE_EDGE, where one
/// record is one type. Taking the type saves a type-matrix scan per edge.
pub fn edge_attrs_set_of_type(
    g: &Graph,
    type_id: TypeId,
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
) -> IndexChanges {
    edge_attrs_set_with(g, attrs, Some(type_id))
}

/// `type_id` is the edges' shared type when the caller knows it; `None` looks it up per edge.
fn edge_attrs_set_with(
    g: &Graph,
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
    type_id: Option<TypeId>,
) -> IndexChanges {
    if !g.falkordb_index.has_columns(EntityType::Relationship) {
        return IndexChanges::new(EntityType::Relationship);
    }
    let mut changes = IndexChanges::new(EntityType::Relationship);
    for (&id, written) in attrs {
        if written.is_empty() {
            continue;
        }
        let type_name = match type_id {
            Some(type_id) => &g.get_types()[type_id.0],
            None => edge_type(g, id),
        };
        for (attr_id, new_value) in written {
            let Some(attr) = g.rel_attr_name(*attr_id) else {
                continue;
            };
            if let Some(old) =
                g.get_relationship_attribute_by_idx(RelationshipId::from(id), *attr_id)
            {
                changes.remove(&g.falkordb_index, type_name, &attr, &old, id);
            }
            changes.add(&g.falkordb_index, type_name, &attr, new_value, id);
        }
    }
    changes
}

/// Edges created in this transaction: every attribute is an add.
pub fn new_edge_attrs(
    g: &Graph,
    attrs: &FxHashMap<u64, Vec<(u16, Value)>>,
) -> IndexChanges {
    if !g.falkordb_index.has_columns(EntityType::Relationship) {
        return IndexChanges::new(EntityType::Relationship);
    }
    let mut changes = IndexChanges::new(EntityType::Relationship);
    for (&id, written) in attrs {
        if written.is_empty() {
            continue;
        }
        let type_name = edge_type(g, id);
        for (attr_id, value) in written {
            if let Some(attr) = g.rel_attr_name(*attr_id) {
                changes.add(&g.falkordb_index, type_name, &attr, value, id);
            }
        }
    }
    changes
}

/// Bulk-loaded edges (`GRAPH.BULK`), all of type `type_id`: every attribute is an add.
///
/// A bulk token carries one type for all its rows, so this takes it rather than looking it up
/// per edge. Each lookup scans the type matrix, and that scan waits on the pending delta.
pub fn bulk_edge_attrs(
    g: &Graph,
    rows: &[(u64, Vec<(u16, Value)>)],
    type_id: TypeId,
) -> IndexChanges {
    if !g.falkordb_index.has_columns(EntityType::Relationship) {
        return IndexChanges::new(EntityType::Relationship);
    }
    let mut changes = IndexChanges::new(EntityType::Relationship);
    let type_name = &g.get_types()[type_id.0];
    for (id, written) in rows {
        for (attr_id, value) in written {
            if let Some(attr) = g.rel_attr_name(*attr_id) {
                changes.add(&g.falkordb_index, type_name, &attr, value, *id);
            }
        }
    }
    changes
}

/// DELETE, explicit or cascading from a deleted node: remove every indexed value of each edge.
/// Called while its attributes and type are still there.
pub fn edges_deleted(
    g: &Graph,
    ids: impl IntoIterator<Item = u64>,
) -> IndexChanges {
    if !g.falkordb_index.has_columns(EntityType::Relationship) {
        return IndexChanges::new(EntityType::Relationship);
    }
    let mut changes = IndexChanges::new(EntityType::Relationship);
    for id in ids {
        let mut attrs = g
            .get_relationship_all_attrs(RelationshipId::from(id))
            .peekable();
        if attrs.peek().is_none() {
            continue;
        }
        let type_name = edge_type(g, id);
        for (attr, value) in attrs {
            changes.remove(&g.falkordb_index, type_name, &attr, &value, id);
        }
    }
    changes
}

// ---- Index DDL ----

/// CREATE INDEX: mark each attribute's column `Building` and return. Only a Range index has
/// native columns; any other type does nothing.
///
/// Nothing is scanned here. The pre-existing entities (BASE) are read from a committed snapshot
/// off the write thread by the module's build controller, which installs them through
/// [`install_base`] and flips the column `Ready`. Until then reads on the column fall back to a
/// scan, and live writes maintain it as usual: the column's own tree is the DELTA, and removes
/// also land in its TOMB. The RDB-load path ([`populate`]) still builds synchronously.
///
/// The build never goes through the RediSearch populate, which reaches the graph through the
/// `Indexer`'s back-pointer. This index is part of the graph version, and keeping that reference
/// out is the point.
pub fn create_index(
    g: &mut Graph,
    index_type: &IndexType,
    entity: EntityType,
    label: &Arc<String>,
    attrs: &[Arc<String>],
) {
    if *index_type == IndexType::Range {
        for attr in attrs {
            g.falkordb_index.create_building(entity, label, attr);
        }
    }
}

// ---- Background (online) build ----
//
// The seam the module's build controller drives. A job reads BASE from a committed snapshot with
// `base_encoded`, off the write thread, then installs it in one commit with `install_base`.

/// Every column currently `Building`, as `(entity, label, attr, epoch)`: the controller's work
/// list, read from a committed snapshot.
#[must_use]
pub fn building_columns(g: &Graph) -> Vec<(EntityType, Arc<String>, Arc<String>, u64)> {
    g.falkordb_index.building_columns()
}

/// BASE for one building column, read from `g` in one pass, encoded, sorted and unique: what
/// the install needs. For a caller that already keeps `g` from being written while it reads; the
/// background build reads a published version and uses [`base_encoded_in_chunks`] instead.
///
/// `None` if the column no longer exists (dropped mid-build).
#[must_use]
pub fn base_encoded(
    g: &Graph,
    entity: EntityType,
    label: &Arc<String>,
    attr: &Arc<String>,
) -> Option<Vec<(u64, u64)>> {
    let column = g.falkordb_index.column(entity, label, attr)?;
    Some(encoded_entries(g, entity, label, attr, |value| {
        column.encode_value(value)
    }))
}

/// Node-id rows a background BASE scan reads per chunk, i.e. per hold of the graph's read lock.
/// A writer waiting for the lock waits at most one chunk. The size is a guess, not a
/// measurement.
pub const BASE_SCAN_CHUNK_ROWS: u64 = 1 << 16;

/// [`base_encoded`] for a **published** version, read off the write thread: everything the
/// install needs, computed so the install pays neither the scan, the encode nor the sort while
/// it holds the write lock and the GIL.
///
/// `snapshot` is pinned for the whole scan, so every chunk reads the same version. It is read
/// in chunks of `chunk_rows` node-id rows, and each chunk borrows it only while the guard
/// `hold()` returns is alive. The caller passes the graph's read lock: a published version is
/// still borrowed mutably by the write path in one place (`Pending::resync_published_indexes`,
/// which counts on the write lock to keep readers out), and `AtomicRefCell` reports a borrow
/// that collides with it by panicking. Taking the lock per chunk keeps the scan from colliding,
/// without holding writers off for the whole scan.
///
/// `None` if the column does not exist in `snapshot` (dropped before the job started).
#[must_use]
pub fn base_encoded_in_chunks<G>(
    snapshot: &AtomicRefCell<Graph>,
    entity: EntityType,
    label: &Arc<String>,
    attr: &Arc<String>,
    chunk_rows: u64,
    mut hold: impl FnMut() -> G,
) -> Option<Vec<(u64, u64)>> {
    let mut pairs = Vec::new();
    let mut lo = 0u64;
    loop {
        // Declared before the borrow, so the borrow ends before the lock is released.
        let _guard = hold();
        let g = snapshot.borrow();
        let column = g.falkordb_index.column(entity, label, attr)?;
        // Every node id, and so every edge's source row, is below the id space's bound.
        let bound = g.node_id_space().bound();
        if lo >= bound {
            break;
        }
        let hi = lo.saturating_add(chunk_rows).min(bound) - 1;
        scan_rows(
            &g,
            entity,
            label,
            attr,
            (lo, hi),
            |value| column.encode_value(value),
            &mut pairs,
        );
        lo = hi + 1;
    }
    pairs.sort_unstable();
    pairs.dedup();
    Some(pairs)
}

/// Install a background-built BASE and flip the column `Ready`, in one commit. Returns whether
/// it installed: `false` means the column is gone, already `Ready`, or on a different epoch
/// (dropped and re-created since the job started), and the caller should not commit.
///
/// The free-id filter is a cheap backstop, not the real defence: a freed id is no longer free
/// once it is reused, so an id recycled during the build passes it. TOMB is what makes the stale
/// BASE safe; this only drops the easy cases before the tree build.
pub fn install_base(
    g: &mut Graph,
    entity: EntityType,
    label: &Arc<String>,
    attr: &Arc<String>,
    epoch: u64,
    base: Vec<(u64, u64)>,
) -> bool {
    let survivors: Vec<(u64, u64)> = match entity {
        EntityType::Node => base
            .into_iter()
            .filter(|(_, id)| !g.is_node_deleted(NodeId::from(*id)))
            .collect(),
        EntityType::Relationship => base
            .into_iter()
            .filter(|(_, id)| !g.is_relationship_deleted(RelationshipId::from(*id)))
            .collect(),
    };
    g.falkordb_index
        .install_base(entity, label, attr, epoch, survivors)
}

/// Index population after an RDB load: build every column that a population snapshot of
/// `label` covers. `fields` maps each attribute to its index fields, and an attribute has a
/// native column when one of its fields is a Range field.
pub fn populate(
    g: &mut Graph,
    entity: EntityType,
    label: &Arc<String>,
    fields: &HashMap<Arc<String>, Vec<Arc<Field>>>,
) {
    for (attr, attr_fields) in fields {
        if attr_fields.iter().any(|f| f.ty == IndexType::Range) {
            build_column(g, entity, label, attr);
        }
    }
}

// ---- Shared ----

/// Every indexed value node `id` carries, added under the one label `label_id`. Label add and
/// remove change only that label's columns.
fn add_node_under_label(
    changes: &mut IndexChanges,
    g: &Graph,
    id: u64,
    label_id: u64,
) {
    let label = &g.get_labels()[label_id as usize];
    for (attr, value) in g.get_node_all_attrs(NodeId::from(id)) {
        changes.add(&g.falkordb_index, label, &attr, &value, id);
    }
}

/// As [`add_node_under_label`], for removal.
fn remove_node_under_label(
    changes: &mut IndexChanges,
    g: &Graph,
    id: u64,
    label_id: u64,
) {
    let label = &g.get_labels()[label_id as usize];
    for (attr, value) in g.get_node_all_attrs(NodeId::from(id)) {
        changes.remove(&g.falkordb_index, label, &attr, &value, id);
    }
}

/// Build the column for `(entity, label, attr)` from the live graph, replacing any existing one.
///
/// One attribute at a time: a column's tuples live only until it is built. Collecting every
/// attribute first would hold one tuple per entity per attribute at once, multiplying the peak
/// for no benefit.
fn build_column(
    g: &mut Graph,
    entity: EntityType,
    label: &Arc<String>,
    attr: &Arc<String>,
) {
    let pairs = encoded_entries(g, entity, label, attr, encode_numeric);
    g.falkordb_index
        .build_numeric_sorted(entity, label, attr, pairs);
}

/// The node's label ids, reusing `out`'s allocation across nodes.
///
/// Exists so a call resolves labels **once per node**. [`Graph::get_node_label_ids`] walks the
/// label matrix, whose `wait` forces a pending-delta merge that costs O(accumulated delta), and a
/// node's labels do not depend on which attribute is being written. Same defect, and same fix,
/// as the edge path in #2344. A function that is handed the labels must not call this.
fn node_label_ids_into(
    g: &Graph,
    id: u64,
    out: &mut Vec<LabelId>,
) {
    out.clear();
    out.extend(g.get_node_label_ids(NodeId::from(id)));
}

/// The type of edge `id`, looked up once per edge rather than once per attribute.
///
/// [`Graph::get_relationship_type_id`] `expect`s the edge to be in the type matrix. Every caller
/// upholds that: create and SET touch a live edge, and the delete paths call in before the graph
/// tears the type matrix down. A panic here is a tripwire for a caller that breaks the order.
/// That beats skipping maintenance silently and leaving a stale entry.
fn edge_type(
    g: &Graph,
    id: u64,
) -> &Arc<String> {
    &g.get_types()[g.get_relationship_type_id(RelationshipId::from(id)).0]
}

/// The tuples of column `(entity, label, attr)` as the live graph holds them, encoded with
/// `encode`, sorted and unique: one `(key, id)` for every live `label` node (or `label`-typed
/// edge) whose `attr` value the column indexes.
///
/// Each value is encoded as it is read and dropped straight away, so the scan never holds the
/// entities' `Value`s, only the encoded tuples. A value the column does not index (a string on a
/// Range index) costs nothing past the read.
fn encoded_entries(
    g: &Graph,
    entity: EntityType,
    label: &Arc<String>,
    attr: &Arc<String>,
    encode: impl Fn(&Value) -> Option<u64>,
) -> Vec<(u64, u64)> {
    let mut pairs = Vec::new();
    scan_rows(g, entity, label, attr, (0, u64::MAX), encode, &mut pairs);
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// Append to `out` the encoded tuples of column `(entity, label, attr)` whose node-id row lies
/// in `rows` (inclusive): a node's own id, or an edge's source node id. Unsorted.
fn scan_rows(
    g: &Graph,
    entity: EntityType,
    label: &Arc<String>,
    attr: &Arc<String>,
    (lo, hi): (u64, u64),
    encode: impl Fn(&Value) -> Option<u64>,
    out: &mut Vec<(u64, u64)>,
) {
    match entity {
        EntityType::Node => {
            if let (Some(lm), Some(idx)) =
                (g.get_label_matrix(label), g.get_node_attribute_id(attr))
            {
                let idx = idx as u16;
                for (n, _) in lm.iter(lo, hi) {
                    if let Some(key) = g
                        .get_node_attribute_by_idx(NodeId::from(n), idx)
                        .and_then(|value| encode(&value))
                    {
                        out.push((key, n));
                    }
                }
            }
        }
        EntityType::Relationship => {
            if let (Some(t), Some(idx)) = (
                g.get_relationship_matrix(label),
                g.get_relationship_attribute_id(attr),
            ) {
                let idx = idx as u16;
                for (_, _, eid) in t.iter(lo, hi, false) {
                    if let Some(key) = g
                        .get_relationship_attribute_by_idx(RelationshipId::from(eid), idx)
                        .and_then(|value| encode(&value))
                    {
                        out.push((key, eid));
                    }
                }
            }
        }
    }
}
