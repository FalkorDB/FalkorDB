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
use std::sync::{Arc, OnceLock};

use roaring::RoaringTreemap;
use rustc_hash::FxHashMap;

use crate::entity_type::EntityType;
use crate::graph::graph::{Graph, LabelId, NodeId, RelationshipId, TypeId};
use crate::index::{Field, IndexType};
use crate::runtime::value::Value;

use super::falkordb_index::{BuildCursor, ColumnState, IndexColumn, IndexKey, IndexerBatch};
use super::numeric::EncodedTuples;
use super::pending_writes::IndexChanges;

/// Measurement gate: `true` when `FALKORDB_INDEX_ONLY=1`. The native index then maintains node
/// Range indexes alone, and the graph skips the redundant RediSearch feed on the write path.
/// Read once and cached.
///
/// This exists only to benchmark the native write cost against RediSearch without the double
/// write. It is not a shipping mode: it drops the string and geo entries RediSearch still
/// serves.
pub fn index_only_writes() -> bool {
    static INDEX_ONLY: OnceLock<bool> = OnceLock::new();
    *INDEX_ONLY.get_or_init(|| {
        std::env::var("FALKORDB_INDEX_ONLY")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

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

/// CREATE INDEX: build each attribute's column from the live graph, so the index serves reads as
/// soon as CREATE INDEX returns. Only a Range index has native columns; any other type does
/// nothing.
///
/// Each column is created `Building` and filled by one indexer batch with no limit on its size
/// ([`read_indexer_batch_of`]), which reaches the end and flips it `Ready`. A batch into an empty
/// column bulk-builds its tree, so this costs what building the column in one pass does.
///
/// Built here on the write thread, with the graph borrowed. It is never built through the
/// RediSearch populate, which reaches the graph through the `Indexer`'s back-pointer. This index
/// is part of the graph version, and keeping that reference out is the point.
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
            g.index_batch(entity, &(label.clone(), attr.clone()), usize::MAX);
        }
    }
}

// ---- Indexer batches ----
//
// A `Building` column is filled by indexer batches. Each reads the label's next entities from the
// cursor, at their values in a version, and adds them to the column in that same version. See
// `ColumnState`.

/// One indexer batch: index the next entities of one `Building` column, at most `entities` of
/// them, at their current values, and move its cursor past them, or flip the column `Ready` if the
/// batch reached the end. Returns whether a column is still `Building` afterwards, i.e. whether to
/// run another batch. Tests pass small batches to force cursors through every boundary.
#[cfg(test)]
pub(crate) fn run_indexer_batch(
    g: &mut Graph,
    entities: usize,
) -> bool {
    g.index_next_batch(entities);
    g.falkordb_index.has_building_columns()
}

/// [`read_indexer_batch_of`] of the first `Building` column. For tests of one column.
#[cfg(test)]
pub(crate) fn read_indexer_batch(
    g: &Graph,
    entities: usize,
) -> Option<IndexerBatch> {
    let (entity, key, from) = g.falkordb_index.next_building()?;
    Some(read_batch(g, entity, key, from, entities))
}

/// Read an indexer batch of column `key` of `entity` from `g`: its next `entities` nodes or edges,
/// at their values in `g`, encoded and sorted. `None` if the column is not `Building`. The batch
/// is added to `g` itself (`Graph::index_batch`).
pub(crate) fn read_indexer_batch_of(
    g: &Graph,
    entity: EntityType,
    key: &IndexKey,
    entities: usize,
) -> Option<IndexerBatch> {
    let ColumnState::Building { next } = g.falkordb_index.column_state(entity, &key.0, &key.1)?
    else {
        return None;
    };
    Some(read_batch(g, entity, key.clone(), next, entities))
}

/// The batch of column `key` of `entity` from cursor `from` on.
fn read_batch(
    g: &Graph,
    entity: EntityType,
    key: IndexKey,
    from: BuildCursor,
    entities: usize,
) -> IndexerBatch {
    let column = g
        .falkordb_index
        .column(entity, &key.0, &key.1)
        .expect("a Building column exists");
    let (tuples, next) = scan(g, &key.0, &key.1, from, entities, column);
    IndexerBatch {
        entity,
        key,
        tuples,
        next,
    }
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
/// One attribute at a time: the `(value, id)` pairs live only until their column is built.
/// Collecting every attribute first would hold one `Value` per entity per attribute at once,
/// multiplying the peak for no benefit. The build steps' tests use it as their reference: it
/// shares no code with them.
pub(crate) fn build_column(
    g: &mut Graph,
    entity: EntityType,
    label: &Arc<String>,
    attr: &Arc<String>,
) {
    let pairs = match entity {
        EntityType::Node => collect_node_entries(g, label, attr),
        EntityType::Relationship => collect_edge_entries(g, label, attr),
    };
    g.falkordb_index
        .build_numeric(entity, label, attr, pairs.iter().map(|(v, id)| (v, *id)));
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

/// The `(value, node_id)` entries of column `(label, attr)`: every live `label` node that has
/// `attr`.
fn collect_node_entries(
    g: &Graph,
    label: &Arc<String>,
    attr: &Arc<String>,
) -> Vec<(Value, u64)> {
    let mut pairs = Vec::new();
    if let (Some(lm), Some(idx)) = (g.get_label_matrix(label), g.get_node_attribute_id(attr)) {
        let idx = idx as u16;
        for (n, _) in lm.iter(0, u64::MAX) {
            if let Some(value) = g.get_node_attribute_by_idx(NodeId::from(n), idx) {
                pairs.push((value, n));
            }
        }
    }
    pairs
}

/// The `(value, edge_id)` entries of column `(type_name, attr)`: every live edge of that type
/// that has `attr`.
fn collect_edge_entries(
    g: &Graph,
    type_name: &Arc<String>,
    attr: &Arc<String>,
) -> Vec<(Value, u64)> {
    let Some(idx) = g.get_relationship_attribute_id(attr) else {
        return Vec::new();
    };
    let idx = idx as u16;
    let edge_ids: Vec<u64> = g
        .get_relationship_matrix(type_name)
        .map_or_else(Vec::new, |t| {
            t.iter(0, u64::MAX, false).map(|(_, _, eid)| eid).collect()
        });
    edge_ids
        .into_iter()
        .filter_map(|eid| {
            g.get_relationship_attribute_by_idx(RelationshipId::from(eid), idx)
                .map(|value| (value, eid))
        })
        .collect()
}

/// The tuples of `column` for up to `entities` `label` nodes, or `label`-typed edges, from `from`
/// on, as `g` holds them: sorted and unique, the tuples each `attr` value takes in the column
/// ([`IndexColumn::push_encoded`]). Also returns where the next batch resumes, or `None` when the
/// scan reached the end.
///
/// Each value is encoded as it is read and dropped straight away, so the scan never holds the
/// entities' `Value`s, only the encoded tuples. A value the column does not index (a string on a
/// Range index) costs nothing past the read.
fn scan(
    g: &Graph,
    label: &Arc<String>,
    attr: &Arc<String>,
    from: BuildCursor,
    entities: usize,
    column: &IndexColumn,
) -> (EncodedTuples, Option<BuildCursor>) {
    let mut tuples = EncodedTuples::default();
    let next = match from {
        BuildCursor::Node(from) => {
            scan_nodes(g, label, attr, from, entities, column, &mut tuples).map(BuildCursor::Node)
        }
        BuildCursor::Edge { src, dst } => {
            scan_edges(g, label, attr, (src, dst), entities, column, &mut tuples)
                .map(|(src, dst)| BuildCursor::Edge { src, dst })
        }
    };
    tuples.sort_dedup();
    (tuples, next)
}

/// Append to `out` the tuples of node column `(label, attr)` for up to `budget` `label` nodes from
/// node `from` on, in id order: what each node's current `attr` value takes in `column`. Unsorted.
/// Returns the node to resume from, or `None` when no `label` node is left.
fn scan_nodes(
    g: &Graph,
    label: &Arc<String>,
    attr: &Arc<String>,
    from: NodeId,
    budget: usize,
    column: &IndexColumn,
    out: &mut EncodedTuples,
) -> Option<NodeId> {
    // No such label, or no entity ever had the attribute: nothing to index. A later write that
    // gives one the attribute is indexed by the write path.
    let (Some(lm), Some(idx)) = (g.get_label_matrix(label), g.get_node_attribute_id(attr)) else {
        return None;
    };
    let idx = idx as u16;
    let mut nodes = lm
        .iter(u64::from(from), u64::MAX)
        .map(|(n, _)| n)
        .peekable();
    for n in nodes.by_ref().take(budget) {
        if let Some(value) = g.get_node_attribute_by_idx(NodeId::from(n), idx) {
            column.push_encoded(&value, n, out);
        }
    }
    nodes.peek().copied().map(NodeId::from)
}

/// The edge counterpart of [`scan_nodes`]: up to `budget` `label`-typed edges from pair `from` on,
/// in `(src, dst)` order, plus the rest of the multi-edge pair the batch stops in, so no pair is
/// ever split between steps. Edges of `from`'s source with a destination below `from`'s are
/// skipped: an earlier step indexed them. Returns the pair to resume from, or `None` when no edge
/// is left.
///
/// Resuming partway through a source's edges re-walks its earlier destinations, as C's populate
/// does, so a step that resumes inside a source with very many edges pays for the part already
/// done.
fn scan_edges(
    g: &Graph,
    label: &Arc<String>,
    attr: &Arc<String>,
    (from_src, from_dst): (NodeId, NodeId),
    budget: usize,
    column: &IndexColumn,
    out: &mut EncodedTuples,
) -> Option<(NodeId, NodeId)> {
    let (Some(t), Some(idx)) = (
        g.get_relationship_matrix(label),
        g.get_relationship_attribute_id(attr),
    ) else {
        return None;
    };
    let idx = idx as u16;
    let mut index = |eid: u64| {
        if let Some(value) = g.get_relationship_attribute_by_idx(RelationshipId::from(eid), idx) {
            column.push_encoded(&value, eid, out);
        }
    };
    let (from_src, from_dst) = (u64::from(from_src), u64::from(from_dst));
    let mut edges = t.iter(from_src, u64::MAX, false).peekable();
    while edges
        .next_if(|&(src, dst, _)| src == from_src && dst < from_dst)
        .is_some()
    {}
    let mut read = 0usize;
    while let Some((src, dst, eid)) = edges.next() {
        index(eid);
        read += 1;
        if read >= budget {
            while let Some((_, _, eid)) = edges.next_if(|&(s, d, _)| s == src && d == dst) {
                index(eid);
            }
            break;
        }
    }
    edges
        .peek()
        .map(|&(src, dst, _)| (NodeId::from(src), NodeId::from(dst)))
}
