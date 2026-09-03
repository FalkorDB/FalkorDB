//! Deferred write operations for transactional semantics.
//!
//! This module provides [`Pending`], which batches write operations during
//! query execution. This enables:
//!
//! - Read-your-writes within a query (created nodes visible to later clauses)
//! - Atomic commit/rollback of all changes
//! - Efficient bulk updates to indexes
//!
//! ## Batched Operations
//!
//! - `created_nodes`: Nodes created in this query
//! - `deleted_nodes`: Nodes marked for deletion
//! - `created_rels_by_type`: Edges created in this query, grouped by type
//! - `deleted_relationships`: Edges marked for deletion
//! - `set_*_attrs`: Property updates by entity ID
//! - `set/remove_node_labels`: Label changes
//!
//! ## Commit Flow
//!
//! ```text
//! Query execution → accumulate in Pending → apply_all() → update Graph
//! ```
//!
//! On error or ROLLBACK, the Pending is simply dropped without applying.

use std::{cell::RefCell, sync::Arc};

use rustc_hash::FxHashMap;

use atomic_refcell::AtomicRefCell;
use roaring::RoaringTreemap;

use crate::graph::graph::{DeletedEdge, DeletedNodeLabel};

use crate::{
    entity_type::EntityType,
    graph::{
        constraint::{ConstraintStatus, ConstraintType},
        graph::{Graph, LabelId, NodeId, RelationshipId},
    },
    runtime::{ordermap::OrderMap, orderset::OrderSet, runtime::QueryStatistics, value::Value},
};

/// Flatten a node_id → [label_ids] map into parallel (rows, cols) arrays.
fn flatten_label_map(map: &FxHashMap<u64, Vec<u64>>) -> (Vec<u64>, Vec<u64>) {
    let total: usize = map.values().map(std::vec::Vec::len).sum();
    let mut rows = Vec::with_capacity(total);
    let mut cols = Vec::with_capacity(total);
    for (&node_id, label_ids) in map {
        for &label_id in label_ids {
            rows.push(node_id);
            cols.push(label_id);
        }
    }
    (rows, cols)
}

const INVALID_PROPERTY_MSG: &str =
    "Property values can only be of primitive types or arrays of primitive types";

fn is_valid_property(
    value: &Value,
    allow_null: bool,
) -> bool {
    match value {
        Value::Null => allow_null,
        Value::Bool(_)
        | Value::Int(_)
        | Value::Float(_)
        | Value::String(_)
        | Value::Point(_)
        | Value::VecF32(_)
        | Value::Datetime(_)
        | Value::Date(_)
        | Value::Time(_)
        | Value::Duration(_) => true,
        Value::List(items) => items.iter().all(|v| is_valid_property(v, false)),
        _ => false,
    }
}

/// Validate that a value is a valid node property type.
fn validate_node_property(value: &Value) -> Result<(), String> {
    if !is_valid_property(value, true) {
        return Err(INVALID_PROPERTY_MSG.into());
    }
    Ok(())
}

/// Validate that a value is a valid relationship property type.
fn validate_relationship_property(value: &Value) -> Result<(), String> {
    if !is_valid_property(value, true) {
        return Err(INVALID_PROPERTY_MSG.into());
    }
    Ok(())
}

/// Binary-search a sorted `(attr_id, Value)` slice.
fn lookup_sorted(
    attrs: &[(u16, Value)],
    attr_id: u16,
) -> Option<&Value> {
    attrs
        .binary_search_by_key(&attr_id, |(k, _)| *k)
        .ok()
        .map(|pos| &attrs[pos].1)
}

/// Accumulated write operations for deferred application.
///
/// All mutations during query execution are collected here and applied
/// atomically at the end. This enables transactional semantics.
/// All mutations during query execution, applied atomically at the end.
///
/// The fields are `pub(crate)` rather than hidden behind per-format view
/// structs. Those views were a projection — every field a reference to the same
/// type — so they documented the dependency surface and did nothing else, while
/// costing a struct and a constructor per wire format. What states that surface
/// now is `effects_v3_emit::digest`, whose return type says precisely what
/// replication makes of this.
pub struct Pending {
    /// Nodes created in this transaction
    pub(crate) created_nodes: RoaringTreemap,
    /// Relationships created, grouped by type: type_name → [(rel_id, from, to)]
    pub(crate) created_rels_by_type: FxHashMap<Arc<String>, Vec<(RelationshipId, NodeId, NodeId)>>,
    /// Reverse index: rel_id → type_name for O(1) existence/type lookups
    pub(crate) created_rel_types: FxHashMap<RelationshipId, Arc<String>>,
    /// Nodes to be deleted
    pub(crate) deleted_nodes: RoaringTreemap,
    /// Relationships to be deleted
    pub(crate) deleted_relationships: RoaringTreemap,
    /// Endpoints for deleted relationships — populated by commit(), used by build_effects_buffer().
    pub(crate) deleted_endpoints: Vec<DeletedEdge>,
    /// `(node_id, label_id)` for every deleted node that carried a label —
    /// populated by commit(), used by the effects emitter.
    ///
    /// Captured here because `delete_nodes` clears the label matrices, so by
    /// the time effects are encoded the labels are unrecoverable; and a label
    /// set is what tells a replica which label-scoped indexes to clear. Stored
    /// as flat pairs rather than grouped: `delete_nodes` already built this
    /// exact vector, so keeping it costs a move, while grouping would charge
    /// every delete for a partitioning only a replicating server reads.
    pub(crate) deleted_node_labels: Vec<DeletedNodeLabel>,
    /// Property updates for newly created nodes (fast path: skip fjall).
    /// Values are attribute-id-resolved, sorted by id, unique.
    pub(crate) new_nodes_attrs: FxHashMap<u64, Vec<(u16, Value)>>,
    /// Property updates for existing nodes (full merge path)
    pub(crate) existing_nodes_attrs: FxHashMap<u64, Vec<(u16, Value)>>,
    /// Property updates for newly created relationships (fast path)
    pub(crate) new_relationships_attrs: FxHashMap<u64, Vec<(u16, Value)>>,
    /// Property updates for existing relationships (full merge path)
    pub(crate) existing_relationships_attrs: FxHashMap<u64, Vec<(u16, Value)>>,
    /// Labels to add: node_id → [label_ids]
    pub(crate) set_labels: FxHashMap<u64, Vec<u64>>,
    /// Labels to remove: node_id → [label_ids]
    pub(crate) remove_labels: FxHashMap<u64, Vec<u64>>,
    /// Index documents this `Commit` produced.
    pub(crate) index_docs: IndexDocs,
    /// Index documents accumulated across the query's commits, applied only
    /// once the whole query succeeds so a failed one never leaves stale
    /// entries in RediSearch.
    pub(crate) deferred_docs: IndexDocs,
    /// Union of every index document this query has published, accumulated across
    /// all of its `Commit`s, so a later failure can resync them against committed
    /// state (see [`Self::resync_published_indexes`]).
    ///
    /// Deliberately **not** reset by [`Self::clear`], which runs after every
    /// `Commit`: the undo has to cover the whole query, not just the last `Commit`.
    /// Per-query state — `Pending` belongs to one `Runtime`.
    published: IndexDocs,
    /// Schema baseline: number of labels when the current commit window started.
    pub(crate) schema_label_count: usize,
    /// Schema baseline: number of relationship types when the current commit window started.
    pub(crate) schema_rel_type_count: usize,
    /// Schema baseline: number of node attribute names when the current commit window started.
    pub(crate) schema_node_attr_count: usize,
    /// Schema baseline: number of relationship attribute names when the current commit window started.
    pub(crate) schema_rel_attr_count: usize,
}

/// Edge documents to remove, keyed by relationship type: `type_id -> { edge_id
/// -> (src, dst) }`.
///
/// The endpoints ride along because they are captured at deletion time — the
/// edge is gone from the tensor by the time the 24-byte RediSearch key has to
/// be rebuilt, so they cannot be looked up again.
pub type EdgeDocRemovals = FxHashMap<u64, FxHashMap<u64, (u64, u64)>>;

/// The index documents a unit of work produced, in both directions and for
/// both entity kinds.
///
/// One type rather than four parallel maps, and used for both the per-commit
/// set and the deferred one it folds into. They have to stay in
/// correspondence — a document added to the wrong half is a stale RediSearch
/// entry that nothing later notices — and four loose fields on `Pending` made
/// that the caller's job at every site.
#[derive(Default)]
pub struct IndexDocs {
    /// Node documents to add, keyed by label id.
    pub node_adds: FxHashMap<u64, RoaringTreemap>,
    /// Node documents to remove, keyed by label id.
    pub node_removes: FxHashMap<u64, RoaringTreemap>,
    /// Edge documents to add, keyed by relationship type id.
    pub edge_adds: FxHashMap<u64, RoaringTreemap>,
    /// Edge documents to remove — see [`EdgeDocRemovals`].
    pub edge_removes: EdgeDocRemovals,
}

impl IndexDocs {
    /// True when nothing was produced, so a caller can skip the work of
    /// publishing an empty set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.node_adds.is_empty()
            && self.node_removes.is_empty()
            && self.edge_adds.is_empty()
            && self.edge_removes.is_empty()
    }

    /// Fold `other` in, leaving it empty.
    ///
    /// Replaces four hand-written loops that folded a commit's documents into
    /// the query's deferred set — one per map, each a chance to fold the wrong
    /// pair together.
    pub fn absorb(
        &mut self,
        other: &mut Self,
    ) {
        for (slot, ids) in other.node_adds.drain() {
            *self.node_adds.entry(slot).or_default() |= ids;
        }
        for (slot, ids) in other.node_removes.drain() {
            *self.node_removes.entry(slot).or_default() |= ids;
        }
        for (slot, ids) in other.edge_adds.drain() {
            *self.edge_adds.entry(slot).or_default() |= ids;
        }
        for (slot, ids) in other.edge_removes.drain() {
            self.edge_removes.entry(slot).or_default().extend(ids);
        }
    }

    /// Fold `other` in, for accumulating everything one query published.
    fn merge(
        &mut self,
        other: &Self,
    ) {
        for (slot, ids) in &other.node_adds {
            *self.node_adds.entry(*slot).or_default() |= ids;
        }
        for (slot, ids) in &other.node_removes {
            *self.node_removes.entry(*slot).or_default() |= ids;
        }
        for (slot, ids) in &other.edge_adds {
            *self.edge_adds.entry(*slot).or_default() |= ids;
        }
        for (slot, ids) in &other.edge_removes {
            self.edge_removes.entry(*slot).or_default().extend(ids);
        }
    }

    /// Every id in this set per slot, regardless of direction — all the undo path
    /// needs, since it re-derives each document from committed state.
    fn ids_by_slot(
        &self
    ) -> (
        FxHashMap<u64, RoaringTreemap>,
        FxHashMap<u64, RoaringTreemap>,
    ) {
        let mut nodes: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        let mut edges: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        for docs in [&self.node_adds, &self.node_removes] {
            for (slot, ids) in docs {
                *nodes.entry(*slot).or_default() |= ids;
            }
        }
        for (slot, ids) in &self.edge_adds {
            *edges.entry(*slot).or_default() |= ids;
        }
        for (slot, ids) in &self.edge_removes {
            edges.entry(*slot).or_default().extend(ids.keys().copied());
        }
        (nodes, edges)
    }

    /// Write the batch to the shared RediSearch index.
    ///
    /// Only valid in writer mode: the index is not MVCC, so readers must be
    /// excluded (mirrors C, which updates index docs inside the graph write lock).
    pub fn commit(
        &mut self,
        g: &AtomicRefCell<Graph>,
    ) {
        // This is the query's own private version, which nothing else can borrow, so
        // graph-then-indexer is safe here — see `Pending::resync_published_indexes`
        // for the published version, where the order has to be the other way round.
        let mut g = g.borrow_mut();
        g.commit_index(&mut self.node_adds, &mut self.node_removes);
        g.commit_edge_index(&mut self.edge_adds, &mut self.edge_removes);
    }
}

impl Default for Pending {
    fn default() -> Self {
        Self::new()
    }
}

impl Pending {
    #[must_use]
    pub fn new() -> Self {
        Self {
            created_nodes: RoaringTreemap::new(),
            created_rels_by_type: FxHashMap::default(),
            created_rel_types: FxHashMap::default(),
            deleted_nodes: RoaringTreemap::new(),
            deleted_relationships: RoaringTreemap::new(),
            deleted_endpoints: Vec::new(),
            deleted_node_labels: Vec::new(),
            new_nodes_attrs: FxHashMap::default(),
            existing_nodes_attrs: FxHashMap::default(),
            new_relationships_attrs: FxHashMap::default(),
            existing_relationships_attrs: FxHashMap::default(),
            set_labels: FxHashMap::default(),
            remove_labels: FxHashMap::default(),
            index_docs: IndexDocs::default(),
            published: IndexDocs::default(),
            deferred_docs: IndexDocs::default(),
            schema_label_count: 0,
            schema_rel_type_count: 0,
            schema_node_attr_count: 0,
            schema_rel_attr_count: 0,
        }
    }

    /// Record the current dictionary sizes, so an effects emitter can tell
    /// which labels, types and attributes this query added.
    pub fn set_schema_baseline(
        &mut self,
        g: &AtomicRefCell<Graph>,
    ) {
        let graph = g.borrow();
        self.schema_label_count = graph.get_labels().len();
        self.schema_rel_type_count = graph.get_types().len();
        self.schema_node_attr_count = graph.get_node_attribute_names().len();
        self.schema_rel_attr_count = graph.get_relationship_attribute_names().len();
    }

    pub fn created_nodes(
        &mut self,
        ids: &[NodeId],
    ) {
        for id in ids {
            self.created_nodes.insert((*id).into());
        }
    }

    /// Set all attributes for a node. `attrs` must be attribute-id-resolved,
    /// sorted by id, and unique (last-wins dedup done by the caller).
    pub fn set_node_attributes(
        &mut self,
        id: NodeId,
        attrs: Vec<(u16, Value)>,
    ) -> Result<(), String> {
        for (_, value) in &attrs {
            validate_node_property(value)?;
        }
        // Empty attribute maps from CREATE without `{...}` props would otherwise
        // create an empty pinned cache entry per node on commit; skip them.
        if attrs.is_empty() {
            return Ok(());
        }
        // Strict: the doc above requires unique ids too, and
        // `AttributeStore::insert_attrs_rows` asserts strictly ascending at the
        // other end, so a non-strict check here would pass a duplicate along to
        // an assert that rejects it.
        debug_assert!(attrs.windows(2).all(|w| w[0].0 < w[1].0));
        let is_new = self.created_nodes.contains(id.into());
        if is_new {
            self.new_nodes_attrs.insert(id.into(), attrs);
        } else {
            self.existing_nodes_attrs.insert(id.into(), attrs);
        }
        Ok(())
    }

    pub fn set_node_attribute(
        &mut self,
        id: NodeId,
        attr_id: u16,
        value: Value,
    ) -> Result<(), String> {
        validate_node_property(&value)?;
        let map = if self.created_nodes.contains(id.into()) {
            &mut self.new_nodes_attrs
        } else {
            &mut self.existing_nodes_attrs
        };
        let entry = map.entry(id.into()).or_default();
        match entry.binary_search_by_key(&attr_id, |(k, _)| *k) {
            Ok(pos) => entry[pos].1 = value,
            Err(pos) => entry.insert(pos, (attr_id, value)),
        }
        Ok(())
    }

    pub fn clear_node_attributes(
        &mut self,
        id: NodeId,
    ) {
        self.new_nodes_attrs.remove(&id.into());
        self.existing_nodes_attrs.remove(&id.into());
    }

    #[must_use]
    #[inline]
    pub fn has_node_attrs(&self) -> bool {
        !self.new_nodes_attrs.is_empty() || !self.existing_nodes_attrs.is_empty()
    }

    #[must_use]
    #[inline]
    pub fn has_relationship_attrs(&self) -> bool {
        !self.new_relationships_attrs.is_empty() || !self.existing_relationships_attrs.is_empty()
    }

    #[must_use]
    pub fn get_node_attribute(
        &self,
        id: NodeId,
        attr_id: u16,
    ) -> Option<&Value> {
        if !self.has_node_attrs() {
            return None;
        }
        self.new_nodes_attrs
            .get(&id.into())
            .and_then(|attrs| lookup_sorted(attrs, attr_id))
            .or_else(|| {
                self.existing_nodes_attrs
                    .get(&id.into())
                    .and_then(|attrs| lookup_sorted(attrs, attr_id))
            })
    }

    pub fn update_node_attrs(
        &self,
        id: NodeId,
        attrs: &mut OrderMap<Arc<String>, Value>,
        g: &Graph,
    ) {
        let added = self
            .new_nodes_attrs
            .get(&id.into())
            .or_else(|| self.existing_nodes_attrs.get(&id.into()));
        if let Some(added) = added {
            for (attr_id, value) in added {
                let Some(key) = g.node_attr_name(*attr_id) else {
                    continue;
                };
                if matches!(value, Value::Null) {
                    attrs.remove(&key);
                } else {
                    attrs.insert(key, value.clone());
                }
            }
        }
    }

    pub fn set_node_labels(
        &mut self,
        id: NodeId,
        labels: &OrderSet<LabelId>,
    ) {
        let entry = self.set_labels.entry(id.into()).or_default();
        for label in labels.iter() {
            entry.push(usize::from(*label) as u64);
        }
    }

    pub fn set_nodes_labels(
        &mut self,
        ids: &[NodeId],
        labels: &OrderSet<LabelId>,
    ) {
        for id in ids {
            let entry = self.set_labels.entry((*id).into()).or_default();
            for label in labels.iter() {
                entry.push(usize::from(*label) as u64);
            }
        }
    }

    pub fn remove_node_labels(
        &mut self,
        id: NodeId,
        labels: &[LabelId],
    ) {
        let raw_id: u64 = id.into();
        for label in labels {
            let label_id = usize::from(*label) as u64;
            // Remove from pending set labels
            if let Some(set) = self.set_labels.get_mut(&raw_id) {
                set.retain(|&l| l != label_id);
            }
            self.remove_labels.entry(raw_id).or_default().push(label_id);
        }
    }

    /// What this query has staged about `label` on `id`: `Some(true)` if it was
    /// added, `Some(false)` if it was removed, `None` if this query says nothing
    /// about it and the committed label matrix is the answer.
    ///
    /// The precedence is [`Self::update_node_labels`]'s, which applies the adds
    /// and then the removals, so a removal wins — the two must agree, since they
    /// answer the same question for the same node.
    pub fn node_has_label(
        &self,
        id: NodeId,
        label: LabelId,
    ) -> Option<bool> {
        let raw_id: u64 = id.into();
        let label_id = usize::from(label) as u64;
        if self
            .remove_labels
            .get(&raw_id)
            .is_some_and(|removed| removed.contains(&label_id))
        {
            return Some(false);
        }
        if self
            .set_labels
            .get(&raw_id)
            .is_some_and(|set| set.contains(&label_id))
        {
            return Some(true);
        }
        None
    }

    pub fn update_node_labels(
        &self,
        id: NodeId,
        labels: &mut OrderSet<LabelId>,
    ) {
        let raw_id: u64 = id.into();
        if let Some(set) = self.set_labels.get(&raw_id) {
            for &label_id in set {
                labels.insert(LabelId(label_id as usize));
            }
        }
        if let Some(removed) = self.remove_labels.get(&raw_id) {
            for &label_id in removed {
                labels.remove(&LabelId(label_id as usize));
            }
        }
    }

    pub fn deleted_node(
        &mut self,
        id: NodeId,
    ) {
        self.deleted_nodes.insert(id.into());
    }

    /// Delete a pending-created node: mark it deleted, collect its labels and attrs,
    /// and also mark any pending-created relationships connected to it for deletion.
    /// Returns (label_ids, attrs, connected_pending_rels) — the relationships carry
    /// their type and staged attrs so callers can snapshot them for later reads.
    pub fn delete_pending_node(
        &mut self,
        id: NodeId,
    ) -> (
        OrderSet<LabelId>,
        Vec<(u16, Value)>,
        Vec<(
            RelationshipId,
            NodeId,
            NodeId,
            Arc<String>,
            Option<Vec<(u16, Value)>>,
        )>,
    ) {
        self.created_nodes.remove(id.into());
        // Collect pending labels
        let mut label_ids = OrderSet::default();
        self.update_node_labels(id, &mut label_ids);
        self.set_labels.remove(&id.into());

        // Collect pending attrs
        let attrs = self
            .new_nodes_attrs
            .remove(&id.into())
            .or_else(|| self.existing_nodes_attrs.remove(&id.into()))
            .unwrap_or_default();

        let rels = self.remove_pending_relationships_for_node(id);

        (label_ids, attrs, rels)
    }

    /// Remove and return all pending-created relationships incident on the
    /// given node, along with their staged attributes. Also cleans up
    /// `new_relationships_attrs` and `deleted_relationships` entries for
    /// each removed relationship so that `commit()` has no stale state.
    pub fn remove_pending_relationships_for_node(
        &mut self,
        id: NodeId,
    ) -> Vec<(
        RelationshipId,
        NodeId,
        NodeId,
        Arc<String>,
        Option<Vec<(u16, Value)>>,
    )> {
        let mut rels = Vec::new();
        for (type_name, entries) in &self.created_rels_by_type {
            for &(rel_id, from, to) in entries {
                if from == id || to == id {
                    rels.push((rel_id, from, to, type_name.clone()));
                }
            }
        }

        let mut result = Vec::with_capacity(rels.len());
        for (rel_id, from, to, type_name) in rels {
            self.created_rel_types.remove(&rel_id);
            if let Some(entries) = self.created_rels_by_type.get_mut(&type_name) {
                entries.retain(|(rid, _, _)| *rid != rel_id);
            }
            let attrs = self.new_relationships_attrs.remove(&rel_id.into());
            self.deleted_relationships.remove(rel_id.into());
            result.push((rel_id, from, to, type_name, attrs));
        }

        result
    }

    pub fn created_relationship(
        &mut self,
        id: RelationshipId,
        from: NodeId,
        to: NodeId,
        type_name: Arc<String>,
    ) {
        self.created_rels_by_type
            .entry(type_name.clone())
            .or_default()
            .push((id, from, to));
        self.created_rel_types.insert(id, type_name);
    }

    /// Set all attributes for a relationship. `attrs` must be
    /// attribute-id-resolved, sorted by id, and unique.
    pub fn set_relationship_attributes(
        &mut self,
        id: RelationshipId,
        attrs: Vec<(u16, Value)>,
    ) -> Result<(), String> {
        for (_, value) in &attrs {
            validate_relationship_property(value)?;
        }
        // Empty attribute maps from CREATE without `{...}` props would otherwise
        // create an empty pinned cache entry per relationship on commit; skip them.
        if attrs.is_empty() {
            return Ok(());
        }
        // Strict: the doc above requires unique ids too, and
        // `AttributeStore::insert_attrs_rows` asserts strictly ascending at the
        // other end, so a non-strict check here would pass a duplicate along to
        // an assert that rejects it.
        debug_assert!(attrs.windows(2).all(|w| w[0].0 < w[1].0));
        if self.created_rel_types.contains_key(&id) {
            self.new_relationships_attrs.insert(id.into(), attrs);
        } else {
            self.existing_relationships_attrs.insert(id.into(), attrs);
        }
        Ok(())
    }

    pub fn set_relationship_attribute(
        &mut self,
        id: RelationshipId,
        attr_id: u16,
        value: Value,
    ) -> Result<(), String> {
        validate_relationship_property(&value)?;
        let map = if self.created_rel_types.contains_key(&id) {
            &mut self.new_relationships_attrs
        } else {
            &mut self.existing_relationships_attrs
        };
        let entry = map.entry(id.into()).or_default();
        match entry.binary_search_by_key(&attr_id, |(k, _)| *k) {
            Ok(pos) => entry[pos].1 = value,
            Err(pos) => entry.insert(pos, (attr_id, value)),
        }
        Ok(())
    }

    #[must_use]
    pub fn get_relationship_attribute(
        &self,
        id: RelationshipId,
        attr_id: u16,
    ) -> Option<&Value> {
        if !self.has_relationship_attrs() {
            return None;
        }
        self.new_relationships_attrs
            .get(&id.into())
            .and_then(|attrs| lookup_sorted(attrs, attr_id))
            .or_else(|| {
                self.existing_relationships_attrs
                    .get(&id.into())
                    .and_then(|attrs| lookup_sorted(attrs, attr_id))
            })
    }

    pub fn update_relationship_attrs(
        &self,
        id: RelationshipId,
        attrs: &mut OrderMap<Arc<String>, Value>,
        g: &Graph,
    ) {
        let added = self
            .new_relationships_attrs
            .get(&id.into())
            .or_else(|| self.existing_relationships_attrs.get(&id.into()));
        if let Some(added) = added {
            for (attr_id, value) in added {
                let Some(key) = g.rel_attr_name(*attr_id) else {
                    continue;
                };
                if matches!(value, Value::Null) {
                    attrs.remove(&key);
                } else {
                    attrs.insert(key, value.clone());
                }
            }
        }
    }

    pub fn deleted_relationship(
        &mut self,
        id: RelationshipId,
    ) {
        self.deleted_relationships.insert(id.into());
    }

    pub fn deleted_relationships_bulk(
        &mut self,
        rels: &[RelationshipId],
    ) {
        for &id in rels {
            self.deleted_relationships.insert(id.into());
        }
    }

    #[must_use]
    pub fn get_relationship_type(
        &self,
        id: RelationshipId,
    ) -> Option<Arc<String>> {
        self.created_rel_types.get(&id).cloned()
    }

    #[must_use]
    pub fn is_node_created(
        &self,
        id: NodeId,
    ) -> bool {
        self.created_nodes.contains(id.into())
    }

    /// Returns pending-created node IDs that have ALL of the given labels.
    /// When `label_ids` is empty, returns all pending-created nodes.
    #[must_use]
    pub fn get_pending_nodes_with_labels(
        &self,
        label_ids: &[LabelId],
    ) -> RoaringTreemap {
        if label_ids.is_empty() {
            return self.created_nodes.clone();
        }
        let label_ids_u64: Vec<u64> = label_ids.iter().map(|l| usize::from(*l) as u64).collect();
        let mut result = RoaringTreemap::new();
        for (&node_id, node_labels) in &self.set_labels {
            if label_ids_u64.iter().all(|lid| node_labels.contains(lid)) {
                result.insert(node_id);
            }
        }
        result
    }

    /// Returns existing (non-created) node IDs that have pending label REMOVEs
    /// for ANY of the given label_ids.
    #[must_use]
    pub fn nodes_with_pending_label_removes(
        &self,
        label_ids: &[LabelId],
    ) -> RoaringTreemap {
        if label_ids.is_empty() {
            return RoaringTreemap::new();
        }
        let label_ids_u64: Vec<u64> = label_ids.iter().map(|l| usize::from(*l) as u64).collect();
        let mut result = RoaringTreemap::new();
        for (&node_id, removed_labels) in &self.remove_labels {
            if label_ids_u64.iter().any(|lid| removed_labels.contains(lid)) {
                result.insert(node_id);
            }
        }
        result
    }

    #[must_use]
    pub fn is_relationship_created(
        &self,
        id: RelationshipId,
    ) -> bool {
        self.created_rel_types.contains_key(&id)
    }

    /// Returns (src, dst) for a pending-created relationship, or None if not found.
    #[must_use]
    pub fn get_created_relationship_endpoints(
        &self,
        id: RelationshipId,
    ) -> Option<(NodeId, NodeId)> {
        let type_name = self.created_rel_types.get(&id)?;
        self.created_rels_by_type
            .get(type_name)?
            .iter()
            .find(|(rid, _, _)| *rid == id)
            .map(|&(_, from, to)| (from, to))
    }

    #[must_use]
    pub fn has_deleted_nodes(&self) -> bool {
        !self.deleted_nodes.is_empty()
    }

    #[must_use]
    pub fn has_deleted_relationships(&self) -> bool {
        !self.deleted_relationships.is_empty()
    }

    #[must_use]
    pub fn has_created_relationships(&self) -> bool {
        !self.created_rel_types.is_empty()
    }

    #[must_use]
    pub fn is_node_deleted(
        &self,
        id: NodeId,
    ) -> bool {
        self.deleted_nodes.contains(id.into())
    }

    /// Returns a clone of the pending-deleted nodes bitmap.
    #[must_use]
    pub fn deleted_nodes(&self) -> RoaringTreemap {
        self.deleted_nodes.clone()
    }

    #[must_use]
    pub fn is_relationship_deleted(
        &self,
        id: RelationshipId,
    ) -> bool {
        self.deleted_relationships.contains(id.into())
    }

    /// Count pending-created relationships whose destination is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    #[must_use]
    pub fn pending_indegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
    ) -> usize {
        if types.is_empty() {
            self.created_rels_by_type
                .values()
                .flat_map(|v| v.iter())
                .filter(|(_, _, to)| *to == node_id)
                .count()
        } else {
            types
                .iter()
                .filter_map(|t| self.created_rels_by_type.get(t))
                .flat_map(|v| v.iter())
                .filter(|(_, _, to)| *to == node_id)
                .count()
        }
    }

    /// Count pending-created relationships whose source is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    #[must_use]
    pub fn pending_outdegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
    ) -> usize {
        if types.is_empty() {
            self.created_rels_by_type
                .values()
                .flat_map(|v| v.iter())
                .filter(|(_, from, _)| *from == node_id)
                .count()
        } else {
            types
                .iter()
                .filter_map(|t| self.created_rels_by_type.get(t))
                .flat_map(|v| v.iter())
                .filter(|(_, from, _)| *from == node_id)
                .count()
        }
    }

    /// Count pending-deleted relationships whose destination is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    /// Requires access to the graph to resolve relationship type IDs.
    #[must_use]
    pub fn pending_deleted_indegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
        g: &Graph,
    ) -> usize {
        self.deleted_relationships
            .iter()
            .filter(|rel_id| {
                let (_from, to) = g.get_relationship_endpoints(RelationshipId::from(*rel_id));
                to == node_id
                    && (types.is_empty()
                        || g.get_type(g.get_relationship_type_id(RelationshipId::from(*rel_id)))
                            .is_some_and(|t| types.contains(&t)))
            })
            .count()
    }

    /// Count pending-deleted relationships whose source is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    /// Requires access to the graph to resolve relationship type IDs.
    #[must_use]
    pub fn pending_deleted_outdegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
        g: &Graph,
    ) -> usize {
        self.deleted_relationships
            .iter()
            .filter(|rel_id| {
                let (from, _to) = g.get_relationship_endpoints(RelationshipId::from(*rel_id));
                from == node_id
                    && (types.is_empty()
                        || g.get_type(g.get_relationship_type_id(RelationshipId::from(*rel_id)))
                            .is_some_and(|t| types.contains(&t)))
            })
            .count()
    }

    pub fn commit(
        &mut self,
        g: &AtomicRefCell<Graph>,
        stats: &RefCell<QueryStatistics>,
    ) -> Result<(), String> {
        if !self.created_nodes.is_empty() {
            stats.borrow_mut().nodes_created += self.created_nodes.len();
            g.borrow_mut().create_nodes(&self.created_nodes);
        }
        if !self.created_rel_types.is_empty() {
            stats.borrow_mut().relationships_created += self.created_rel_types.len();
            let mut g = g.borrow_mut();
            for (type_name, rel_ids) in &self.created_rels_by_type {
                let mut srcs = Vec::with_capacity(rel_ids.len());
                let mut dsts = Vec::with_capacity(rel_ids.len());
                let mut ids = Vec::with_capacity(rel_ids.len());
                for &(rel_id, from, to) in rel_ids {
                    srcs.push(from.into());
                    dsts.push(to.into());
                    ids.push(rel_id.into());
                }
                g.create_relationships_bulk(type_name, &srcs, &dsts, &ids);
            }
        }
        if !self.set_labels.is_empty() {
            let (rows, cols) = flatten_label_map(&self.set_labels);
            // Pairs for nodes created in this transaction cannot already be
            // committed (fresh ids; a reclaimed id's stale entries carry dm
            // tombstones), so a pure-create batch takes the unchecked insert.
            let all_new = self
                .set_labels
                .keys()
                .all(|id| self.created_nodes.contains(*id));
            g.borrow_mut().set_nodes_labels_bulk(
                &rows,
                &cols,
                &mut self.index_docs.node_adds,
                all_new,
            );
        }
        if !self.remove_labels.is_empty() {
            let (rows, cols) = flatten_label_map(&self.remove_labels);
            stats.borrow_mut().labels_removed += rows.len();
            g.borrow_mut()
                .remove_nodes_labels(&rows, &cols, &mut self.index_docs.node_removes);
        }
        if !self.new_nodes_attrs.is_empty() || !self.existing_nodes_attrs.is_empty() {
            let mut g = g.borrow_mut();
            if !self.new_nodes_attrs.is_empty() {
                let nset = g.import_node_attrs(
                    &self.new_nodes_attrs,
                    &self.set_labels,
                    &mut self.index_docs.node_adds,
                );
                stats.borrow_mut().properties_set += nset;
            }
            if !self.existing_nodes_attrs.is_empty() {
                let (nremoved, nset) = g.set_nodes_attributes(
                    &self.existing_nodes_attrs,
                    &mut self.index_docs.node_adds,
                )?;
                let mut s = stats.borrow_mut();
                s.properties_set += nset;
                s.properties_removed += nremoved;
            }
        }

        if !self.new_relationships_attrs.is_empty() || !self.existing_relationships_attrs.is_empty()
        {
            let mut g = g.borrow_mut();
            if !self.new_relationships_attrs.is_empty() {
                let nset = g.import_relationship_attrs(
                    &self.new_relationships_attrs,
                    &mut self.index_docs.edge_adds,
                );
                stats.borrow_mut().properties_set += nset;
            }
            if !self.existing_relationships_attrs.is_empty() {
                let (nremoved, nset) = g.set_relationships_attributes(
                    &self.existing_relationships_attrs,
                    &mut self.index_docs.edge_adds,
                )?;
                let mut s = stats.borrow_mut();
                s.properties_set += nset;
                s.properties_removed += nremoved;
            }
        }
        if !self.deleted_nodes.is_empty() {
            stats.borrow_mut().nodes_deleted += self.deleted_nodes.len();
            self.deleted_node_labels = g
                .borrow_mut()
                .delete_nodes(&self.deleted_nodes, &mut self.index_docs.node_removes)?;
        }
        // Take relationship deletions BEFORE implicit edge processing
        // so we can pass them to delete_implicit_edges for dedup.
        let explicit_rels = std::mem::take(&mut self.deleted_relationships);

        // Bulk cascade-delete edges for implicitly deleted nodes.
        // This must run after delete_nodes so that node matrices are already
        // cleaned up, and before delete_relationships so explicit edges are
        // still tracked separately.
        if !self.deleted_nodes.is_empty() {
            let implicit_edges = g.borrow_mut().delete_implicit_edges(
                &self.deleted_nodes,
                &explicit_rels,
                &mut self.index_docs.edge_removes,
            )?;
            let count = implicit_edges.len();
            stats.borrow_mut().relationships_deleted += count;
            // Record in deleted_relationships so effects buffer can serialize them
            for DeletedEdge {
                id: rel_id,
                type_id,
                src: from,
                dst: to,
            } in implicit_edges
            {
                self.deleted_relationships.insert(u64::from(rel_id));
                self.deleted_endpoints.push(DeletedEdge {
                    id: rel_id,
                    type_id,
                    src: from,
                    dst: to,
                });
            }
        }
        if !explicit_rels.is_empty() {
            let endpoints = g
                .borrow_mut()
                .delete_relationships(&explicit_rels, &mut self.index_docs.edge_removes)?;
            // Use the actually-removed relationships (delete_relationships skips
            // stale/missing ids) for stats and effects/constraint bookkeeping.
            stats.borrow_mut().relationships_deleted += endpoints.len();
            self.deleted_relationships
                .extend(endpoints.iter().map(|e| u64::from(e.id)));
            self.deleted_endpoints.extend(endpoints);
        }
        // Enforce constraints before accumulating index operations.
        // The constraint checks read the attribute store, which already has
        // the mutations from this transaction (writes apply immediately to
        // the private MVCC graph).
        self.enforce_constraints(g)?;

        // Index operations are deferred — they will be applied only after
        // the full query succeeds to avoid stale RediSearch entries on
        // rollback.

        // Accumulate this commit's documents into the query's deferred set.
        self.deferred_docs.absorb(&mut self.index_docs);

        Ok(())
    }

    /// Enforce graph constraints on all entities affected by this transaction.
    fn enforce_constraints(
        &self,
        g: &AtomicRefCell<Graph>,
    ) -> Result<(), String> {
        let g = g.borrow();
        let constraints = g.constraints();
        if constraints.is_empty() {
            return Ok(());
        }

        // Collect affected node IDs and their labels
        // Affected = created + had attributes modified + had labels added
        let mut affected_node_ids = RoaringTreemap::new();
        affected_node_ids |= &self.created_nodes;
        for &id in self.new_nodes_attrs.keys() {
            affected_node_ids.insert(id);
        }
        for &id in self.existing_nodes_attrs.keys() {
            affected_node_ids.insert(id);
        }
        for &id in self.set_labels.keys() {
            affected_node_ids.insert(id);
        }

        // Collect affected edge IDs
        let mut affected_edge_ids = RoaringTreemap::new();
        for rels in self.created_rels_by_type.values() {
            for &(rel_id, _, _) in rels {
                affected_edge_ids.insert(rel_id.into());
            }
        }
        for &id in self.new_relationships_attrs.keys() {
            affected_edge_ids.insert(id);
        }
        for &id in self.existing_relationships_attrs.keys() {
            affected_edge_ids.insert(id);
        }

        // Remove deleted entities from affected sets
        for id in &self.deleted_nodes {
            affected_node_ids.remove(id);
        }
        for rel_id in &self.deleted_relationships {
            affected_edge_ids.remove(rel_id);
        }

        // Check each OPERATIONAL constraint
        for constraint in constraints {
            if constraint.status != ConstraintStatus::Operational {
                continue;
            }

            match constraint.entity_type {
                EntityType::Node => {
                    self.check_node_constraint(&g, constraint, &affected_node_ids)?;
                }
                EntityType::Relationship => {
                    self.check_edge_constraint(&g, constraint, &affected_edge_ids)?;
                }
            }
        }

        Ok(())
    }

    /// Label check for constraint enforcement that answers from this
    /// transaction's own bookkeeping first. Reading the label matrix here
    /// would force a pending-tuple materialization of its delta on every
    /// commit (`O(|delta|)` per write query, quadratic between folds) —
    /// measured as the dominant cost of small repeated creates. Mirrors
    /// [`Self::update_node_labels`] semantics: a removed label wins over a
    /// pending set.
    fn constraint_node_has_label(
        &self,
        g: &Graph,
        node_id: u64,
        label_id: u64,
    ) -> bool {
        if let Some(removed) = self.remove_labels.get(&node_id)
            && removed.contains(&label_id)
        {
            return false;
        }
        if let Some(set) = self.set_labels.get(&node_id)
            && set.contains(&label_id)
        {
            return true;
        }
        if self.created_nodes.contains(node_id) {
            // Labels of nodes created this transaction live entirely in
            // set_labels — no need to touch the graph.
            return false;
        }
        g.node_has_label_id(node_id.into(), LabelId(label_id as usize))
    }

    fn check_node_constraint(
        &self,
        g: &Graph,
        constraint: &crate::graph::constraint::Constraint,
        affected_node_ids: &RoaringTreemap,
    ) -> Result<(), String> {
        let label = &constraint.label;
        let Some(label_id) = g.get_label_id(label) else {
            // Label doesn't exist yet — no node can carry it.
            return Ok(());
        };
        let label_id = usize::from(label_id) as u64;

        for node_id in affected_node_ids {
            // Check if this node has the constrained label
            if !self.constraint_node_has_label(g, node_id, label_id) {
                continue;
            }

            match constraint.ct {
                ConstraintType::Mandatory => {
                    for prop in &constraint.properties {
                        let has_prop = g
                            .get_node_attribute(node_id.into(), prop)
                            .is_some_and(|val| !matches!(val, Value::Null));
                        if !has_prop {
                            return Err(format!(
                                "mandatory constraint violation: node with label {label} missing property {prop}"
                            ));
                        }
                    }
                }
                ConstraintType::Unique => {
                    let key = Graph::build_composite_key(&constraint.properties, |prop| {
                        g.get_node_attribute(node_id.into(), prop)
                    });
                    if key.is_empty() {
                        continue; // All NULL → no violation
                    }

                    // Build a set of all existing keys for this label in one pass
                    if let Some(lm) = g.get_label_matrix(label) {
                        let mut seen: FxHashMap<Vec<u8>, u64> = FxHashMap::default();
                        for (other_id, _) in lm.iter(0, u64::MAX) {
                            let other_key =
                                Graph::build_composite_key(&constraint.properties, |prop| {
                                    g.get_node_attribute(other_id.into(), prop)
                                });
                            if other_key.is_empty() {
                                continue;
                            }
                            if let Some(&existing_id) = seen.get(&other_key)
                                && existing_id != other_id
                            {
                                return Err(format!(
                                    "unique constraint violation on node of type {label}"
                                ));
                            }
                            seen.insert(other_key, other_id);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn check_edge_constraint(
        &self,
        g: &Graph,
        constraint: &crate::graph::constraint::Constraint,
        affected_edge_ids: &RoaringTreemap,
    ) -> Result<(), String> {
        let type_name = &constraint.label;

        for edge_id in affected_edge_ids {
            // Edges created this transaction resolve their type from
            // Pending's reverse index — no relationship-matrix read, which
            // would materialize the delta's pending tuples on every commit.
            // An edge's type never changes, so created under a different
            // type means not a member.
            let has_type = match self.created_rel_types.get(&edge_id.into()) {
                Some(created_type) => created_type.as_str() == type_name.as_str(),
                None => g.edge_has_type(edge_id.into(), type_name),
            };
            if !has_type {
                continue;
            }

            match constraint.ct {
                ConstraintType::Mandatory => {
                    for prop in &constraint.properties {
                        let has_prop = g
                            .get_relationship_attribute(edge_id.into(), prop)
                            .is_some_and(|val| !matches!(val, Value::Null));
                        if !has_prop {
                            return Err(format!(
                                "mandatory constraint violation: edge with relationship-type {type_name} missing property {prop}"
                            ));
                        }
                    }
                }
                ConstraintType::Unique => {
                    let key = Graph::build_composite_key(&constraint.properties, |prop| {
                        g.get_relationship_attribute(edge_id.into(), prop)
                    });
                    if key.is_empty() {
                        continue;
                    }

                    // Build a set of all existing keys for this type in one pass
                    if let Some(tensor) = g.get_relationship_matrix(type_name) {
                        let mut seen: FxHashMap<Vec<u8>, u64> = FxHashMap::default();
                        for (_, _, other_eid) in tensor.iter(0, u64::MAX, false) {
                            let other_key =
                                Graph::build_composite_key(&constraint.properties, |prop| {
                                    g.get_relationship_attribute(other_eid.into(), prop)
                                });
                            if other_key.is_empty() {
                                continue;
                            }
                            if let Some(&existing_id) = seen.get(&other_key)
                                && existing_id != other_eid
                            {
                                return Err(format!(
                                    "unique constraint violation, on edge of relationship-type {type_name}"
                                ));
                            }
                            seen.insert(other_key, other_eid);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Take the accumulated index document changes, leaving pending empty.
    pub fn take_deferred_indexes(&mut self) -> IndexDocs {
        std::mem::take(&mut self.deferred_docs)
    }

    /// Undo the index documents earlier `Commit`s published, after this query
    /// failed, by re-synchronising them against committed state.
    ///
    /// Mirrors C's undo log (`_UndoLog_Rollback_Update_Entity`): a failed *create*
    /// loses its document, a failed *update* is re-indexed from the entity's
    /// previous values — deleting it instead would drop a live entity out of the
    /// index. No undo log is needed for that, because MVCC still holds the previous
    /// state: `committed` *is* the old value, and re-adding rewrites the document.
    ///
    /// `private` is the version this query built and is about to discard; a removed
    /// edge's document key comes from there, since committed state no longer has the
    /// edge at all.
    ///
    /// Writer mode only, and before the write lock is released.
    pub fn resync_published_indexes(
        &mut self,
        committed: &AtomicRefCell<Graph>,
        private: &AtomicRefCell<Graph>,
    ) {
        let (nodes, edges) = std::mem::take(&mut self.published).ids_by_slot();
        if nodes.is_empty() && edges.is_empty() {
            return;
        }

        let mut node_adds: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        let mut node_removes: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        let mut edge_adds: FxHashMap<u64, RoaringTreemap> = FxHashMap::default();
        let mut edge_removes: FxHashMap<u64, FxHashMap<u64, (u64, u64)>> = FxHashMap::default();
        {
            let g = committed.borrow();
            for (slot, ids) in &nodes {
                for id in ids {
                    if g.node_has_label_id(NodeId::from(id), LabelId(*slot as usize)) {
                        // Still live: rewrite the document from committed values.
                        node_adds.entry(*slot).or_default().insert(id);
                    } else {
                        // Never existed outside the discarded version.
                        node_removes.entry(*slot).or_default().insert(id);
                    }
                }
            }
            let p = private.borrow();
            for (slot, ids) in &edges {
                for id in ids {
                    if g.endpoints_for_edge(id).is_some() {
                        edge_adds.entry(*slot).or_default().insert(id);
                    } else if let Some(endpoints) = p.endpoints_for_edge(id) {
                        // The doc key is `(src, dst, edge_id)` as written, and only the
                        // discarded version still knows the endpoints.
                        edge_removes.entry(*slot).or_default().insert(id, endpoints);
                    }
                }
            }
        }

        // Indexer locks *before* the graph borrow — the order `populate_index_batch`
        // uses, and the only one that keeps this mutable borrow of the published
        // version from colliding with a populate batch reading it (`AtomicRefCell`
        // reports that collision by panicking). Concurrent *readers* are excluded
        // instead by the write lock this query holds.
        let (node_lock, edge_lock) = committed.borrow().index_locks();
        let node_guard = node_lock.lock();
        let edge_guard = edge_lock.lock();
        let mut g = committed.borrow_mut();
        g.commit_index_locked(&node_guard, &mut node_adds, &mut node_removes);
        g.commit_edge_index_locked(&edge_guard, &mut edge_adds, &mut edge_removes);
    }

    /// Write this `Commit`'s index documents to RediSearch. Writer mode only.
    pub fn commit_deferred_indexes(
        &mut self,
        g: &AtomicRefCell<Graph>,
    ) {
        let mut deferred = self.take_deferred_indexes();
        // Fold the batch into the published log *before* writing it, because `commit`
        // drains the batch. Removals matter as much as additions: a rolled-back
        // DELETE must get its document back, or the entity silently disappears from
        // the index.
        self.published.merge(&deferred);
        deferred.commit(g);
    }

    /// Clear all pending mutation state.
    ///
    /// Runs after every `Commit`, so it must NOT touch `published` — a later failure
    /// has to undo the documents *all* of this query's `Commit`s published, not just
    /// the last one's.
    pub fn clear(&mut self) {
        // Dropping millions of per-entity Vec allocations is O(n) frees and
        // stalls the serialized write thread; move large maps to a background
        // thread and let it pay the deallocation cost.
        const OFFLOAD_THRESHOLD: usize = 4096;
        let big_entries = self.new_nodes_attrs.len()
            + self.existing_nodes_attrs.len()
            + self.new_relationships_attrs.len()
            + self.existing_relationships_attrs.len()
            + self.set_labels.len()
            + self.remove_labels.len()
            + self
                .created_rels_by_type
                .values()
                .map(Vec::len)
                .sum::<usize>();
        if big_entries >= OFFLOAD_THRESHOLD {
            let maps = (
                std::mem::take(&mut self.new_nodes_attrs),
                std::mem::take(&mut self.existing_nodes_attrs),
                std::mem::take(&mut self.new_relationships_attrs),
                std::mem::take(&mut self.existing_relationships_attrs),
                std::mem::take(&mut self.set_labels),
                std::mem::take(&mut self.remove_labels),
                std::mem::take(&mut self.created_rels_by_type),
            );
            std::thread::spawn(move || drop(maps));
        } else {
            self.new_nodes_attrs.clear();
            self.existing_nodes_attrs.clear();
            self.new_relationships_attrs.clear();
            self.existing_relationships_attrs.clear();
            self.set_labels.clear();
            self.remove_labels.clear();
            self.created_rels_by_type.clear();
        }
        self.created_nodes.clear();
        self.created_rel_types.clear();
        self.deleted_nodes.clear();
        self.deleted_relationships.clear();
        self.deleted_endpoints.clear();
        self.deleted_node_labels.clear();
        self.index_docs.node_adds.clear();
        self.index_docs.node_removes.clear();
        self.index_docs.edge_adds.clear();
        self.index_docs.edge_removes.clear();
    }

    /// Returns the number of effects (operations) tracked in this Pending.
    #[must_use]
    pub fn effects_count(&self) -> u64 {
        self.created_nodes.len()
            + self.created_rel_types.len() as u64
            + self.deleted_nodes.len()
            + self.deleted_relationships.len()
            + self.new_nodes_attrs.len() as u64
            + self.existing_nodes_attrs.len() as u64
            + self.new_relationships_attrs.len() as u64
            + self.existing_relationships_attrs.len() as u64
            + self
                .set_labels
                .values()
                .map(|v| v.len() as u64)
                .sum::<u64>()
            + self
                .remove_labels
                .values()
                .map(|v| v.len() as u64)
                .sum::<u64>()
    }
}
