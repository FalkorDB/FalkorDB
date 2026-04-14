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
//! - `created_relationships`: Edges created in this query
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

use std::{cell::RefCell, collections::HashMap, sync::Arc};

use atomic_refcell::AtomicRefCell;
use roaring::RoaringTreemap;

use crate::{
    graph::{
        graph::{Graph, LabelId, NodeId, RelationshipId},
        graphblas::matrix::{Matrix, New, Remove, Set, Size},
    },
    runtime::{ordermap::OrderMap, orderset::OrderSet, runtime::QueryStatistics, value::Value},
};

/// A relationship waiting to be created.
pub struct PendingRelationship {
    pub from: NodeId,
    pub to: NodeId,
    pub type_name: Arc<String>,
}

impl PendingRelationship {
    #[must_use]
    pub const fn new(
        from: NodeId,
        to: NodeId,
        type_name: Arc<String>,
    ) -> Self {
        Self {
            from,
            to,
            type_name,
        }
    }
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

/// Accumulated write operations for deferred application.
///
/// All mutations during query execution are collected here and applied
/// atomically at the end. This enables transactional semantics.
pub struct Pending {
    /// Nodes created in this transaction
    created_nodes: RoaringTreemap,
    /// Relationships created (id → pending relationship data)
    created_relationships: HashMap<RelationshipId, PendingRelationship>,
    /// Nodes to be deleted
    deleted_nodes: RoaringTreemap,
    /// Relationships to be deleted (edge_id, src, dst)
    deleted_relationships: HashMap<RelationshipId, (NodeId, NodeId)>,
    /// Property updates for newly created nodes (fast path: skip fjall)
    new_nodes_attrs: HashMap<u64, OrderMap<Arc<String>, Value>>,
    /// Property updates for existing nodes (full merge path)
    existing_nodes_attrs: HashMap<u64, OrderMap<Arc<String>, Value>>,
    /// Property updates for newly created relationships (fast path: skip fjall)
    new_relationships_attrs: HashMap<u64, OrderMap<Arc<String>, Value>>,
    /// Property updates for existing relationships (full merge path)
    existing_relationships_attrs: HashMap<u64, OrderMap<Arc<String>, Value>>,
    /// Labels to add (node_id × label_id matrix)
    set_node_labels: Matrix,
    /// Labels to remove
    remove_node_labels: Matrix,
    /// Documents to add to indexes (keyed by label id)
    index_add_docs: HashMap<u64, RoaringTreemap>,
    /// Documents to remove from indexes (keyed by label id)
    index_remove_docs: HashMap<u64, RoaringTreemap>,
    /// Schema baseline: number of labels when the current commit window started.
    schema_label_count: usize,
    /// Schema baseline: number of relationship types when the current commit window started.
    schema_rel_type_count: usize,
    /// Schema baseline: number of node attribute names when the current commit window started.
    schema_node_attr_count: usize,
    /// Schema baseline: number of relationship attribute names when the current commit window started.
    schema_rel_attr_count: usize,
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
            created_relationships: HashMap::new(),
            deleted_nodes: RoaringTreemap::new(),
            deleted_relationships: HashMap::new(),
            new_nodes_attrs: HashMap::new(),
            existing_nodes_attrs: HashMap::new(),
            new_relationships_attrs: HashMap::new(),
            existing_relationships_attrs: HashMap::new(),
            set_node_labels: Matrix::new(0, 0),
            remove_node_labels: Matrix::new(0, 0),
            index_add_docs: HashMap::new(),
            index_remove_docs: HashMap::new(),
            schema_label_count: 0,
            schema_rel_type_count: 0,
            schema_node_attr_count: 0,
            schema_rel_attr_count: 0,
        }
    }

    /// Record the current schema sizes so `build_effects_buffer` can emit
    /// EFFECT_ADD_SCHEMA / EFFECT_ADD_ATTRIBUTE for newly added entries.
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

    pub fn resize(
        &mut self,
        node_cap: u64,
        labels_count: usize,
    ) {
        // Use max dimensions from both set and remove matrices to avoid shrinking either
        let new_nrows = node_cap
            .max(self.set_node_labels.nrows())
            .max(self.remove_node_labels.nrows());
        let new_ncols = (labels_count as u64)
            .max(self.set_node_labels.ncols())
            .max(self.remove_node_labels.ncols());
        self.set_node_labels.resize(new_nrows, new_ncols);
        self.remove_node_labels.resize(new_nrows, new_ncols);
    }

    pub fn created_nodes(
        &mut self,
        ids: &[NodeId],
    ) {
        let max_id = ids.iter().map(|id| u64::from(*id)).max();
        for id in ids {
            self.created_nodes.insert((*id).into());
        }
        if let Some(max_id) = max_id {
            let mut cap = self.set_node_labels.nrows();
            if cap <= max_id {
                if cap == 0 {
                    cap = 1;
                }
                while cap <= max_id {
                    cap *= 2;
                }
                self.set_node_labels
                    .resize(cap, self.set_node_labels.ncols());
            }
        }
    }

    pub fn set_node_attributes(
        &mut self,
        id: NodeId,
        attrs: OrderMap<Arc<String>, Value>,
    ) -> Result<(), String> {
        for value in attrs.values() {
            validate_node_property(value)?;
        }
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
        key: Arc<String>,
        value: Value,
    ) -> Result<(), String> {
        validate_node_property(&value)?;
        let map = if self.created_nodes.contains(id.into()) {
            &mut self.new_nodes_attrs
        } else {
            &mut self.existing_nodes_attrs
        };
        let entry = map.entry(id.into()).or_default();
        entry.insert(key, value);
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
    pub fn get_node_attribute(
        &self,
        id: NodeId,
        key: &Arc<String>,
    ) -> Option<&Value> {
        self.new_nodes_attrs
            .get(&id.into())
            .and_then(|attrs| attrs.get(key))
            .or_else(|| {
                self.existing_nodes_attrs
                    .get(&id.into())
                    .and_then(|attrs| attrs.get(key))
            })
    }

    pub fn update_node_attrs(
        &self,
        id: NodeId,
        attrs: &mut OrderMap<Arc<String>, Value>,
    ) {
        let added = self
            .new_nodes_attrs
            .get(&id.into())
            .or_else(|| self.existing_nodes_attrs.get(&id.into()));
        if let Some(added) = added {
            for (key, value) in added.iter() {
                if matches!(value, Value::Null) {
                    attrs.remove(key);
                } else {
                    attrs.insert(key.clone(), value.clone());
                }
            }
        }
    }

    pub fn set_node_labels(
        &mut self,
        id: NodeId,
        labels: &OrderSet<LabelId>,
    ) {
        for label in labels.iter() {
            self.set_node_labels
                .set(id.into(), usize::from(*label) as u64, true);
        }
    }

    pub fn set_nodes_labels(
        &mut self,
        ids: &[NodeId],
        labels: &OrderSet<LabelId>,
    ) {
        for id in ids {
            for label in labels.iter() {
                self.set_node_labels
                    .set((*id).into(), usize::from(*label) as u64, true);
            }
        }
    }

    pub fn remove_node_labels(
        &mut self,
        id: NodeId,
        labels: &[LabelId],
    ) {
        for label in labels {
            self.set_node_labels
                .remove(id.into(), usize::from(*label) as u64);
            self.remove_node_labels
                .set(id.into(), usize::from(*label) as u64, true);
        }
    }

    pub fn update_node_labels(
        &self,
        id: NodeId,
        labels: &mut OrderSet<LabelId>,
    ) {
        labels.extend(
            self.set_node_labels
                .iter(id.into(), id.into())
                .map(|(_, label_id)| LabelId(label_id as usize)),
        );

        for (_, label) in self.remove_node_labels.iter(id.into(), id.into()) {
            labels.remove(&LabelId(label as usize));
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
    /// Returns (label_ids, attrs, connected_pending_rels).
    pub fn delete_pending_node(
        &mut self,
        id: NodeId,
    ) -> (
        OrderSet<LabelId>,
        OrderMap<Arc<String>, Value>,
        Vec<(RelationshipId, NodeId, NodeId)>,
    ) {
        self.created_nodes.remove(id.into());
        // Collect pending labels
        let mut label_ids = OrderSet::default();
        self.update_node_labels(id, &mut label_ids);
        for label in label_ids.iter() {
            self.set_node_labels
                .remove(id.into(), usize::from(*label) as u64);
        }

        // Collect pending attrs
        let attrs = self
            .new_nodes_attrs
            .remove(&id.into())
            .or_else(|| self.existing_nodes_attrs.remove(&id.into()))
            .unwrap_or_default();

        // Find pending-created relationships connected to this node
        let rels: Vec<_> = self
            .created_relationships
            .iter()
            .filter(|(_, r)| r.from == id || r.to == id)
            .map(|(rid, r)| (*rid, r.from, r.to))
            .collect();

        for (rel_id, _, _) in &rels {
            self.created_relationships.remove(rel_id);
        }

        (label_ids, attrs, rels)
    }

    /// Remove and return all pending-created relationships incident on the
    /// given node, along with their staged attributes. Also cleans up
    /// `new_relationships_attrs` and `deleted_relationships` entries for
    /// each removed relationship so that commit() has no stale state.
    pub fn remove_pending_relationships_for_node(
        &mut self,
        id: NodeId,
    ) -> Vec<(
        RelationshipId,
        NodeId,
        NodeId,
        Arc<String>,
        Option<OrderMap<Arc<String>, Value>>,
    )> {
        let rels: Vec<_> = self
            .created_relationships
            .iter()
            .filter(|(_, r)| r.from == id || r.to == id)
            .map(|(rid, r)| (*rid, r.from, r.to, r.type_name.clone()))
            .collect();

        let mut result = Vec::with_capacity(rels.len());
        for (rel_id, from, to, type_name) in rels {
            self.created_relationships.remove(&rel_id);
            let attrs = self.new_relationships_attrs.remove(&rel_id.into());
            self.deleted_relationships.remove(&rel_id);
            result.push((rel_id, from, to, type_name, attrs));
        }

        result
    }

    pub fn created_relationships(
        &mut self,
        rels: Vec<(RelationshipId, NodeId, NodeId, Arc<String>)>,
    ) {
        for (id, from, to, type_name) in rels {
            self.created_relationships
                .insert(id, PendingRelationship::new(from, to, type_name));
        }
    }

    pub fn set_relationship_attributes(
        &mut self,
        id: RelationshipId,
        attrs: OrderMap<Arc<String>, Value>,
    ) -> Result<(), String> {
        for value in attrs.values() {
            validate_relationship_property(value)?;
        }
        if self.created_relationships.contains_key(&id) {
            self.new_relationships_attrs.insert(id.into(), attrs);
        } else {
            self.existing_relationships_attrs.insert(id.into(), attrs);
        }
        Ok(())
    }

    pub fn set_relationship_attribute(
        &mut self,
        id: RelationshipId,
        key: Arc<String>,
        value: Value,
    ) -> Result<(), String> {
        validate_relationship_property(&value)?;
        let map = if self.created_relationships.contains_key(&id) {
            &mut self.new_relationships_attrs
        } else {
            &mut self.existing_relationships_attrs
        };
        let entry = map.entry(id.into()).or_default();
        entry.insert(key, value);
        Ok(())
    }

    #[must_use]
    pub fn get_relationship_attribute(
        &self,
        id: RelationshipId,
        key: &Arc<String>,
    ) -> Option<&Value> {
        self.new_relationships_attrs
            .get(&id.into())
            .and_then(|attrs| attrs.get(key))
            .or_else(|| {
                self.existing_relationships_attrs
                    .get(&id.into())
                    .and_then(|attrs| attrs.get(key))
            })
    }

    pub fn update_relationship_attrs(
        &self,
        id: RelationshipId,
        attrs: &mut OrderMap<Arc<String>, Value>,
    ) {
        let added = self
            .new_relationships_attrs
            .get(&id.into())
            .or_else(|| self.existing_relationships_attrs.get(&id.into()));
        if let Some(added) = added {
            for (key, value) in added.iter() {
                if matches!(value, Value::Null) {
                    attrs.remove(key);
                } else {
                    attrs.insert(key.clone(), value.clone());
                }
            }
        }
    }

    pub fn deleted_relationship(
        &mut self,
        id: RelationshipId,
        from: NodeId,
        to: NodeId,
    ) {
        self.deleted_relationships.insert(id, (from, to));
    }

    #[must_use]
    pub fn get_relationship_type(
        &self,
        id: RelationshipId,
    ) -> Option<Arc<String>> {
        self.created_relationships
            .get(&id)
            .map(|r| r.type_name.clone())
    }

    #[must_use]
    pub fn is_node_created(
        &self,
        id: NodeId,
    ) -> bool {
        self.created_nodes.contains(id.into())
    }

    #[must_use]
    pub fn is_relationship_created(
        &self,
        id: RelationshipId,
    ) -> bool {
        self.created_relationships.contains_key(&id)
    }

    #[must_use]
    pub fn is_node_deleted(
        &self,
        id: NodeId,
    ) -> bool {
        self.deleted_nodes.contains(id.into())
    }

    #[must_use]
    pub fn is_relationship_deleted(
        &self,
        id: RelationshipId,
        from: NodeId,
        to: NodeId,
    ) -> bool {
        self.deleted_relationships
            .get(&id)
            .is_some_and(|(from_id, to_id)| *from_id == from && *to_id == to)
    }

    /// Count pending-created relationships whose destination is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    #[must_use]
    pub fn pending_indegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
    ) -> usize {
        self.created_relationships
            .values()
            .filter(|r| r.to == node_id && (types.is_empty() || types.contains(&r.type_name)))
            .count()
    }

    /// Count pending-created relationships whose source is `node_id` and
    /// whose type name matches one of `types` (or all if `types` is empty).
    #[must_use]
    pub fn pending_outdegree(
        &self,
        node_id: NodeId,
        types: &[Arc<String>],
    ) -> usize {
        self.created_relationships
            .values()
            .filter(|r| r.from == node_id && (types.is_empty() || types.contains(&r.type_name)))
            .count()
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
            .filter(|(rel_id, (_from, to))| {
                *to == node_id
                    && (types.is_empty()
                        || g.get_type(g.get_relationship_type_id(**rel_id))
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
            .filter(|(rel_id, (from, _to))| {
                *from == node_id
                    && (types.is_empty()
                        || g.get_type(g.get_relationship_type_id(**rel_id))
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
        if !self.created_relationships.is_empty() {
            stats.borrow_mut().relationships_created += self.created_relationships.len();
            g.borrow_mut()
                .create_relationships(&self.created_relationships);
        }
        if self.set_node_labels.nvals() > 0 {
            g.borrow_mut()
                .set_nodes_labels(&mut self.set_node_labels, &mut self.index_add_docs);
        }
        if self.remove_node_labels.nvals() > 0 {
            stats.borrow_mut().labels_removed += self.remove_node_labels.nvals() as usize;
            g.borrow_mut()
                .remove_nodes_labels(&mut self.remove_node_labels, &mut self.index_remove_docs);
        }
        if !self.new_nodes_attrs.is_empty() || !self.existing_nodes_attrs.is_empty() {
            let count_properties = |map: &HashMap<u64, OrderMap<Arc<String>, Value>>| -> usize {
                map.values()
                    .flat_map(super::ordermap::OrderMap::values)
                    .map(|v| match *v {
                        Value::Null => 0,
                        _ => 1,
                    })
                    .sum()
            };
            stats.borrow_mut().properties_set += count_properties(&self.new_nodes_attrs)
                + count_properties(&self.existing_nodes_attrs);
            let mut g = g.borrow_mut();
            if !self.new_nodes_attrs.is_empty() {
                g.import_node_attrs(&self.new_nodes_attrs, &mut self.index_add_docs);
            }
            if !self.existing_nodes_attrs.is_empty() {
                stats.borrow_mut().properties_removed +=
                    g.set_nodes_attributes(&self.existing_nodes_attrs, &mut self.index_add_docs)?;
            }
        }

        if !self.new_relationships_attrs.is_empty() || !self.existing_relationships_attrs.is_empty()
        {
            let count_properties = |map: &HashMap<u64, OrderMap<Arc<String>, Value>>| -> usize {
                map.values()
                    .flat_map(super::ordermap::OrderMap::values)
                    .map(|v| match *v {
                        Value::Null => 0,
                        _ => 1,
                    })
                    .sum()
            };
            stats.borrow_mut().properties_set += count_properties(&self.new_relationships_attrs)
                + count_properties(&self.existing_relationships_attrs);
            let mut g = g.borrow_mut();
            if !self.new_relationships_attrs.is_empty() {
                g.import_relationship_attrs(&self.new_relationships_attrs);
            }
            if !self.existing_relationships_attrs.is_empty() {
                stats.borrow_mut().properties_removed +=
                    g.set_relationships_attributes(&self.existing_relationships_attrs)?;
            }
        }
        if !self.deleted_nodes.is_empty() {
            stats.borrow_mut().nodes_deleted += self.deleted_nodes.len();
            g.borrow_mut()
                .delete_nodes(&self.deleted_nodes, &mut self.index_remove_docs)?;
        }
        if !self.deleted_relationships.is_empty() {
            stats.borrow_mut().relationships_deleted += self.deleted_relationships.len();
            let rels = std::mem::take(&mut self.deleted_relationships);
            g.borrow_mut().delete_relationships(rels)?;
        }
        // Commit attribute changes and indexes after all deletions have been
        // applied. This ensures relationship_attrs.remove() pending_deletes
        // (staged by delete_relationships) are included in commit_attrs().
        {
            let mut g = g.borrow_mut();
            g.commit_attrs()?;
            g.commit_index(&mut self.index_add_docs, &mut self.index_remove_docs);
        }

        Ok(())
    }

    /// Clear all pending mutation state.
    pub fn clear(&mut self) {
        self.created_nodes.clear();
        self.created_relationships.clear();
        self.set_node_labels.clear();
        self.remove_node_labels.clear();
        self.new_nodes_attrs.clear();
        self.existing_nodes_attrs.clear();
        self.new_relationships_attrs.clear();
        self.existing_relationships_attrs.clear();
        self.deleted_nodes.clear();
        self.deleted_relationships.clear();
    }

    /// Returns the number of effects (operations) tracked in this Pending.
    #[must_use]
    pub fn effects_count(&self) -> u64 {
        self.created_nodes.len()
            + self.created_relationships.len() as u64
            + self.deleted_nodes.len()
            + self.deleted_relationships.len() as u64
            + self.new_nodes_attrs.len() as u64
            + self.existing_nodes_attrs.len() as u64
            + self.new_relationships_attrs.len() as u64
            + self.existing_relationships_attrs.len() as u64
            + self.set_node_labels.nvals()
            + self.remove_node_labels.nvals()
    }

    /// Build a binary effects buffer from the accumulated mutations.
    /// Must be called before `clear()` resets the pending data.
    /// Appends to an existing buffer if provided, so multiple commits
    /// in the same query accumulate into a single effects buffer.
    /// Returns the number of effect records written.
    pub fn build_effects_buffer(
        &self,
        g: &AtomicRefCell<Graph>,
        buf: &mut Vec<u8>,
    ) -> u64 {
        let mut n_effects = 0u64;

        // Version header (only write once at the start)
        if buf.is_empty() {
            buf.push(EFFECTS_VERSION);
        }

        // --- Schema additions (new labels, relationship types) ---
        {
            let graph = g.borrow();
            let labels = graph.get_labels();
            for label in labels.iter().skip(self.schema_label_count) {
                buf.push(EFFECT_ADD_SCHEMA);
                buf.push(SCHEMA_NODE_LABEL);
                write_string(buf, label);
                n_effects += 1;
            }
            let types = graph.get_types();
            for rel_type in types.iter().skip(self.schema_rel_type_count) {
                buf.push(EFFECT_ADD_SCHEMA);
                buf.push(SCHEMA_REL_TYPE);
                write_string(buf, rel_type);
                n_effects += 1;
            }

            // --- Attribute additions (new node/rel attribute names) ---
            let node_attrs = graph.get_node_attribute_names();
            for attr in node_attrs.iter().skip(self.schema_node_attr_count) {
                buf.push(EFFECT_ADD_ATTRIBUTE);
                buf.push(ATTR_NODE);
                write_string(buf, attr);
                n_effects += 1;
            }
            let rel_attrs = graph.get_relationship_attribute_names();
            for attr in rel_attrs.iter().skip(self.schema_rel_attr_count) {
                buf.push(EFFECT_ADD_ATTRIBUTE);
                buf.push(ATTR_REL);
                write_string(buf, attr);
                n_effects += 1;
            }
        }

        // --- Created nodes ---
        for node_id in &self.created_nodes {
            buf.push(EFFECT_CREATE_NODE);
            buf.extend_from_slice(&node_id.to_le_bytes());

            // Labels: iterate the set_node_labels matrix for this row
            let label_entries: Vec<u64> = self
                .set_node_labels
                .iter(node_id, node_id)
                .map(|(_, col)| col)
                .collect();
            write_u16(buf, label_entries.len() as u16);
            let graph = g.borrow();
            for label_id in &label_entries {
                let label_name = graph.get_label_by_id(LabelId(*label_id as usize));
                write_string(buf, &label_name);
            }
            drop(graph);

            // Attributes
            if let Some(attrs) = self.new_nodes_attrs.get(&node_id) {
                write_u16(buf, attrs.len() as u16);
                for (key, value) in attrs.iter() {
                    write_string(buf, key);
                    write_value(buf, value);
                }
            } else {
                write_u16(buf, 0);
            }
            n_effects += 1;
        }

        // --- Created relationships ---
        for (rel_id, rel) in &self.created_relationships {
            buf.push(EFFECT_CREATE_EDGE);
            buf.extend_from_slice(&u64::from(*rel_id).to_le_bytes());
            buf.extend_from_slice(&u64::from(rel.from).to_le_bytes());
            buf.extend_from_slice(&u64::from(rel.to).to_le_bytes());
            write_string(buf, &rel.type_name);

            if let Some(attrs) = self.new_relationships_attrs.get(&u64::from(*rel_id)) {
                write_u16(buf, attrs.len() as u16);
                for (key, value) in attrs.iter() {
                    write_string(buf, key);
                    write_value(buf, value);
                }
            } else {
                write_u16(buf, 0);
            }
            n_effects += 1;
        }

        // --- Updated node attributes (existing nodes only) ---
        for (node_id, attrs) in &self.existing_nodes_attrs {
            buf.push(EFFECT_UPDATE_NODE);
            buf.extend_from_slice(&node_id.to_le_bytes());
            write_u16(buf, attrs.len() as u16);
            for (key, value) in attrs.iter() {
                write_string(buf, key);
                write_value(buf, value);
            }
            n_effects += 1;
        }

        // --- Updated relationship attributes (existing rels only) ---
        for (rel_id, attrs) in &self.existing_relationships_attrs {
            buf.push(EFFECT_UPDATE_EDGE);
            buf.extend_from_slice(&rel_id.to_le_bytes());
            write_u16(buf, attrs.len() as u16);
            for (key, value) in attrs.iter() {
                write_string(buf, key);
                write_value(buf, value);
            }
            n_effects += 1;
        }

        // --- Set labels (non-created nodes only) ---
        {
            let nrows = self.set_node_labels.nrows();
            if nrows > 0 {
                let mut label_map: HashMap<u64, Vec<u64>> = HashMap::new();
                for (node_id, label_id) in self.set_node_labels.iter(0, nrows - 1) {
                    if !self.created_nodes.contains(node_id) {
                        label_map.entry(node_id).or_default().push(label_id);
                    }
                }
                let graph = g.borrow();
                for (node_id, label_ids) in &label_map {
                    buf.push(EFFECT_SET_LABELS);
                    buf.extend_from_slice(&node_id.to_le_bytes());
                    write_u16(buf, label_ids.len() as u16);
                    for label_id in label_ids {
                        let label_name = graph.get_label_by_id(LabelId(*label_id as usize));
                        write_string(buf, &label_name);
                    }
                    n_effects += 1;
                }
            }
        }

        // --- Remove labels ---
        {
            let nrows = self.remove_node_labels.nrows();
            if nrows > 0 {
                let mut label_map: HashMap<u64, Vec<u64>> = HashMap::new();
                for (node_id, label_id) in self.remove_node_labels.iter(0, nrows - 1) {
                    label_map.entry(node_id).or_default().push(label_id);
                }
                let graph = g.borrow();
                for (node_id, label_ids) in &label_map {
                    buf.push(EFFECT_REMOVE_LABELS);
                    buf.extend_from_slice(&node_id.to_le_bytes());
                    write_u16(buf, label_ids.len() as u16);
                    for label_id in label_ids {
                        let label_name = graph.get_label_by_id(LabelId(*label_id as usize));
                        write_string(buf, &label_name);
                    }
                    n_effects += 1;
                }
            }
        }

        // --- Deleted relationships (before nodes, so replica removes edges first) ---
        for (rel_id, (from, to)) in &self.deleted_relationships {
            buf.push(EFFECT_DELETE_EDGE);
            buf.extend_from_slice(&u64::from(*rel_id).to_le_bytes());
            buf.extend_from_slice(&u64::from(*from).to_le_bytes());
            buf.extend_from_slice(&u64::from(*to).to_le_bytes());
            n_effects += 1;
        }

        // --- Deleted nodes ---
        for node_id in &self.deleted_nodes {
            buf.push(EFFECT_DELETE_NODE);
            buf.extend_from_slice(&node_id.to_le_bytes());
            n_effects += 1;
        }

        n_effects
    }
}

// ── Effects buffer constants and helpers ──

pub const EFFECTS_VERSION: u8 = 1;

pub const EFFECT_UPDATE_NODE: u8 = 1;
pub const EFFECT_UPDATE_EDGE: u8 = 2;
pub const EFFECT_CREATE_NODE: u8 = 3;
pub const EFFECT_CREATE_EDGE: u8 = 4;
pub const EFFECT_DELETE_NODE: u8 = 5;
pub const EFFECT_DELETE_EDGE: u8 = 6;
pub const EFFECT_SET_LABELS: u8 = 7;
pub const EFFECT_REMOVE_LABELS: u8 = 8;
pub const EFFECT_ADD_SCHEMA: u8 = 9;
pub const EFFECT_ADD_ATTRIBUTE: u8 = 10;
pub const EFFECT_CREATE_INDEX: u8 = 11;
pub const EFFECT_DROP_INDEX: u8 = 12;

// Schema type tags (used in EFFECT_ADD_SCHEMA)
pub const SCHEMA_NODE_LABEL: u8 = 0;
pub const SCHEMA_REL_TYPE: u8 = 1;

// Attribute type tags (used in EFFECT_ADD_ATTRIBUTE)
pub const ATTR_NODE: u8 = 0;
pub const ATTR_REL: u8 = 1;

// Value type tags for effect serialization
const VALUE_NULL: u8 = 0;
const VALUE_BOOL: u8 = 1;
const VALUE_INT: u8 = 2;
const VALUE_FLOAT: u8 = 3;
const VALUE_STRING: u8 = 4;
const VALUE_LIST: u8 = 5;
const VALUE_POINT: u8 = 6;
const VALUE_VECF32: u8 = 7;
const VALUE_DATETIME: u8 = 8;
const VALUE_DATE: u8 = 9;
const VALUE_TIME: u8 = 10;
const VALUE_DURATION: u8 = 11;

pub fn write_u16(
    buf: &mut Vec<u8>,
    v: u16,
) {
    buf.extend_from_slice(&v.to_le_bytes());
}

pub fn write_string(
    buf: &mut Vec<u8>,
    s: &str,
) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn write_value(
    buf: &mut Vec<u8>,
    value: &Value,
) {
    match value {
        Value::Null => buf.push(VALUE_NULL),
        Value::Bool(b) => {
            buf.push(VALUE_BOOL);
            buf.push(u8::from(*b));
        }
        Value::Int(i) => {
            buf.push(VALUE_INT);
            buf.extend_from_slice(&i.to_le_bytes());
        }
        Value::Float(f) => {
            buf.push(VALUE_FLOAT);
            buf.extend_from_slice(&f.to_le_bytes());
        }
        Value::String(s) => {
            buf.push(VALUE_STRING);
            write_string(buf, s);
        }
        Value::List(items) => {
            buf.push(VALUE_LIST);
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items.iter() {
                write_value(buf, item);
            }
        }
        Value::Point(p) => {
            buf.push(VALUE_POINT);
            buf.extend_from_slice(&(p.latitude as f64).to_le_bytes());
            buf.extend_from_slice(&(p.longitude as f64).to_le_bytes());
        }
        Value::VecF32(v) => {
            buf.push(VALUE_VECF32);
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for f in v.iter() {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        Value::Datetime(ts) => {
            buf.push(VALUE_DATETIME);
            buf.extend_from_slice(&ts.to_le_bytes());
        }
        Value::Date(ts) => {
            buf.push(VALUE_DATE);
            buf.extend_from_slice(&ts.to_le_bytes());
        }
        Value::Time(ts) => {
            buf.push(VALUE_TIME);
            buf.extend_from_slice(&ts.to_le_bytes());
        }
        Value::Duration(dur) => {
            buf.push(VALUE_DURATION);
            buf.extend_from_slice(&dur.to_le_bytes());
        }
        _ => {
            debug_assert!(false, "Unsupported value type in effects buffer: {value:?}");
            buf.push(VALUE_NULL); // Fallback for unsupported types
        }
    }
}

pub fn read_string(
    buf: &[u8],
    offset: &mut usize,
) -> Result<Arc<String>, String> {
    if *offset + 8 > buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let len = u64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap()) as usize;
    *offset += 8;
    if *offset + len > buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let s = std::str::from_utf8(&buf[*offset..*offset + len])
        .map_err(|e| format!("invalid utf8 in effects buffer: {e}"))?;
    *offset += len;
    Ok(Arc::new(s.to_string()))
}

pub fn read_u16(
    buf: &[u8],
    offset: &mut usize,
) -> Result<u16, String> {
    if *offset + 2 > buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let v = u16::from_le_bytes(buf[*offset..*offset + 2].try_into().unwrap());
    *offset += 2;
    Ok(v)
}

pub fn read_u64(
    buf: &[u8],
    offset: &mut usize,
) -> Result<u64, String> {
    if *offset + 8 > buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let v = u64::from_le_bytes(buf[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    Ok(v)
}

pub fn read_value(
    buf: &[u8],
    offset: &mut usize,
) -> Result<Value, String> {
    if *offset >= buf.len() {
        return Err("effects buffer truncated".to_string());
    }
    let tag = buf[*offset];
    *offset += 1;
    match tag {
        VALUE_NULL => Ok(Value::Null),
        VALUE_BOOL => {
            if *offset >= buf.len() {
                return Err("effects buffer truncated".to_string());
            }
            let b = buf[*offset] != 0;
            *offset += 1;
            Ok(Value::Bool(b))
        }
        VALUE_INT => {
            let v = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Int(v))
        }
        VALUE_FLOAT => {
            let v = f64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Float(v))
        }
        VALUE_STRING => {
            let s = read_string(buf, offset)?;
            Ok(Value::String(s))
        }
        VALUE_LIST => {
            let len = read_u64(buf, offset)? as usize;
            let mut items = thin_vec::ThinVec::with_capacity(len);
            for _ in 0..len {
                items.push(read_value(buf, offset)?);
            }
            Ok(Value::List(Arc::new(items)))
        }
        VALUE_POINT => {
            let lat = f64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            let lon = f64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Point(crate::runtime::value::Point {
                latitude: lat as f32,
                longitude: lon as f32,
            }))
        }
        VALUE_VECF32 => {
            let len = read_u64(buf, offset)? as usize;
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                let f = f32::from_le_bytes(
                    buf.get(*offset..*offset + 4)
                        .ok_or("truncated")?
                        .try_into()
                        .unwrap(),
                );
                *offset += 4;
                v.push(f);
            }
            Ok(Value::VecF32(Arc::new(v.into())))
        }
        VALUE_DATETIME => {
            let ts = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Datetime(ts))
        }
        VALUE_DATE => {
            let ts = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Date(ts))
        }
        VALUE_TIME => {
            let ts = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Time(ts))
        }
        VALUE_DURATION => {
            let dur = i64::from_le_bytes(
                buf.get(*offset..*offset + 8)
                    .ok_or("truncated")?
                    .try_into()
                    .unwrap(),
            );
            *offset += 8;
            Ok(Value::Duration(dur))
        }
        _ => Err(format!("unknown value tag in effects buffer: {tag}")),
    }
}
