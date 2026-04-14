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
    /// Property updates for nodes
    set_nodes_attrs: HashMap<u64, OrderMap<Arc<String>, Value>>,
    /// Property updates for relationships
    set_relationships_attrs: HashMap<u64, OrderMap<Arc<String>, Value>>,
    /// Labels to add (node_id × label_id matrix)
    set_node_labels: Matrix,
    /// Labels to remove
    remove_node_labels: Matrix,
    /// Documents to add to indexes (keyed by label id)
    index_add_docs: HashMap<u64, RoaringTreemap>,
    /// Documents to remove from indexes (keyed by label id)
    index_remove_docs: HashMap<u64, RoaringTreemap>,
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
            set_nodes_attrs: HashMap::new(),
            set_relationships_attrs: HashMap::new(),
            set_node_labels: Matrix::new(0, 0),
            remove_node_labels: Matrix::new(0, 0),
            index_add_docs: HashMap::new(),
            index_remove_docs: HashMap::new(),
        }
    }

    pub fn resize(
        &mut self,
        node_cap: u64,
        labels_count: usize,
    ) {
        self.set_node_labels.resize(node_cap, labels_count as u64);
        self.remove_node_labels
            .resize(node_cap, labels_count as u64);
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
        for (_, value) in attrs.iter() {
            validate_node_property(value)?;
        }
        self.set_nodes_attrs.insert(id.into(), attrs);
        Ok(())
    }

    pub fn set_node_attribute(
        &mut self,
        id: NodeId,
        key: Arc<String>,
        value: Value,
    ) -> Result<(), String> {
        validate_node_property(&value)?;
        self.set_nodes_attrs
            .entry(id.into())
            .or_default()
            .insert(key, value);
        Ok(())
    }

    pub fn clear_node_attributes(
        &mut self,
        id: NodeId,
    ) {
        self.set_nodes_attrs.remove(&id.into());
    }

    #[must_use]
    pub fn get_node_attribute(
        &self,
        id: NodeId,
        key: &Arc<String>,
    ) -> Option<&Value> {
        self.set_nodes_attrs
            .get(&id.into())
            .and_then(|attrs| attrs.get(key))
    }

    pub fn update_node_attrs(
        &self,
        id: NodeId,
        attrs: &mut OrderMap<Arc<String>, Value>,
    ) {
        if let Some(added) = self.set_nodes_attrs.get(&id.into()) {
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
        let attrs = self.set_nodes_attrs.remove(&id.into()).unwrap_or_default();

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
    /// `set_relationships_attrs` and `deleted_relationships` entries for
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
            let attrs = self.set_relationships_attrs.remove(&rel_id.into());
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
        for (_, value) in attrs.iter() {
            validate_relationship_property(value)?;
        }
        self.set_relationships_attrs.insert(id.into(), attrs);
        Ok(())
    }

    pub fn set_relationship_attribute(
        &mut self,
        id: RelationshipId,
        key: Arc<String>,
        value: Value,
    ) -> Result<(), String> {
        validate_relationship_property(&value)?;
        self.set_relationships_attrs
            .entry(id.into())
            .or_default()
            .insert(key, value);
        Ok(())
    }

    #[must_use]
    pub fn get_relationship_attribute(
        &self,
        id: RelationshipId,
        key: &Arc<String>,
    ) -> Option<&Value> {
        self.set_relationships_attrs
            .get(&id.into())
            .and_then(|attrs| attrs.get(key))
    }

    pub fn update_relationship_attrs(
        &self,
        id: RelationshipId,
        attrs: &mut OrderMap<Arc<String>, Value>,
    ) {
        if let Some(added) = self.set_relationships_attrs.get(&id.into()) {
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
            self.created_nodes.clear();
        }
        if !self.created_relationships.is_empty() {
            stats.borrow_mut().relationships_created += self.created_relationships.len();
            g.borrow_mut()
                .create_relationships(&self.created_relationships);
            self.created_relationships.clear();
        }
        if self.set_node_labels.nvals() > 0 {
            g.borrow_mut()
                .set_nodes_labels(&mut self.set_node_labels, &mut self.index_add_docs);

            self.set_node_labels.clear();
        }
        if self.remove_node_labels.nvals() > 0 {
            stats.borrow_mut().labels_removed += self.remove_node_labels.nvals() as usize;
            g.borrow_mut()
                .remove_nodes_labels(&mut self.remove_node_labels, &mut self.index_remove_docs);

            self.remove_node_labels.clear();
        }
        if !self.set_nodes_attrs.is_empty() {
            stats.borrow_mut().properties_set += self
                .set_nodes_attrs
                .values()
                .flat_map(super::ordermap::OrderMap::values)
                .map(|v| match *v {
                    Value::Null => 0,
                    _ => 1,
                })
                .sum::<usize>();
            stats.borrow_mut().properties_removed += g
                .borrow_mut()
                .set_nodes_attributes(&self.set_nodes_attrs, &mut self.index_add_docs)?;
            self.set_nodes_attrs.clear();
        }

        if !self.set_relationships_attrs.is_empty() {
            stats.borrow_mut().properties_set += self
                .set_relationships_attrs
                .values()
                .flat_map(super::ordermap::OrderMap::values)
                .map(|v| match *v {
                    Value::Null => 0,
                    _ => 1,
                })
                .sum::<usize>();
            stats.borrow_mut().properties_removed += g
                .borrow_mut()
                .set_relationships_attributes(&self.set_relationships_attrs)?;
            self.set_relationships_attrs.clear();
        }
        if !self.deleted_nodes.is_empty() {
            stats.borrow_mut().nodes_deleted += self.deleted_nodes.len();
            g.borrow_mut()
                .delete_nodes(&self.deleted_nodes, &mut self.index_remove_docs)?;
            self.deleted_nodes.clear();
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
}
