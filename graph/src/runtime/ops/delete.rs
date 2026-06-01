//! Batch-mode delete operator — marks nodes and relationships for deletion.
//!
//! For each active row in each input batch, evaluates the delete expressions
//! and records deletions in the pending batch.
//!
//! ```text
//!  Input batch
//!       │
//!  ┌────▼──────────────────────────────────────┐
//!  │ For each delete expr:                      │
//!  │   fast path: simple Variable ──► read_columns (bulk)
//!  │   slow path: complex expr   ──► eval per row       │
//!  │                                                     │
//!  │ Node deletion cascades:                             │
//!  │   mark all connected relationships for deletion     │
//!  │   mark the node itself for deletion                 │
//!  │                                                     │
//!  │ Relationship deletion:                              │
//!  │   mark the single relationship for deletion         │
//!  └────┬──────────────────────────────────────┘
//!       │
//!  output batch (unchanged, mutations in Pending)
//! ```

use crate::graph::graph::{NodeId, RelationshipId};
use crate::parser::ast::{ExprIR, QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::Runtime,
    value::{DeletedNode, DeletedRelationship, Value},
};
use orx_tree::{Dyn, NodeIdx, NodeRef};

pub struct DeleteOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    trees: &'a Vec<QueryExpr<Variable>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> DeleteOp<'a> {
    pub const fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        trees: &'a Vec<QueryExpr<Variable>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            trees,
            idx,
        }
    }
}

impl<'a> Iterator for DeleteOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = match self.child.next()? {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };

        if let Err(e) = self.runtime.delete_batch(self.trees, &batch) {
            return Some(Err(e));
        }

        Some(Ok(batch))
    }
}
impl Runtime<'_> {
    pub fn delete_batch(
        &self,
        trees: &Vec<QueryExpr<Variable>>,
        batch: &Batch<'_>,
    ) -> Result<(), String> {
        // Partition trees: collect var IDs for simple variable references (fast path),
        // and keep references to non-variable trees (slow path).
        let mut var_ids = Vec::new();
        let mut expr_trees = Vec::new();

        for tree in trees {
            match tree.root().data() {
                ExprIR::Variable(var) => var_ids.push(var.id),
                _ => expr_trees.push(tree),
            }
        }

        // Fast path: read all simple variable columns at once, no env needed
        if !var_ids.is_empty() {
            // Collect all node IDs and relationship tuples for bulk deletion
            let mut node_ids = Vec::new();
            let mut rel_ids = Vec::new();
            let rows = batch.read_columns(&var_ids);
            for row in rows {
                for val in row {
                    match val {
                        Value::Node(id) => node_ids.push(*id),
                        Value::Relationship(rel) => rel_ids.push(**rel),
                        _ => {
                            // Paths, etc. go through per-entity path
                            self.delete_entity(val)?;
                        }
                    }
                }
            }
            if !node_ids.is_empty() {
                self.delete_nodes_bulk(&node_ids)?;
            }
            if !rel_ids.is_empty() {
                self.delete_relationships_bulk(&rel_ids)?;
            }
        }

        // Slow path: evaluate remaining expression trees via env_ref
        if !expr_trees.is_empty() {
            for row in batch.active_indices() {
                let env = batch.env_ref(row);
                for tree in &expr_trees {
                    let value = ExprEval::from_runtime(self).eval(
                        tree,
                        tree.root().idx(),
                        Some(env),
                        None,
                    )?;
                    self.delete_entity(&value)?;
                }
            }
        }

        Ok(())
    }

    /// Bulk delete committed nodes — avoids per-node iterator creation for
    /// relationship lookups, label collection, and attribute snapshotting.
    fn delete_nodes_bulk(
        &self,
        node_ids: &[NodeId],
    ) -> Result<(), String> {
        // First pass: partition into already-deleted, pending-created, and committed
        let mut committed = Vec::with_capacity(node_ids.len());
        for &id in node_ids {
            if self.pending.borrow().is_node_deleted(id) {
                continue;
            }
            if self.pending.borrow().is_node_created(id) {
                // Created in this txn — use existing per-node path
                self.delete_entity(&Value::Node(id))?;
            } else if !self.g.borrow().is_node_deleted(id) {
                committed.push(id);
            }
        }

        if committed.is_empty() {
            return Ok(());
        }

        // Snapshot implicit-edge type/attrs only when a RETURN clause may
        // reference them. The actual cascade delete and effects/replication
        // bookkeeping is handled at commit time by `delete_implicit_edges`,
        // which is O(E) once instead of O(E) per batch.
        if !self.return_names.is_empty() {
            // Build a set of committed IDs for O(1) lookups
            let committed_set: std::collections::HashSet<NodeId> =
                committed.iter().copied().collect();
            let g = self.g.borrow();
            let n = g.node_cap();
            for tensor in g.relationship_matrices_iter() {
                for (src, dest, rel_id) in tensor.iter(0, n, false) {
                    let src_node: NodeId = src.into();
                    let dest_node: NodeId = dest.into();
                    let rel: RelationshipId = rel_id.into();
                    let src_deleted = committed_set.contains(&src_node);
                    let dest_deleted = committed_set.contains(&dest_node);
                    if !src_deleted && !dest_deleted {
                        continue;
                    }
                    // Skip if other endpoint is already pending-deleted
                    if !src_deleted && self.pending.borrow().is_node_deleted(src_node) {
                        continue;
                    }
                    if !dest_deleted && self.pending.borrow().is_node_deleted(dest_node) {
                        continue;
                    }
                    let type_name = self.get_relationship_type(rel).unwrap();
                    let attrs = self.get_relationship_attrs(rel);
                    self.deleted_relationships
                        .borrow_mut()
                        .insert(rel, DeletedRelationship::new(type_name, attrs));
                }
            }
        }

        // Cascade-delete pending-created relationships and mark nodes for deletion.
        // Batch the pending_deletes and deleted_nodes insertions.
        {
            let mut pending = self.pending.borrow_mut();
            if pending.has_created_relationships() {
                // Only scan for pending rels if there are any
                drop(pending);
                for &id in &committed {
                    let pending_rels = self
                        .pending
                        .borrow_mut()
                        .remove_pending_relationships_for_node(id);
                    for (rel_id, _src, _dest, type_name, attrs) in pending_rels {
                        let attrs = attrs.unwrap_or_default();
                        self.g.borrow_mut().return_relationship_id(rel_id);
                        self.deleted_relationships
                            .borrow_mut()
                            .insert(rel_id, DeletedRelationship::new(type_name, attrs));
                    }
                    self.pending.borrow_mut().deleted_node(id);
                }
            } else {
                // Fast path: no pending created relationships — just mark all as deleted
                for &id in &committed {
                    pending.deleted_node(id);
                }
            }
        }

        // Snapshot labels and attrs only when needed (i.e., a RETURN clause
        // references the deleted nodes).  For the common write-only pattern
        // `MATCH (n) DELETE n` this saves ~30-40ms on 100K nodes.
        if !self.return_names.is_empty() {
            let mut deleted_nodes = self.deleted_nodes.borrow_mut();
            deleted_nodes.reserve(committed.len());
            let g = self.g.borrow();
            for &id in &committed {
                let labels = g.get_node_label_ids(id).collect();
                let mut actual =
                    crate::runtime::ordermap::OrderMap::from_vec(g.get_node_all_attrs(id));
                self.pending.borrow().update_node_attrs(id, &mut actual);
                deleted_nodes.insert(id, DeletedNode::new(labels, actual));
            }
        }

        Ok(())
    }

    /// Bulk delete committed relationships — avoids per-edge iterator creation
    /// for type lookups and reduces RefCell borrow overhead.
    fn delete_relationships_bulk(
        &self,
        rels: &[(RelationshipId, NodeId, NodeId)],
    ) -> Result<(), String> {
        if rels.is_empty() {
            return Ok(());
        }

        // Filter out already-deleted, pending-created, and duplicate relationships
        let mut committed = Vec::with_capacity(rels.len());
        let mut seen = rustc_hash::FxHashSet::default();
        let mut pending_created = Vec::new();
        {
            let pending = self.pending.borrow();
            let g = self.g.borrow();
            for &(rel_id, src, dst) in rels {
                if !seen.insert(rel_id) {
                    continue;
                }
                if pending.is_relationship_deleted(rel_id, src, dst) {
                    continue;
                }
                if pending.is_relationship_created(rel_id) {
                    // Created in this txn — use per-entity path
                    pending_created.push((rel_id, src, dst));
                } else if !g.is_relationship_deleted(rel_id) {
                    committed.push((rel_id, src, dst));
                }
            }
        }

        // Handle pending-created relationships via the per-entity path
        for (rel_id, src, dst) in pending_created {
            self.delete_entity(&Value::Relationship(Box::new((rel_id, src, dst))))?;
        }

        if committed.is_empty() {
            return Ok(());
        }

        // Mark for deletion and optionally snapshot data.
        {
            let mut pending = self.pending.borrow_mut();

            if self.return_names.is_empty() {
                // Fast path: no RETURN clause — skip snapshotting, just mark for deletion
                pending.deleted_relationships_bulk(&committed);
            } else {
                // Need snapshot for RETURN to reference deleted relationship data.
                // Build edge_id -> type_id mapping using a single type matrix scan
                // instead of N individual GraphBLAS iterators.
                let edge_set: rustc_hash::FxHashSet<u64> =
                    committed.iter().map(|(r, _, _)| u64::from(*r)).collect();
                let mut edge_type_map: rustc_hash::FxHashMap<u64, usize> =
                    rustc_hash::FxHashMap::with_capacity_and_hasher(
                        committed.len(),
                        rustc_hash::FxBuildHasher,
                    );
                let g = self.g.borrow();
                let min_id = committed
                    .iter()
                    .map(|(r, _, _)| u64::from(*r))
                    .min()
                    .unwrap();
                let max_id = committed
                    .iter()
                    .map(|(r, _, _)| u64::from(*r))
                    .max()
                    .unwrap();
                #[allow(clippy::cast_possible_truncation)]
                for (row, col) in g.relationship_type_matrix_iter(min_id, max_id) {
                    if edge_set.contains(&row) {
                        edge_type_map.insert(row, col as usize);
                    }
                }

                let mut deleted_rels = self.deleted_relationships.borrow_mut();
                for &(rel_id, src, dst) in &committed {
                    let type_idx = edge_type_map
                        .get(&u64::from(rel_id))
                        .expect("relationship must have a type");
                    let type_name = g
                        .get_type(crate::graph::graph::TypeId(*type_idx))
                        .expect("type must exist");
                    let mut actual = crate::runtime::ordermap::OrderMap::from_vec(
                        g.get_relationship_all_attrs(rel_id),
                    );
                    pending.update_relationship_attrs(rel_id, &mut actual);

                    pending.deleted_relationship(rel_id, src, dst);
                    deleted_rels.insert(rel_id, DeletedRelationship::new(type_name, actual));
                }
            }
        }

        Ok(())
    }

    pub fn delete_entity(
        &self,
        value: &Value,
    ) -> Result<(), String> {
        match value {
            Value::Node(id) => {
                let id = *id;
                if self.pending.borrow().is_node_deleted(id) {
                    // Already pending deletion, nothing to do
                } else if self.pending.borrow().is_node_created(id) {
                    // Node was created in this transaction but not yet committed.
                    let (label_ids, attrs, pending_rels) =
                        self.pending.borrow_mut().delete_pending_node(id);
                    // Return the node ID and relationship IDs to the graph for reuse.
                    self.g.borrow_mut().return_node_id(id);
                    for (rel_id, _, _) in &pending_rels {
                        self.g.borrow_mut().return_relationship_id(*rel_id);
                    }
                    self.deleted_nodes.borrow_mut().insert(
                        id,
                        DeletedNode::new(
                            label_ids.into_iter().collect(),
                            attrs.into_iter().collect(),
                        ),
                    );
                } else if !self.g.borrow().is_node_deleted(id) {
                    // Snapshot committed relationships for effects/replication
                    for (src, _dest, rel_id) in self.g.borrow().get_node_relationships(id) {
                        // Skip edges whose other endpoint is already being
                        // deleted — they will be discovered from that node's
                        // perspective too. Only snapshot once.
                        if src != id && self.pending.borrow().is_node_deleted(src) {
                            continue;
                        }
                        let type_name = self.get_relationship_type(rel_id).unwrap();
                        let attrs = self.get_relationship_attrs(rel_id);
                        self.deleted_relationships
                            .borrow_mut()
                            .insert(rel_id, DeletedRelationship::new(type_name, attrs));
                    }
                    // Cascade-delete pending-created relationships incident on this node
                    let pending_rels = self
                        .pending
                        .borrow_mut()
                        .remove_pending_relationships_for_node(id);
                    for (rel_id, _src, _dest, type_name, attrs) in pending_rels {
                        let attrs = attrs.unwrap_or_default();
                        self.g.borrow_mut().return_relationship_id(rel_id);
                        self.deleted_relationships
                            .borrow_mut()
                            .insert(rel_id, DeletedRelationship::new(type_name, attrs));
                    }
                    // Mark as implicit — edge cleanup deferred to commit time
                    self.pending.borrow_mut().deleted_node(id);
                    if !self.return_names.is_empty() {
                        let labels = self.g.borrow().get_node_label_ids(id).collect();
                        let attrs = self.get_node_attrs(id);
                        self.deleted_nodes
                            .borrow_mut()
                            .insert(id, DeletedNode::new(labels, attrs));
                    }
                }
            }
            Value::Relationship(rel) => {
                let (rel_id, src, dest) = **rel;
                if self
                    .pending
                    .borrow()
                    .is_relationship_deleted(rel_id, src, dest)
                {
                    // Already pending deletion, nothing to do
                } else if !self.g.borrow().is_relationship_deleted(rel_id) {
                    // Snapshot attrs BEFORE marking as deleted so pending data
                    // is still accessible via get_relationship_attrs.
                    let type_name = self.get_relationship_type(rel_id).unwrap();
                    let attrs = self.get_relationship_attrs(rel_id);
                    self.pending
                        .borrow_mut()
                        .deleted_relationship(rel_id, src, dest);
                    self.deleted_relationships
                        .borrow_mut()
                        .insert(rel_id, DeletedRelationship::new(type_name, attrs));
                }
            }
            Value::Path(values) => {
                for value in values.iter() {
                    self.delete_entity(value)?;
                }
            }
            Value::Null => {}
            _ => {
                return Err(String::from(
                    "Delete type mismatch, expecting either Node or Relationship.",
                ));
            }
        }
        Ok(())
    }
}
