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

use crate::graph::graph::{Graph, NodeId, RelationshipId};
use crate::parser::ast::{ExprIR, QueryExpr, Variable};
use crate::planner::IR;
use crate::runtime::eval::ExprEval;
use crate::runtime::ordermap::OrderMap;
use crate::runtime::{
    batch::{Batch, BatchOp, BatchRow},
    runtime::Runtime,
    value::{DeletedNode, DeletedRelationship, Value},
};
use orx_tree::{Dyn, DynTree, NodeIdx, NodeRef};
use std::sync::Arc;

fn node_attrs_to_map(
    g: &Graph,
    attrs: Vec<(u16, Value)>,
) -> OrderMap<Arc<String>, Value> {
    OrderMap::from_unique_keys(
        attrs
            .into_iter()
            .filter_map(|(id, v)| g.node_attr_name(id).map(|k| (k, v))),
    )
}

fn rel_attrs_to_map(
    g: &Graph,
    attrs: Vec<(u16, Value)>,
) -> OrderMap<Arc<String>, Value> {
    OrderMap::from_unique_keys(
        attrs
            .into_iter()
            .filter_map(|(id, v)| g.rel_attr_name(id).map(|k| (k, v))),
    )
}

/// Does anything above this `Delete` still get to look at the rows it passes
/// through? A deleted entity keeps its labels/type/endpoints/attributes only in
/// the runtime's `deleted_*` snapshot maps, so every operator layered on top of
/// the delete — a later `WITH`, `RETURN`, filter, procedure call — needs those
/// snapshots to be taken. Only `Commit` is transparent here: it neither reads
/// nor forwards entity data to a consumer.
///
/// For the common write-only `MATCH (n) DELETE n` the delete is the last thing
/// that happens, and snapshotting would cost an O(E) tensor scan for nothing.
fn snapshot_required(
    plan: &DynTree<IR>,
    idx: NodeIdx<Dyn<IR>>,
) -> bool {
    let mut current = plan.node(idx);
    while let Some(parent) = current.parent() {
        if !matches!(parent.data(), IR::Commit) {
            return true;
        }
        current = parent;
    }
    false
}

pub struct DeleteOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    trees: &'a Vec<QueryExpr<Variable>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
    /// Whether deleted entities must be snapshotted for later reads.
    snapshot: bool,
}

impl<'a> DeleteOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        trees: &'a Vec<QueryExpr<Variable>>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        let snapshot = snapshot_required(&runtime.plan, idx);
        Self {
            runtime,
            child,
            trees,
            idx,
            snapshot,
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

        if let Err(e) = self.runtime.delete_batch(self.trees, &batch, self.snapshot) {
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
        snapshot: bool,
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
            for row in batch.active_indices() {
                for &id in &var_ids {
                    match batch.value_at(id, row).unwrap_or(Value::Null) {
                        Value::Node(id) => node_ids.push(id),
                        Value::Relationship(rel) => rel_ids.push(rel),
                        val => {
                            // Paths, etc. go through per-entity path
                            self.delete_entity(&val, snapshot)?;
                        }
                    }
                }
            }
            if !node_ids.is_empty() {
                self.delete_nodes_bulk(&node_ids, snapshot)?;
            }
            if !rel_ids.is_empty() {
                self.delete_relationships_bulk(&rel_ids, snapshot)?;
            }
        }

        // Slow path: evaluate remaining expression trees via env_ref
        if !expr_trees.is_empty() {
            for row in batch.active_indices() {
                let env = BatchRow::new(batch, row);
                for tree in &expr_trees {
                    let value = ExprEval::from_runtime(self).eval(
                        tree,
                        tree.root().idx(),
                        Some(&env),
                        None,
                    )?;
                    self.delete_entity(&value, snapshot)?;
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
        snapshot: bool,
    ) -> Result<(), String> {
        // First pass: partition into already-deleted, pending-created, and committed
        let mut committed = Vec::with_capacity(node_ids.len());
        for &id in node_ids {
            if self.pending.borrow().is_node_deleted(id) {
                continue;
            }
            if self.pending.borrow().is_node_created(id) {
                // Created in this txn — use existing per-node path
                self.delete_entity(&Value::Node(id), snapshot)?;
            } else if !self.g.borrow().is_node_deleted(id) {
                committed.push(id);
            }
        }

        if committed.is_empty() {
            return Ok(());
        }

        // Snapshot implicit-edge type/attrs only when a later clause may
        // reference them. The actual cascade delete and effects/replication
        // bookkeeping is handled at commit time by `delete_implicit_edges`,
        // which is O(E) once instead of O(E) per batch.
        if snapshot {
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
                    let (src, dst) = (src_node, dest_node);
                    self.deleted_relationships
                        .borrow_mut()
                        .insert(rel, DeletedRelationship::new(src, dst, type_name, attrs));
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
                    for (rel_id, src, dest, type_name, attrs) in pending_rels {
                        self.g.borrow_mut().return_relationship_id(rel_id);
                        let attrs = rel_attrs_to_map(&self.g.borrow(), attrs.unwrap_or_default());
                        self.deleted_relationships.borrow_mut().insert(
                            rel_id,
                            DeletedRelationship::new(src, dest, type_name, attrs),
                        );
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

        // Snapshot labels and attrs only when needed (i.e., a later clause can
        // still reference the deleted nodes).  For the common write-only
        // pattern `MATCH (n) DELETE n` this saves ~30-40ms on 100K nodes.
        if snapshot {
            let mut deleted_nodes = self.deleted_nodes.borrow_mut();
            deleted_nodes.reserve(committed.len());
            let g = self.g.borrow();
            for &id in &committed {
                let labels = g.get_node_label_ids(id).collect();
                let mut actual =
                    crate::runtime::ordermap::OrderMap::from_unique_keys(g.get_node_all_attrs(id));
                self.pending.borrow().update_node_attrs(id, &mut actual, &g);
                deleted_nodes.insert(id, DeletedNode::new(labels, actual));
            }
        }

        Ok(())
    }

    /// Bulk delete committed relationships — avoids per-edge iterator creation
    /// for type lookups and reduces RefCell borrow overhead.
    fn delete_relationships_bulk(
        &self,
        rels: &[RelationshipId],
        snapshot: bool,
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
            for rel_id in rels {
                if !seen.insert(rel_id) {
                    continue;
                }
                if pending.is_relationship_deleted(*rel_id) {
                    continue;
                }
                if pending.is_relationship_created(*rel_id) {
                    // Created in this txn — use per-entity path
                    pending_created.push(*rel_id);
                } else if !g.is_relationship_deleted(*rel_id) {
                    committed.push(*rel_id);
                }
            }
        }

        // Handle pending-created relationships via the per-entity path
        for rel_id in pending_created {
            self.delete_entity(&Value::Relationship(rel_id), snapshot)?;
        }

        if committed.is_empty() {
            return Ok(());
        }

        // Mark for deletion and optionally snapshot data.
        {
            let mut pending = self.pending.borrow_mut();

            if snapshot {
                // Need snapshot for later clauses to reference deleted relationship data.
                // Build edge_id -> type_id mapping using a single type matrix scan
                // instead of N individual GraphBLAS iterators.
                let edge_set: rustc_hash::FxHashSet<u64> =
                    committed.iter().map(|r| u64::from(*r)).collect();
                let mut edge_type_map: rustc_hash::FxHashMap<u64, usize> =
                    rustc_hash::FxHashMap::with_capacity_and_hasher(
                        committed.len(),
                        rustc_hash::FxBuildHasher,
                    );
                let g = self.g.borrow();
                let min_id = committed.iter().map(|r| u64::from(*r)).min().unwrap();
                let max_id = committed.iter().map(|r| u64::from(*r)).max().unwrap();
                #[allow(clippy::cast_possible_truncation)]
                for (row, col) in g.relationship_type_matrix_iter(min_id, max_id) {
                    if edge_set.contains(&row) {
                        edge_type_map.insert(row, col as usize);
                    }
                }

                let mut deleted_rels = self.deleted_relationships.borrow_mut();
                for &rel_id in &committed {
                    let type_idx = edge_type_map
                        .get(&u64::from(rel_id))
                        .expect("relationship must have a type");
                    let type_name = g
                        .get_type(crate::graph::graph::TypeId(*type_idx))
                        .expect("type must exist");
                    let mut actual = crate::runtime::ordermap::OrderMap::from_unique_keys(
                        g.get_relationship_all_attrs(rel_id),
                    );
                    let (src, dst) = g.get_relationship_endpoints(rel_id);
                    pending.update_relationship_attrs(rel_id, &mut actual, &g);

                    pending.deleted_relationship(rel_id);
                    deleted_rels.insert(
                        rel_id,
                        DeletedRelationship::new(src, dst, type_name, actual),
                    );
                }
            } else {
                // Fast path: nothing above can read them — skip snapshotting,
                // just mark for deletion.
                pending.deleted_relationships_bulk(&committed);
            }
        }

        Ok(())
    }

    pub fn delete_entity(
        &self,
        value: &Value,
        snapshot: bool,
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
                    // Snapshot the cascaded relationships so expressions still
                    // holding them (e.g. `startNode(r)`) can be evaluated after
                    // the node — and with it the edge — is gone.
                    for (rel_id, src, dest, type_name, rel_attrs) in pending_rels {
                        self.g.borrow_mut().return_relationship_id(rel_id);
                        let rel_attrs =
                            rel_attrs_to_map(&self.g.borrow(), rel_attrs.unwrap_or_default());
                        self.deleted_relationships.borrow_mut().insert(
                            rel_id,
                            DeletedRelationship::new(src, dest, type_name, rel_attrs),
                        );
                    }
                    self.deleted_nodes.borrow_mut().insert(
                        id,
                        DeletedNode::new(
                            label_ids.into_iter().collect(),
                            node_attrs_to_map(&self.g.borrow(), attrs),
                        ),
                    );
                } else if !self.g.borrow().is_node_deleted(id) {
                    // Snapshot committed relationships for effects/replication
                    for (src, dest, rel_id) in self.g.borrow().get_node_relationships(id) {
                        // Skip edges whose other endpoint is already being
                        // deleted — they will be discovered from that node's
                        // perspective too. Only snapshot once.
                        if src != id && self.pending.borrow().is_node_deleted(src) {
                            continue;
                        }
                        let type_name = self.get_relationship_type(rel_id).unwrap();
                        let attrs = self.get_relationship_attrs(rel_id);
                        let (src_node, dest_node) = (src, dest);
                        self.deleted_relationships.borrow_mut().insert(
                            rel_id,
                            DeletedRelationship::new(src_node, dest_node, type_name, attrs),
                        );
                    }
                    // Cascade-delete pending-created relationships incident on this node
                    let pending_rels = self
                        .pending
                        .borrow_mut()
                        .remove_pending_relationships_for_node(id);
                    for (rel_id, src, dest, type_name, attrs) in pending_rels {
                        self.g.borrow_mut().return_relationship_id(rel_id);
                        let attrs = rel_attrs_to_map(&self.g.borrow(), attrs.unwrap_or_default());
                        self.deleted_relationships.borrow_mut().insert(
                            rel_id,
                            DeletedRelationship::new(src, dest, type_name, attrs),
                        );
                    }
                    // Mark as implicit — edge cleanup deferred to commit time
                    self.pending.borrow_mut().deleted_node(id);
                    if snapshot {
                        let labels = self.g.borrow().get_node_label_ids(id).collect();
                        let attrs = self.get_node_attrs(id);
                        self.deleted_nodes
                            .borrow_mut()
                            .insert(id, DeletedNode::new(labels, attrs));
                    }
                }
            }
            Value::Relationship(rel) => {
                if self.pending.borrow().is_relationship_deleted(*rel) {
                    // Already pending deletion, nothing to do
                } else if !self.g.borrow().is_relationship_deleted(*rel) {
                    // Snapshot attrs BEFORE marking as deleted so pending data
                    // is still accessible via get_relationship_attrs.
                    let type_name = self.get_relationship_type(*rel).unwrap();
                    let attrs = self.get_relationship_attrs(*rel);
                    let (src, dst) = self.get_relationship_endpoints(*rel);
                    self.pending.borrow_mut().deleted_relationship(*rel);
                    self.deleted_relationships
                        .borrow_mut()
                        .insert(*rel, DeletedRelationship::new(src, dst, type_name, attrs));
                }
            }
            Value::Path(values) => {
                for value in values.iter() {
                    self.delete_entity(value, snapshot)?;
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
