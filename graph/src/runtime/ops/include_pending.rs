//! Composable operator that augments a child scan with pending mutations.
//!
//! Wraps any scan operator (NodeByLabelScan, NodeByIndexScan, etc.) and:
//! 1. Filters out pending-deleted nodes and nodes with pending label removals
//! 2. After child exhaustion, emits pending-created nodes and nodes with
//!    pending label additions
//!
//! Used inside MERGE match sub-plans so they see in-flight mutations from
//! prior clauses in the same query.

use std::sync::Arc;

use crate::graph::graph::NodeId;
use crate::parser::ast::{QueryNode, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{BATCH_SIZE, Batch, BatchOp},
    env::Env,
    runtime::Runtime,
    value::Value,
};
use orx_tree::{Dyn, NodeIdx};
use roaring::RoaringTreemap;

pub struct IncludePendingOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    node_pattern: &'a QueryNode<Arc<String>, Variable>,
    /// Extra pending nodes to emit after child exhaustion.
    pending_extra: Option<Box<dyn Iterator<Item = NodeId> + 'a>>,
    /// Nodes to filter out of child results.
    deleted_nodes: Option<RoaringTreemap>,
    label_removed_nodes: Option<RoaringTreemap>,
    /// Whether we've finished pulling from the child.
    child_exhausted: bool,
    /// Saved parent env for emitting pending nodes.
    parent_env: Option<Env<'a>>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> IncludePendingOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        node_pattern: &'a QueryNode<Arc<String>, Variable>,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            node_pattern,
            pending_extra: None,
            deleted_nodes: None,
            label_removed_nodes: None,
            child_exhausted: false,
            parent_env: None,
            idx,
        }
    }

    /// Initialize pending state (deleted/removed sets and extra nodes to emit).
    fn init_pending_state(&mut self) {
        let g = self.runtime.g.borrow();
        let label_ids: Vec<_> = self
            .node_pattern
            .labels
            .iter()
            .filter_map(|l| g.get_label_id(l))
            .collect();
        drop(g);

        let pending = self.runtime.pending.borrow();

        self.deleted_nodes = Some(pending.deleted_nodes());
        self.label_removed_nodes = Some(pending.nodes_with_pending_label_removes(&label_ids));

        let pending_nodes = if label_ids.len() == self.node_pattern.labels.len() {
            pending.get_pending_nodes_with_labels(&label_ids)
        } else {
            RoaringTreemap::new()
        };

        drop(pending);

        self.pending_extra = Some(Box::new(pending_nodes.into_iter().map(NodeId::from)));
    }

    /// Emit a batch from pending_extra nodes.
    fn emit_pending_batch(&mut self) -> Option<Result<Batch<'a>, String>> {
        let iter = self.pending_extra.as_mut()?;
        let alias = &self.node_pattern.alias;

        let mut envs = Vec::with_capacity(BATCH_SIZE);
        for id in iter.by_ref() {
            let mut row = match &self.parent_env {
                Some(env) => env.clone_pooled(self.runtime.env_pool),
                None => Env::new(self.runtime.env_pool),
            };
            row.insert(alias, Value::Node(id));
            envs.push(row);
            if envs.len() >= BATCH_SIZE {
                break;
            }
        }

        if envs.is_empty() {
            None
        } else {
            Some(Ok(Batch::from_envs(envs)))
        }
    }
}

impl<'a> Iterator for IncludePendingOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        // Lazily initialize pending state on first call.
        if self.deleted_nodes.is_none() {
            self.init_pending_state();
        }

        // Phase 1: Pull from child, filtering out deleted/removed nodes.
        if !self.child_exhausted {
            loop {
                match self.child.next() {
                    Some(Ok(batch)) => {
                        // Save a parent env for later pending emission.
                        if self.parent_env.is_none() && batch.len() > 0 {
                            // Use the first row (without the node binding) as template.
                            // The child scan already inserted the node var, but we need
                            // the base env. We'll just use the first env as-is since
                            // pending nodes will overwrite the node alias anyway.
                            self.parent_env =
                                Some(batch.env_ref(0).clone_pooled(self.runtime.env_pool));
                        }

                        let deleted = self.deleted_nodes.as_ref().unwrap();
                        let removed = self.label_removed_nodes.as_ref().unwrap();
                        let alias = &self.node_pattern.alias;

                        // Filter the batch: keep only rows whose node is not deleted/removed
                        let mut envs = Vec::with_capacity(batch.len());
                        for env in batch.active_env_iter() {
                            match env.get(alias) {
                                Some(Value::Node(nid)) => {
                                    let raw: u64 = (*nid).into();
                                    if deleted.contains(raw) || removed.contains(raw) {
                                        continue;
                                    }
                                }
                                _ => {
                                    debug_assert!(
                                        false,
                                        "IncludePendingOp: missing node binding for alias"
                                    );
                                    continue;
                                }
                            }
                            envs.push(env.clone_pooled(self.runtime.env_pool));
                        }

                        if !envs.is_empty() {
                            return Some(Ok(Batch::from_envs(envs)));
                        }
                        // All rows filtered, pull next batch from child
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => {
                        self.child_exhausted = true;
                        break;
                    }
                }
            }
        }

        // Phase 2: Emit pending-created and label-added nodes.
        self.emit_pending_batch()
    }
}
