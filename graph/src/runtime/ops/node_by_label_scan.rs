//! Batch-mode label scan operator — iterates all nodes with a given label.
//!
//! For each parent row, collects up to [`BATCH_SIZE`](super::super::batch::BATCH_SIZE)
//! matching node IDs into a [`Column::NodeIds`] vector. This avoids cloning the
//! parent env per node — the parent env columns are copied once per batch.
//!
//! ```text
//!  parent BatchOp ──► parent_batch
//!                          │
//!             for each parent row:
//!               label_matrix.get_nodes(labels) ──► node iterator
//!                          │
//!              ┌───────────┴───────────┐
//!              │  collect ≤ BATCH_SIZE │
//!              │  node IDs per batch   │
//!              └───────────┬───────────┘
//!                          │
//!          Batch { parent_env + Node(id) per row }
//!                          │
//!                    yield Batch ──► parent
//! ```

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

pub struct NodeByLabelScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    /// Current parent batch being expanded.
    parent_batch: Option<Batch<'a>>,
    /// Index of the current parent row within `parent_batch`.
    parent_row: usize,
    /// Cached env for the current parent row.
    parent_env: Option<Env<'a>>,
    /// Iterator over node IDs for the current parent row.
    node_iter: Option<Box<dyn Iterator<Item = NodeId> + 'a>>,
    node_pattern: &'a QueryNode<Arc<String>, Variable>,
    /// When true, include pending-created nodes and exclude pending-deleted
    /// nodes from scan results. Set only for MERGE match sub-plans.
    include_pending: bool,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> NodeByLabelScanOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        node_pattern: &'a QueryNode<Arc<String>, Variable>,
        include_pending: bool,
        idx: NodeIdx<Dyn<IR>>,
    ) -> Self {
        Self {
            runtime,
            child,
            parent_batch: None,
            parent_row: 0,
            parent_env: None,
            node_iter: None,
            node_pattern,
            include_pending,
            idx,
        }
    }
}

impl<'a> Iterator for NodeByLabelScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have an active node iterator, collect up to BATCH_SIZE nodes
            if let Some(ref mut node_iter) = self.node_iter {
                let parent_env = self.parent_env.as_ref().unwrap();
                let mut node_ids: Vec<NodeId> = Vec::with_capacity(BATCH_SIZE);

                for id in node_iter.by_ref() {
                    node_ids.push(id);
                    if node_ids.len() >= BATCH_SIZE {
                        break;
                    }
                }

                if !node_ids.is_empty() {
                    let batch_len = node_ids.len();
                    // Build output batch: clone parent env for each row, set node column
                    let mut envs = Vec::with_capacity(batch_len);
                    let alias = &self.node_pattern.alias;
                    for id in node_ids {
                        let mut row = parent_env.clone_pooled(self.runtime.env_pool);
                        row.insert(alias, Value::Node(id));
                        envs.push(row);
                    }
                    return Some(Ok(Batch::from_envs(envs)));
                }

                // Node iterator exhausted; fall through to get next parent row
                self.node_iter = None;
                self.parent_env = None;
            }

            // Get the next parent row
            loop {
                // If we have a parent batch, try the next row
                if let Some(ref parent_batch) = self.parent_batch {
                    if self.parent_row < parent_batch.len() {
                        let env = parent_batch.env_ref(self.parent_row);
                        self.parent_row += 1;
                        let g = self.runtime.g.borrow();
                        let graph_iter = g.get_nodes(&self.node_pattern.labels, 0);
                        // When inside a MERGE match sub-plan, include pending-created
                        // nodes and exclude pending-deleted nodes so MERGE can see
                        // in-flight mutations from prior clauses.
                        if self.include_pending {
                            let pending = self.runtime.pending.borrow();
                            let label_ids: Vec<_> = self
                                .node_pattern
                                .labels
                                .iter()
                                .filter_map(|l| g.get_label_id(l))
                                .collect();
                            let pending_nodes = if label_ids.len() == self.node_pattern.labels.len()
                            {
                                pending.get_pending_nodes_with_labels(&label_ids)
                            } else {
                                Vec::new()
                            };
                            let deleted_nodes = pending.deleted_nodes();
                            drop(pending);
                            drop(g);
                            let filtered_graph_iter =
                                graph_iter.filter(move |id| !deleted_nodes.contains((*id).into()));
                            self.parent_env = Some(env.clone_pooled(self.runtime.env_pool));
                            self.node_iter = Some(Box::new(
                                filtered_graph_iter.chain(pending_nodes.into_iter()),
                            ));
                        } else {
                            drop(g);
                            self.parent_env = Some(env.clone_pooled(self.runtime.env_pool));
                            self.node_iter = Some(graph_iter);
                        }
                        break;
                    }
                    // Parent batch exhausted
                    self.parent_batch = None;
                    self.parent_row = 0;
                }

                // Pull next parent batch from child
                match self.child.next() {
                    Some(Ok(batch)) => {
                        self.parent_batch = Some(batch);
                        self.parent_row = 0;
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None,
                }
            }
        }
    }
}
