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
    batch::{BATCH_SIZE, Batch, BatchBuilder, BatchOp, BatchRow},
    row::{Row, RowView},
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
    parent_env: Option<Row>,
    /// Iterator over node IDs for the current parent row.
    node_iter: Option<Box<dyn Iterator<Item = NodeId> + 'a>>,
    node_pattern: &'a QueryNode<Arc<String>, Variable>,
    pub(crate) idx: NodeIdx<Dyn<IR>>,
}

impl<'a> NodeByLabelScanOp<'a> {
    pub fn new(
        runtime: &'a Runtime<'a>,
        child: Box<BatchOp<'a>>,
        node_pattern: &'a QueryNode<Arc<String>, Variable>,
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
                    let alias_id = self.node_pattern.alias.id;
                    // Fast path: when the parent row carries no bindings, emit a
                    // columnar batch of node IDs and skip building one env per
                    // node. Downstream operators materialize envs/properties
                    // only for the rows they actually need.
                    if !parent_env.has_bindings() {
                        return Some(Ok(Batch::from_node_ids(alias_id, node_ids)));
                    }
                    // Build output batch: clone parent env for each row, set node column
                    let mut builder = BatchBuilder::new();
                    for id in node_ids {
                        builder.push_row(&parent_env.clone_with(alias_id, Value::Node(id)));
                    }
                    return Some(Ok(builder.finish()));
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
                        let env = BatchRow::new(parent_batch, self.parent_row).to_owned_row();
                        self.parent_row += 1;
                        let g = self.runtime.g.borrow();
                        let graph_iter = g.get_nodes(&self.node_pattern.labels, 0);
                        drop(g);
                        self.parent_env = Some(env);
                        self.node_iter = Some(graph_iter);
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
