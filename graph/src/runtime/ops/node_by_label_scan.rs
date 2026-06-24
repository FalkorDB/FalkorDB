//! Batch-mode label scan operator — iterates all nodes with a given label.
//!
//! For each active parent row, queues an iterator over the label's node IDs and
//! defers to the shared [`BatchedResultEmitter`], which packs up to
//! [`BATCH_SIZE`](super::super::batch::BATCH_SIZE) `(parent_row, node_id)` pairs
//! into one columnar batch. This avoids cloning the parent env per node — the
//! parent columns are replicated once per batch via `gather`.
//!
//! ```text
//!  parent BatchOp ──► parent_batch ──► BatchedResultEmitter::set_batch
//!                          │
//!             for each active parent row:
//!               g.get_nodes(labels) ──► BatchedResultEmitter::push(row, iter)
//!                          │
//!              ┌───────────┴───────────┐
//!              │  emit(): pack ≤       │
//!              │  BATCH_SIZE node IDs  │
//!              └───────────┬───────────┘
//!                          │
//!          Batch { parent columns + Node(id) per row }
//!                          │
//!                    yield Batch ──► parent
//! ```

use std::sync::Arc;

use crate::graph::graph::NodeId;
use crate::parser::ast::{QueryNode, Variable};
use crate::planner::IR;
use crate::runtime::{
    batch::{Batch, BatchOp},
    runtime::Runtime,
};
use orx_tree::{Dyn, NodeIdx};

use super::batched_result_emitter::BatchedResultEmitter;

pub struct NodeByLabelScanOp<'a> {
    pub(crate) runtime: &'a Runtime<'a>,
    pub(crate) child: Box<BatchOp<'a>>,
    /// Holds the parent batch being expanded and the per-row node iterators, and
    /// performs the shared pack-and-gather emit.
    emitter: BatchedResultEmitter<'a, NodeId>,
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
            emitter: BatchedResultEmitter::new(node_pattern.alias.id),
            node_pattern,
            idx,
        }
    }
}

impl<'a> Iterator for NodeByLabelScanOp<'a> {
    type Item = Result<Batch<'a>, String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Refill the per-row scans from the child when we've run dry. Every
            // active parent row expands to the full label node set, so queue one
            // `get_nodes` iterator per row.
            if self.emitter.needs_refill() {
                match self.child.next() {
                    Some(Ok(batch)) => {
                        for row in batch.active_indices() {
                            let iter = self
                                .runtime
                                .g
                                .borrow()
                                .get_nodes(&self.node_pattern.labels, 0);
                            self.emitter.push(row, iter);
                        }
                        self.emitter.set_batch(batch);
                        continue;
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None,
                }
            }

            if let Some(out) = self.emitter.emit() {
                return Some(Ok(out));
            }
        }
    }
}
