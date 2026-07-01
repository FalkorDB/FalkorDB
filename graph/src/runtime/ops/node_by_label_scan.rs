//! Batch-mode label scan operator — iterates all nodes with a given label.
//!
//! For each active parent row, queues an iterator over the label's node IDs and
//! defers to the shared [`BatchedResultEmitter`], which packs up to
//! [`BATCH_SIZE`](super::super::batch::BATCH_SIZE) `(parent_row, node_id)` pairs
//! into one columnar batch. This avoids cloning the parent env per node — the
//! parent columns are replicated once per batch via `gather`.
//!
//! ```text
//!  parent BatchOp ──► parent_batch ──► BatchedResultEmitter::seed
//!                          │
//!             for each active parent row (on demand):
//!               g.get_nodes(labels) ──► RowIter::many(iter)
//!                          │
//!              ┌───────────┴───────────┐
//!              │  emit_lazy: pack ≤    │
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

use super::batched_result_emitter::{BatchedResultEmitter, RowIter};

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
            // The lazy emit builds one `get_nodes` iterator at a time as it walks
            // the active parent rows, so no per-row iterator is queued up front.
            // When the batch is exhausted (`Ok(None)`), pull and seed the next
            // child batch.
            let labels = &self.node_pattern.labels;
            let runtime = self.runtime;
            match self.emitter.emit_lazy(|_b, _row| {
                Ok(Some(RowIter::many(runtime.g.borrow().get_nodes(labels, 0))))
            }) {
                Ok(Some(out)) => return Some(Ok(out)),
                Ok(None) => match self.child.next() {
                    Some(Ok(batch)) => self.emitter.seed(batch),
                    Some(Err(e)) => return Some(Err(e)),
                    None => return None,
                },
                Err(e) => return Some(Err(e)),
            }
        }
    }
}
